"""Event-driven backtester for STRATEGY/Contrarian.py signal ledger.

Signals are decided at close t and executed at open t+1. Every rebalance first
closes all existing positions at that open, then opens the new basket. Positions
removed from the point-in-time top-200 universe are closed at the first market
open on/after the new list's effective date. Missing execution prices are never
silently replaced with a previous close.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNALS = ROOT / "STRATEGY" / "OUTPUT" / "CONTRARIAN" / "signals.csv"
DEFAULT_PRICE_DIR = ROOT / "DATA" / "TICKER_DATA"
DEFAULT_MEMBERSHIP_DIR = ROOT / "DATA" / "NIFTY_200_FROM_MCAP"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "OUTPUT" / "CONTRARIAN"


@dataclass
class Position:
    symbol: str
    side: int
    quantity: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_notional: float
    entry_cost: float
    universe: str


def load_signals(path: Path) -> pd.DataFrame:
    required = {
        "period_start",
        "period_end",
        "universe",
        "signal_date",
        "execution_date",
        "symbol",
        "position",
    }
    signals = pd.read_csv(path)
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"Signal file missing columns: {sorted(missing)}")
    for column in ("period_start", "period_end", "signal_date", "execution_date"):
        signals[column] = pd.to_datetime(signals[column], errors="raise")
    signals["symbol"] = signals["symbol"].astype(str).str.strip().str.upper()
    signals["position"] = pd.to_numeric(signals["position"], errors="raise").astype(int)

    if not (signals["execution_date"] > signals["signal_date"]).all():
        raise ValueError("Lookahead violation: execution_date must be after signal_date")
    inside = (signals["signal_date"] >= signals["period_start"]) & (
        signals["execution_date"] <= signals["period_end"]
    )
    if not inside.all():
        raise ValueError("Signal dates cross a declared membership boundary")
    if signals.duplicated(["execution_date", "symbol"]).any():
        raise ValueError("Duplicate symbol on same execution date")
    if not signals["position"].isin((-1, 1)).all():
        raise ValueError("Position must be -1 or 1")
    return signals.sort_values(["execution_date", "symbol"]).reset_index(drop=True)


def load_memberships(directory: Path) -> dict[pd.Timestamp, set[str]]:
    memberships: dict[pd.Timestamp, set[str]] = {}
    for path in sorted(directory.glob("*_top200.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if len(rows) != 200:
            raise ValueError(f"{path}: expected 200 membership rows, got {len(rows)}")
        dates = {row["available_date"] for row in rows}
        if len(dates) != 1:
            raise ValueError(f"{path}: inconsistent available_date")
        effective = pd.Timestamp(dates.pop())
        symbols = {row["symbol"].strip().upper() for row in rows}
        if len(symbols) != 200:
            raise ValueError(f"{path}: membership symbols are not unique")
        memberships[effective] = symbols
    if not memberships:
        raise RuntimeError(f"No top-200 membership files found in {directory}")
    return dict(sorted(memberships.items()))


def load_prices(directory: Path) -> dict[str, pd.DataFrame]:
    """Load adjusted open/close series consistently across period folders."""
    pieces: dict[str, list[pd.DataFrame]] = {}
    for path in directory.glob("*/*.csv"):
        try:
            frame = pd.read_csv(
                path, usecols=["Date", "Open", "Close", "Adj Close"]
            )
        except (ValueError, pd.errors.EmptyDataError):
            continue
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        for column in ("Open", "Close", "Adj Close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["Date", "Open", "Close", "Adj Close"])
        frame = frame[(frame["Open"] > 0) & (frame["Close"] > 0) & (frame["Adj Close"] > 0)]
        if frame.empty:
            continue
        factor = frame["Adj Close"] / frame["Close"]
        frame = pd.DataFrame(
            {
                "Date": frame["Date"],
                "Open": frame["Open"] * factor,
                "Close": frame["Adj Close"],
            }
        )
        pieces.setdefault(path.stem.upper(), []).append(frame)

    prices: dict[str, pd.DataFrame] = {}
    for symbol, frames in pieces.items():
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates("Date", keep="last").set_index("Date").sort_index()
        prices[symbol] = combined
    if not prices:
        raise RuntimeError(f"No ticker prices found in {directory}")
    return prices


def load_supplemental_cache(path: Path) -> dict[tuple[str, pd.Timestamp], float]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    frame["Date"] = pd.to_datetime(frame["Date"])
    return {
        (row.Symbol.upper(), row.Date): float(row.Open)
        for row in frame.itertuples(index=False)
    }


def save_supplemental_cache(
    path: Path, cache: dict[tuple[str, pd.Timestamp], float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"Symbol": symbol, "Date": day.date().isoformat(), "Open": price}
        for (symbol, day), price in sorted(cache.items(), key=lambda item: (item[0][1], item[0][0]))
    ]
    pd.DataFrame(rows, columns=["Symbol", "Date", "Open"]).to_csv(path, index=False)


def fetch_exact_adjusted_open(symbol: str, day: pd.Timestamp) -> float | None:
    """Fetch exact missing open; reject later-day or prior-close substitutions."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Missing yfinance. Run: python -m pip install yfinance") from exc

    data = yf.download(
        tickers=[f"{symbol}.NS"],
        start=day.date().isoformat(),
        end=(day.date() + timedelta(days=5)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        repair=True,
        threads=False,
        progress=False,
        timeout=30,
        multi_level_index=False,
    )
    if data is None or data.empty:
        return None
    data.index = pd.to_datetime(data.index).tz_localize(None)
    if day not in data.index:
        return None
    row = data.loc[day]
    raw_open = float(row["Open"])
    raw_close = float(row["Close"])
    adjusted_close = float(row["Adj Close"])
    if min(raw_open, raw_close, adjusted_close) <= 0:
        return None
    return raw_open * adjusted_close / raw_close


def annualized_metrics(equity: pd.Series) -> dict[str, float | None]:
    returns = equity.pct_change().dropna()
    if equity.empty:
        return {"cagr": None, "annualized_volatility": None, "sharpe": None, "max_drawdown": None}
    elapsed_days = max((equity.index[-1] - equity.index[0]).days, 1)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0 if 1.0 + total_return > 0 else None
    volatility = float(returns.std(ddof=1) * math.sqrt(252)) if len(returns) > 1 else None
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else None
    )
    drawdown = equity / equity.cummax() - 1.0
    return {
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def run_backtest(
    signals: pd.DataFrame,
    memberships: dict[pd.Timestamp, set[str]],
    prices: dict[str, pd.DataFrame],
    output_dir: Path,
    trade_notional: float,
    transaction_cost_rate: float,
    initial_capital: float,
    allow_fractional: bool,
    fetch_missing: bool,
) -> dict[str, Any]:
    all_dates = sorted({day for frame in prices.values() for day in frame.index})
    calendar = pd.DatetimeIndex(all_dates)
    if calendar.empty:
        raise RuntimeError("Price calendar is empty")

    effective_memberships: dict[pd.Timestamp, set[str]] = {}
    for source_date, symbols in memberships.items():
        candidates = calendar[calendar >= source_date]
        if len(candidates):
            effective_memberships[candidates[0]] = symbols

    targets = {
        day: {row.symbol: int(row.position) for row in group.itertuples(index=False)}
        for day, group in signals.groupby("execution_date")
    }
    target_universe = {
        day: str(group["universe"].iloc[0])
        for day, group in signals.groupby("execution_date")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "supplemental_open_prices.csv"
    supplemental = load_supplemental_cache(cache_path)

    positions: dict[str, Position] = {}
    trades: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    realized_pnl = 0.0
    total_costs = 0.0
    current_membership: set[str] = set()

    def price(symbol: str, day: pd.Timestamp, field: str, permit_fetch: bool = False) -> float:
        frame = prices.get(symbol)
        if frame is not None and day in frame.index:
            value = frame.at[day, field]
            if pd.notna(value) and float(value) > 0:
                return float(value)
        key = (symbol, day)
        if field == "Open" and key in supplemental:
            return supplemental[key]
        if field == "Open" and permit_fetch and fetch_missing:
            fetched = fetch_exact_adjusted_open(symbol, day)
            if fetched is not None:
                supplemental[key] = fetched
                save_supplemental_cache(cache_path, supplemental)
                return fetched
        raise RuntimeError(
            f"Missing exact {field} for {symbol} on {day.date()}; "
            "refusing prior-close substitution"
        )

    def close_position(symbol: str, day: pd.Timestamp, reason: str, permit_fetch: bool) -> None:
        nonlocal realized_pnl, total_costs
        position = positions[symbol]
        exit_price = price(symbol, day, "Open", permit_fetch=permit_fetch)
        positions.pop(symbol)
        exit_notional = position.quantity * exit_price
        exit_cost = exit_notional * transaction_cost_rate
        gross_pnl = position.side * position.quantity * (exit_price - position.entry_price)
        net_pnl = gross_pnl - position.entry_cost - exit_cost
        realized_pnl += net_pnl
        total_costs += exit_cost
        trades.append(
            {
                **asdict(position),
                "side": "LONG" if position.side == 1 else "SHORT",
                "exit_date": day.date().isoformat(),
                "exit_price": exit_price,
                "exit_notional": exit_notional,
                "exit_cost": exit_cost,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "holding_days": (day - position.entry_date).days,
                "exit_reason": reason,
            }
        )

    for day in calendar:
        if day in effective_memberships:
            new_membership = effective_memberships[day]
            for symbol in sorted(set(positions) - new_membership):
                close_position(symbol, day, "UNIVERSE_REMOVAL", permit_fetch=True)
            current_membership = new_membership

        if day in targets:
            desired = targets[day]
            invalid = set(desired) - current_membership
            if invalid:
                raise ValueError(
                    f"Signals use non-members on {day.date()}: {sorted(invalid)}"
                )

            # Full monthly reset: square off every surviving position before
            # opening the newly selected basket. Re-selected symbols therefore
            # incur both exit and new-entry costs at this execution open.
            for symbol in sorted(list(positions)):
                close_position(symbol, day, "REBALANCE_EXIT", permit_fetch=False)

            for symbol, side in sorted(desired.items()):
                entry_price = price(symbol, day, "Open", permit_fetch=False)
                quantity = trade_notional / entry_price
                if not allow_fractional:
                    quantity = math.floor(quantity)
                if quantity <= 0:
                    audits.append(
                        {
                            "date": day.date().isoformat(),
                            "symbol": symbol,
                            "event": "SKIPPED_ENTRY",
                            "detail": "price exceeds fixed trade notional",
                        }
                    )
                    continue
                entry_notional = quantity * entry_price
                entry_cost = entry_notional * transaction_cost_rate
                total_costs += entry_cost
                positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_date=day,
                    entry_price=entry_price,
                    entry_notional=entry_notional,
                    entry_cost=entry_cost,
                    universe=target_universe[day],
                )

        open_liquidation_pnl = 0.0
        long_exposure = 0.0
        short_exposure = 0.0
        valued_positions = 0
        for symbol, position in positions.items():
            frame = prices.get(symbol)
            if frame is None:
                continue
            known = frame.loc[:day, "Close"].dropna()
            if known.empty:
                continue
            # Stale close is allowed only for end-of-day valuation. Execution
            # prices above always require exact opens.
            close_price = float(known.iloc[-1])
            notional = position.quantity * close_price
            estimated_exit_cost = notional * transaction_cost_rate
            open_liquidation_pnl += (
                position.side * position.quantity * (close_price - position.entry_price)
                - position.entry_cost
                - estimated_exit_cost
            )
            if position.side == 1:
                long_exposure += notional
            else:
                short_exposure += notional
            valued_positions += 1
        curve.append(
            {
                "Date": day.date().isoformat(),
                "Equity": initial_capital + realized_pnl + open_liquidation_pnl,
                "RealizedPnL": realized_pnl,
                "OpenLiquidationPnL": open_liquidation_pnl,
                "LongExposure": long_exposure,
                "ShortExposure": short_exposure,
                "GrossExposure": long_exposure + short_exposure,
                "OpenPositions": len(positions),
                "ValuedPositions": valued_positions,
            }
        )

    save_supplemental_cache(cache_path, supplemental)
    trades_frame = pd.DataFrame(trades)
    curve_frame = pd.DataFrame(curve)
    curve_frame["Date"] = pd.to_datetime(curve_frame["Date"])
    curve_frame = curve_frame.set_index("Date")
    equity = curve_frame["Equity"]
    extras = annualized_metrics(equity)

    wins = int((trades_frame.get("net_pnl", pd.Series(dtype=float)) > 0).sum())
    losses = int((trades_frame.get("net_pnl", pd.Series(dtype=float)) < 0).sum())
    gross_profit = float(trades_frame.loc[trades_frame.get("net_pnl", 0) > 0, "net_pnl"].sum()) if not trades_frame.empty else 0.0
    gross_loss = float(trades_frame.loc[trades_frame.get("net_pnl", 0) < 0, "net_pnl"].sum()) if not trades_frame.empty else 0.0
    metrics: dict[str, Any] = {
        "start_date": equity.index[0].date().isoformat(),
        "end_date": equity.index[-1].date().isoformat(),
        "initial_capital": initial_capital,
        "final_liquidation_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / initial_capital - 1.0),
        **extras,
        "closed_trades": len(trades_frame),
        "open_positions_at_end": len(positions),
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": wins / len(trades_frame) if len(trades_frame) else None,
        "average_net_pnl": float(trades_frame["net_pnl"].mean()) if not trades_frame.empty else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss < 0 else None,
        "realized_net_pnl": realized_pnl,
        "total_transaction_costs_paid": total_costs,
        "universe_removal_exits": int((trades_frame.get("exit_reason", pd.Series(dtype=str)) == "UNIVERSE_REMOVAL").sum()),
        "skipped_entries": len(audits),
        "trade_notional": trade_notional,
        "transaction_cost_rate_each_end": transaction_cost_rate,
        "fractional_shares": allow_fractional,
        "pricing": "adjusted open for execution; adjusted close for marking",
        "short_borrow_cost_included": False,
    }

    trades_frame.to_csv(output_dir / "trades.csv", index=False)
    curve_frame.to_csv(output_dir / "equity_curve.csv", date_format="%Y-%m-%d")
    pd.DataFrame(audits, columns=["date", "symbol", "event", "detail"]).to_csv(
        output_dir / "audit.csv", index=False
    )
    pd.DataFrame([asdict(position) for position in positions.values()]).to_csv(
        output_dir / "open_positions.csv", index=False
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bias-controlled next-open backtest for contrarian signals."
    )
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--price-dir", type=Path, default=DEFAULT_PRICE_DIR)
    parser.add_argument("--membership-dir", type=Path, default=DEFAULT_MEMBERSHIP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trade-notional", type=float, default=10_000.0)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--initial-capital", type=float, default=400_000.0)
    parser.add_argument("--allow-fractional", action="store_true")
    parser.add_argument(
        "--no-fetch-missing",
        action="store_true",
        help="Fail instead of fetching missing exact universe-removal opens",
    )
    args = parser.parse_args()
    if min(args.trade_notional, args.initial_capital) <= 0:
        parser.error("trade-notional and initial-capital must be positive")
    if not 0 <= args.transaction_cost < 1:
        parser.error("transaction-cost must be in [0, 1)")
    return args


def main() -> int:
    args = parse_args()
    signals = load_signals(args.signals.resolve())
    memberships = load_memberships(args.membership_dir.resolve())
    prices = load_prices(args.price_dir.resolve())
    metrics = run_backtest(
        signals,
        memberships,
        prices,
        args.output_dir.resolve(),
        args.trade_notional,
        args.transaction_cost,
        args.initial_capital,
        args.allow_fractional,
        not args.no_fetch_missing,
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
