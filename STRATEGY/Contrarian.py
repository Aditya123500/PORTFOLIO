"""Point-in-time cross-sectional contrarian momentum strategy.

For every historical universe period, rank stocks by trailing close-to-close
return and buy the worst performers. A signal calculated after close on day t
is executable only on next trading day t+1.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "DATA" / "TICKER_DATA"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "OUTPUT" / "CONTRARIAN"
PERIOD_PATTERN = re.compile(
    r"^(?P<start>\d{4}-\d{2}-\d{2})_to_(?P<end>\d{4}-\d{2}-\d{2})_(?P<name>.+)$"
)


@dataclass(frozen=True)
class Period:
    path: Path
    start: pd.Timestamp
    end: pd.Timestamp
    name: str


def discover_periods(data_dir: Path) -> list[Period]:
    """Discover and validate non-overlapping point-in-time universe folders."""
    periods: list[Period] = []
    for path in data_dir.iterdir():
        if not path.is_dir():
            continue
        match = PERIOD_PATTERN.match(path.name)
        if not match:
            continue
        start = pd.Timestamp(match.group("start"))
        end = pd.Timestamp(match.group("end"))
        if end < start:
            raise ValueError(f"Invalid period dates: {path.name}")
        periods.append(Period(path, start, end, match.group("name")))

    periods.sort(key=lambda period: period.start)
    if not periods:
        raise RuntimeError(f"No dated ticker-data folders found in {data_dir}")
    for previous, current in zip(periods, periods[1:]):
        if current.start <= previous.end:
            raise ValueError(
                f"Overlapping universe periods: {previous.path.name} and {current.path.name}"
            )
    return periods


def load_close_panel(period: Period, price_column: str) -> pd.DataFrame:
    """Load prices while rejecting rows outside folder's declared date range."""
    series: dict[str, pd.Series] = {}
    for path in sorted(period.path.glob("*.csv")):
        try:
            frame = pd.read_csv(path, usecols=["Date", price_column])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
        frame = frame.dropna(subset=["Date", price_column]).drop_duplicates(
            subset="Date", keep="last"
        )
        if frame.empty:
            continue
        outside = (frame["Date"] < period.start) | (frame["Date"] > period.end)
        if outside.any():
            raise ValueError(f"{path}: contains prices outside declared universe period")
        series[path.stem] = frame.set_index("Date")[price_column].sort_index()

    if not series:
        return pd.DataFrame()
    panel = pd.concat(series, axis=1).sort_index()
    panel.index.name = "Date"
    return panel


def period_signals(
    period: Period,
    prices: pd.DataFrame,
    lookback: int,
    long_count: int,
    rebalance_every: int,
) -> list[dict[str, object]]:
    """Calculate close-t signals, stamped for execution on next trading day."""
    if prices.empty:
        return []

    # Explicit shift: score on t uses only prices at t and t-lookback.
    momentum = prices.divide(prices.shift(lookback)).subtract(1.0)
    dates = prices.index[
        (prices.index >= period.start) & (prices.index <= period.end)
    ].unique().sort_values()
    signals: list[dict[str, object]] = []

    # Last period date cannot generate an order: no in-period next trading day.
    for location in range(0, len(dates) - 1, rebalance_every):
        signal_date = dates[location]
        execution_date = dates[location + 1]
        if execution_date > period.end:
            continue

        scores = momentum.loc[signal_date].dropna()
        if len(scores) < long_count:
            continue

        ranked = (
            scores.rename("momentum")
            .rename_axis("symbol")
            .reset_index()
            .sort_values(["momentum", "symbol"], ascending=[True, True])
        )
        losers = ranked.head(long_count)

        for cross_rank, row in enumerate(losers.itertuples(index=False), start=1):
            signals.append(
                {
                    "period_start": period.start.date().isoformat(),
                    "period_end": period.end.date().isoformat(),
                    "universe": period.name,
                    "signal_date": signal_date.date().isoformat(),
                    "execution_date": execution_date.date().isoformat(),
                    "symbol": row.symbol,
                    "side": "LONG",
                    "position": 1,
                    "target_weight": 1.0 / long_count,
                    "momentum_return": row.momentum,
                    "cross_section_rank": cross_rank,
                    "lookback_days": lookback,
                }
            )
    return signals


def generate_signals(
    data_dir: Path,
    output_dir: Path,
    lookback: int,
    long_count: int,
    rebalance_every: int,
    price_column: str,
) -> tuple[Path, Path, int]:
    """Generate one signal ledger plus period diagnostics."""
    periods = discover_periods(data_dir)
    all_signals: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    # Build one chronological panel. Earlier-period prices provide warm-up for
    # later universes. No future rows are used by shift-based momentum.
    period_panels = [(period, load_close_panel(period, price_column)) for period in periods]
    nonempty = [panel for _, panel in period_panels if not panel.empty]
    if not nonempty:
        raise RuntimeError(f"No usable price data found in {data_dir}")
    history = pd.concat(nonempty, axis=0).sort_index()
    history = history[~history.index.duplicated(keep="last")]

    for period, current_prices in period_panels:
        # Only current list symbols are eligible. Their earlier observations may
        # warm up momentum, but rows after signal date remain inaccessible.
        active_symbols = list(current_prices.columns)
        prices = history.loc[: period.end, active_symbols]
        signals = period_signals(
            period,
            prices,
            lookback,
            long_count,
            rebalance_every,
        )
        all_signals.extend(signals)
        diagnostics.append(
            {
                "period_start": period.start.date().isoformat(),
                "period_end": period.end.date().isoformat(),
                "universe": period.name,
                "tickers_loaded": len(current_prices.columns),
                "trading_dates": len(current_prices.index),
                "rebalance_dates": len(signals) // long_count,
                "position_rows": len(signals),
            }
        )
        print(
            f"{period.start.date()} to {period.end.date()}: "
            f"{len(current_prices.columns)} tickers, "
            f"{len(signals) // long_count} rebalances, "
            f"{len(signals)} position rows"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = output_dir / "signals.csv"
    diagnostics_path = output_dir / "diagnostics.csv"
    signal_fields = [
        "period_start",
        "period_end",
        "universe",
        "signal_date",
        "execution_date",
        "symbol",
        "side",
        "position",
        "target_weight",
        "momentum_return",
        "cross_section_rank",
        "lookback_days",
    ]
    with signal_path.with_suffix(".csv.tmp").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=signal_fields)
        writer.writeheader()
        writer.writerows(all_signals)
    signal_path.with_suffix(".csv.tmp").replace(signal_path)

    pd.DataFrame(diagnostics).to_csv(
        diagnostics_path.with_suffix(".csv.tmp"), index=False
    )
    diagnostics_path.with_suffix(".csv.tmp").replace(diagnostics_path)
    return signal_path, diagnostics_path, len(all_signals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Point-in-time long-only contrarian momentum signals."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--long-count", type=int, default=10)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument(
        "--price-column", choices=("Close", "Adj Close"), default="Adj Close"
    )
    args = parser.parse_args()
    if min(args.lookback, args.long_count, args.rebalance_every) < 1:
        parser.error("lookback, long-count, and rebalance-every must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    signal_path, diagnostics_path, count = generate_signals(
        args.data_dir.resolve(),
        args.output_dir.resolve(),
        args.lookback,
        args.long_count,
        args.rebalance_every,
        args.price_column,
    )
    selections_per_rebalance = args.long_count
    print(
        f"Saved {count} position rows across "
        f"{count // selections_per_rebalance} rebalance dates: {signal_path}"
    )
    print(f"Diagnostics: {diagnostics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
