"""Fetch NIFTY 200 constituents and save one point-in-time CSV per year.

NSE's public ``equity-stockIndices`` endpoint exposes the current composition,
not point-in-time historical compositions. Current year comes from NSE. Earlier
years use archived copies of NSE's official constituent CSV; the latest capture
inside each year is selected. Licensed NSE data remains the authoritative source.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener
from zoneinfo import ZoneInfo


NSE_HOME = "https://www.nseindia.com/"
NSE_API = "https://www.nseindia.com/api/equity-stockIndices"
NSE_LIVE_PAGE = "https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%20200"
NSE_CONSTITUENT_CSV = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"
NIFTY_INDICES_CSV = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
LEGACY_NSE_CSV = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
OLD_NSE_CSV = "https://www.nseindia.com/content/indices/ind_nifty200list.csv"
HISTORICAL_CSV_URLS = (
    NIFTY_INDICES_CSV,
    NSE_CONSTITUENT_CSV,
    LEGACY_NSE_CSV,
    OLD_NSE_CSV,
)
WAYBACK_CDX_PATTERNS = (
    "*.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    "*.nseindia.com/content/indices/ind_nifty200list.csv",
)
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK_REPLAY = "https://web.archive.org/web"
WAYBACK_AVAILABLE = "https://archive.org/wayback/available"
INDEX_NAME = "NIFTY 200"
EXPECTED_CONSTITUENTS = 200
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "NIFTY_200"
DEFAULT_LISTS_DIR = Path(__file__).resolve().parent / "NIFTY_LIST"
DEFAULT_TOP200_DIR = Path(__file__).resolve().parent / "NIFTY_200_FROM_MCAP"
DEFAULT_TICKER_DATA_DIR = Path(__file__).resolve().parent / "TICKER_DATA"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_HOME,
    "Connection": "keep-alive",
}


def _request_json_with_browser() -> dict[str, Any] | None:
    """Fetch API using a real browser TLS fingerprint when curl_cffi exists."""
    try:
        from curl_cffi import requests as browser_requests
    except ImportError:
        return None

    with browser_requests.Session(impersonate="chrome", timeout=30) as session:
        landing = session.get(NSE_LIVE_PAGE)
        landing.raise_for_status()
        response = session.get(
            NSE_API,
            params={"index": INDEX_NAME},
            headers={"Accept": "application/json, text/plain, */*", "Referer": NSE_LIVE_PAGE},
        )
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("NSE API returned a non-object JSON response")
    payload["_source"] = "NSE API"
    return payload


def _csv_payload(text: str, as_of: datetime, source: str) -> dict[str, Any]:
    """Convert an official NSE constituent CSV into API-shaped data."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    data = []
    for row in reader:
        data.append(
            {
                "symbol": row.get("Symbol", ""),
                "meta": {
                    "companyName": row.get("Company Name", ""),
                    "industry": row.get("Industry", ""),
                    "isin": row.get("ISIN Code", ""),
                },
            }
        )
    return {
        "timestamp": as_of.strftime("%d-%b-%Y %H:%M:%S"),
        "data": data,
        "_source": source,
    }


def _read_url(url: str, retries: int = 3, headers: dict[str, str] | None = None) -> str:
    """Read text URL with bounded retries."""
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    last_error: Exception | None = None
    request_headers = headers or HEADERS

    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers=request_headers)
            with opener.open(request, timeout=30) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset)
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)

    detail = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
    raise RuntimeError(
        f"Could not download {url} after {retries} attempts ({detail})"
    ) from last_error


def _request_official_csv() -> dict[str, Any]:
    """Fetch NSE's official current constituent file when API blocks access."""
    text = _read_url(NSE_CONSTITUENT_CSV)
    return _csv_payload(
        text,
        datetime.now(ZoneInfo("Asia/Kolkata")),
        "NSE official constituent CSV fallback",
    )


def _request_wayback_year(year: int) -> dict[str, Any]:
    """Fetch latest archived official NSE constituent snapshot for one year."""
    candidates: list[tuple[str, str]] = []
    errors: list[str] = []
    archive_headers = {"User-Agent": HEADERS["User-Agent"], "Accept": "application/json"}

    for archive_pattern in WAYBACK_CDX_PATTERNS:
        query = urlencode(
            {
                "url": archive_pattern,
                "from": year,
                "to": year,
                "output": "json",
                "fl": "timestamp,original,statuscode",
                "filter": "statuscode:200",
                "limit": "500",
            }
        )
        try:
            raw = _read_url(
                f"{WAYBACK_CDX}?{query}", retries=1, headers=archive_headers
            )
            result = json.loads(raw)
        except (RuntimeError, json.JSONDecodeError) as exc:
            errors.append(f"CDX {archive_pattern}: {exc}")
            continue
        if not isinstance(result, list) or len(result) < 2:
            continue
        for row in result[1:]:
            if isinstance(row, list) and len(row) >= 2 and str(row[0]).startswith(str(year)):
                candidates.append((str(row[0]), str(row[1])))

    # CDX is sometimes rate-limited. Availability API supplies nearest capture.
    if not candidates:
        for official_url in HISTORICAL_CSV_URLS:
            query = urlencode(
                {"url": official_url, "timestamp": f"{year}1231235959"}
            )
            try:
                raw = _read_url(
                    f"{WAYBACK_AVAILABLE}?{query}", retries=1, headers=archive_headers
                )
                result = json.loads(raw)
                closest = result.get("archived_snapshots", {}).get("closest", {})
                timestamp = str(closest.get("timestamp", ""))
                if closest.get("available") and timestamp.startswith(str(year)):
                    candidates.append((timestamp, official_url))
            except (RuntimeError, json.JSONDecodeError, AttributeError) as exc:
                errors.append(f"availability {official_url}: {exc}")

    if not candidates:
        detail = " | ".join(errors) if errors else "archive returned no captures"
        raise RuntimeError(f"No archived official NIFTY 200 snapshot for {year}. {detail}")

    for timestamp, original in sorted(set(candidates), reverse=True):
        try:
            replay_url = f"{WAYBACK_REPLAY}/{timestamp}id_/{original}"
            text = _read_url(
                replay_url,
                retries=2,
                headers={"User-Agent": HEADERS["User-Agent"], "Accept": "text/csv,*/*"},
            )
            as_of = datetime.strptime(timestamp[:14], "%Y%m%d%H%M%S").replace(
                tzinfo=ZoneInfo("Asia/Kolkata")
            )
            payload = _csv_payload(
                text, as_of, f"archived official NSE CSV ({original})"
            )
            extract_constituents(payload)  # Reject archive error pages/incomplete files.
            return payload
        except (RuntimeError, ValueError) as exc:
            errors.append(f"replay {timestamp}: {exc}")

    raise RuntimeError(
        f"Archived snapshots for {year} were unusable. " + " | ".join(errors)
    )


def _request_json() -> dict[str, Any]:
    """Prefer NSE API; fall back to NSE's official constituent download."""
    try:
        payload = _request_json_with_browser()
        if payload is not None:
            return payload
    except Exception as exc:
        print(f"NSE API blocked request ({exc}); using official CSV fallback.", file=sys.stderr)
    return _request_official_csv()


def _as_of_datetime(payload: dict[str, Any]) -> datetime:
    raw = str(payload.get("timestamp") or payload.get("timeStamp") or "").strip()
    for fmt in (
        "%d-%b-%Y %H:%M:%S",
        "%d-%b-%Y",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        except ValueError:
            continue
    return datetime.now(ZoneInfo("Asia/Kolkata"))


def extract_constituents(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten constituent membership fields and discard the index summary row."""
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("NSE response has no 'data' list")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip()
        if not symbol or symbol.upper() == INDEX_NAME or symbol in seen:
            continue

        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        rows.append(
            {
                "symbol": symbol,
                "company_name": str(meta.get("companyName") or item.get("companyName") or "").strip(),
                "industry": str(meta.get("industry") or item.get("industry") or "").strip(),
                "isin": str(meta.get("isin") or item.get("isin") or "").strip(),
            }
        )
        seen.add(symbol)

    rows.sort(key=lambda row: row["symbol"])
    if len(rows) != EXPECTED_CONSTITUENTS:
        raise ValueError(
            f"Expected {EXPECTED_CONSTITUENTS} constituents, got {len(rows)}; "
            "refusing to write an incomplete CSV"
        )
    return rows


def write_year_csv(
    rows: list[dict[str, str]], as_of: datetime, output_dir: Path
) -> Path:
    """Atomically write/refresh the snapshot for the response year."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"nifty_200_{as_of.year}.csv"
    temporary = destination.with_suffix(".csv.tmp")
    fieldnames = ["year", "as_of_date", "symbol", "company_name", "industry", "isin"]

    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        as_of_date = as_of.date().isoformat()
        for row in rows:
            writer.writerow({"year": as_of.year, "as_of_date": as_of_date, **row})
    temporary.replace(destination)
    return destination


def valid_existing_year_csv(path: Path, year: int) -> bool:
    """Return true only for a complete 200-symbol snapshot of requested year."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        symbols = {row.get("symbol", "").strip() for row in rows}
        years = {row.get("year", "").strip() for row in rows}
        return len(rows) == EXPECTED_CONSTITUENTS and len(symbols) == EXPECTED_CONSTITUENTS and years == {str(year)}
    except (OSError, csv.Error):
        return False


def _ranking_date(path: Path) -> date:
    """Extract market-cap measurement date from known NSE workbook names."""
    match = re.search(r"MCAP[_ ]?(\d{2})(\d{2})(\d{4})", path.stem, re.IGNORECASE)
    if match:
        return datetime.strptime("".join(match.groups()), "%d%m%Y").date()

    match = re.search(r"ToDecember\s*(\d{4})", path.stem, re.IGNORECASE)
    if match:
        return date(int(match.group(1)), 12, 31)

    raise ValueError(f"Cannot find ranking date in filename: {path.name}")


def build_top200_snapshots(lists_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Read ranked workbooks and persist top-200 membership snapshots."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Run: python -m pip install openpyxl") from exc

    files = sorted(lists_dir.glob("*.xlsx"))
    if not files:
        raise RuntimeError(f"No .xlsx market-cap lists found in {lists_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots: list[dict[str, Any]] = []
    for path in files:
        workbook = load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        header = [str(value or "").strip() for value in next(sheet.iter_rows(values_only=True))]
        normalized = [value.lower().replace(".", "").strip() for value in header]
        try:
            symbol_col = normalized.index("symbol")
            company_col = normalized.index("company name")
        except ValueError as exc:
            workbook.close()
            raise ValueError(f"Missing Symbol/Company Name columns in {path.name}") from exc

        ranking_date = _ranking_date(path)
        modified = workbook.properties.modified
        available_date = max(ranking_date, modified.date() if modified else ranking_date)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for values in sheet.iter_rows(values_only=True):
            rank = values[0] if len(values) > 0 else None
            symbol = str(values[symbol_col] or "").strip().upper()
            if not symbol or symbol in seen or not isinstance(rank, (int, float)):
                continue
            company = str(values[company_col] or "").strip()
            market_cap = values[3] if len(values) > 3 else None
            rows.append(
                {
                    "rank": int(rank),
                    "symbol": symbol,
                    "yahoo_symbol": f"{symbol}.NS",
                    "company_name": company,
                    "market_cap_lakhs": market_cap,
                }
            )
            seen.add(symbol)
        workbook.close()

        rows.sort(key=lambda row: row["rank"])
        rows = rows[:EXPECTED_CONSTITUENTS]
        if len(rows) != EXPECTED_CONSTITUENTS:
            raise ValueError(f"{path.name}: expected 200 ranked stocks, got {len(rows)}")

        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_")
        list_path = output_dir / f"{available_date.isoformat()}_{safe_stem}_top200.csv"
        temporary = list_path.with_suffix(".csv.tmp")
        fields = [
            "available_date",
            "ranking_date",
            "rank",
            "symbol",
            "yahoo_symbol",
            "company_name",
            "market_cap_lakhs",
            "source_file",
        ]
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "available_date": available_date.isoformat(),
                        "ranking_date": ranking_date.isoformat(),
                        **row,
                        "source_file": path.name,
                    }
                )
        temporary.replace(list_path)
        snapshots.append(
            {
                "available_date": available_date,
                "ranking_date": ranking_date,
                "source": path,
                "safe_stem": safe_stem,
                "rows": rows,
                "list_path": list_path,
            }
        )

    snapshots.sort(key=lambda item: (item["available_date"], item["source"].name))
    return snapshots


def _ticker_frame(download: Any, yahoo_symbol: str, ticker_count: int) -> Any:
    """Extract one ticker frame from yfinance single/multi-ticker output."""
    import pandas as pd

    if download is None or download.empty:
        return pd.DataFrame()
    if isinstance(download.columns, pd.MultiIndex):
        level_zero = download.columns.get_level_values(0)
        level_one = download.columns.get_level_values(1)
        if yahoo_symbol in level_zero:
            frame = download[yahoo_symbol].copy()
        elif yahoo_symbol in level_one:
            frame = download.xs(yahoo_symbol, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    elif ticker_count == 1:
        frame = download.copy()
    else:
        return pd.DataFrame()

    wanted = [name for name in ("Open", "High", "Low", "Close", "Adj Close", "Volume") if name in frame]
    if not wanted:
        return pd.DataFrame()
    frame = frame[wanted].dropna(how="all")
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index.name = "Date"
    return frame.sort_index()


def download_snapshot_prices(
    snapshots: list[dict[str, Any]], ticker_data_dir: Path, batch_size: int, refresh: bool
) -> tuple[int, list[dict[str, str]]]:
    """Download union once, then save unbiased [list date, next list date) slices."""
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Run: python -m pip install yfinance pandas"
        ) from exc

    today_end = datetime.now(ZoneInfo("Asia/Kolkata")).date() + timedelta(days=1)
    tasks: dict[str, list[tuple[dict[str, Any], date, date, Path]]] = {}
    for index, snapshot in enumerate(snapshots):
        start = snapshot["available_date"]
        end = snapshots[index + 1]["available_date"] if index + 1 < len(snapshots) else today_end
        if end <= start:
            continue
        period_dir = ticker_data_dir / (
            f"{start.isoformat()}_to_{(end - timedelta(days=1)).isoformat()}_"
            f"{snapshot['safe_stem']}"
        )
        period_dir.mkdir(parents=True, exist_ok=True)
        for row in snapshot["rows"]:
            symbol = row["symbol"]
            safe_symbol = re.sub(r'[<>:"/\\|?*]+', "_", symbol)
            destination = period_dir / f"{safe_symbol}.csv"
            if not refresh and destination.exists() and destination.stat().st_size > 20:
                continue
            tasks.setdefault(row["yahoo_symbol"], []).append(
                (snapshot, start, end, destination)
            )

    if not tasks:
        print("All requested daily ticker files already exist.")
        return 0, []

    symbols = sorted(tasks)
    overall_start = min(task[1] for symbol_tasks in tasks.values() for task in symbol_tasks)
    overall_end = max(task[2] for symbol_tasks in tasks.values() for task in symbol_tasks)
    failures: list[dict[str, str]] = []
    saved = 0

    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset : offset + batch_size]
        print(f"Downloading daily data: {offset + 1}-{offset + len(batch)} of {len(symbols)}")
        try:
            downloaded = yf.download(
                tickers=batch,
                start=overall_start.isoformat(),
                end=overall_end.isoformat(),
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                repair=True,
                keepna=False,
                threads=min(8, len(batch)),
                progress=False,
                timeout=30,
                multi_level_index=True,
            )
        except Exception as exc:
            print(f"Batch failed ({exc}); retrying tickers individually.", file=sys.stderr)
            downloaded = pd.DataFrame()

        for yahoo_symbol in batch:
            frame = _ticker_frame(downloaded, yahoo_symbol, len(batch))
            if frame.empty:
                try:
                    retry = yf.download(
                        tickers=[yahoo_symbol],
                        start=overall_start.isoformat(),
                        end=overall_end.isoformat(),
                        interval="1d",
                        group_by="ticker",
                        auto_adjust=False,
                        actions=False,
                        repair=True,
                        keepna=False,
                        threads=False,
                        progress=False,
                        timeout=30,
                        multi_level_index=True,
                    )
                    frame = _ticker_frame(retry, yahoo_symbol, 1)
                except Exception as exc:
                    failures.append({"ticker": yahoo_symbol, "reason": str(exc)})
                    continue
            if frame.empty:
                failures.append({"ticker": yahoo_symbol, "reason": "Yahoo returned no data"})
                continue
            for snapshot, start, end, destination in tasks[yahoo_symbol]:
                mask = (frame.index.date >= start) & (frame.index.date < end)
                sliced = frame.loc[mask]
                if sliced.empty:
                    failures.append(
                        {
                            "ticker": yahoo_symbol,
                            "reason": f"No rows for {start.isoformat()} to {end.isoformat()}",
                        }
                    )
                    continue
                temporary = destination.with_suffix(".csv.tmp")
                sliced.to_csv(temporary, index=True, date_format="%Y-%m-%d")
                temporary.replace(destination)
                saved += 1
        time.sleep(1)

    ticker_data_dir.mkdir(parents=True, exist_ok=True)
    failure_path = ticker_data_dir / "download_failures.csv"
    pd.DataFrame(failures, columns=["ticker", "reason"]).to_csv(failure_path, index=False)
    return saved, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build top-200 market-cap snapshots and download unbiased daily prices."
    )
    current_year = datetime.now(ZoneInfo("Asia/Kolkata")).year
    parser.add_argument(
        "--mode",
        choices=("prices", "constituents"),
        default="prices",
        help="prices: process XLSX lists and Yahoo data; constituents: old NSE fetcher",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="First snapshot year (default: 2024)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=current_year,
        help=f"Last snapshot year (default: {current_year})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"CSV directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace existing valid output files",
    )
    parser.add_argument(
        "--lists-dir",
        type=Path,
        default=DEFAULT_LISTS_DIR,
        help=f"Ranked market-cap XLSX directory (default: {DEFAULT_LISTS_DIR})",
    )
    parser.add_argument(
        "--top200-dir",
        type=Path,
        default=DEFAULT_TOP200_DIR,
        help=f"Generated top-200 lists directory (default: {DEFAULT_TOP200_DIR})",
    )
    parser.add_argument(
        "--ticker-data-dir",
        type=Path,
        default=DEFAULT_TICKER_DATA_DIR,
        help=f"Daily ticker CSV directory (default: {DEFAULT_TICKER_DATA_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Yahoo tickers per request batch (default: 25)",
    )
    return parser.parse_args()


def fetch_constituent_years(args: argparse.Namespace) -> int:
    """Legacy NSE/archived constituent-list workflow."""
    current_year = datetime.now(ZoneInfo("Asia/Kolkata")).year
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")
    if args.end_year > current_year:
        raise SystemExit(f"--end-year cannot exceed current year ({current_year})")

    failures: list[str] = []
    output_dir = args.output_dir.resolve()
    for year in range(args.start_year, args.end_year + 1):
        existing = output_dir / f"nifty_200_{year}.csv"
        if not args.refresh and valid_existing_year_csv(existing, year):
            print(f"Kept existing valid file: {existing}")
            continue
        try:
            payload = _request_json() if year == current_year else _request_wayback_year(year)
            rows = extract_constituents(payload)
            as_of = _as_of_datetime(payload)
            destination = write_year_csv(rows, as_of, output_dir)
            print(
                f"Saved {len(rows)} constituents to {destination} "
                f"(as of {as_of.date()}, source: {payload.get('_source', 'NSE')})"
            )
        except Exception as exc:
            failures.append(f"{year}: {exc}")
            print(f"FAILED {year}: {exc}", file=sys.stderr)

    if failures:
        print("Failed years: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "constituents":
        return fetch_constituent_years(args)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    snapshots = build_top200_snapshots(
        args.lists_dir.resolve(), args.top200_dir.resolve()
    )
    print(f"Built {len(snapshots)} point-in-time top-200 lists.")
    for index, snapshot in enumerate(snapshots):
        end = (
            snapshots[index + 1]["available_date"]
            if index + 1 < len(snapshots)
            else datetime.now(ZoneInfo("Asia/Kolkata")).date() + timedelta(days=1)
        )
        print(
            f"{snapshot['source'].name}: ranking {snapshot['ranking_date']}, "
            f"available {snapshot['available_date']}, valid until {end} (exclusive)"
        )

    saved, failures = download_snapshot_prices(
        snapshots,
        args.ticker_data_dir.resolve(),
        args.batch_size,
        args.refresh,
    )
    print(f"Saved {saved} daily ticker files. Failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
