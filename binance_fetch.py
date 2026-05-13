#!/usr/bin/env python3
"""
Binance OHLCV klines fetcher.

Usage (library):
    from binance_fetch import fetch_klines_1m
    df = fetch_klines_1m("ACTUSDT", start_dt, end_dt, interval="3m")

Usage (CLI):
    python binance_fetch.py --symbol ACTUSDT --start 2025-11-01 --end 2025-12-01 --interval 3m

Columns returned: open_time, open, high, low, close, volume
  - open_time is UTC-aware datetime
  - numeric columns are float64

Supported intervals: 1m, 3m  (extend _INTERVAL_MS_MAP for others).
"""

import argparse
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

BINANCE_API_BASE = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
MAX_LIMIT = 1000          # Binance max records per request
REQUEST_DELAY_SEC = 0.2   # seconds between paginated requests (avoid rate-limit)

# Bar duration in milliseconds for each supported interval
_INTERVAL_MS_MAP: dict[str, int] = {
    "1m":  1 * 60_000,
    "3m":  3 * 60_000,
    "5m":  5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h":  60 * 60_000,
}

SUPPORTED_INTERVALS = ("1m", "3m")


def _interval_to_ms(interval: str) -> int:
    """Return the duration of one bar in milliseconds for *interval*.

    Falls back to 60 000 ms (1 minute) for unknown intervals so that
    existing callers that pass arbitrary intervals don't break.
    """
    return _INTERVAL_MS_MAP.get(interval, 60_000)


def fetch_klines_1m(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    interval: str = "1m",
) -> pd.DataFrame:
    """
    Fetch OHLCV klines from Binance for the given symbol and time range.

    Args:
        symbol:    Binance symbol, e.g. "ACTUSDT"
        start_dt:  Start datetime (UTC-aware or naive UTC, inclusive)
        end_dt:    End datetime   (UTC-aware or naive UTC, exclusive)
        interval:  Kline interval (default "1m")

    Returns:
        DataFrame with columns: open_time, open, high, low, close, volume
        open_time is UTC-aware datetime64[ns, UTC].
    """

    def _to_ms(dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    start_ms = _to_ms(start_dt)
    end_ms   = _to_ms(end_dt)

    print(
        f"Fetching {symbol} {interval} klines "
        f"from {start_dt.isoformat()} to {end_dt.isoformat()} ..."
    )

    all_rows: list = []
    current_ms = start_ms

    while current_ms < end_ms:
        params = {
            "symbol":    symbol.upper(),
            "interval":  interval,
            "startTime": current_ms,
            "endTime":   end_ms - 1,   # subtract 1 ms to convert exclusive end_dt to inclusive API endTime
            "limit":     MAX_LIMIT,
        }

        resp = requests.get(
            BINANCE_API_BASE + KLINES_ENDPOINT,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            break

        all_rows.extend(rows)

        # Advance to the bar *after* the last returned bar
        last_open_ms = rows[-1][0]
        current_ms = last_open_ms + _interval_to_ms(interval)

        if len(rows) < MAX_LIMIT:
            break  # no more data in range

        time.sleep(REQUEST_DELAY_SEC)

    if not all_rows:
        raise ValueError(
            f"No klines returned for {symbol} between {start_dt} and {end_dt}. "
            "Check symbol name and date range."
        )

    # Binance kline column order:
    # [0]open_time [1]open [2]high [3]low [4]close [5]volume
    # [6]close_time [7]quote_vol [8]num_trades
    # [9]taker_buy_base_vol [10]taker_buy_quote_vol [11]ignore
    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        ],
    )

    # Keep only the OHLCV columns expected by downstream scripts
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("open_time").reset_index(drop=True)

    print(f"Fetched {len(df)} bars for {symbol}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Binance OHLCV klines and print a CSV summary."
    )
    parser.add_argument("--symbol", required=True, help="Binance trading pair, e.g. ACTUSDT")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD", help="Start date (UTC, inclusive)")
    parser.add_argument("--end", required=True, metavar="YYYY-MM-DD", help="End date (UTC, exclusive)")
    parser.add_argument(
        "--interval",
        default="1m",
        choices=list(SUPPORTED_INTERVALS),
        help="Kline interval (default: 1m)",
    )
    cli_args = parser.parse_args()

    start_dt = datetime.strptime(cli_args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(cli_args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    result_df = fetch_klines_1m(cli_args.symbol, start_dt, end_dt, interval=cli_args.interval)
    print(result_df.to_string())
