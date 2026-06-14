import os
import requests
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, timezone

BINANCE_BASE = "https://api.binance.com"
LIMIT = 1000  # Binance REST API max limit per fetch

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Fetches raw klines from Binance public REST API."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_ohlcv_binance(symbol: str, interval: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """Fetches, parses, and normalizes candlestick data for any interval."""
    start_utc = pd.to_datetime(start_utc, utc=True)
    end_utc = pd.to_datetime(end_utc, utc=True)

    start_ms = int(start_utc.value // 10**6)
    end_ms = int(end_utc.value // 10**6)

    rows = []
    cur = start_ms
    while cur < end_ms:
        data = fetch_klines(symbol, interval, cur, end_ms)
        if not data:
            break

        rows.extend(data)
        last_open = data[-1][0]
        cur = last_open + 1  # Paginate to the next millisecond to avoid duplicates

        time.sleep(0.25)
        if len(data) < LIMIT:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time_ms", "open", "high", "low", "close", "volume",
            "close_time_ms", "qav", "num_trades", "tb", "tq", "ignore"
        ],
    )
    df["time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    df = df.dropna().sort_values("time").reset_index(drop=True)
    return df

def main():
    symbol = "DYMUSDT"
    print("======================================================")
    print(f" 🚀 EXTRACTING FORENSIC DATA FOR {symbol}")
    print("======================================================")

    # --------------------------------------------------------
    # 1. Fetch 1-Hour (HTF) Data (For Swing Low/High pivots)
    # --------------------------------------------------------
    # Target Window: 2025-11-10 00:00:00 UTC to 2025-11-24 02:00:00 UTC
    htf_start = pd.Timestamp("2025-11-01 00:00:00", tz="UTC")
    htf_end = pd.Timestamp("2025-11-08 00:00:00", tz="UTC")
    
    print(f"[1/2] Fetching 1H data: {htf_start} -> {htf_end}")
    df_1h = get_ohlcv_binance(symbol, "1h", htf_start, htf_end)
    
    # Save 1H data to CSV
    file_1h = "dymusdt_1h_forensic.csv"
    df_1h.to_csv(file_1h, index=False)
    print(f"✅ Saved 1H data ({len(df_1h)} rows) to: {file_1h}")

    # --------------------------------------------------------
    # 2. Fetch 3-Minute (LTF) Data (For entry/exit execution)
    # --------------------------------------------------------
    # Target Window: 2025-11-24 00:00:00 UTC to 2025-11-27 10:00:00 UTC
    ltf_target_start = pd.Timestamp("2025-11-06 00:00:00", tz="UTC")
    ltf_end = pd.Timestamp("2025-11-08 00:00:00", tz="UTC")
    
    # Fetch starts 6 hours earlier for indicator warmup padding
    ltf_fetch_start = ltf_target_start - pd.Timedelta(hours=6)
    
    print(f"\n[2/2] Fetching 3m data (including warmup): {ltf_fetch_start} -> {ltf_end}")
    df_3m = get_ohlcv_binance(symbol, "3m", ltf_fetch_start, ltf_end)
    
    # Calculate indicators on the 3m LTF
    print("Calculating RSI, RSI_SMA, and EMA50...")
    df_3m["rsi"] = ta.rsi(df_3m["close"], length=14)
    df_3m["rsi_sma"] = ta.sma(df_3m["rsi"], length=14)
    df_3m["ema50"] = df_3m["close"].ewm(span=50, adjust=False).mean()
    
    # Slice DataFrame to only keep the target forensic window (after warmup is complete)
    df_3m_filtered = df_3m[df_3m["time"] >= ltf_target_start][
        ["time", "open", "high", "low", "close", "volume", "rsi", "rsi_sma", "ema50"]
    ].copy()
    
    # Save 3m data to CSV
    file_3m = "dymusdt_3m_forensic.csv"
    df_3m_filtered.to_csv(file_3m, index=False)
    print(f"✅ Saved 3m data ({len(df_3m_filtered)} rows) to: {file_3m}")
    print("\nExtraction complete! Ready for forensic review.")

if __name__ == "__main__":
    main()