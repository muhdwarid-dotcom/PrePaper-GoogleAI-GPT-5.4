import os
import time
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from binance.client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key=api_key, api_secret=api_secret)

# Configuration
RESULTS_DIR = Path("./results_v29R_30d")
STAGE1B_CSV = RESULTS_DIR / "stage1B_behavior.csv"
OUTPUT_CSV = RESULTS_DIR / "stage2_intraday_dual_tf_improved.csv"
INTRADAY_LOOKBACK_DAYS = 7
MIN_BARS = 800
PAUSE_SEC = 0.25

def _previous_monday(dt: datetime) -> datetime:
    """Return the most recent Monday <= dt (UTC-aware)."""
    days_back = (dt.weekday() - 0) % 7
    return (dt - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

def get_windows_from_manual_monday(prompt_msg: str = "Enter Monday date (UTC) [YYYY-MM-DD]: ") -> dict:
    """User supplies a Monday and we return windows using the convention."""
    while True:
        s = input(prompt_msg).strip()
        try:
            dt = datetime.fromisoformat(s)
            dt = dt.replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)
        except Exception:
            print("Invalid format. Use YYYY-MM-DD (e.g., 2025-12-01). Try again.")
            continue

        monday = _previous_monday(dt)
        if monday.date() != dt.date():
            print(f"Input {dt.date()} is not a Monday. Using previous Monday: {monday.date()} (UTC).")

        trade_end = monday
        trade_start = trade_end - timedelta(days=7)
        train_end = trade_start
        train_start = train_end - timedelta(days=30)

        return {
            "train_start": train_start,
            "train_end": train_end,
            "trade_start": trade_start,
            "trade_end": trade_end
        }

def fetch_klines(client, symbol, interval, start, end):
    """Fetch klines for a symbol within the specified window."""
    try:
        klines = client.get_historical_klines(
            symbol,
            interval,
            start.strftime("%d %b %Y %H:%M:%S"),
            end.strftime("%d %b %Y %H:%M:%S")
        )
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(
            klines,
            columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"]
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_coherence_score(df):
    """Calculate how often the symbol met funnel criteria during the training window."""
    if len(df) < 200:
        return 0

    df["rsi"] = ta.rsi(df["close"], length=14)
    df["rsi_sma"] = ta.sma(df["rsi"], length=14)
    df["smma_200"] = df["close"].rolling(window=200).mean()

    rsi_condition = df["rsi_sma"] > 51
    smma_condition = df["close"] > df["smma_200"]
    coherence_score = (rsi_condition & smma_condition).mean()  # Percentage of candles meeting both criteria
    return coherence_score

def trend_consistency(df):
    """Calculate the percentage of time price > SMMA 200."""
    if len(df) < 200:
        return 0
    smma_200 = df["close"].rolling(window=200).mean()
    return (df["close"] > smma_200).mean()

def micro_metrics(df):
    """Calculate microstructural metrics for Stage 2."""
    if len(df) < 200:
        return None

    c = df["close"]
    df["rsi_sma"] = ta.sma(ta.rsi(c, length=14), length=14)
    df["atr_val"] = ta.atr(df["high"], df["low"], c, length=14)

    # Identify journeys (RSI_SMA 51 to 51)
    df["trigger"] = (df["rsi_sma"] >= 51.0).astype(int)
    df["change"] = df["trigger"].diff()

    starts = df[df["change"] == 1].index
    mae_list = []
    mfe_list = []

    for start_idx in starts:
        future = df.loc[start_idx:]
        ends = future[future["rsi_sma"] < 51.0].index
        if not ends.empty:
            end_idx = ends[0]
            journey = df.loc[start_idx:end_idx]
            entry_price = journey["close"].iloc[0]
            entry_atr = journey["atr_val"].iloc[0]
            if entry_atr == 0:
                continue

            low_val = journey["low"].min()
            high_val = journey["high"].max()
            mae_list.append((entry_price - low_val) / entry_atr)
            mfe_list.append((high_val - entry_price) / entry_atr)

    if not mae_list:
        return None

    p80 = np.percentile(mae_list, 80)
    p50 = np.percentile(mae_list, 50)
    suggested_sl = p80 + 0.5
    suggested_trail = (p80 - p50) + 0.5
    avg_atr_pct = (df["atr_val"] / c).mean() * 100
    integrity_ratio = (np.mean(mfe_list) * avg_atr_pct) / 0.20

    return {
        "median_atr_pct": float(avg_atr_pct),
        "suggested_sl": round(float(suggested_sl), 2),
        "suggested_trail": round(float(suggested_trail), 2),
        "integrity_ratio": round(float(integrity_ratio), 2),
        "ema_trend_frac": (c > ta.sma(c, 200)).mean()
    }

def stage2_dual_tf_improved(client, train_start, train_end):
    """Improved Stage 2 with coherence and trend consistency metrics."""
    if not STAGE1B_CSV.exists():
        raise FileNotFoundError(STAGE1B_CSV)

    df1b = pd.read_csv(STAGE1B_CSV)
    symbols = df1b["symbol"].astype(str).tolist()
    print(f"\n[STAGE 2] Loaded {len(symbols)} symbols from Stage 1B.")

    scan_end = train_end
    scan_start = scan_end - timedelta(days=INTRADAY_LOOKBACK_DAYS)
    print(f"[STAGE 2] Scanning window: {scan_start} -> {scan_end}")

    out_rows = []
    for sym in symbols:
        time.sleep(PAUSE_SEC)

        # Fetch data for 1m and 3m
        d1 = fetch_klines(client, sym, "1m", scan_start, scan_end)
        d3 = fetch_klines(client, sym, "3m", scan_start, scan_end)

        if d1.empty and d3.empty:
            continue

        # Calculate micro metrics
        m1 = micro_metrics(d1) if len(d1) >= MIN_BARS else None
        m3 = micro_metrics(d3) if len(d3) >= MIN_BARS else None

        # Calculate coherence and trend consistency for both timeframes
        coherence_1m = calculate_coherence_score(d1)
        coherence_3m = calculate_coherence_score(d3)
        trend_consistency_1m = trend_consistency(d1)
        trend_consistency_3m = trend_consistency(d3)

        # Apply Zombie Gate and coherence filters
        if m1 and (m1["integrity_ratio"] < 4.0 or coherence_1m < 0.2):
            m1 = None
        if m3 and (m3["integrity_ratio"] < 4.0 or coherence_3m < 0.15):
            m3 = None

        if not m1 and not m3:
            continue

        rec = {"symbol": sym}
        if m1:
            for k, v in m1.items():
                rec[f"{k}_1m"] = v
            rec["coherence_1m"] = coherence_1m
            rec["trend_consistency_1m"] = trend_consistency_1m
        if m3:
            for k, v in m3.items():
                rec[f"{k}_3m"] = v
            rec["coherence_3m"] = coherence_3m
            rec["trend_consistency_3m"] = trend_consistency_3m

        out_rows.append(rec)

    df = pd.DataFrame(out_rows)
    if df.empty:
        raise RuntimeError("No valid intraday candidates.")

    # Calculate adjusted scores
    df["score_1m"] = (
        df.get("median_atr_pct_1m", 0) * df.get("coherence_1m", 0) * df.get("trend_consistency_1m", 0)
    )
    df["score_3m"] = (
        df.get("median_atr_pct_3m", 0) * df.get("coherence_3m", 0) * df.get("trend_consistency_3m", 0)
    )

    # Determine best timeframe
    df["best_tf"] = np.where(df["score_1m"] > df["score_3m"], "1m", "3m")
    df["score_final"] = df[["score_1m", "score_3m"]].max(axis=1)

    # Flatten multipliers for the best timeframe
    df["suggested_sl"] = np.where(df["best_tf"] == "1m", df["suggested_sl_1m"], df["suggested_sl_3m"])
    df["suggested_trail"] = np.where(df["best_tf"] == "1m", df["suggested_trail_1m"], df["suggested_trail_3m"])
    df["integrity_ratio"] = np.where(df["best_tf"] == "1m", df["integrity_ratio_1m"], df["integrity_ratio_3m"])
    df["coherence_score"] = np.where(df["best_tf"] == "1m", df["coherence_1m"], df["coherence_3m"])
    df["trend_consistency"] = np.where(df["best_tf"] == "1m", df["trend_consistency_1m"], df["trend_consistency_3m"])

    df = df.sort_values("score_final", ascending=False)
    OUTPUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[STAGE 2] Saved improved results to {OUTPUT_CSV}")
    print(df[["symbol", "best_tf", "score_final", "coherence_score", "trend_consistency"]].head(10).to_string(index=False))

    return df

def main():
    load_dotenv()
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        raise RuntimeError("Set BINANCE_API_KEY / BINANCE_API_SECRET in .env")

    # Get training window
    env_train_start = os.getenv("TRAIN_START")
    env_train_end = os.getenv("TRAIN_END")

    if env_train_start and env_train_end:
        print("[STAGE 2] AutoRun provided TRAIN window.")
        train_start = pd.to_datetime(env_train_start, utc=True)
        train_end = pd.to_datetime(env_train_end, utc=True)
    else:
        print("[STAGE 2] AutoRun vars missing → asking for manual Monday.")
        w = get_windows_from_manual_monday()
        train_start = pd.to_datetime(w["train_start"], utc=True)
        train_end = pd.to_datetime(w["train_end"], utc=True)

    print(f"\n[STAGE 2] TRAIN_START = {train_start}")
    print(f"[STAGE 2] TRAIN_END   = {train_end}")

    stage2_dual_tf_improved(client, train_start, train_end)

if __name__ == "__main__":
    main()
