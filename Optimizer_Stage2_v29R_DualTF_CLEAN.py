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
STAGE1B_CSV = RESULTS_DIR / "stage1B_behavior_top20.csv"   # <-- Top20 only
OUTPUT_CSV = RESULTS_DIR / "stage2_intraday_dual_tf_improved.csv"
INTRADAY_LOOKBACK_DAYS = 7
MIN_BARS = 800
PAUSE_SEC = 0.25

def _wilders_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder RMA (aka SMMA) to align with Stage1 and live engine."""
    return series.ewm(alpha=1 / length, adjust=False).mean()

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

def fetch_klines(client, symbol, interval, start, end, limit=1000):
    """Fetch klines for a symbol within the specified window (paged up to end)."""
    start_ms = int(pd.Timestamp(start).tz_convert("UTC").timestamp() * 1000) if pd.Timestamp(start).tzinfo else int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).tz_convert("UTC").timestamp() * 1000) if pd.Timestamp(end).tzinfo else int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    all_rows = []
    cur = start_ms

    while cur < end_ms:
        try:
            # Binance client supports start_str/end_str, but paging is more reliable using ms params.
            klines = client.get_klines(symbol=symbol, interval=interval, startTime=cur, endTime=end_ms, limit=limit)
        except Exception as e:
            print(f"Error fetching data for {symbol} {interval}: {e}")
            break

        if not klines:
            break

        all_rows.extend(klines)

        last_open = klines[-1][0]
        next_cur = last_open + 1  # advance at least 1ms to avoid repeating last candle
        if next_cur <= cur:
            break
        cur = next_cur

        time.sleep(PAUSE_SEC)

        if len(klines) < limit:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades",
                 "taker_base_vol", "taker_quote_vol", "ignore"]
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df

def calculate_coherence_score(df):
    """Calculate how often the symbol met funnel criteria during the window."""
    if len(df) < 200:
        return np.nan  # don't force-kill candidates just because bars are short

    c = df["close"]
    rsi = ta.rsi(c, length=14)
    rsi_sma = ta.sma(rsi, length=14)

    smma_200 = _wilders_rma(c, 200)

    rsi_condition = rsi_sma > 51
    smma_condition = c > smma_200

    coherence_score = (rsi_condition & smma_condition).mean()
    return float(coherence_score)

def trend_consistency(df):
    """Calculate the percentage of time price > SMMA(RMA) 200."""
    if len(df) < 200:
        return np.nan

    c = df["close"]
    smma_200 = _wilders_rma(c, 200)
    return float((c > smma_200).mean())

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
        coherence_1m = calculate_coherence_score(d1) if not d1.empty else np.nan
        coherence_3m = calculate_coherence_score(d3) if not d3.empty else np.nan
        trend_consistency_1m = trend_consistency(d1) if not d1.empty else np.nan
        trend_consistency_3m = trend_consistency(d3) if not d3.empty else np.nan

        # --- Gates ---
        INTEGRITY_MIN = 4.0
        COH_MIN_1M = 0.20
        COH_MIN_3M = 0.15

        def _passes_coherence(coh: float, min_coh: float) -> bool:
            """If coherence is NaN (unknown), do not reject. Otherwise require coh >= min."""
            if coh is None or (isinstance(coh, float) and np.isnan(coh)):
                return True
            return float(coh) >= float(min_coh)

        # Apply integrity + coherence gates
        if m1:
            if float(m1.get("integrity_ratio", 0.0)) < INTEGRITY_MIN:
                m1 = None
            elif not _passes_coherence(coherence_1m, COH_MIN_1M):
                m1 = None

        if m3:
            if float(m3.get("integrity_ratio", 0.0)) < INTEGRITY_MIN:
                m3 = None
            elif not _passes_coherence(coherence_3m, COH_MIN_3M):
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
    
    # ---------------------------------------------------------------------
    # Auto-run safety gate: drop symbols with incomplete downstream fields
    # ---------------------------------------------------------------------
    required_cols = [
        "symbol", "best_tf", "score_final",
        "suggested_sl", "suggested_trail",
        "integrity_ratio", "coherence_score", "trend_consistency",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[STAGE 2] Missing required columns for downstream: {missing}")

    before_n = len(df)
    df["_eligible_downstream"] = df[required_cols].notna().all(axis=1)

    ineligible_df = df.loc[~df["_eligible_downstream"], ["symbol", "best_tf", "score_final"] + required_cols]
    if len(ineligible_df) > 0:
        print(f"[STAGE 2][GATE] Dropping {len(ineligible_df)} ineligible symbols due to NaNs in required fields.")
        # optional: write a debug CSV
        ineligible_df.to_csv("results_v29R_30d/stage2_ineligible_symbols.csv", index=False)

    df = df.loc[df["_eligible_downstream"]].drop(columns=["_eligible_downstream"])
    after_n = len(df)
    print(f"[STAGE 2][GATE] Eligible symbols: {after_n}/{before_n}")
    
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
