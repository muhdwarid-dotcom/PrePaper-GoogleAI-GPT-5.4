#!/usr/bin/env python3
"""
Funnel Data V30 Event Study — one-pair-at-a-time CLI tool.

Fetches OHLCV data from Binance (1m or 3m), runs V30 funnel event-study logic,
and writes a window-stamped CSV for the requested pair.

USAGE (3-step pipeline for one pair):
  Step 1 — Generate events CSV:
    python Funnel_Data_Test_V30_EventStudy.py --pair ACTUSDT --prepaper-start 2025-12-01
    python Funnel_Data_Test_V30_EventStudy.py --pair ACTUSDT --prepaper-start 2025-12-01 --interval 3m

  Step 2 — Analyse events, generate candidates CSV:
    python eventstudy_analysis.py forwardtest/v30_eventstudy_ACTUSDT_1m_rsi_sma_cross_gt51_prepaper_2025-12-01.csv \\
        --pair ACTUSDT --prepaper-start 2025-12-01 --interval 1m --grid

  Step 3 — Derive k/t exit parameters:
    python Derive_k_t_from_PQ_windows.py --pair ACTUSDT --prepaper-start 2025-12-01 --interval 1m

  Then copy/rename the output JSON to candidate_for_TRADE.json and run:
    python 7_day_trade_window_forward_livefetch_v6+PrePaper.py

WINDOW DERIVATION (from --prepaper-start P):
  PREPAPER : [P,          P + 7 d)
  TRADE    : [P - 7 d,    P)
  TRAIN    : [P - 37 d,   P - 7 d)   (TRADE start − 30 d)
  Fetch    : [P - 44 d,   P - 7 d)   (TRAIN + 7-day warmup buffer)

OUTPUT:
  forwardtest/v30_eventstudy_{PAIR}_{INTERVAL}_rsi_sma_cross_gt51_prepaper_{DATE}.csv
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from binance_fetch import fetch_klines_1m, SUPPORTED_INTERVALS

RSI_LEN = 14
RSI_SMA_LEN = 14
ATR_LEN = 14
RSI_SMA_LEVEL = 51.0

SMMA_LEN = 200           # as requested
VOL_SMA_LEN = 20         # default for "vol_sma" (change if you want)

STOP_ON_CLOSE_BELOW_ENTRY = True

TRADE_SIZE_USDT = 1000.0
FEE_RATE_PER_SIDE = 0.001  # 0.1% per side

def compute_windows(prepaper_start_str: str) -> dict:
    """
    Derive PREPAPER, TRADE, TRAIN, and fetch-range windows from a PrePaper start date.

    Args:
        prepaper_start_str: 'YYYY-MM-DD' string, treated as 00:00 UTC.

    Returns:
        dict with keys 'prepaper', 'trade', 'train', 'fetch', each a (start, end)
        tuple of UTC-aware datetimes.  Ranges are half-open: [start, end).
    """
    prepaper_start = datetime.strptime(prepaper_start_str, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    trade_start   = prepaper_start - timedelta(days=7)
    train_start   = trade_start    - timedelta(days=30)
    warmup_start  = train_start    - timedelta(days=7)   # extra buffer for indicators

    return {
        "prepaper": (prepaper_start,       prepaper_start + timedelta(days=7)),
        "trade":    (trade_start,           prepaper_start),
        "train":    (train_start,           trade_start),
        "fetch":    (warmup_start,          trade_start),   # covers warmup + TRAIN
    }


def prepare_df_from_binance(symbol: str, fetch_start: datetime, fetch_end: datetime, interval: str = "1m") -> pd.DataFrame:
    """Fetch klines from Binance and return a normalised DataFrame ready for indicators."""
    raw = fetch_klines_1m(symbol, fetch_start, fetch_end, interval=interval)

    # Rename open_time -> time to match existing indicator/event logic
    df = raw.rename(columns={"open_time": "time"})

    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df["volume"] = df["volume"].fillna(0.0)
    df = df.sort_values("time").reset_index(drop=True)

    return df


def safe_round(value, decimals=8):
    """Round a value to specified decimals, returning np.nan for NaN values."""
    return round(float(value), decimals) if pd.notna(value) else np.nan


def wilders_rma(series: pd.Series, length: int) -> pd.Series:
    """
    Compute Wilder's RMA (also known as SMMA - Smoothed Moving Average).
    Compatible with TradingView's RMA behavior.
    
    Formula: RMA(length) = EWM with alpha=1/length, adjust=False
    
    This is equivalent to the recursive formula:
        RMA[i] = (RMA[i-1] * (length-1) + value[i]) / length
    
    Args:
        series: Input price series
        length: RMA period length
        
    Returns:
        Series with RMA values
    """
    return series.ewm(alpha=1/length, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rsi"] = ta.rsi(df["close"], length=RSI_LEN)
    df["rsi_sma"] = ta.sma(df["rsi"], length=RSI_SMA_LEN)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LEN)

    # SMMA(200) - Wilder's RMA/Smoothed Moving Average
    # NOTE: Prior versions incorrectly used SMA (rolling mean) instead of SMMA.
    # This change aligns with Wilder's RMA/SMMA used in TradingView and elsewhere in this repo.
    df["smma_200"] = wilders_rma(df["close"], SMMA_LEN)

    # Volume SMA for vol > vol_sma
    df["vol_sma"] = df["volume"].rolling(window=VOL_SMA_LEN).mean()

    return df


def cross_up_mask(df: pd.DataFrame) -> pd.Series:
    prev = df["rsi_sma"].shift(1)
    curr = df["rsi_sma"]
    return (prev <= RSI_SMA_LEVEL) & (curr > RSI_SMA_LEVEL)


def analyze_events(df: pd.DataFrame) -> pd.DataFrame:
    mask = cross_up_mask(df)
    event_idxs = df.index[mask].tolist()

    rows = []
    for idx in event_idxs:
        entry_time = df.at[idx, "time"]
        entry_close = float(df.at[idx, "close"])
        entry_atr = df.at[idx, "atr"]
        entry_rsi_sma = df.at[idx, "rsi_sma"]

        smma_200 = df.at[idx, "smma_200"]
        vol_sma = df.at[idx, "vol_sma"]
        volume = float(df.at[idx, "volume"])

        close_gt_smma_200 = bool(pd.notna(smma_200) and (entry_close > float(smma_200)))
        vol_gt_vol_sma = bool(pd.notna(vol_sma) and (volume > float(vol_sma)))

        if pd.notna(vol_sma) and float(vol_sma) > 0:
            vol_ratio = volume / float(vol_sma)
        else:
            vol_ratio = np.nan

        vol_ratio_ge_15 = bool(pd.notna(vol_ratio) and vol_ratio >= 1.5)

        # Skip events before ATR/RSI is warmed up
        if pd.isna(entry_atr) or float(entry_atr) == 0 or pd.isna(entry_rsi_sma):
            continue

        entry_atr = float(entry_atr)
        entry_rsi_sma = float(entry_rsi_sma)

        max_high = float(df.at[idx, "high"])
        max_high_time = entry_time

        # NEW: track min low in the SAME window you already use
        min_low = float(df.at[idx, "low"])
        min_low_time = entry_time

        stop_idx = None

        # Scan forward until stop condition (close below entry)
        for j in range(idx + 1, len(df)):
            h = float(df.at[j, "high"])
            if h > max_high:
                max_high = h
                max_high_time = df.at[j, "time"]

            # NEW: update min low during the same scan
            l = float(df.at[j, "low"])
            if l < min_low:
                min_low = l
                min_low_time = df.at[j, "time"]

            if STOP_ON_CLOSE_BELOW_ENTRY:
                if float(df.at[j, "close"]) < entry_close:
                    stop_idx = j
                    break

        open_ended = stop_idx is None
        end_idx = (len(df) - 1) if open_ended else stop_idx
        stop_time = df.at[end_idx, "time"]

        time_to_max_high_min = (max_high_time - entry_time).total_seconds() / 60.0
        time_to_stop_min = (stop_time - entry_time).total_seconds() / 60.0

        atr_multiple_to_max = (max_high - entry_close) / entry_atr

        # NEW: adverse excursion (how far price went below entry) in ATR units
        # If min_low >= entry_close, this becomes <= 0; clamp to 0 for readability.
        atr_multiple_to_min = (entry_close - min_low) / entry_atr
        atr_multiple_to_min = max(0.0, atr_multiple_to_min)

        raw_move = max_high - entry_close

        # --- PnL model (fixed trade size, fees on both buy & sell) ---
        buy_px = entry_close
        sell_px = max_high

        qty = TRADE_SIZE_USDT / buy_px
        gross_pnl_usdt = (sell_px - buy_px) * qty

        buy_fee_usdt = (TRADE_SIZE_USDT) * FEE_RATE_PER_SIDE
        sell_notional_usdt = sell_px * qty
        sell_fee_usdt = sell_notional_usdt * FEE_RATE_PER_SIDE

        net_pnl_usdt = gross_pnl_usdt - buy_fee_usdt - sell_fee_usdt

        # Compute audit columns for debugging/validation
        close_minus_smma_200 = (entry_close - float(smma_200)) if pd.notna(smma_200) else np.nan
        vol_minus_vol_sma = (volume - float(vol_sma)) if pd.notna(vol_sma) else np.nan

        rows.append(
            {
                "event_time": entry_time,

                "entry_rsi_sma": entry_rsi_sma,

                "close_gt_smma_200": close_gt_smma_200,
                "vol_gt_vol_sma": vol_gt_vol_sma,
                "vol_ratio": safe_round(vol_ratio, 4),
                "vol_ratio_ge_15": vol_ratio_ge_15,

                "entry_close": entry_close,
                "entry_atr": entry_atr,

                # Audit columns for verifying calculations
                "smma_200": safe_round(smma_200, 8),
                "vol_sma": safe_round(vol_sma, 8),
                "close_minus_smma_200": safe_round(close_minus_smma_200, 8),
                "vol_minus_vol_sma": safe_round(vol_minus_vol_sma, 8),

                "max_high_before_stop": max_high,
                "max_high_time": max_high_time,
                "time_to_max_high_min": round(time_to_max_high_min, 2),

                # NEW fields
                "min_low_before_stop": min_low,
                "min_low_time": min_low_time,
                "atr_multiple_to_min": round(atr_multiple_to_min, 4),

                "stop_time": stop_time,
                "time_to_stop_min": round(time_to_stop_min, 2),
                "open_ended": bool(open_ended),

                "atr_multiple_to_max": round(atr_multiple_to_max, 4),
                "raw_move": raw_move,

                "qty": qty,
                "buy_fee_usdt": round(buy_fee_usdt, 6),
                "sell_fee_usdt": round(sell_fee_usdt, 6),
                "net_pnl_usdt": round(net_pnl_usdt, 6),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("event_time").set_index("event_time")
    return out


def summarize(out: pd.DataFrame) -> None:
    print("\n=== V30 Funnel Event Study: RSI_SMA cross-up > 51 ===")
    print(f"Events (after indicator warmup): {len(out)}")

    if out.empty:
        return

    # Self-check: Report NaN counts for audit columns
    print("\n--- Indicator Warmup Check ---")
    if "smma_200" in out.columns:
        nan_count = out["smma_200"].isna().sum()
        print(f"smma_200 NaN count: {nan_count} / {len(out)} ({100*nan_count/len(out):.1f}%)")
    if "vol_sma" in out.columns:
        nan_count = out["vol_sma"].isna().sum()
        print(f"vol_sma NaN count: {nan_count} / {len(out)} ({100*nan_count/len(out):.1f}%)")

    print("\n--- Filter counts ---")
    print(f"close_gt_smma_200 TRUE: {int(out['close_gt_smma_200'].sum())} / {len(out)}")
    print(f"vol_gt_vol_sma   TRUE: {int(out['vol_gt_vol_sma'].sum())} / {len(out)}")
    print(f"both TRUE: {int((out['close_gt_smma_200'] & out['vol_gt_vol_sma']).sum())} / {len(out)}")
    print(f"both TRUE + vol_ratio_ge_15 TRUE: {int((out['close_gt_smma_200'] & out['vol_gt_vol_sma'] & out['vol_ratio_ge_15']).sum())} / {len(out)}")

    print("\n--- ATR multiple to max (summary) ---")
    print(out["atr_multiple_to_max"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
    
    # --- Net PnL (USDT, trade size $1000, fees 0.1% per side) ---
    print("\n--- Net PnL (USDT, trade size $1000, fees 0.1% per side) ---")

    def pnl_sum(df, mask, label):
        sub = df[mask]
        total = float(sub["net_pnl_usdt"].sum()) if len(sub) else 0.0
        avg = float(sub["net_pnl_usdt"].mean()) if len(sub) else 0.0
        print(f"{label}: trades={len(sub)} | total_net_pnl_usdt={total:.2f} | avg_net_pnl_usdt={avg:.2f}")

    BOTH_TRUE = out["close_gt_smma_200"] & out["vol_gt_vol_sma"]

    pnl_sum(out, pd.Series(True, index=out.index), "ALL events (RSI_SMA cross-up > 51)")
    pnl_sum(out, BOTH_TRUE, "both TRUE (close>smma_200 AND vol>vol_sma)")
    pnl_sum(out, BOTH_TRUE & out["vol_ratio_ge_15"], "both TRUE + vol_ratio_ge_15")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "V30 Event Study — fetch klines from Binance for one pair and "
            "generate a window-stamped events CSV."
        )
    )
    parser.add_argument(
        "--pair",
        required=True,
        help="Binance trading pair symbol, e.g. ACTUSDT",
    )
    parser.add_argument(
        "--prepaper-start",
        default="2025-12-01",
        metavar="YYYY-MM-DD",
        help="PrePaper window start date (00:00 UTC). Default: 2025-12-01",
    )
    parser.add_argument(
        "--interval",
        default="1m",
        choices=list(SUPPORTED_INTERVALS),
        help="Kline interval to fetch from Binance (default: 1m)",
    )
    parser.add_argument(
        "--out-dir",
        default="forwardtest",
        help="Output directory (default: forwardtest)",
    )
    args = parser.parse_args()

    pair = args.pair.upper()
    prepaper_start_str = args.prepaper_start
    interval = args.interval
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive windows
    windows = compute_windows(prepaper_start_str)
    fetch_start, fetch_end = windows["fetch"]
    train_start, train_end = windows["train"]
    trade_start, _         = windows["trade"]

    print(f"Pair           : {pair}")
    print(f"Interval       : {interval}")
    print(f"PrePaper start : {prepaper_start_str}")
    print(f"TRAIN window   : {train_start.date()} → {train_end.date()}")
    print(f"TRADE window   : {train_end.date()} → {windows['trade'][1].date()}")
    print(f"Fetch range    : {fetch_start.date()} → {fetch_end.date()} (incl. warmup)")

    # Fetch from Binance
    try:
        df = prepare_df_from_binance(pair, fetch_start, fetch_end, interval=interval)
    except Exception as e:
        print(f"Error fetching data from Binance: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("Error: No data returned from Binance.", file=sys.stderr)
        sys.exit(1)

    # Compute indicators on full fetched range (so warmup rows are included)
    df = compute_indicators(df)

    # Run event study on full range; then filter output to TRAIN window
    all_events = analyze_events(df)

    if all_events.empty:
        print("Warning: No events found in the fetched range.")
        out = all_events
    else:
        # Filter to TRAIN window: [train_start, train_end)
        event_index = pd.to_datetime(all_events.index, utc=True)
        mask = (event_index >= train_start) & (event_index < train_end)
        out = all_events.loc[mask]
        print(
            f"\nEvents in full fetch range : {len(all_events)}"
            f"\nEvents in TRAIN window     : {len(out)}"
        )

    # Window-stamped output filename (interval is dynamic)
    out_filename = (
        f"v30_eventstudy_{pair}_{interval}_rsi_sma_cross_gt51"
        f"_prepaper_{prepaper_start_str}.csv"
    )
    out_path = out_dir / out_filename
    out.to_csv(out_path, index=True)

    summarize(out)
    print(f"\nSaved: {out_path}")
    if not out.empty:
        print("\nFirst 10 events (preview):")
        print(out.head(10).to_string())


if __name__ == "__main__":
    main()