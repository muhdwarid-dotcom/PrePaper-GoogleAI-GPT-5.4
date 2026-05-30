#!/usr/bin/env python3
"""
Funnel Data V30 Event Study — one-pair-at-a-time CLI tool.

Fetches OHLCV data from Binance (1m or 3m), runs V30 funnel event-study logic,
and writes a window-stamped CSV for the requested pair.

USAGE (3-step pipeline for one pair):
  Step 1 — Generate events CSV:
    python Funnel_Data_Test_V30_EventStudy.py --pair ACTUSDT --prepaper-start 2025-12-01

  Step 2 — Analyse events, generate candidates CSV:
    python eventstudy_analysis.py --pair ACTUSDT --prepaper-start 2025-12-01 --grid

  Step 3 — Derive k/t exit parameters:
    python Derive_k_t_from_PQ_windows.py --pair ACTUSDT --prepaper-start 2025-12-01

WINDOW DERIVATION (from --prepaper-start P):
  PREPAPER : [P,          P + 7 d)
  TRADE    : [P - 7 d,    P)
  TRAIN    : [P - 37 d,   P - 7 d)   (TRADE start − 30 d)
  Fetch    : [P - 44 d,   P - 7 d)   (TRAIN + 7-day warmup buffer)

STABLE OUTPUT CONTRACT:
- Downstream scripts (eventstudy_analysis/metrics/transform/derive) remain untouched.
- This script preserves the legacy column schema in its output CSV.
- The ONLY behavioral refactor is that trade outcomes are now simulated using
  MtfFibClusterEngine rather than the legacy ATR-band forward scan.

ENGINE INTEGRATION NOTE (intentional design):
- We run a local forward simulation per event. If the RSI_SMA cross hoards a ticket
  but never validates an entry (no kill-zone touch + bounce), we SKIP the event.
- Each event uses fixed sizing: TRADE_SIZE_USDT = 1000.0 (one slot per event).
- If a trade remains open through end-of-data, we mark-to-market at the final close
  and set open_ended=True. This matches the legacy contract.

LOOKAHEAD SAFETY (CRITICAL):
- The fib engine internally uses HTF (1h) bars for pivots.
- To avoid lookahead bias, the engine must only see HTF candles up to the current
  simulated timestamp.
- PERFORMANCE OPTIMIZATION (Option A): precompute htf_full once (from full LTF df)
  and, at each bar j, slice htf_full[htf_full['time'] <= ts]. This avoids expensive
  resampling inside the per-bar loop while still eliminating lookahead.

OUTPUT:
  forwardtest/v30_eventstudy_{PAIR}_{INTERVAL}_rsi_sma_cross_gt51_prepaper_{DATE}.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from binance_fetch import SUPPORTED_INTERVALS, fetch_klines_1m
from mtf_fib_cluster_engine import MtfFibClusterEngine, build_binance_aligned_1h

RSI_LEN = 14
RSI_SMA_LEN = 14
ATR_LEN = 14
RSI_SMA_LEVEL = 51.0

SMMA_LEN = 200  # as requested
VOL_SMA_LEN = 20

TRADE_SIZE_USDT = 1000.0
FEE_RATE_PER_SIDE = 0.001  # 0.1% per side


def compute_windows(prepaper_start_str: str) -> dict:
    """Derive PREPAPER, TRADE, TRAIN, and fetch-range windows from a PrePaper start date."""
    prepaper_start = datetime.strptime(prepaper_start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    trade_start = prepaper_start - timedelta(days=7)
    train_start = trade_start - timedelta(days=30)
    warmup_start = train_start - timedelta(days=7)  # extra buffer for indicators

    return {
        "prepaper": (prepaper_start, prepaper_start + timedelta(days=7)),
        "trade": (trade_start, prepaper_start),
        "train": (train_start, trade_start),
        "fetch": (warmup_start, trade_start),
    }


def prepare_df_from_binance(symbol: str, fetch_start: datetime, fetch_end: datetime, interval: str = "1m") -> pd.DataFrame:
    """Fetch klines from Binance and return a normalised DataFrame ready for indicators."""
    raw = fetch_klines_1m(symbol, fetch_start, fetch_end, interval=interval)

    df = raw.rename(columns={"open_time": "time"})
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df["volume"] = df["volume"].fillna(0.0)

    # ✅ TZ ALIGNMENT: make all timestamps tz-naive for MtfFibClusterEngine compatibility
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

    df = df.sort_values("time").reset_index(drop=True)
    return df

def safe_round(value, decimals=8):
    return round(float(value), decimals) if pd.notna(value) else np.nan


def wilders_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA (SMMA) compatible with TradingView."""
    return series.ewm(alpha=1 / length, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rsi"] = ta.rsi(df["close"], length=RSI_LEN)
    df["rsi_sma"] = ta.sma(df["rsi"], length=RSI_SMA_LEN)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LEN)

    df["smma_200"] = wilders_rma(df["close"], SMMA_LEN)
    df["vol_sma"] = df["volume"].rolling(window=VOL_SMA_LEN).mean()

    # Engine uses EMA50 bounce validation
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    return df


def cross_up_mask(df: pd.DataFrame) -> pd.Series:
    prev = df["rsi_sma"].shift(1)
    curr = df["rsi_sma"]
    return (prev <= RSI_SMA_LEVEL) & (curr > RSI_SMA_LEVEL)


def _net_pnl_usdt_for_trade(*, entry_px: float, exit_px: float, trade_size_usdt: float) -> tuple[float, float, float, float]:
    """Return (net_pnl_usdt, qty, buy_fee_usdt, sell_fee_usdt)."""
    qty = float(trade_size_usdt / entry_px) if entry_px > 0 else 0.0
    gross_pnl = (float(exit_px) - float(entry_px)) * qty

    buy_fee_usdt = float(trade_size_usdt) * float(FEE_RATE_PER_SIDE)
    sell_notional_usdt = float(exit_px) * qty
    sell_fee_usdt = sell_notional_usdt * float(FEE_RATE_PER_SIDE)

    net_pnl_usdt = float(gross_pnl - buy_fee_usdt - sell_fee_usdt)
    return net_pnl_usdt, qty, buy_fee_usdt, sell_fee_usdt


def simulate_fib_trade_from_event(df: pd.DataFrame, idx: int, pair: str) -> dict | None:
    """Simulate one fixed-size fib trade outcome starting at an RSI_SMA cross event.

    Contract:
    - If no entry occurs for the ticket hoarded at idx, return None (skip event).
    - If trade remains open through end-of-data, MTM at final close, open_ended=True.
    - Fixed trade size per event (TRADE_SIZE_USDT). No scaling with ticket count.

    Lookahead safety:
    - Precompute htf_full once (from the full df). At each bar j, slice htf_full by time
      so that the engine only sees HTF candles with time <= current ts.
    """

    # Need warmed indicators
    if pd.isna(df.at[idx, "atr"]) or pd.isna(df.at[idx, "ema50"]) or pd.isna(df.at[idx, "rsi_sma"]):
        return None

    # Precompute full HTF once for speed (Option A). We'll slice by time within loop.
    htf_full = build_binance_aligned_1h(df[["time", "open", "high", "low", "close", "volume"]])
    htf_full = htf_full.sort_values("time").reset_index(drop=True)

    # Initialize engine with history up to the event index.
    hist0 = df.iloc[: idx + 1].copy()
    fib = MtfFibClusterEngine(
        symbol=pair,
        ohlcv_1m=hist0[["time", "open", "high", "low", "close", "volume"]],
    )

    in_position = False
    entry_idx: int | None = None
    entry_time = None
    entry_close = np.nan
    entry_atr = np.nan

    max_high = np.nan
    max_high_time = None
    min_low = np.nan
    min_low_time = None

    exit_time = None
    exit_px = np.nan
    open_ended = False

    # Forward sim from idx onward
    for j in range(idx, len(df)):
        ts = df.at[j, "time"]

        # ---- Lookahead elimination with speed-up: slice precomputed HTF by time ----
        fib.htf_1h = htf_full.loc[htf_full["time"] <= ts].copy()
        fib._htf_opens = pd.to_datetime(fib.htf_1h["time"]).dt.tz_localize(None).to_numpy()

        o = float(df.at[j, "open"])
        h = float(df.at[j, "high"])
        l = float(df.at[j, "low"])
        c = float(df.at[j, "close"])
        ema50 = float(df.at[j, "ema50"]) if pd.notna(df.at[j, "ema50"]) else np.nan

        # ---- In-trade management ----
        if in_position:
            if not np.isfinite(max_high) or h > max_high:
                max_high = h
                max_high_time = ts
            if not np.isfinite(min_low) or l < min_low:
                min_low = l
                min_low_time = ts

            cluster_sl = fib.update_cluster_sl(ts=ts, bar_high=h, ltf_ema50=ema50)
            if np.isfinite(cluster_sl) and l <= cluster_sl:
                exit_px = max(o, float(cluster_sl))
                exit_time = ts
                open_ended = False
                break

            continue

        # ---- Pre-entry management ----
        if fib.cooldown_active:
            fib.maybe_release_cooldown(ts=ts, ltf_price=c)
        else:
            fib.apply_pre_entry_wipes(ts=ts, ltf_high=h, ltf_low=l, ltf_price=c)

        if fib.cooldown_active:
            continue

        route = fib.on_spearhead(
            ts=ts,
            ltf_open=o,
            ltf_high=h,
            ltf_low=l,
            ltf_close=c,
            ltf_ema50=ema50,
        )
        immediate_entry = bool(route.get("immediate_entry", False))

        if immediate_entry or fib.should_enter(ltf_low=l, ltf_close=c, ltf_ema50=ema50):
            if int(fib.pending_triggers) <= 0:
                continue

            fib.lock_cluster(cluster_id=f"{pair}_EVENT_{idx}", ts=ts, entry_price=c, ltf_ema50=ema50)

            in_position = True
            entry_idx = j
            entry_time = ts
            entry_close = float(c)
            entry_atr = float(df.at[j, "atr"]) if pd.notna(df.at[j, "atr"]) else np.nan

            max_high = float(h)
            max_high_time = ts
            min_low = float(l)
            min_low_time = ts

    if not in_position or entry_idx is None:
        return None

    if exit_time is None:
        last_ts = df.at[len(df) - 1, "time"]
        last_close = float(df.at[len(df) - 1, "close"])
        exit_time = last_ts
        exit_px = last_close
        open_ended = True

    net_pnl_usdt, qty, buy_fee_usdt, sell_fee_usdt = _net_pnl_usdt_for_trade(
        entry_px=entry_close,
        exit_px=exit_px,
        trade_size_usdt=TRADE_SIZE_USDT,
    )

    time_to_max_high_min = (max_high_time - entry_time).total_seconds() / 60.0 if max_high_time else np.nan
    time_to_stop_min = (exit_time - entry_time).total_seconds() / 60.0 if exit_time else np.nan

    return {
        "entry_time": entry_time,
        "entry_close": float(entry_close),
        "entry_atr": float(entry_atr),
        "exit_time": exit_time,
        "exit_price": float(exit_px),
        "open_ended": bool(open_ended),
        "max_high_before_stop": float(max_high),
        "max_high_time": max_high_time,
        "min_low_before_stop": float(min_low),
        "min_low_time": min_low_time,
        "time_to_max_high_min": float(time_to_max_high_min),
        "time_to_stop_min": float(time_to_stop_min),
        "qty": float(qty),
        "buy_fee_usdt": float(buy_fee_usdt),
        "sell_fee_usdt": float(sell_fee_usdt),
        "net_pnl_usdt": float(net_pnl_usdt),
    }


def analyze_events(df: pd.DataFrame, pair: str) -> pd.DataFrame:
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

        if pd.isna(entry_atr) or float(entry_atr) == 0 or pd.isna(entry_rsi_sma):
            continue

        entry_atr = float(entry_atr)
        entry_rsi_sma = float(entry_rsi_sma)

        sim = simulate_fib_trade_from_event(df, idx, pair)
        if sim is None:
            continue

        max_high = float(sim["max_high_before_stop"])
        min_low = float(sim["min_low_before_stop"])

        atr_multiple_to_max = (max_high - float(sim["entry_close"])) / float(sim["entry_atr"]) if float(sim["entry_atr"]) else np.nan
        atr_multiple_to_min = (float(sim["entry_close"]) - min_low) / float(sim["entry_atr"]) if float(sim["entry_atr"]) else np.nan
        if pd.notna(atr_multiple_to_min):
            atr_multiple_to_min = max(0.0, float(atr_multiple_to_min))

        raw_move = max_high - float(sim["entry_close"])

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
                "entry_close": float(sim["entry_close"]),
                "entry_atr": float(sim["entry_atr"]),
                "smma_200": safe_round(smma_200, 8),
                "vol_sma": safe_round(vol_sma, 8),
                "close_minus_smma_200": safe_round(close_minus_smma_200, 8),
                "vol_minus_vol_sma": safe_round(vol_minus_vol_sma, 8),
                "max_high_before_stop": max_high,
                "max_high_time": sim["max_high_time"],
                "time_to_max_high_min": round(float(sim["time_to_max_high_min"]), 2),
                "min_low_before_stop": min_low,
                "min_low_time": sim["min_low_time"],
                "atr_multiple_to_min": round(float(atr_multiple_to_min), 4) if pd.notna(atr_multiple_to_min) else np.nan,
                "stop_time": sim["exit_time"],
                "time_to_stop_min": round(float(sim["time_to_stop_min"]), 2),
                "open_ended": bool(sim["open_ended"]),
                "atr_multiple_to_max": round(float(atr_multiple_to_max), 4) if pd.notna(atr_multiple_to_max) else np.nan,
                "raw_move": raw_move,
                "qty": float(sim["qty"]),
                "buy_fee_usdt": round(float(sim["buy_fee_usdt"]), 6),
                "sell_fee_usdt": round(float(sim["sell_fee_usdt"]), 6),
                "net_pnl_usdt": round(float(sim["net_pnl_usdt"]), 6),
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
    print(
        f"both TRUE + vol_ratio_ge_15 TRUE: "
        f"{int((out['close_gt_smma_200'] & out['vol_gt_vol_sma'] & out['vol_ratio_ge_15']).sum())} / {len(out)}"
    )

    print("\n--- ATR multiple to max (summary) ---")
    print(out["atr_multiple_to_max"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())

    print("\n--- Net PnL (USDT, trade size $1000, fees 0.1% per side) ---")

    def pnl_sum(df, mask, label):
        sub = df[mask]
        total = float(sub["net_pnl_usdt"].sum()) if len(sub) else 0.0
        avg = float(sub["net_pnl_usdt"].mean()) if len(sub) else 0.0
        print(f"{label}: trades={len(sub)} | total_net_pnl_usdt={total:.2f} | avg_net_pnl_usdt={avg:.2f}")

    both_true = out["close_gt_smma_200"] & out["vol_gt_vol_sma"]

    pnl_sum(out, pd.Series(True, index=out.index), "ALL events (RSI_SMA cross-up > 51)")
    pnl_sum(out, both_true, "both TRUE (close>smma_200 AND vol>vol_sma)")
    pnl_sum(out, both_true & out["vol_ratio_ge_15"], "both TRUE + vol_ratio_ge_15")


def resolve_best_tf_from_stage2(pair: str, stage2_csv_path: str) -> str:
    """Resolve best_tf with Stage4B-selected preference, Stage2 fallback."""

    preferred_stage4b = Path("results_v29R_30d/stage4b_intraday_dual_tf_selected.csv")
    fallback_stage2 = Path(stage2_csv_path)

    csv_path = preferred_stage4b if preferred_stage4b.exists() else fallback_stage2

    try:
        df = pd.read_csv(str(csv_path))
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {csv_path} ({e})")

    if "symbol" not in df.columns or "best_tf" not in df.columns:
        raise RuntimeError(
            f"CSV missing required columns. Need ['symbol','best_tf'].\n"
            f"Got columns: {list(df.columns)}\n"
            f"CSV path: {csv_path}"
        )

    pair_u = pair.upper()
    row = df.loc[df["symbol"].astype(str).str.upper() == pair_u]
    if row.empty:
        raise RuntimeError(
            f"Pair {pair_u} not found in CSV: {csv_path}\n"
            "Run Stage1A/1B + Stage2 (and Stage4B optional) first, or check eligibility."
        )

    tf = str(row.iloc[0]["best_tf"]).strip().lower()
    if tf not in SUPPORTED_INTERVALS:
        raise RuntimeError(
            f"Invalid best_tf='{tf}' for {pair_u} in CSV. Supported: {SUPPORTED_INTERVALS}"
        )

    return tf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V30 Event Study — fetch klines from Binance for one pair and "
            "generate a window-stamped events CSV."
        )
    )
    parser.add_argument("--pair", required=True, help="Binance trading pair symbol, e.g. ACTUSDT")
    parser.add_argument(
        "--prepaper-start",
        default="2025-12-01",
        metavar="YYYY-MM-DD",
        help="PrePaper window start date (00:00 UTC). Default: 2025-12-01",
    )
    parser.add_argument(
        "--stage2-csv",
        default="results_v29R_30d/stage2_intraday_dual_tf_improved.csv",
        help="Stage2 output CSV used as fallback to resolve best_tf.",
    )
    parser.add_argument("--out-dir", default="forwardtest", help="Output directory (default: forwardtest)")
    args = parser.parse_args()

    pair = args.pair.upper()
    prepaper_start_str = args.prepaper_start

    interval = resolve_best_tf_from_stage2(pair, args.stage2_csv)
    print(f"[AUTO-TF] Interval resolved from best_tf CSV: {pair} -> {interval}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = compute_windows(prepaper_start_str)
    fetch_start, fetch_end = windows["fetch"]
    train_start, train_end = windows["train"]

    print(f"Pair           : {pair}")
    print(f"Interval       : {interval}")
    print(f"PrePaper start : {prepaper_start_str}")
    print(f"TRAIN window   : {train_start.date()} → {train_end.date()}")
    print(f"TRADE window   : {train_end.date()} → {windows['trade'][1].date()}")
    print(f"Fetch range    : {fetch_start.date()} → {fetch_end.date()} (incl. warmup)")

    try:
        df = prepare_df_from_binance(pair, fetch_start, fetch_end, interval=interval)
        
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    except Exception as e:
        print(f"Error fetching data from Binance: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("Error: No data returned from Binance.", file=sys.stderr)
        sys.exit(1)

    df = compute_indicators(df)

    all_events = analyze_events(df, pair)

    if all_events.empty:
        print("Warning: No events found in the fetched range.")
        out = all_events
    else:
        event_index = pd.to_datetime(all_events.index).tz_localize(None)

        ts0 = pd.to_datetime(train_start).tz_localize(None)
        ts1 = pd.to_datetime(train_end).tz_localize(None)

        mask = (event_index >= ts0) & (event_index < ts1)
        out = all_events.loc[mask]
        print(f"\nEvents in full fetch range : {len(all_events)}\nEvents in TRAIN window     : {len(out)}")

    out_filename = f"v30_eventstudy_{pair}_{interval}_rsi_sma_cross_gt51_prepaper_{prepaper_start_str}.csv"
    out_path = out_dir / out_filename
    out.to_csv(out_path, index=True)

    summarize(out)
    print(f"\nSaved: {out_path}")
    if not out.empty:
        print("\nFirst 10 events (preview):")
        print(out.head(10).to_string())


if __name__ == "__main__":
    main()
