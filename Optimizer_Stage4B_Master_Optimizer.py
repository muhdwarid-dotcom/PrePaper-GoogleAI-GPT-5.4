"""Optimizer_Stage4B_Master_Optimizer.py

Stage 4B — Stage 2.5 Darwinian Filter for MTF Fibonacci Cluster Engine.

Approved refactor (Option i):
- Stage 4B reads directly from Stage 2 output CSV:
    results_v29R_30d/stage2_intraday_dual_tf_improved.csv
- Stage 4B produces a filtered survivor CSV preserving all Stage 2 columns:
    results_v29R_30d/stage4b_intraday_dual_tf_selected.csv
  This preserves the EventStudy funnel's required columns and avoids legacy JSON.

Operational requirements:
- No scenario configuration / selection.
- No hourly pruning.
- Uses fib_train_verifier.verify_symbol_fib_train as a structural stress-test:
  on_spearhead() is called on every bar during the 7-day TRAIN verification window.

Time alignment:
- Adds --prepaper-start (Monday anchor, YYYY-MM-DD) to derive windows using the
  same convention as the rest of the pipeline:
    trade_end   = Monday 00:00 UTC
    trade_start = trade_end - 7d
    train_end   = trade_start
    train_start = train_end - 30d
  Stage4B verifies last 7d of TRAIN: [train_end-7d, train_end)

Sizing defaults:
- initial_capital default: 10_000.0
- trade_size per slot default: 1_000.0

Outputs:
- results_v29R_30d/stage4b_fib_verify.csv (audit metrics)
- results_v29R_30d/stage4b_intraday_dual_tf_selected.csv (survivor list)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from fib_train_verifier import verify_symbol_fib_train


# ============================================================
# CONFIG (Paths)
# ============================================================
RESULTS_DIR = Path("./results_v29R_30d")

STAGE2_IN_CSV = RESULTS_DIR / "stage2_intraday_dual_tf_improved.csv"
STAGE4B_OUT_CSV = RESULTS_DIR / "stage4b_intraday_dual_tf_selected.csv"

VERIFY_OUT_CSV = RESULTS_DIR / "stage4b_fib_verify.csv"


# ============================================================
# SAFETY GATES (Audited constants)
# ============================================================
NET_FLOOR_PCT = 0.0
MIN_CLUSTERS_COMPLETED = 1
MAX_DD_PCT = 0.10

# Stage4B verifies last 30 days of TRAIN
VERIFY_DAYS = 30

# Risk model defaults
DEFAULT_INITIAL_CAPITAL = 10_000.0
DEFAULT_TRADE_SIZE = 1_000.0


# ============================================================
# Window derivation (Monday anchor)
# ============================================================

def _previous_monday(dt: datetime) -> datetime:
    """Return the most recent Monday <= dt (UTC-aware)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days_back = (dt.weekday() - 0) % 7
    monday = (dt - timedelta(days=days_back)).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )
    return monday


def derive_windows_from_prepaper_start(prepaper_start: str) -> dict:
    """Derive train/trade windows from a Monday anchor (YYYY-MM-DD)."""
    dt = datetime.fromisoformat(prepaper_start)
    dt = dt.replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)

    monday = _previous_monday(dt)
    if monday.date() != dt.date():
        raise ValueError(f"--prepaper-start must be a Monday (UTC). Got {dt.date()}, nearest Monday is {monday.date()}.")

    trade_end = monday
    trade_start = trade_end - timedelta(days=7)
    train_end = trade_start
    train_start = train_end - timedelta(days=30)

    return {
        "train_start": pd.to_datetime(train_start, utc=True),
        "train_end": pd.to_datetime(train_end, utc=True),
        "trade_start": pd.to_datetime(trade_start, utc=True),
        "trade_end": pd.to_datetime(trade_end, utc=True),
    }


# ============================================================
# Stage2 interface helpers
# ============================================================

def _detect_symbol_column(df: pd.DataFrame) -> str:
    for c in ("symbol", "Symbol"):
        if c in df.columns:
            return c
    raise RuntimeError(f"Stage2 CSV missing symbol column. Columns={list(df.columns)}")


def _detect_best_tf_column(df: pd.DataFrame) -> str:
    for c in ("best_tf", "best_TF", "BestTF"):
        if c in df.columns:
            return c
    raise RuntimeError(f"Stage2 CSV missing best_tf column. Columns={list(df.columns)}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B — Stage 2.5 Darwinian Filter (MTF Fib Cluster)")
    parser.add_argument(
        "--prepaper-start",
        required=True,
        help="Monday anchor date (UTC) in YYYY-MM-DD. Must be a Monday.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help=f"Initial capital for verification accounting (default: {DEFAULT_INITIAL_CAPITAL}).",
    )
    parser.add_argument(
        "--trade-size",
        type=float,
        default=DEFAULT_TRADE_SIZE,
        help=f"Unit trade size per slot (default: {DEFAULT_TRADE_SIZE}).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not STAGE2_IN_CSV.exists():
        raise FileNotFoundError(f"Stage2 CSV not found: {STAGE2_IN_CSV}")

    windows = derive_windows_from_prepaper_start(args.prepaper_start)
    train_start = windows["train_start"]
    train_end = windows["train_end"]

    # Verify last 7 days of TRAIN
    verify_end = train_end
    verify_start = verify_end - pd.Timedelta(days=int(VERIFY_DAYS))

    print("======================================================")
    print(" 🚀 STAGE 4B — Stage 2.5 Darwinian Filter (MTF Fib Cluster)")
    print("======================================================")
    print(f"[4B] prepaper_start = {args.prepaper_start}")
    print(f"[4B] TRAIN window   = {train_start} -> {train_end}")
    print(f"[4B] VERIFY window  = {verify_start} -> {verify_end} (last {VERIFY_DAYS}d of TRAIN)")
    print(f"[4B] Gates: net>={NET_FLOOR_PCT:.2%}, clusters>={MIN_CLUSTERS_COMPLETED}, maxDD<={MAX_DD_PCT:.2%}")
    print(f"[4B] initial_capital={args.initial_capital} trade_size={args.trade_size}")
    print(f"[4B] Stage2 in:  {STAGE2_IN_CSV}")
    print(f"[4B] Stage4B out: {STAGE4B_OUT_CSV}")

    s2 = pd.read_csv(STAGE2_IN_CSV)
    if s2.empty:
        raise RuntimeError(f"Stage2 CSV is empty: {STAGE2_IN_CSV}")

    symbol_col = _detect_symbol_column(s2)
    tf_col = _detect_best_tf_column(s2)

    # Normalize minimal columns for iteration
    s2_iter = s2.copy()
    s2_iter[symbol_col] = s2_iter[symbol_col].astype(str).str.strip().str.upper()
    s2_iter[tf_col] = s2_iter[tf_col].astype(str).str.strip()

    rows = []
    for _, row in s2_iter.iterrows():
        pair = str(row[symbol_col]).strip().upper()
        interval = str(row[tf_col]).strip()

        if interval not in {"1m", "3m"}:
            # Stage2 should only emit 1m/3m, but be defensive
            continue

        print("-" * 80)
        print(f"[4B] Verifying {pair} interval={interval}")

        try:
            r = verify_symbol_fib_train(
                pair=pair,
                interval=interval,
                train_start=verify_start,
                train_end=verify_end,
                initial_capital=float(args.initial_capital),
                trade_size=float(args.trade_size),
            )
            r["gate_net_ok"] = bool(float(r.get("net_profit_pct", 0.0)) >= float(NET_FLOOR_PCT))
            r["gate_clusters_ok"] = bool(int(r.get("clusters_completed", 0)) >= int(MIN_CLUSTERS_COMPLETED))
            r["gate_dd_ok"] = bool(float(r.get("max_dd_pct", 0.0)) <= float(MAX_DD_PCT))
            r["gate_pass"] = bool(r["gate_net_ok"] and r["gate_clusters_ok"] and r["gate_dd_ok"])
        except Exception as e:
            r = {
                "pair": pair,
                "interval": interval,
                "train_start": verify_start,
                "train_end": verify_end,
                "bars": 0,
                "net_profit_usdt": 0.0,
                "net_profit_pct": -1.0,
                "max_dd_pct": 1.0,
                "max_dd_usdt": 0.0,
                "clusters_completed": 0,
                "trades_closed": 0,
                "error": str(e),
                "gate_net_ok": False,
                "gate_clusters_ok": False,
                "gate_dd_ok": False,
                "gate_pass": False,
            }

        rows.append(r)

    verify_df = pd.DataFrame(rows)
    if verify_df.empty:
        raise RuntimeError("No verification rows produced. Check Stage2 input columns and intervals.")

    verify_df = verify_df.sort_values(by=["gate_pass", "net_profit_pct"], ascending=[False, False])
    verify_df.to_csv(VERIFY_OUT_CSV, index=False)
    print(f"\n[4B] Wrote audit CSV: {VERIFY_OUT_CSV.resolve()}")

    survivors = set(
        verify_df.loc[verify_df["gate_pass"] == True, "pair"].astype(str).str.upper().tolist()
    )

    # Preserve Stage2 columns fully for downstream EventStudy funnel.
    selected = s2_iter[s2_iter[symbol_col].astype(str).str.upper().isin(survivors)].copy()
    selected.to_csv(STAGE4B_OUT_CSV, index=False)

    print("\n" + "=" * 80)
    print(f"[4B] Survivors: {len(selected)}/{len(s2_iter)}")
    print(f"[4B] Selected CSV written: {STAGE4B_OUT_CSV.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
