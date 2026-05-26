"""Optimizer_Stage4B_Master_Optimizer.py

Stage 4B Master Optimizer — Darwinian Gate for MTF Fibonacci Cluster Engine.

Audited requirements satisfied:
- No scenario configuration / selection in Stage 4B.
- Ingest portfolio_plan_v29R_auto.json and preserve trade sizes as-is.
  (Note: portfolio entries do NOT contain trade_size; Stage 4B does not add or scale sizes.)
- Inject best_tf (1m/3m) from Stage 2 output to portfolio entries.
- Remove ALL hourly pruning logic.
- Run a 7-day TRAIN verification per pair using fib_train_verifier.verify_symbol_fib_train.
- Apply safety gates:
    * NET_FLOOR_PCT = 0.0
    * MIN_CLUSTERS_COMPLETED = 1
    * MAX_DD_PCT = 0.10

Outputs:
- portfolio_plan_v29R_selected.json (survivors only)
- results_v29R_30d/stage4b_fib_verify.csv (audit trail)
"""

import json
from pathlib import Path

import pandas as pd

from fib_train_verifier import verify_symbol_fib_train


# ============================================================
# CONFIG (Paths)
# ============================================================
RESULTS_DIR = Path("./results_v29R_30d")

# Must match Optimizer_Stage2_v29R_DualTF_CLEAN.py
STAGE2_CSV = RESULTS_DIR / "stage2_intraday_dual_tf_improved.csv"

PLAN_IN_PATH = Path("./portfolio_plan_v29R_auto.json")
PLAN_FINAL_OUT = Path("./portfolio_plan_v29R_selected.json")

VERIFY_OUT_CSV = RESULTS_DIR / "stage4b_fib_verify.csv"


# ============================================================
# SAFETY GATES (Audited constants)
# ============================================================
NET_FLOOR_PCT = 0.0
MIN_CLUSTERS_COMPLETED = 1
MAX_DD_PCT = 0.10

VERIFY_DAYS = 7


def inject_stage2_timeframes(plan_base: dict) -> dict:
    """Inject Stage 2 best_tf (1m/3m) into each plan entry as `interval`.

    - Keeps only assets that exist in Stage 2 output (Stage 2 is the source of truth for best_tf).
    - Does NOT inject any ATR/trail/scenario params.
    """
    if not STAGE2_CSV.exists():
        raise FileNotFoundError(f"[4B] Stage 2 CSV not found: {STAGE2_CSV}")

    s2 = pd.read_csv(STAGE2_CSV)
    s2.columns = [c.strip() for c in s2.columns]

    required = ["symbol", "best_tf"]
    missing = [c for c in required if c not in s2.columns]
    if missing:
        raise RuntimeError(f"[4B] Stage 2 CSV missing columns {missing}. Found={list(s2.columns)}")

    tf_map = {
        str(r["symbol"]).strip().upper(): str(r["best_tf"]).strip()
        for _, r in s2.iterrows()
    }

    updated = []
    dropped = []

    for entry in plan_base.get("portfolio", []):
        sym = str(entry.get("pair", "")).strip().upper()
        if not sym:
            dropped.append({"pair": sym, "reason": "missing_pair"})
            continue

        best_tf = tf_map.get(sym)
        if not best_tf:
            dropped.append({"pair": sym, "reason": "not_in_stage2"})
            continue

        new_entry = dict(entry)
        new_entry["interval"] = best_tf
        updated.append(new_entry)

    plan_base["portfolio"] = updated

    if dropped:
        print(f"[4B] Dropped {len(dropped)} entries without Stage2 mapping:")
        for d in dropped:
            print(f"  - {d['pair']!r}: {d['reason']}")

    return plan_base


def main() -> None:
    print("======================================================")
    print(" 🚀 STAGE 4B MASTER — Darwinian Gate (MTF Fib Cluster)")
    print("======================================================")

    if not PLAN_IN_PATH.exists():
        raise FileNotFoundError(f"[4B] Missing plan: {PLAN_IN_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plan = json.loads(PLAN_IN_PATH.read_text(encoding="utf-8"))

    # Inject best_tf timeframes (1m/3m)
    plan = inject_stage2_timeframes(plan)

    if not plan.get("portfolio"):
        print("[4B] No portfolio entries after Stage2 injection. Exiting.")
        return

    meta = plan.get("meta") or {}
    if "train_start" not in meta or "train_end" not in meta:
        raise KeyError("[4B] plan.meta must include train_start and train_end")

    train_start = pd.to_datetime(meta["train_start"], utc=True)
    train_end = pd.to_datetime(meta["train_end"], utc=True)

    verify_end = train_end
    verify_start = verify_end - pd.Timedelta(days=int(VERIFY_DAYS))

    initial_capital = float(meta.get("initial_capital", 10_000.0))

    print(f"[4B] VERIFY window (UTC): {verify_start} -> {verify_end} (last {VERIFY_DAYS}d of TRAIN)")
    print(f"[4B] Gates: net>={NET_FLOOR_PCT:.2%}, clusters>={MIN_CLUSTERS_COMPLETED}, maxDD<={MAX_DD_PCT:.2%}")
    print(f"[4B] Stage2 CSV: {STAGE2_CSV}")

    rows = []
    for entry in plan["portfolio"]:
        pair = str(entry.get("pair", "")).strip().upper()
        interval = str(entry.get("interval", "")).strip()

        # Trade sizing is not present in the provided plan schema; do not add or modify it.
        trade_size = float(meta.get("trade_size", 1000.0))

        print("-" * 80)
        print(f"[4B] Verifying {pair} interval={interval}")

        try:
            r = verify_symbol_fib_train(
                pair=pair,
                interval=interval,
                train_start=verify_start,
                train_end=verify_end,
                initial_capital=initial_capital,
                trade_size=trade_size,
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

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["gate_pass", "net_profit_pct"], ascending=[False, False])
        df.to_csv(VERIFY_OUT_CSV, index=False)
        print(f"\n[4B] Wrote audit CSV: {VERIFY_OUT_CSV.resolve()}")
        print(df[["pair", "interval", "net_profit_pct", "clusters_completed", "max_dd_pct", "gate_pass"]].to_string(index=False))

    survivors = set(df.loc[df["gate_pass"] == True, "pair"].astype(str).str.upper().tolist())

    print("\n" + "=" * 80)
    print(f"[4B] Survivors: {len(survivors)}/{len(df)}")
    killed = [p for p in df["pair"].astype(str).str.upper().tolist() if p not in survivors]
    print(f"[4B] Killed: {killed}")
    print("=" * 80)

    plan["portfolio"] = [p for p in plan["portfolio"] if str(p.get("pair", "")).strip().upper() in survivors]

    plan.setdefault("meta", {})
    plan["meta"]["mode"] = "DARWINIAN_PRUNE_MTF_FIB_CLUSTER"
    plan["meta"]["stage4b_verify_days"] = int(VERIFY_DAYS)
    plan["meta"]["stage4b_gates"] = {
        "NET_FLOOR_PCT": float(NET_FLOOR_PCT),
        "MIN_CLUSTERS_COMPLETED": int(MIN_CLUSTERS_COMPLETED),
        "MAX_DD_PCT": float(MAX_DD_PCT),
    }

    PLAN_FINAL_OUT.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    print(f"\n[4B] MASTER PLAN GENERATED: {PLAN_FINAL_OUT.resolve()}")


if __name__ == "__main__":
    main()
