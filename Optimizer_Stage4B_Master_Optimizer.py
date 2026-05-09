import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

import Run_Final_Verification_RAW_4A as RV
from Hunter_Engine_v29R_next import HunterTactics as Engine

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR      = Path("./results_v29R_30d")
STAGE2_CSV       = RESULTS_DIR / "stage2_intraday_dual_tf.csv"
PLAN_IN_PATH     = Path("./portfolio_plan_v29R_auto.json") # From Stage 1/2
PLAN_FINAL_OUT   = Path("./portfolio_plan_v29R_selected.json") # The final result
TRADES_OUT       = RESULTS_DIR / "trades_4B_master.csv"

# Pruning Settings (Aligned with your v29R setup)
MIN_PRUNE_PF = 1.1
MIN_PRUNE_NET = 0.0
MIN_ABSOLUTE_TRADES = 5

# ============================================================
# SURGERY 1: THE INJECTOR LOGIC
# ============================================================
def inject_stage2_data(plan_base: dict) -> dict:
    if not STAGE2_CSV.exists():
        print(f"⚠️ [WARN] Stage 2 CSV not found.")
        return plan_base

    s2_df = pd.read_csv(STAGE2_CSV)
    
    # Safety Check: Ensure the required columns exist
    required = ["suggested_sl", "suggested_trail", "symbol"]
    missing = [col for col in required if col not in s2_df.columns]
    if missing:
        print(f"❌ [ERROR] Stage 2 CSV is missing columns: {missing}")
        print("Please rerun Stage 2 with the updated flattening logic.")
        return plan_base

    print(f"[4B] Injecting surgical multipliers for {len(s2_df)} assets...")
    
    updated_portfolio = []
    for entry in plan_base.get("portfolio", []):
        sym = str(entry["pair"]).upper().strip()
        profile = s2_df[s2_df["symbol"] == sym]
        
        if not profile.empty:
            row = profile.iloc[0]
            pk = entry.get("perfect_key", {}).copy()
            # Successfully map the flattened columns
            pk["atr_multiplier"] = float(row["suggested_sl"])      
            pk["trail_multiplier"] = float(row["suggested_trail"]) 
            pk["adx_threshold"] = 26.0
            
            new_entry = entry.copy()
            new_entry["perfect_key"] = pk
            new_entry["interval"] = row["best_tf"]
            updated_portfolio.append(new_entry)

    plan_base["portfolio"] = updated_portfolio
    return plan_base

# ============================================================
# SURGERY 2: THE PRUNING LOGIC (Maintained from your version)
# ============================================================
def select_keep_hours(trades_df: pd.DataFrame) -> list[int]:
    if trades_df.empty: return list(range(24))
    
    trades_df["hour"] = pd.to_datetime(trades_df["entry_time"]).dt.hour
    stats = trades_df.groupby("hour")["net_profit"].agg(
        pf=lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if x[x < 0].sum() != 0 else 999,
        net="sum",
        count="count"
    )
    
    keep = stats[(stats["pf"] >= MIN_PRUNE_PF) & (stats["count"] >= MIN_ABSOLUTE_TRADES)].index.tolist()
    return keep if keep else list(range(24))

def main():
    print("======================================================")
    print(" 🚀 STAGE 4B MASTER — The Scientific Cut v30.41")
    print("======================================================")

    # 1. Load the raw plan (now containing all 42 symbols from Stage 2)
    if not PLAN_IN_PATH.exists():
        print(f"❌ Error: {PLAN_IN_PATH} missing.")
        return
    plan = json.loads(PLAN_IN_PATH.read_text())

    # 2. INJECT Stage 2 Surgical Multipliers (Matches 1m/3m intervals)
    plan = inject_stage2_data(plan)

    # 3. VERIFY (The run you just did)
    train_start = pd.to_datetime(plan["meta"]["train_start"], utc=True)
    train_end   = pd.to_datetime(plan["meta"]["train_end"], utc=True)

    print("[4B] Running verification simulation with Surgical Multipliers...")
    trades_df = RV.run_simulation_generic(
        engine_class=Engine,
        simulation_name="4B_Master_Verify",
        portfolio_plan=plan,
        use_perfect_keys=True, 
        engine_kwargs={},
        train_start=train_start,
        train_end=train_end,
        hour_prune=False
    )

    if trades_df is None or trades_df.empty:
        print("❌ Verification failed: No trades.")
        return

    # --- SURGERY: THE DARWINIAN PURGE ---
    # We look at the results per symbol
    symbol_stats = trades_df.groupby("symbol")["net_profit"].sum()
    
    # QUALITY GATE: Keep coins that made money or lost very little (Soft Gate)
    # We kill DASH, FUN, WAN, THETA, ALGO, etc.
    survivors = symbol_stats[symbol_stats > -10.0].index.tolist()
    
    print(f"\n[PURGE] Initial: {len(symbol_stats)} assets | Survivors: {len(survivors)}")
    print(f"☠️  Killed: {list(set(symbol_stats.index) - set(survivors))}")
    
    # 4. Filter the portfolio to only include the Winners/Survivors
    plan["portfolio"] = [p for p in plan["portfolio"] if p["pair"] in survivors]

    # 5. PRUNE: Calculate Best Hours for ONLY the survivors
    survivor_trades = trades_df[trades_df["symbol"].isin(survivors)]
    keep_hours = select_keep_hours(survivor_trades)
    print(f"✅ Hourly Optimization Complete. Keeping: {keep_hours}")

    # 6. ASSEMBLE FINAL PLAN
    plan["meta"]["keep_hours"] = keep_hours
    plan["meta"]["mode"] = "SURGICAL_ASSET_AND_HOUR_PRUNED"
    
    for entry in plan["portfolio"]:
        entry["perfect_key"]["keep_hours"] = keep_hours

    # Save to the 'Selected' JSON for PrePaper
    PLAN_FINAL_OUT.write_text(json.dumps(plan, indent=2))
    print(f"🏁 MASTER PLAN GENERATED: {PLAN_FINAL_OUT.resolve()}")
    
    # Save trades for your audit
    trades_df.to_csv(TRADES_OUT, index=False)

if __name__ == "__main__":
    main()