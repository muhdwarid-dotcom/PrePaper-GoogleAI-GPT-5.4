from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests
import pandas_ta as ta
import sys
from mtf_fib_cluster_engine import MtfFibClusterEngine
from fib_train_verifier import verify_symbol_fib_train
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] v6 started; python={sys.version}", flush=True)

from pathlib import Path
import re

import datetime


class DualLogger:
    def __init__(self, filepath="runlog.txt"):
        self.terminal = sys.stdout
        # overwrite every run
        self.log = open(filepath, "w", encoding="utf-8")
        self.log.write(f"{'='*100}\n--- NEW RUN START: {datetime.datetime.now()} ---\n{'='*100}\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ============================================================
# v30 Trade Window Portfolio Simulator (EventStudy-driven)
# - Fetch OHLCV 1m from Binance for 7-day trade window (+ warmup)
# - Build entry signals on-the-fly (A1 / C0) with v30 cross logic
# - Exit model uses ATR-based k/t + optional barrier x_bars
# - Portfolio:
#     initial_capital = 10,000 USDT
#     trade_size      = 1,000 USDT
#     max positions   = capital-only (maxAvail = floor(capital / trade_size))
#     realized PnL on close only
# - Runs baseline and barrier as separate portfolios
# - AUTO_CYCLE:
#     * runs both scenarios (A1 + C0)
#     * best-of(baseline, barrier) per scenario
#     * select winner across scenarios
#     * PrePaper (next week) runs BARRIER ONLY for winner
#     * writes selection JSON so PrePaper can run without prompts
# - PREPAPER_FROM_JSON:
#     * runs PrePaper (barrier only) from selection JSON, no prompts
# - Console format:
#     SIGNAL (blue) -> OPEN (blue) -> STOP/WINDOW_END (green/red)
# ============================================================

TRAILING_MODE = "immediate"   # "immediate" or "x_bars"

BINANCE_BASE = "https://api.binance.com"
LIMIT = 1000

FEE_RATE = 0.001  # 0.1% buy + 0.1% sell
WARMUP_DAYS = 7
ATR_LEN = 14

# ----------------------------
# Default portfolio
# ----------------------------
DEFAULT_INITIAL_CAPITAL = 10_000.0
DEFAULT_TRADE_SIZE = 1_000.0

# ----------------------------
# REGIME GUARD: 15m ADX gate
# ----------------------------
ADX_GATE_ENABLE = False        # set False to disable without code changes
ADX_TF = "15T"                # 15-minute
ADX_LEN = 14
ADX_MIN = 26.0                # gate threshold

# Apply gate to pyramiding too?
ADX_GATE_APPLY_TO_PYRAMID = True
ADX_PYR_MIN = 30.0            # PYRAMID gate threshold (stricter)

# ----------------------------
# DIRECTION FILTER (15m +DI/-DI)
# ----------------------------
DI_FILTER_ENABLE = False
DI_GATE_APPLY_TO_PYRAMID = True

# ----------------------------
# PYRAMIDING CONFIG
# ----------------------------
PYRAMID_ENABLE = True

# Maximum pyramid adds per BASE position (base itself is not counted).
# Example: 5 => base + up to 5 pyramid legs.
PYR_MAX_ADDS_CAP = 5

# Pyramid vol threshold rule:
# - If scenario has a min vol rule (A1 => >=1.5), use it.
# - If scenario vol_rule == ALL (C0), pyramid vol threshold is >= 1.0
PYR_VOL_THRESHOLD_ALL = 1.0

# Pyramiding requires RSI_SMA incremental AND vol_ratio threshold.
# If either fails once for a base, pyramiding ceases permanently for that base.

OUT_DIR = "forwardtest"
os.makedirs(OUT_DIR, exist_ok=True)

# Console colors (match your old preference)
COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_RESET = "\033[0m"

# Set to a path string to force events from CSV; set to None to disable override.
EVENTS_CSV_OVERRIDE: str | None = None

# Optional: enable override by setting an environment variable:
#   PowerShell: $env:EVENTS_CSV_OVERRIDE=".\some_events.csv"
_env = os.getenv("EVENTS_CSV_OVERRIDE", "").strip()
if _env:
    EVENTS_CSV_OVERRIDE = _env

# If False: do NOT force-close open positions at trade_end.
# Open positions are carried beyond the window, but their P/L is NOT counted in the window summary.
FORCE_CLOSE_AT_WINDOW_END = False

TRAILING_MODE = "immediate"   # "immediate" or "x_bars"

# ----------------------------
# Walk-forward robustness (TRAIN slicing)
# ----------------------------
ROBUST_ENABLE = True
ROBUST_TRAIN_SLICES = 4              # 4 weeks inside TRAIN
ROBUST_MIN_POS_WEEKS = 3             # R1: >=3/4 positive weeks
ROBUST_MEAN_NET_MIN = 150.0          # R1: mean weekly net profit must be >= this
ROBUST_WORST_WEEK_NET_MIN = -200.0   # R2 gate: Average Weekly Drag/Loss (sum negative / 4) must be > -X (tune)
ROBUST_RANK_PRIMARY = "median_profit_over_maxdd"  # for display; we will rank by this after gates
ROBUST_MEAN_POMDD_MIN = 0.8    # R3: mean_profit_over_maxdd must be >= this

# ----------------------------
#  OFF because x<60
# ----------------------------
#X_BARS_MIN_DELAY = 60  # below this, treat as OFF (immediate trailing)
X_BARS_MIN_DELAY = 0  # disable minimum delay; any x_bars (even 0) triggers immediate trailing

def load_event_times_from_csv(path: str) -> set[pd.Timestamp]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "event_time" in df.columns:
        col = "event_time"
    elif "time" in df.columns:
        col = "time"
    else:
        raise ValueError(
            f"events csv must contain 'event_time' or 'time'. columns={list(df.columns)}"
        )

    t = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
    return set(t.tolist())

# ----------------------------
# Console logging
# ----------------------------

# The Master Print Switch
PRINT_PLAY_BY_PLAY = True

def log_line(ts, action: str, symbol: str, price: float, extra: str = "", color: str = ""):
    global PRINT_PLAY_BY_PLAY
    if not PRINT_PLAY_BY_PLAY: 
        return
    
    ts_str = str(pd.to_datetime(ts, utc=True))[:16]
    msg = f"{ts_str} | {action:<10} | {symbol:<9} | Price {price:<10.6f} {extra}".rstrip()
    if color:
        print(f"{color}{msg}{COLOR_RESET}")
    else:
        print(msg)

def format_trade_id(pid: str) -> str:
    parts = pid.split("_")
    if len(parts) >= 4:
        return "_".join(parts[-3:])  # v30_6_PYR1
    if len(parts) >= 3:
        return "_".join(parts[-2:])  # v30_6
    return pid


# ----------------------------
# JSON helpers
# ----------------------------
def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_candidate_json(data: dict) -> dict:
    """
    Safely normalizes 'possibility' and 'scenario' keys in the loaded candidate JSON
    across all finalists, cycle, and candidate blocks.
    """
    if not isinstance(data, dict):
        return data

    for key in ("finalists", "cycle", "candidates"):
        collection = data.get(key, [])
        if not isinstance(collection, list):
            continue
        for x in collection:
            if isinstance(x, dict):
                if "possibility" in x and "scenario" not in x:
                    x["scenario"] = x["possibility"]
                elif "scenario" in x and "possibility" not in x:
                    x["possibility"] = x["scenario"]

                # Case-insensitivity normalization
                if "scenario" in x and "possibility" in x:
                    x["scenario"] = str(x["scenario"]).strip().upper()
                    x["possibility"] = str(x["possibility"]).strip().upper()
    return data


# ----------------------------
# Portfolio helpers
# ----------------------------
def max_avail_slots(current_capital: float, trade_size: float) -> int:
    """Capital-only max positions availability (your rule #2)."""
    if trade_size <= 0:
        return 0
    return int(max(0, np.floor(current_capital / trade_size)))


def can_open_position(current_capital: float, trade_size: float, open_positions: int) -> bool:
    """
    Capital-only capacity:
      free_capital = capital - (open_positions * trade_size)
      must be >= trade_size
    """
    capital_in_use = open_positions * trade_size
    free_capital = current_capital - capital_in_use
    return free_capital >= trade_size

def compute_max_drawdown(eq: pd.DataFrame, capital_col: str = "capital_usdt") -> tuple[float, float]:
    """
    Returns (max_dd_pct, max_dd_usdt)
      - max_dd_pct is negative (e.g., -0.12 for -12%)
      - max_dd_usdt is negative (e.g., -350.0)
    """
    if eq is None or len(eq) == 0 or capital_col not in eq.columns:
        return (0.0, 0.0)

    s = pd.to_numeric(eq[capital_col], errors="coerce").dropna()
    if len(s) == 0:
        return (0.0, 0.0)

    peak = s.cummax()
    dd_usdt = s - peak
    dd_pct = (s / peak) - 1.0

    max_dd_usdt = float(dd_usdt.min())  # most negative
    max_dd_pct = float(dd_pct.min())    # most negative
    return (max_dd_pct, max_dd_usdt)


def add_drawdown_to_summary(summary_df: pd.DataFrame, eq: pd.DataFrame, label: str) -> pd.DataFrame:
    max_dd_pct, max_dd_usdt = compute_max_drawdown(eq, "capital_usdt")
    summary_df = summary_df.copy()
    summary_df["max_dd_pct"] = max_dd_pct
    summary_df["max_dd_usdt"] = max_dd_usdt
    # Optional: simple risk-adjusted score
    denom = abs(max_dd_usdt) if abs(max_dd_usdt) > 1e-9 else 0.0
    summary_df["profit_over_maxdd"] = (summary_df["net_profit"] / denom) if denom else float("inf")
    return summary_df

# ----------------------------
# PrePaper Monday 0800 UTC Helper
# ----------------------------
def next_monday_0800_utc(after_ts: pd.Timestamp) -> pd.Timestamp:
    """
    Returns the next Monday 08:00 UTC strictly AFTER after_ts.
    Rule A: PrePaper must start Monday 08:00 UTC.
    """
    t = pd.to_datetime(after_ts, utc=True)

    # move to next day 00:00 to ensure "strictly after"
    t = (t + pd.Timedelta(minutes=1)).floor("min")

    # pandas: Monday=0 ... Sunday=6
    days_ahead = (0 - t.weekday()) % 7
    candidate = t.normalize() + pd.Timedelta(days=days_ahead) + pd.Timedelta(hours=8)

    # if candidate is not strictly after t, jump 7 days
    if candidate <= t:
        candidate = candidate + pd.Timedelta(days=7)

    return candidate

# ----------------------------
# Monday-aligned 7d slices within TRAIN Helpers
# ----------------------------

def iter_monday_week_slices(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return list of [week_start, week_end) slices aligned to Monday 00:00 UTC.
    Only returns full 7-day slices fully contained in [start_utc, end_utc).
    """
    start_utc = pd.to_datetime(start_utc, utc=True)
    end_utc = pd.to_datetime(end_utc, utc=True)

    # find first Monday 00:00 >= start_utc
    t = start_utc.floor("D")
    while t.weekday() != 0:
        t += pd.Timedelta(days=1)
    t = t.replace(hour=0, minute=0, second=0, microsecond=0)

    slices: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while True:
        w0 = t
        w1 = w0 + pd.Timedelta(days=7)
        if w0 < start_utc:
            t += pd.Timedelta(days=7)
            continue
        if w1 > end_utc:
            break
        slices.append((w0, w1))
        t += pd.Timedelta(days=7)

    return slices

# ----------------------------
# build_events_all_for_robustness
# ----------------------------
def build_events_all_for_robustness(
    d_features: pd.DataFrame,
    train_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    scenario: str,
) -> pd.DataFrame:
    """
    Full-range events used for robustness slicing.
    If EVENTS_CSV_OVERRIDE is set, uses CSV times (unfiltered).
    Otherwise uses build_events over [train_start, trade_end).
    """
    if EVENTS_CSV_OVERRIDE:
        forced_times = load_event_times_from_csv(EVENTS_CSV_OVERRIDE)
        ev = pd.DataFrame({"event_time": sorted(forced_times)})
        ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
        return ev.reset_index(drop=True)

    # Non-override: generate from features across the full range
    ev = build_events(d_features, train_start, trade_end, scenario)
    ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
    return ev.reset_index(drop=True)

# ----------------------------
# Indicators (Wilder smoothing)
# ----------------------------
def wilders_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = wilders_rma(delta.clip(lower=0), length)
    avg_loss = wilders_rma(-delta.clip(upper=0), length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


# ----------------------------
# Binance OHLCV fetch
# ----------------------------
def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
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


def get_ohlcv_binance(symbol: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    start_utc = pd.to_datetime(start_utc, utc=True)
    end_utc = pd.to_datetime(end_utc, utc=True)

    start_ms = int(start_utc.value // 10**6)
    end_ms = int(end_utc.value // 10**6)

    rows = []
    cur = start_ms
    while cur < end_ms:
        data = fetch_klines(symbol, INTERVAL, cur, end_ms)
        if not data:
            break

        rows.extend(data)
        last_open = data[-1][0]
        cur = last_open + 60_000  # next minute

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

# ----------------------------
# Evaluate ONE candidate across TRAIN slices
# ----------------------------

def eval_candidate_robustness_over_train(
    *,
    pair: str,
    scenario: str,
    slices: list[tuple[pd.Timestamp, pd.Timestamp]],
    initial_capital: float,
    trade_size: float,
    interval: str = "1m",
    close_gate: Any = "ALL",
    vol_gate: Any = "ALL",
    vol_rule: str = "ALL",
) -> Dict[str, Any]:
    """Run Fibonacci engine verification across each TRAIN weekly slice.

    Replaces legacy ATR-based simulation.  For each slice, calls
    verify_symbol_fib_train and collects Fibonacci metrics.
    """
    rows: list[dict[str, Any]] = []

    for i, (w0, w1) in enumerate(slices, start=1):
        fib_result = verify_symbol_fib_train(
            pair=pair,
            interval=interval,
            train_start=w0,
            train_end=w1,
            initial_capital=initial_capital,
            trade_size=trade_size,
            close_gate=close_gate,
            vol_gate=vol_gate,
            vol_rule=vol_rule,
            verbose=True,
        )

        net = float(fib_result.get("net_profit_usdt", 0.0))
        mdd_usdt = float(fib_result.get("max_dd_usdt", 0.0))
        clusters = int(fib_result.get("clusters_completed", 0))
        trades_cl = int(fib_result.get("trades_closed", 0))

        pomdd = (net / abs(mdd_usdt)) if mdd_usdt != 0.0 else float("inf")

        rows.append({
            "slice_ix": int(i),
            "slice_start": w0,
            "slice_end": w1,
            "net_profit": net,
            "max_dd_usdt": mdd_usdt,
            "profit_over_maxdd": pomdd,
            "clusters_completed": clusters,
            "trades_closed": trades_cl,
        })

    df = pd.DataFrame(rows)

    # Per-slice flags + aggregates
    if not df.empty:
        df["r2_week_pass"] = df["net_profit"].astype(float) > float(ROBUST_WORST_WEEK_NET_MIN)

        # R1: count winning weeks (net_profit >= 0)
        pos_weeks = int((df["net_profit"].astype(float) >= 0.0).sum())
        worst_week_net = float(df["net_profit"].astype(float).min())
        total_net_profit = float(df["net_profit"].astype(float).sum())
        mean_net_profit = float(df["net_profit"].astype(float).mean())

        # R2: Average Weekly Drag/Loss over 4 weeks (sum of negative weeks / 4)
        total_loss = float(df[df["net_profit"].astype(float) < 0]["net_profit"].astype(float).sum())
        avg_weekly_loss = float(total_loss / len(df)) if len(df) > 0 else 0.0

        # R1: pos_weeks >= ROBUST_MIN_POS_WEEKS AND mean_net_profit >= ROBUST_MEAN_NET_MIN
        passes_robust_r1 = bool(
            pos_weeks >= int(ROBUST_MIN_POS_WEEKS)
            and mean_net_profit >= float(ROBUST_MEAN_NET_MIN)
        )

        # R3: Mean profit_over_maxdd
        mean_profit_over_maxdd = float(df["profit_over_maxdd"].astype(float).mean())
    else:
        pos_weeks = 0
        worst_week_net = 0.0
        passes_robust_r1 = False
        total_net_profit = 0.0
        mean_net_profit = 0.0
        avg_weekly_loss = 0.0
        mean_profit_over_maxdd = 0.0

    out: Dict[str, Any] = {
        "pair": pair,
        "scenario": scenario,

        "n_slices": int(len(df)),
        "pos_weeks": int(pos_weeks),
        "passes_robust_r1": bool(passes_robust_r1),
        "total_net_profit": float(total_net_profit),

        "worst_week_net_profit": float(worst_week_net),
        "mean_net_profit": float(mean_net_profit),
        "avg_weekly_loss": float(avg_weekly_loss),
        "mean_profit_over_maxdd": float(mean_profit_over_maxdd),

        "df_slices": df,
    }
    return out

# ----------------------------
# Entry features + events (A1 / C0)
# ----------------------------
def compute_entry_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    d = ohlcv.copy()

    d["rsi"] = rsi_wilder(d["close"], 14)
    d["rsi_sma"] = sma(d["rsi"], 14)

    # SMMA200 (Wilder RMA(200))
    d["smma_200"] = wilders_rma(d["close"], 200)
    d["close_gt_smma_200"] = d["close"] > d["smma_200"]

    # Volume SMA(20) + ratio
    d["vol_sma_20"] = sma(d["volume"], 20)
    d["vol_gt_vol_sma"] = d["volume"] > d["vol_sma_20"]
    d["vol_ratio"] = d["volume"] / d["vol_sma_20"]

    # keep only valid rows
    return d.dropna().reset_index(drop=True)

def compute_adx_15m_maps(ohlcv_1m: pd.DataFrame) -> Dict[str, Dict[pd.Timestamp, float]]:
    """
    Compute ADX(+DI/-DI) on 15m bars from 1m OHLCV.
    Returns dict of maps keyed by 1m timestamps:
      {
        "adx_15m": {ts: val},
        "dmp_15m": {ts: val},  # +DI
        "dmn_15m": {ts: val},  # -DI
      }
    """
    d = ohlcv_1m[["time", "open", "high", "low", "close", "volume"]].copy()
    d = d.sort_values("time").set_index("time")

    o15 = d.resample(ADX_TF).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    if o15.empty:
        return {"adx_15m": {}, "dmp_15m": {}, "dmn_15m": {}}

    adx_df = ta.adx(o15["high"], o15["low"], o15["close"], length=ADX_LEN)
    adx_col = f"ADX_{ADX_LEN}"
    dmp_col = f"DMP_{ADX_LEN}"
    dmn_col = f"DMN_{ADX_LEN}"

    if (
        adx_df is None
        or adx_df.empty
        or adx_col not in adx_df.columns
        or dmp_col not in adx_df.columns
        or dmn_col not in adx_df.columns
    ):
        return {"adx_15m": {}, "dmp_15m": {}, "dmn_15m": {}}

    o15["adx_15m"] = adx_df[adx_col]
    o15["dmp_15m"] = adx_df[dmp_col]
    o15["dmn_15m"] = adx_df[dmn_col]
    o15 = o15.dropna(subset=["adx_15m", "dmp_15m", "dmn_15m"]).copy()

    adx_series = o15["adx_15m"].reindex(d.index, method="ffill").dropna()
    dmp_series = o15["dmp_15m"].reindex(d.index, method="ffill").dropna()
    dmn_series = o15["dmn_15m"].reindex(d.index, method="ffill").dropna()

    return {
        "adx_15m": adx_series.to_dict(),
        "dmp_15m": dmp_series.to_dict(),
        "dmn_15m": dmn_series.to_dict(),
    }

def _parse_possibility(possibility: str) -> dict:
    s = possibility.strip().upper()

    # A) R_ALL
    m_all = re.fullmatch(r"C_(TRUE|FALSE|ALL)__V_(TRUE|FALSE|ALL)__R_ALL", s)
    if m_all:
        close_s, vol_s = m_all.group(1), m_all.group(2)
        return {"close": close_s, "vol": vol_s, "r_op": "ALL", "r_value": None}

    # B) R_(LT|GE)_<number>
    m = re.fullmatch(
        r"C_(TRUE|FALSE|ALL)__V_(TRUE|FALSE|ALL)__R_(LT|GE)_([0-9]+(?:\.[0-9]+)?)",
        s
    )
    if m:
        close_s, vol_s, r_op, r_val = m.group(1), m.group(2), m.group(3), float(m.group(4))
        return {"close": close_s, "vol": vol_s, "r_op": r_op, "r_value": r_val}

    # C) BIN: R_<low>_<high>
    m_bin = re.fullmatch(
        r"C_(TRUE|FALSE|ALL)__V_(TRUE|FALSE|ALL)__R_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)",
        s
    )
    if m_bin:
        close_s, vol_s = m_bin.group(1), m_bin.group(2)
        lo, hi = float(m_bin.group(3)), float(m_bin.group(4))
        if not (hi > lo):
            raise ValueError(f"Invalid R bin (high must be > low): {possibility}")
        return {"close": close_s, "vol": vol_s, "r_op": "BIN", "r_low": lo, "r_high": hi, "r_value": None}

    raise ValueError(f"Invalid possibility format: {possibility}")

def get_exit_params_from_finalist(f: Dict[str, Any]) -> tuple[float, float, int]:
    """
    finalist["exit_params"] = {"k": float, "t": float, "x_bars": Optional[int]}
    x_bars semantics:
      - missing / None / <= 0 => disable time barrier
      - > 0 => enable time barrier
    """
    exitp = f.get("exit_params", {}) or {}

    k = float(exitp["k"])
    t = float(exitp["t"])

    x_raw = exitp.get("x_bars", 0)
    if x_raw in (None, "", "None", "null"):
        x_bars = 0
    else:
        x_bars = int(x_raw)

    if x_bars < 0:
        x_bars = 0

    return k, t, x_bars

def build_events(d, trade_start, trade_end, scenario):
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("scenario must be a non-empty string")

    scenario = scenario.strip().upper()

    # Base event generator (same as before): RSI_SMA cross up 51
    prev = d["rsi_sma"].shift(1)
    curr = d["rsi_sma"]
    cross_up_51 = (prev < 51.0) & (curr >= 51.0)

    ev = d.loc[cross_up_51].copy()
    ev = ev[(ev["time"] >= trade_start) & (ev["time"] < trade_end)].copy()
    
    print(f"[DEBUG][events] base cross_up_51 in-window count = {len(ev)}")

    if scenario == "A1":
        ev = ev[
            (ev["close_gt_smma_200"] == True) &
            (ev["vol_gt_vol_sma"] == True) &
            (ev["vol_ratio"] >= 1.5)
        ].copy()

    elif scenario == "C0":
        ev = ev[
            (ev["close_gt_smma_200"] == False) &
            (ev["vol_gt_vol_sma"] == True)
        ].copy()

    elif scenario.startswith("C_"):
        p = _parse_possibility(scenario)
        print(f"[DEBUG][events] parsed possibility = {p}")

        # close filter
        if p["close"] == "TRUE":
            before = len(ev)
            ev = ev[ev["close_gt_smma_200"] == True].copy()
            print(f"[DEBUG][events] close TRUE: {before} -> {len(ev)}")
        elif p["close"] == "FALSE":
            before = len(ev)
            ev = ev[ev["close_gt_smma_200"] == False].copy()
            print(f"[DEBUG][events] close FALSE: {before} -> {len(ev)}")
        else:
            print(f"[DEBUG][events] close ALL: {len(ev)} (no filter)")

        # vol filter
        if p["vol"] == "TRUE":
            before = len(ev)
            ev = ev[ev["vol_gt_vol_sma"] == True].copy()
            print(f"[DEBUG][events] vol TRUE: {before} -> {len(ev)}")
        elif p["vol"] == "FALSE":
            before = len(ev)
            ev = ev[ev["vol_gt_vol_sma"] == False].copy()
            print(f"[DEBUG][events] vol FALSE: {before} -> {len(ev)}")
        else:
            print(f"[DEBUG][events] vol ALL: {len(ev)} (no filter)")

        # R (ratio_vol) filter -> your column vol_ratio
        before = len(ev)

        if p["r_op"] == "ALL":
            # no filter
            print(f"[DEBUG][events] R ALL: {before} -> {len(ev)}")

        elif p["r_op"] == "LT":
            ev = ev[ev["vol_ratio"] < float(p["r_value"])].copy()
            print(f"[DEBUG][events] R LT {p['r_value']}: {before} -> {len(ev)}")

        elif p["r_op"] == "GE":
            ev = ev[ev["vol_ratio"] >= float(p["r_value"])].copy()
            print(f"[DEBUG][events] R GE {p['r_value']}: {before} -> {len(ev)}")

        elif p["r_op"] == "BIN":
            lo = float(p["r_low"])
            hi = float(p["r_high"])
            ev = ev[(ev["vol_ratio"] >= lo) & (ev["vol_ratio"] < hi)].copy()
            print(f"[DEBUG][events] R BIN {lo}_{hi}: {before} -> {len(ev)}")

        else:
            raise ValueError(f"Unknown r_op={p['r_op']}")
        print(f"[DEBUG][events] R {p['r_op']} {p['r_value']}: {before} -> {len(ev)}")

    else:
        raise ValueError(f"Unsupported scenario='{scenario}'. Expected 'A1', 'C0', or a 'C_*' possibility.")

    out = ev[[
        "time", "close", "rsi_sma", "smma_200",
        "close_gt_smma_200", "vol_gt_vol_sma", "vol_ratio"
    ]].rename(columns={"time": "event_time", "close": "entry_close"}).reset_index(drop=True)

    return out

# ----------------------------
# Positions
# ----------------------------
@dataclass
class Position:
    pid: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    atr_entry: float
    fixed_stop: float
    trail_dist: float
    peak_high: float
    bars_held: int = 0
    trailing_active: bool = False

    # --- pyramiding identity ---
    base_id: str = ""          # base leg id; base leg has base_id == pid
    is_pyramid: bool = False   # True for pyramid legs
    pyr_level: int = 0         # 0=base, 1..N pyramid level

    # --- pyramiding control (base leg only) ---
    pyr_ceased: bool = False   # once True: never pyramid again for this base
    pyr_adds_done: int = 0     # how many pyramid legs have been opened for this base

    # --- MTF Fib clustered engine ---
    cluster_id: str = ""
    fib_000_locked: float = np.nan
    fib_100_locked: float = np.nan
    current_cluster_sl: float = np.nan
    highest_price_since_entry: float = np.nan


def open_position(
    pid: str,
    ts: pd.Timestamp,
    entry_price: float,
    atr_entry: float,
    k: float,
    t: float,
    trade_size: float,
    *,
    base_id: str,
    is_pyramid: bool,
    pyr_level: int,
    cluster_id: str = "",
    fib_000_locked: float = np.nan,
    fib_100_locked: float = np.nan,
    current_cluster_sl: float = np.nan,
    highest_price_since_entry: float = np.nan,
) -> Position:
    qty = trade_size / entry_price
    fixed_stop = entry_price - (k * atr_entry)
    trail_dist = t * atr_entry

    return Position(
        pid=pid,
        entry_time=ts,
        entry_price=entry_price,
        qty=qty,
        atr_entry=atr_entry,
        fixed_stop=fixed_stop,
        trail_dist=trail_dist,
        peak_high=entry_price,

        base_id=base_id,
        is_pyramid=is_pyramid,
        pyr_level=pyr_level,

        # base leg defaults:
        pyr_ceased=False if not is_pyramid else True,   # pyramids don't control pyramiding
        pyr_adds_done=0,

        cluster_id=cluster_id,
        fib_000_locked=fib_000_locked,
        fib_100_locked=fib_100_locked,
        current_cluster_sl=current_cluster_sl,
        highest_price_since_entry=(
            highest_price_since_entry if np.isfinite(highest_price_since_entry) else entry_price
        ),
    )


def close_position(pos: Position, ts: pd.Timestamp, exit_price: float, reason: str, trade_size: float) -> Dict[str, Any]:
    entry_val = trade_size
    exit_val = pos.qty * exit_price
    buy_fee = entry_val * FEE_RATE
    sell_fee = exit_val * FEE_RATE
    pnl = (exit_val - entry_val) - (buy_fee + sell_fee)

    return {
        "position_id": pos.pid,
        "entry_time": pos.entry_time,
        "exit_time": ts,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "qty": pos.qty,
        "atr_entry": pos.atr_entry,
        "fixed_stop": pos.fixed_stop,
        "trail_dist": pos.trail_dist,
        "bars_held": pos.bars_held,
        "reason": reason,
        "buy_fee_usdt": buy_fee,
        "sell_fee_usdt": sell_fee,
        "net_pnl_usdt": pnl,
    }


# ----------------------------
# Portfolio simulation (bar-by-bar)
# ----------------------------
def run_portfolio_sim(
    *,
    mode: str,
    pair: str,
    scenario: str,
    ohlcv: pd.DataFrame,
    d_features: pd.DataFrame,
    events: pd.DataFrame,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    k: float,
    t: float,
    x_bars: int,
    initial_capital: float,
    trade_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    mode = mode.strip().lower()
    if mode not in {"baseline", "barrier", "mtf_fib_cluster"}:
        raise ValueError("mode must be baseline, barrier, or mtf_fib_cluster")
    fib_mode = mode == "mtf_fib_cluster"

    # event times for O(1) check
    event_times = set(pd.to_datetime(events["event_time"], utc=True).tolist())

    # feature lookup by timestamp for SIGNAL printing
    feat = d_features.set_index("time")[[
        "rsi_sma", "smma_200", "close_gt_smma_200",
        "vol_gt_vol_sma", "vol_ratio"
    ]]
    feat_map = feat.to_dict("index")

    window = ohlcv[(ohlcv["time"] >= trade_start) & (ohlcv["time"] <= trade_end)].copy()
    if window.empty:
        return pd.DataFrame(), pd.DataFrame(), {"opens_count": 0, "closes_count": 0, "open_positions_end": 0}

    ema50_map: Dict[pd.Timestamp, float] = {}
    fib_engine = None
    if fib_mode:
        ema50_map = (
            ohlcv.sort_values("time")
            .assign(ema50=lambda x: x["close"].ewm(span=50, adjust=False).mean())
            .set_index("time")["ema50"]
            .to_dict()
        )
        fib_engine = MtfFibClusterEngine(symbol=pair, ohlcv_1m=ohlcv)

    current_capital = float(initial_capital)
    positions: Dict[str, Position] = {}
    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    next_id = 1
    opens_count = 0
    closes_count = 0
        
    stops_with_trailing = 0
    stops_without_trailing = 0
    max_bars_held_at_stop = 0

    for _, bar in window.iterrows():
        ts = bar["time"]
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        atr = float(bar.get("atr", np.nan))
        ema50 = float(ema50_map.get(ts, np.nan)) if fib_mode else np.nan
        fib_immediate_entry = False

        # 1) exits
        if fib_mode:
            if positions:
                cluster_sl = fib_engine.update_cluster_sl(ts=ts, bar_high=h, ltf_ema50=ema50)
                for pos in positions.values():
                    pos.bars_held += 1
                    pos.highest_price_since_entry = max(float(pos.highest_price_since_entry), h)
                    pos.current_cluster_sl = cluster_sl

                if np.isfinite(cluster_sl) and l <= cluster_sl:
                    exit_price = max(o, cluster_sl)
                    for pid, pos in list(positions.items()):
                        tr = close_position(pos, ts, exit_price, "FIB_CLUSTER_SL", trade_size)
                        trades.append(tr)
                        closes_count += 1
                        current_capital += float(tr["net_pnl_usdt"])
                        del positions[pid]
                        pnl = float(tr["net_pnl_usdt"])
                        color = COLOR_GREEN if pnl > 0 else COLOR_RED
                        open_cnt = len(positions)
                        avail = max_avail_slots(current_capital, trade_size)
                        log_line(
                            ts, "STOP", pair, exit_price,
                            extra=f"| ID {format_trade_id(pos.pid):<10} | P/L ${pnl:>8.2f} | Cap ${current_capital:>10.2f} | Port {open_cnt:02d}/{avail:02d}",
                            color=color
                        )
                    fib_engine.trigger_cooldown(ts=ts)
        else:
            for pid, pos in list(positions.items()):
                pos.bars_held += 1
                pos.peak_high = max(pos.peak_high, h)
                                  
                # Runtime policy:
                # trailing activates immediately in production flow.
                # x_bars is retained for record/research only unless TRAILING_MODE == "x_bars".
                if TRAILING_MODE == "immediate":
                    pos.trailing_active = True
                elif TRAILING_MODE == "x_bars":
                    effective_x = x_bars if x_bars >= X_BARS_MIN_DELAY else 0
                    if effective_x == 0:
                        pos.trailing_active = True
                    elif pos.bars_held > effective_x:
                        pos.trailing_active = True
                else:
                    raise ValueError(f"Unsupported TRAILING_MODE={TRAILING_MODE!r}")

                trail_stop = -np.inf
                if pos.trailing_active:
                    trail_stop = pos.peak_high - pos.trail_dist

                stop_level = max(pos.fixed_stop, trail_stop)

                if l <= stop_level:
                    max_bars_held_at_stop = max(max_bars_held_at_stop, pos.bars_held)
                    if pos.trailing_active:
                        stops_with_trailing += 1
                    else:
                        stops_without_trailing += 1

                    exit_price = max(o, stop_level)
                    tr = close_position(pos, ts, exit_price, "STOP", trade_size)
                    trades.append(tr)
                    closes_count += 1

                    current_capital += float(tr["net_pnl_usdt"])
                    del positions[pid]

                    pnl = float(tr["net_pnl_usdt"])
                    color = COLOR_GREEN if pnl > 0 else COLOR_RED
                    open_cnt = len(positions)
                    avail = max_avail_slots(current_capital, trade_size)

                    log_line(
                        ts, "STOP", pair, exit_price,
                        extra=f"| ID {format_trade_id(pos.pid):<10} | P/L ${pnl:>8.2f} | Cap ${current_capital:>10.2f} | Port {open_cnt:02d}/{avail:02d}",
                        color=color
                    )

        # 2) entries
        if ts in event_times:
            f = feat_map.get(ts, {})
            rsi_sma = float(f.get("rsi_sma", np.nan))
            smma200 = float(f.get("smma_200", np.nan))
            c_gt = f.get("close_gt_smma_200", None)
            v_gt = f.get("vol_gt_vol_sma", None)
            vr = float(f.get("vol_ratio", np.nan))
                        # ADX/DI values (only used if gates are enabled)
            adx15 = np.nan
            dmp15 = np.nan
            dmn15 = np.nan
            if ADX_GATE_ENABLE or DI_FILTER_ENABLE:
                adx15 = float(f.get("adx_15m", np.nan))
                dmp15 = float(f.get("dmp_15m", np.nan))  # +DI
                dmn15 = float(f.get("dmn_15m", np.nan))  # -DI
            # adx15 = float(f.get("adx_15m", np.nan))
            # dmp15 = float(f.get("dmp_15m", np.nan))  # +DI
            # dmn15 = float(f.get("dmn_15m", np.nan))  # -DI

            # SIGNAL row (blue)
            log_line(
                ts, "SIGNAL", pair, c,
                extra=f"| RSI_SMA {rsi_sma:5.2f} | SMMA {smma200:.5f} | "
                    f"C>SMMA {str(c_gt):<5} | V>VSMA {str(v_gt):<5} | "
                    f"VR {vr:>5.2f}",
                color=COLOR_BLUE
            )

            # 1) ADX strength gate (existing)
            if ADX_GATE_ENABLE:
                if not np.isfinite(adx15) or adx15 < ADX_MIN:
                    log_line(ts, "SKIP_ADX", pair, c, extra=f"| ADX15 {adx15:>5.2f} < {ADX_MIN:.2f}")
                    continue

            # 2) DI direction gate (NEW)
            if DI_FILTER_ENABLE:
                if (not np.isfinite(dmp15)) or (not np.isfinite(dmn15)) or (dmp15 <= dmn15):
                    log_line(ts, "SKIP_DI", pair, c, extra=f"| DMP {dmp15:>5.2f} <= DMN {dmn15:>5.2f}")
                    continue

            if fib_mode:
                if len(positions) == 0:
                    route_result = fib_engine.on_spearhead(
                        ts=ts,
                        ltf_open=o,
                        ltf_high=h,
                        ltf_low=l,
                        ltf_close=c,
                        ltf_ema50=ema50,
                    )
                    fib_immediate_entry = bool(route_result.get("immediate_entry", False))
            else:
                # ATR must exist
                if not np.isfinite(atr) or atr <= 0:
                    continue

                if can_open_position(current_capital, trade_size, open_positions=len(positions)):
                    posid = f"{pair}_v30_{next_id}"
                    pos = open_position(
                        posid, ts, entry_price=c, atr_entry=atr, k=k, t=t, trade_size=trade_size,
                        base_id=posid, is_pyramid=False, pyr_level=0
                    )
                    positions[posid] = pos
                    opens_count += 1

                    open_cnt = len(positions)
                    avail = max_avail_slots(current_capital, trade_size)

                    # OPEN row (blue), compact (no repeated k/t/SL)
                    log_line(
                        ts, "OPEN", pair, c,
                        extra=f"| PosID {posid:<18} | Port {open_cnt:02d}/{avail:02d}",
                        color=COLOR_BLUE
                    )
                    next_id += 1
                else:
                    open_cnt = len(positions)
                    avail = max_avail_slots(current_capital, trade_size)
                    log_line(
                        ts, "SKIP", pair, c,
                        extra=f"| no capital | Port {open_cnt:02d}/{avail:02d}"
                    )

        if fib_mode:
            if fib_engine.cooldown_active:
                fib_engine.maybe_release_cooldown(ts=ts, ltf_price=c)
            elif len(positions) == 0:
                fib_engine.apply_pre_entry_wipes(ts=ts, ltf_high=h, ltf_low=l, ltf_price=c)
                entry_window_open = (ts not in event_times) or fib_immediate_entry
                if entry_window_open and fib_engine.should_enter(ltf_low=l, ltf_close=c, ltf_ema50=ema50):
                    tickets = int(fib_engine.pending_triggers)
                    free_slots = max_avail_slots(current_capital, trade_size) - len(positions)
                    if tickets > 0 and free_slots >= tickets:
                        cluster_id = f"{pair}_FIBCL_{next_id}"
                        fib_engine.lock_cluster(cluster_id=cluster_id, ts=ts, entry_price=c, ltf_ema50=ema50)
                        for _ in range(tickets):
                            posid = f"{pair}_v30_{next_id}"
                            next_id += 1
                            atr_for_book = atr if np.isfinite(atr) else 0.0
                            pos = open_position(
                                posid,
                                ts,
                                entry_price=c,
                                atr_entry=atr_for_book,
                                k=0.0,
                                t=0.0,
                                trade_size=trade_size,
                                base_id=cluster_id,
                                is_pyramid=False,
                                pyr_level=0,
                                cluster_id=cluster_id,
                                fib_000_locked=fib_engine.locked_fib_000,
                                fib_100_locked=fib_engine.locked_fib_100,
                                current_cluster_sl=fib_engine.current_cluster_sl,
                                highest_price_since_entry=fib_engine.highest_price_since_entry,
                            )
                            positions[posid] = pos
                            opens_count += 1
                            open_cnt = len(positions)
                            avail = max_avail_slots(current_capital, trade_size)
                            log_line(
                                ts, "OPEN", pair, c,
                                extra=f"| PosID {posid:<18} | Cluster {cluster_id} | Port {open_cnt:02d}/{avail:02d}",
                                color=COLOR_BLUE
                            )
                        print(
                            f"[FIB_MTF][{pair}] clustered_entry ts={pd.to_datetime(ts, utc=True)} "
                            f"cluster={cluster_id} tickets={tickets} entry={c:.8f}",
                            flush=True,
                        )
                    elif tickets > 0:
                        print(
                            f"[FIB_MTF][{pair}] entry_blocked_no_capital ts={pd.to_datetime(ts, utc=True)} "
                            f"pending={tickets} free_slots={free_slots}",
                            flush=True,
                        )

        # ============================================================
        # PYRAMIDING: attempt every bar after base OPEN until failure.
        # Conditions to add (MUST satisfy BOTH):
        #   1) rsi_sma(now) > rsi_sma(prev)
        #   2) vol_ratio(now) >= pyr_vol_min
        # If either fails ONCE for a base => base.pyr_ceased=True forever.
        #
        # Vol threshold rule:
        #   - A1 => >=1.5
        #   - C0 => >=1.0
        # ============================================================
        if PYRAMID_ENABLE and not fib_mode:
            scen = scenario.upper()
            pyr_vol_min = 1.5 if scen == "A1" else PYR_VOL_THRESHOLD_ALL  # C0 => 1.0

            f_now = feat_map.get(ts, {})
                        # ADX/DI values (only used if gates are enabled)
            adx_now = np.nan
            dmp_now = np.nan
            dmn_now = np.nan
            if ADX_GATE_ENABLE or DI_FILTER_ENABLE:
                adx_now = float(f_now.get("adx_15m", np.nan))
                dmp_now = float(f_now.get("dmp_15m", np.nan))
                dmn_now = float(f_now.get("dmn_15m", np.nan))

            # adx_now = float(f_now.get("adx_15m", np.nan))
            # dmp_now = float(f_now.get("dmp_15m", np.nan))
            # dmn_now = float(f_now.get("dmn_15m", np.nan))

            # ADX gate for pyramiding (strength)
            if ADX_GATE_ENABLE and ADX_GATE_APPLY_TO_PYRAMID:
                if not np.isfinite(adx_now) or adx_now < ADX_PYR_MIN:
                    continue

            # DI gate for pyramiding (direction)
            if DI_FILTER_ENABLE and DI_GATE_APPLY_TO_PYRAMID:
                if (not np.isfinite(dmp_now)) or (not np.isfinite(dmn_now)) or (dmp_now <= dmn_now):
                    continue
            
            f_prev = feat_map.get(ts - pd.Timedelta(minutes=1), {})

            vr_now = float(f_now.get("vol_ratio", np.nan))
            rsi_now = float(f_now.get("rsi_sma", np.nan))
            rsi_prev = float(f_prev.get("rsi_sma", np.nan))

            ok_vol = np.isfinite(vr_now) and (vr_now >= pyr_vol_min)
            ok_rsi = np.isfinite(rsi_now) and np.isfinite(rsi_prev) and (rsi_now > rsi_prev)

            # iterate base legs only (not pyramids)
            for base in [p for p in positions.values() if (not p.is_pyramid and p.base_id == p.pid)]:
                if base.pyr_ceased:
                    continue

                # only start trying after at least 1 bar passed since base entry
                if base.bars_held < 1:
                    continue

                # cap number of adds
                if base.pyr_adds_done >= PYR_MAX_ADDS_CAP:
                    base.pyr_ceased = True
                    continue

                # if either condition fails once => cease forever for this base
                if not (ok_vol and ok_rsi):
                    base.pyr_ceased = True
                    continue

                # must have ATR for the pyramid leg at this bar
                if not (np.isfinite(atr) and atr > 0):
                    continue

                # capital check
                if not can_open_position(current_capital, trade_size, open_positions=len(positions)):
                    break

                # open next pyramid leg
                level = base.pyr_adds_done + 1
                posid = f"{base.pid}_PYR{level}"

                pos = open_position(
                    posid, ts, entry_price=c, atr_entry=atr, k=k, t=t, trade_size=trade_size,
                    base_id=base.pid, is_pyramid=True, pyr_level=level
                )
                positions[posid] = pos
                opens_count += 1
                base.pyr_adds_done += 1

                open_cnt = len(positions)
                avail = max_avail_slots(current_capital, trade_size)

                log_line(
                    ts, f"PYR{level}", pair, c,
                    extra=f"| PosID {posid:<18} | RSI {rsi_now:5.2f}>{rsi_prev:5.2f} | "
                        f"VR {vr_now:>4.2f}>={pyr_vol_min:.2f} | Port {open_cnt:02d}/{avail:02d}",
                    color=COLOR_BLUE
                )

        # 3) equity snapshot (realized only)
        equity_rows.append({
            "time": ts,
            "capital_usdt": current_capital,
            "open_positions": len(positions),
        })

    if fib_mode and len(positions) > 0:
        final_ts = window.iloc[-1]["time"]
        final_close = float(window.iloc[-1]["close"])
        for pid, pos in list(positions.items()):
            tr = close_position(pos, final_ts, final_close, "WINDOW_END_MTM", trade_size)
            trades.append(tr)
            closes_count += 1
            current_capital += float(tr["net_pnl_usdt"])
            del positions[pid]
            pnl = float(tr["net_pnl_usdt"])
            color = COLOR_GREEN if pnl > 0 else COLOR_RED
            log_line(
                final_ts, "WINDOW_END", pair, final_close,
                extra=f"| ID {format_trade_id(pos.pid):<10} | P/L ${pnl:>8.2f} | Cap ${current_capital:>10.2f}",
                color=color
            )
        equity_rows.append({"time": final_ts, "capital_usdt": current_capital, "open_positions": 0})

    # 4) end-of-window forced closes (optional)
    open_positions_end = len(positions)
    sim_counts = {
        "opens_count": opens_count,
        "closes_count": closes_count,
        "open_positions_end": open_positions_end,
    }
    
    # In quiet mode (Plan B), suppress this noise.
    # In verbose mode (Plan A), keep it for diagnostics.
    if PRINT_PLAY_BY_PLAY:
        print(
            f"[TRAIL_PROOF] x_bars={x_bars} "
            f"stops_with_trailing={stops_with_trailing} "
            f"stops_without_trailing={stops_without_trailing} "
            f"max_bars_held_at_stop={max_bars_held_at_stop}",
            flush=True,
        )
    
    return pd.DataFrame(trades), pd.DataFrame(equity_rows), sim_counts


def summarize_trades(trades_df: pd.DataFrame, label: str) -> Dict[str, Any]:
    if trades_df is None or trades_df.empty:
        return {"label": label, "trades": 0, "net_profit": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "avg_pnl": 0.0}

    net = float(trades_df["net_pnl_usdt"].sum())
    wins = trades_df[trades_df["net_pnl_usdt"] > 0]["net_pnl_usdt"]
    losses = trades_df[trades_df["net_pnl_usdt"] <= 0]["net_pnl_usdt"]

    gross_win = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses.sum())) if not losses.empty else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
    wr = float((trades_df["net_pnl_usdt"] > 0).mean() * 100.0)
    avg = float(trades_df["net_pnl_usdt"].mean())

    return {"label": label, "trades": int(len(trades_df)), "net_profit": net, "win_rate": wr, "profit_factor": pf, "avg_pnl": avg}


def pick_best_mode_for_scenario(
    scenario: str,
    summary_baseline: Dict[str, Any],
    summary_barrier: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Best-of(baseline, barrier) for a single scenario.
    Primary: higher net_profit
    Tie-breakers: higher profit_factor, higher win_rate, higher avg_pnl, more trades
    """
    a = dict(summary_baseline)
    b = dict(summary_barrier)

    a["scenario"] = scenario
    b["scenario"] = scenario

    for k in ["profit_over_maxdd", "net_profit", "profit_factor", "win_rate", "avg_pnl", "trades"]:
        a[k] = float(a.get(k, 0) or 0)
        b[k] = float(b.get(k, 0) or 0)

    def score(x: Dict[str, Any]) -> tuple:
        return (
            x["profit_over_maxdd"],
            x["net_profit"],
            x["profit_factor"],
            x["win_rate"],
            x["avg_pnl"],
            x["trades"],
        )

    return a if score(a) >= score(b) else b


def choose_winner_across_scenarios(best_a1: Dict[str, Any], best_c0: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare best-of per scenario and pick winner.
    Uses same scoring rules as pick_best_mode_for_scenario.
    """
    def score(x: Dict[str, Any]) -> tuple:
        return (
            float(x.get("profit_over_maxdd", 0) or 0),
            float(x.get("net_profit", 0) or 0),
            float(x.get("profit_factor", 0) or 0),
            float(x.get("win_rate", 0) or 0),
            float(x.get("avg_pnl", 0) or 0),
            float(x.get("trades", 0) or 0),
        )

    return best_a1 if score(best_a1) >= score(best_c0) else best_c0
          
def choose_winner_across_candidates(best_list: list[Dict[str, Any]], *, pair_label: str, min_trades: int = 80) -> Dict[str, Any]:
    """
    Robust winner selection (hard constraints + maximize profit).

    Rule:
      - If no candidate passes constraints, DO NOT pick a winner for this pair/week.
        (Caller should treat this as "no trade".)
    """
    if not best_list:
        raise ValueError("best_list is empty")

    # ----------------------------
    # Robustness constraints (tuneable)
    # ----------------------------
    WIN_MIN_TRADES = min_trades  # Dynamically assigned!
    WIN_MIN_PROFIT_FACTOR = 1.25
    WIN_MAX_ABS_DD_USDT = 800.0   # must have max_dd_usdt >= -800
    WIN_REQUIRE_POSITIVE_NET = True

    def passes_constraints(x: Dict[str, Any]) -> bool:
        trades = int(x.get("trades", 0) or 0)
        net = float(x.get("net_profit", 0) or 0)
        pf = float(x.get("profit_factor", 0) or 0)
        dd = float(x.get("max_dd_usdt", 0) or 0)  # negative drawdown
        cand_name = x.get("scenario", "Unknown")

        print("\n" + "-" * 100)
        print(f"TRADE WINDOW EVALUATION (1-Week Forward Test): {cand_name}")
        print(f"  -> Trades: {trades} | Net PnL: {net:.2f} | Profit Factor: {pf:.4f} | Max DD: {dd:.2f}")
        
        # Evaluate all constraints without stopping at the first failure
        failed = False
        
        if trades < WIN_MIN_TRADES:
            print(f"  [REJECTED] Trades ({trades}) < Required ({WIN_MIN_TRADES})")
            failed = True
        if pf < WIN_MIN_PROFIT_FACTOR:
            print(f"  [REJECTED] Profit Factor ({pf:.2f}) < Required ({WIN_MIN_PROFIT_FACTOR})")
            failed = True
        if dd < -WIN_MAX_ABS_DD_USDT:
            print(f"  [REJECTED] Drawdown ({dd:.2f}) exceeded Limit ({-WIN_MAX_ABS_DD_USDT})")
            failed = True
        if WIN_REQUIRE_POSITIVE_NET and net <= 0:
            print(f"  [REJECTED] Net Profit ({net:.2f}) is not positive.")
            failed = True
            
        if not failed:
            print(f"  [PASSED] Candidate meets all TRADE week constraints!")
            return True
            
        return False

    eligible = [x for x in best_list if passes_constraints(x)]

    if not eligible:
        print("\n" + "=" * 100)
        print(f"[WINNER][GATE] {pair_label}: NO TRADE FOR PREPAPER — no candidate met winner constraints "
              f"(min_trades={WIN_MIN_TRADES}, min_pf={WIN_MIN_PROFIT_FACTOR}, "
              f"max_abs_dd_usdt={WIN_MAX_ABS_DD_USDT}, require_pos_net={WIN_REQUIRE_POSITIVE_NET}).")
        print("=" * 100)
        return None

    # If constraints pass: maximize net_profit (primary), then profit_over_maxdd, then PF, then win_rate.
    def score_profit_first(x: Dict[str, Any]) -> tuple:
        return (
            float(x.get("net_profit", 0) or 0),
            float(x.get("profit_over_maxdd", 0) or 0),
            float(x.get("profit_factor", 0) or 0),
            float(x.get("win_rate", 0) or 0),
            float(x.get("avg_pnl", 0) or 0),
            float(x.get("trades", 0) or 0),
        )

    winner = eligible[0]
    for b in eligible[1:]:
        if score_profit_first(b) >= score_profit_first(winner):
            winner = b
    return winner

def run_one_scenario_both_modes(
    *,
    pair: str,
    scenario: str,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    ohlcv: pd.DataFrame,
    d_features: pd.DataFrame,
    k: float,
    t: float,
    x_bars: int,
    initial_capital: float,
    trade_size: float,
) -> Dict[str, Any]:
    """
    Runs baseline + barrier for one scenario and returns:
      - events
      - trades/eq for both modes
      - summaries for both modes
      - best-of selection for that scenario
    """
    events = build_events(d_features, trade_start, trade_end, scenario)
    
    events_all = events  # default (non-override): full-range events

    # --- OVERRIDE: replace events with CSV times if configured ---
    if EVENTS_CSV_OVERRIDE:
        forced_times = load_event_times_from_csv(EVENTS_CSV_OVERRIDE)

        events_all = pd.DataFrame({"event_time": sorted(forced_times)})
        events_all["event_time"] = pd.to_datetime(events_all["event_time"], utc=True)

        events = events_all[(events_all["event_time"] >= trade_start) &
                            (events_all["event_time"] < trade_end)].reset_index(drop=True)

        print(f"[events override] Using events from CSV: {EVENTS_CSV_OVERRIDE} | "
              f"events_all={len(events_all)} | events_trade_window={len(events)}")
    else:
        events_all = events
    # --- END OVERRIDE ---
    
    # ADD THESE LINES HERE
    print(f"[DEBUG] EVENTS USED FOR ENTRY (show first 10):")
    print(events.head(10).to_string(index=False))
    print(f"[DEBUG] TOTAL EVENTS USED: {len(events)}")
    
    trades_base, eq_base, _ = run_portfolio_sim(
        mode="baseline",
        pair=pair,
        scenario=scenario,
        ohlcv=ohlcv,
        d_features=d_features,
        events=events,
        trade_start=trade_start,
        trade_end=trade_end,
        k=k,
        t=t,
        x_bars=x_bars,
        initial_capital=initial_capital,
        trade_size=trade_size,
    )

    trades_barr, eq_barr, _ = run_portfolio_sim(
        mode="barrier",
        pair=pair,
        scenario=scenario,
        ohlcv=ohlcv,
        d_features=d_features,
        events=events,
        trade_start=trade_start,
        trade_end=trade_end,
        k=k,
        t=t,
        x_bars=x_bars,
        initial_capital=initial_capital,
        trade_size=trade_size,
    )

    s_base = summarize_trades(trades_base, f"{scenario}-baseline")
    dd_pct_base, dd_usdt_base = compute_max_drawdown(eq_base, "capital_usdt")
    s_base["max_dd_pct"] = dd_pct_base
    s_base["max_dd_usdt"] = dd_usdt_base
    s_base["profit_over_maxdd"] = (s_base["net_profit"] / abs(dd_usdt_base)) if dd_usdt_base != 0 else float("inf")

    s_barr = summarize_trades(trades_barr, f"{scenario}-barrier")
    dd_pct_barr, dd_usdt_barr = compute_max_drawdown(eq_barr, "capital_usdt")
    s_barr["max_dd_pct"] = dd_pct_barr
    s_barr["max_dd_usdt"] = dd_usdt_barr
    s_barr["profit_over_maxdd"] = (s_barr["net_profit"] / abs(dd_usdt_barr)) if dd_usdt_barr != 0 else float("inf")
    best = pick_best_mode_for_scenario(scenario, s_base, s_barr)

    return dict(
        scenario=scenario,
        events=events,
        trades_baseline=trades_base,
        trades_barrier=trades_barr,
        equity_baseline=eq_base,
        equity_barrier=eq_barr,
        summary_baseline=s_base,
        summary_barrier=s_barr,
        best=best,
    )


def strip_tz(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s) and getattr(s.dt, "tz", None) is not None:
                df[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return df

def regime_score_block(
    *,
    label: str,
    d_features: pd.DataFrame,
    events: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    adx_col: str = "adx_15m",
    adx_min: float = ADX_MIN,
) -> pd.DataFrame:
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)

    w = d_features[(d_features["time"] >= start) & (d_features["time"] < end)].copy()
    if w.empty or adx_col not in w.columns:
        return pd.DataFrame([{
            "window": label,
            "scope": "all_minutes",
            "minutes": 0,
            "pct_adx_ge_min": 0.0,
            "adx_p10": np.nan,
            "adx_p25": np.nan,
            "adx_p50": np.nan,
            "adx_p75": np.nan,
            "adx_p90": np.nan,
        }])

    adx_all = pd.to_numeric(w[adx_col], errors="coerce").dropna()
    minutes_all = int(len(adx_all))
    pct_all = float((adx_all >= adx_min).mean() * 100.0) if minutes_all else 0.0

    def q(s: pd.Series, p: float) -> float:
        return float(s.quantile(p)) if len(s) else np.nan

    rows = [{
        "window": label,
        "scope": "all_minutes",
        "minutes": minutes_all,
        "pct_adx_ge_min": pct_all,
        "adx_p10": q(adx_all, 0.10),
        "adx_p25": q(adx_all, 0.25),
        "adx_p50": q(adx_all, 0.50),
        "adx_p75": q(adx_all, 0.75),
        "adx_p90": q(adx_all, 0.90),
    }]

    # SIGNAL-time stats
    if events is not None and (not events.empty) and ("event_time" in events.columns):
        ev = events.copy()
        ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
        ev = ev[(ev["event_time"] >= start) & (ev["event_time"] < end)].copy()

        adx_map = w.set_index("time")[adx_col].to_dict()
        ev["adx_at_signal"] = ev["event_time"].map(adx_map)
        adx_sig = pd.to_numeric(ev["adx_at_signal"], errors="coerce").dropna()

        minutes_sig = int(len(adx_sig))
        pct_sig = float((adx_sig >= adx_min).mean() * 100.0) if minutes_sig else 0.0

        rows.append({
            "window": label,
            "scope": "signal_minutes",
            "minutes": minutes_sig,
            "pct_adx_ge_min": pct_sig,
            "adx_p10": q(adx_sig, 0.10),
            "adx_p25": q(adx_sig, 0.25),
            "adx_p50": q(adx_sig, 0.50),
            "adx_p75": q(adx_sig, 0.75),
            "adx_p90": q(adx_sig, 0.90),
        })

    return pd.DataFrame(rows)

def print_regime_score(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if df is None or df.empty:
        print("(no regime score)")
    else:
        print(df.to_string(index=False))
     
def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

def main():
    global PRINT_PLAY_BY_PLAY    
    
    global INTERVAL

    # Start logging immediately for the whole run
    sys.stdout = DualLogger("runlog.txt")

    # Load candidate JSON once at runtime
    CANDIDATE_TRADE_JSON = os.getenv("CANDIDATE_TRADE_JSON", "candidate_for_TRADE.json")
    with open(CANDIDATE_TRADE_JSON, "r", encoding="utf-8") as f:
        trade_json = json.load(f)
    trade_json = normalize_candidate_json(trade_json)

    minutes = trade_json.get("metadata", {}).get("timeframe_minutes")
    if minutes not in (1, 3):
        raise ValueError(f"metadata.timeframe_minutes must be 1 or 3, got: {minutes!r}")

    INTERVAL = f"{minutes}m"
    print(f"[CONFIG] Using INTERVAL = {INTERVAL} from {CANDIDATE_TRADE_JSON}", flush=True)
    
    # --- Pair ---
    pair = (os.getenv("PAIR") or "").strip().upper()
    if not pair:
        pair = input("Pair (e.g. ACTUSDT): ").strip().upper()
    pair_label = pair

    # --- Run mode ---
    run_mode_env = (os.getenv("RUN_MODE") or "").strip()
    if run_mode_env:
        run_mode_raw = run_mode_env
    else:
        run_mode_raw = input("Mode (MANUAL / AUTO_CYCLE): ")

    run_mode = (run_mode_raw or "").strip().upper().replace("-", "_") or "MANUAL"
    print(f"[CONFIG] run_mode_raw={run_mode_raw!r}  run_mode={run_mode!r}", flush=True)

    # -----------------------------
    # Schedule anchor: PrePaper start (Rule A)
    # -----------------------------
    
    prepaper_start_str = (os.getenv("PREPAPER_START") or "").strip()
    if not prepaper_start_str:
        prepaper_start_str = input("PrePaper START Monday (UTC) [YYYY-MM-DD]: ").strip()

    # strict YYYY-MM-DD validation
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", prepaper_start_str):
        raise ValueError(f"PREPAPER_START must be YYYY-MM-DD, got: {prepaper_start_str!r}")

    pre_start = pd.to_datetime(prepaper_start_str + " 00:00", utc=True)

    if pre_start.weekday() != 0:
        raise ValueError(f"PrePaper START must be a Monday date. Got: {pre_start}")

    pre_end = pre_start + pd.Timedelta(days=7)
    trade_end = pre_start
    trade_start = trade_end - pd.Timedelta(days=7)

    train_end = trade_start
    train_start = train_end - pd.Timedelta(days=30)

    print_section("SCHEDULE")
    print(f"[SCHEDULE] TRAIN (UTC):    {train_start} -> {train_end}   (30d)")
    print(f"[SCHEDULE] TRADE (UTC):    {trade_start} -> {trade_end}   (7d)")
    print(f"[SCHEDULE] PREPAPER (UTC): {pre_start} -> {pre_end}   (7d)")
    
    print_section("CONFIG")
    print(f"[CONFIG] Pair: {pair}")
    print(f"[CONFIG] Run mode: {run_mode}")
    print(f"[CONFIG] Interval: {INTERVAL}")
    print(f"[CONFIG] Trailing mode: {TRAILING_MODE}")
    print(f"[CONFIG] x_bars runtime: record-only / ignored operationally")
    print(f"[CONFIG] FORCE_CLOSE_AT_WINDOW_END: {FORCE_CLOSE_AT_WINDOW_END}")   
    print(f"[CONFIG] TRADE gate net > 0 required: True")

    if run_mode != "AUTO_CYCLE":
        # v30-driven candidates (non-interactive):
        # - Loads finalists from candidate_for_TRADE.json
        # - Picks a candidate via env var CANDIDATE, otherwise defaults to rank #1 (finalists[0])
        # - Uses the k/t/x_bars determined by the quantile selection code (already stored in JSON)
        
        CANDIDATE_TRADE_JSON = os.getenv("CANDIDATE_TRADE_JSON", "candidate_for_TRADE.json")
        with open(CANDIDATE_TRADE_JSON, "r", encoding="utf-8") as f:
            d = json.load(f)
        d = normalize_candidate_json(d)

        finalists = d.get("finalists", [])
        if not finalists:
            raise ValueError(f"No finalists found in {CANDIDATE_TRADE_JSON}")
        
        if run_mode != "AUTO_CYCLE":
            AUTO_TOP_N = int(os.getenv("AUTO_TOP_N", "3"))  # cycle top N finalists in AUTO_CYCLE

            allowed = [f["possibility"] for f in finalists]
            requested = os.getenv("CANDIDATE", "").strip()

            if requested:
                match = next((f for f in finalists if f["possibility"] == requested), None)
                if match is None:
                    raise ValueError(f"Invalid CANDIDATE='{requested}'. Allowed: {allowed}")
                chosen = match
            else:
                chosen = finalists[0]  # default: best-ranked finalist

            scenario = chosen["possibility"]  # keep variable name 'scenario' so the rest of v6 still works
            k, t, x_bars = get_exit_params_from_finalist(chosen)

            print(f"Scenario (candidate possibility): {scenario}")
            print(f"k (ATR-mult for fixed stop): {k}")
            print(f"t (ATR-mult for trailing dist): {t}")
            print(f"x_bars (barrier activation bars): {x_bars}")

    initial_capital = DEFAULT_INITIAL_CAPITAL
    trade_size = DEFAULT_TRADE_SIZE

    print("\n" + "=" * 100)
    if run_mode == "AUTO_CYCLE":
        print(f"AUTO_CYCLE PIPELINE (UTC) | Pair={pair}")
        print("INTENDED ORDER:")
        print("  FINALISTS LOADED")
        print("  ROBUSTNESS CHECK")
        print("  TRAIN - ALL FINALISTS")
        print("  TRAIN SUMMARY")
        print("  ROBUST SURVIVORS")
        print("  TRADE - SURVIVORS")
        print("  TRADE SUMMARY")
        print("  WINNER / NO TRADE")
        print("  PREPAPER")
        print(f"TRAIN (UTC):    {train_start} -> {train_end}   (30d)")
        print(f"TRADE (UTC):    {trade_start} -> {trade_end}   (7d)")
        print(f"PREPAPER (UTC): {pre_start} -> {pre_end}   (7d)")
    else:
        print(f"TRADE WINDOW (UTC): {trade_start} -> {trade_end} | Pair={pair} | Scenario={scenario}")
    print(f"[PORT] initial=${initial_capital:,.2f} | trade_size=${trade_size:,.2f} | maxAvail starts at {int(initial_capital // trade_size)}")
    print("=" * 100)

    warmup_start = train_start - pd.Timedelta(days=WARMUP_DAYS)
    fetch_end = trade_end + (pd.Timedelta(days=7) if run_mode == "AUTO_CYCLE" else pd.Timedelta(days=0))
    print(f"[FETCH] {pair} {INTERVAL} (TRAIN warmup included): {warmup_start} -> {fetch_end}")

    ohlcv = get_ohlcv_binance(pair, warmup_start, fetch_end)
    if ohlcv.empty:
        print("No OHLCV fetched. Check pair/date.")
        return

    ohlcv["atr"] = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=ATR_LEN)
    ohlcv = ohlcv.dropna(subset=["atr"]).reset_index(drop=True)
          
    d = compute_entry_features(ohlcv)
    
    # -----------------------------
    # 15m ADX regime map (for gating) - attach for ALL modes (manual/auto + prepaper)
    # -----------------------------
    maps = {"adx_15m": {}, "dmp_15m": {}, "dmn_15m": {}}

    if ADX_GATE_ENABLE:
        maps = compute_adx_15m_maps(ohlcv)
        if not maps or not maps.get("adx_15m"):
            print("[ADX] WARNING: ADX/DI maps are empty; gates may block entries if enabled.")
        else:
            print("[ADX] 15m ADX/DI maps ready.")

        d["adx_15m"] = d["time"].map(maps.get("adx_15m", {}))
        d["dmp_15m"] = d["time"].map(maps.get("dmp_15m", {}))  # +DI
        d["dmn_15m"] = d["time"].map(maps.get("dmn_15m", {}))  # -DI
    else:
        d["adx_15m"] = np.nan
        d["dmp_15m"] = np.nan
        d["dmn_15m"] = np.nan

    if run_mode == "AUTO_CYCLE":

        # v30 finalists (self-contained load for AUTO_CYCLE)
        CANDIDATE_TRADE_JSON = os.getenv("CANDIDATE_TRADE_JSON", "candidate_for_TRADE.json")
        with open(CANDIDATE_TRADE_JSON, "r", encoding="utf-8") as f:
            cand_data = json.load(f)
        cand_data = normalize_candidate_json(cand_data)

        finalists = cand_data.get("finalists", [])
        if not finalists:
            raise ValueError(f"No finalists found in {CANDIDATE_TRADE_JSON}")

        AUTO_TOP_N = int(os.getenv("AUTO_TOP_N", "3"))
        cycle = finalists[:AUTO_TOP_N]  # ranked order from json (top N)
        print(f"[AUTO_CYCLE] AUTO_TOP_N={AUTO_TOP_N} | finalists={len(finalists)} | cycle_scenarios={[f['possibility'] for f in cycle]}", flush=True)

        # -------------------------------------------------------------------
        # AUTO_CYCLE evaluation verbosity (A vs B)
        #   - Default (B): quiet evaluation (no per-bar SIGNAL/OPEN/STOP spam)
        #   - Plan A: verbose evaluation logs (for debugging/verification)
        # -------------------------------------------------------------------
        ans = input("Verbose eval logs (Plan A)? (y/N): ").strip().lower()
        EVAL_VERBOSE = ans in {"y", "yes", "1", "true", "on"}
        print(f"[CONFIG] EVAL_VERBOSE={EVAL_VERBOSE} (AUTO_CYCLE evaluation play-by-play)", flush=True)

        # Apply to evaluation runs only; winner replay + prepaper will force it back to True later.
        PRINT_PLAY_BY_PLAY = bool(EVAL_VERBOSE)
        
        ##################
        # FINALISTS LOADED
        ##################

        print_section("FINALISTS LOADED")
        for i, f in enumerate(cycle, 1):
            ep = f.get("exit_params", {}) or {}
            print(
                f"[FINALIST {i}] scen={f.get('possibility')} | "
                f"score={f.get('score')} | trades={f.get('trades')} | "
                f"k={ep.get('k')} | t={ep.get('t')} | x_bars={ep.get('x_bars')} (record-only)"
            )              
            
        # -----------------------------
        # ROBUSTNESS (TRAIN walk-forward on finalists via Fibonacci engine)
        # Gate R1: pos_weeks >= ROBUST_MIN_POS_WEEKS AND mean weekly net >= ROBUST_MEAN_NET_MIN
        # Gate R2: Average Weekly Drag/Loss (sum negative / 4) > ROBUST_WORST_WEEK_NET_MIN
        # Gate R3: mean_profit_over_maxdd >= ROBUST_MEAN_POMDD_MIN
        # Ranking: mean profit_over_maxdd, tie-break mean net_profit
        # -----------------------------

        def _print_robust_block(rob: dict) -> None:
            """Print the full detailed robustness block for one candidate."""
            scen = rob["scenario"]
            r1 = bool(rob.get("passes_robust_r1", False))
            r2 = float(rob.get("avg_weekly_loss", 0.0)) > float(ROBUST_WORST_WEEK_NET_MIN)
            r3 = float(rob.get("mean_profit_over_maxdd", 0.0)) >= float(ROBUST_MEAN_POMDD_MIN)
            print("\n" + "-" * 100)
            print(f"[ROBUST] {scen}")
            print(
                f"  R1 (pos_weeks>={ROBUST_MIN_POS_WEEKS} AND mean_net>={ROBUST_MEAN_NET_MIN}): "
                f"{rob['pos_weeks']}/{rob['n_slices']} weeks positive, Mean Net={rob.get('mean_net_profit', 0.0):.2f} "
                f"=> {'PASS' if r1 else 'FAIL'}"
            )
            print(
                f"  R2 (Average Weekly Drag/Loss over 4 weeks > {ROBUST_WORST_WEEK_NET_MIN}): "
                f"{rob.get('avg_weekly_loss', 0.0):.2f} => {'PASS' if r2 else 'FAIL'}"
            )
            print(f"  R3 (mean_profit_over_maxdd>={ROBUST_MEAN_POMDD_MIN}): {rob['mean_profit_over_maxdd']:.4f} => {'PASS' if r3 else 'FAIL'}")
            print(f"     worst_week_net_profit: {rob['worst_week_net_profit']:.2f}")
            dfw = rob["df_slices"]
            if dfw is not None and not dfw.empty:
                for _, row in dfw.iterrows():
                    w0 = row["slice_start"]
                    w1 = row["slice_end"]
                    net = float(row.get("net_profit", 0.0))
                    mdd = float(row.get("max_dd_usdt", 0.0))
                    pomdd = float(row.get("profit_over_maxdd", 0.0))
                    clusters = int(row.get("clusters_completed", 0))
                    trades_cl = int(row.get("trades_closed", 0))
                    r2p = "PASS" if bool(row.get("r2_week_pass", False)) else "FAIL"
                    print(
                        f"   - W{int(row['slice_ix'])} {w0} -> {w1}"
                        f" | net={net:+10.2f} | maxDD={mdd:+10.2f} | P/MaxDD={pomdd:+8.4f}"
                        f" | clusters={clusters:2d} | trades={trades_cl:3d} | R2_week={r2p}"
                    )

        print(f"[CONFIG] ROBUST_ENABLE={ROBUST_ENABLE!r}")
        if ROBUST_ENABLE:
            train_slices = iter_monday_week_slices(train_start, trade_start)
            if len(train_slices) > ROBUST_TRAIN_SLICES:
                train_slices = train_slices[-ROBUST_TRAIN_SLICES:]

            print_section("ROBUSTNESS CHECK (TRAIN weekly slices on finalists via Fibonacci engine)")
            print(
                f"R1 (pos_weeks>={ROBUST_MIN_POS_WEEKS} AND mean_week_net>={ROBUST_MEAN_NET_MIN}): checked | "
                f"R2 (Average Weekly Drag/Loss over 4 weeks > {ROBUST_WORST_WEEK_NET_MIN}): applied after R1 | "
                f"R3 (mean_profit_over_maxdd>={ROBUST_MEAN_POMDD_MIN}): applied after R2"
            )
            for (w0, w1) in train_slices:
                print(f"  - {w0} -> {w1}")

            robust_rows = []

            params_by_scen = {str(f["possibility"]).strip().upper(): f for f in cycle}

            #######################
            # TRAIN - ALL FINALISTS
            #######################
                    
            print_section("TRAIN - ALL FINALISTS")
            for f in cycle:
                scen = str(f["possibility"]).strip().upper()
                parsed = {}
                try:
                    parsed = _parse_possibility(scen)
                except Exception:
                    parsed = {}

                close_gate = f.get("close", parsed.get("close", "ALL"))
                vol_gate = f.get("vol", parsed.get("vol", "ALL"))
                vol_rule = f.get("vol_rule", "ALL")
                if vol_rule in (None, "", "None", "null"):
                    r_op = parsed.get("r_op")
                    r_value = pd.to_numeric(parsed.get("r_value", np.nan), errors="coerce")
                    if r_op == "GE":
                        vol_rule = f">={float(r_value)}" if np.isfinite(r_value) else "ALL"
                    elif r_op == "LT":
                        vol_rule = f"<{float(r_value)}" if np.isfinite(r_value) else "ALL"
                    else:
                        vol_rule = "ALL"

                rob = eval_candidate_robustness_over_train(
                    pair=pair,
                    scenario=scen,
                    slices=train_slices,
                    initial_capital=initial_capital,
                    trade_size=trade_size,
                    interval=INTERVAL,
                    close_gate=close_gate,
                    vol_gate=vol_gate,
                    vol_rule=vol_rule,
                )
                robust_rows.append(rob)
                _print_robust_block(rob)             
            
            # Apply gates (ROBUST)                
            gated =[
                r for r in robust_rows
                if bool(r.get("passes_robust_r1", False)) 
                and (float(r.get("avg_weekly_loss", 0.0)) > float(ROBUST_WORST_WEEK_NET_MIN))  # R2
                and (float(r.get("mean_profit_over_maxdd", 0.0)) >= float(ROBUST_MEAN_POMDD_MIN))  # R3
            ]

            # --- END-OF-ROBUSTNESS RECAP: full detailed blocks for ALL finalists (always printed) ---
            print("\n" + "=" * 100)
            
            ###############
            # TRAIN SUMMARY
            ###############
            
            print_section("TRAIN SUMMARY")
            print("ROBUSTNESS RECAP — ALL FINALISTS (W1..W4 DETAIL)")
            print("=" * 100)
            for rob in robust_rows:
                _print_robust_block(rob)
            print("=" * 100)

            if not gated:
                print("\n" + "=" * 100)
                print(f"[ROBUST][GATE] {pair_label}: NO TRADE FOR TRADE WINDOW — no finalist passed TRAIN robustness gates (R1 + R2 + R3).")
                print(f"R1  (pos_weeks>={ROBUST_MIN_POS_WEEKS} AND mean_net>={ROBUST_MEAN_NET_MIN}): Needs >= {ROBUST_MIN_POS_WEEKS} positive weeks and mean net >= ${ROBUST_MEAN_NET_MIN}.")
                print(f"R2  (Average Weekly Drag/Loss over 4 weeks > {ROBUST_WORST_WEEK_NET_MIN}): Average weekly drag/loss must exceed threshold.")
                print(f"R3  (mean_profit_over_maxdd>={ROBUST_MEAN_POMDD_MIN}): Mean profit/maxdd must exceed threshold.")
                print("=" * 100)
                return

            gated_sorted = sorted(
                gated,
                key=lambda r: (float(r["mean_profit_over_maxdd"]), float(r["mean_net_profit"])),
                reverse=True,
            )
            robust_pick = gated_sorted[0]  # <--- RESTORED: Prevents the variable error!

            print("\n" + "=" * 100)
            print("ROBUST SURVIVORS (Moving to TRADE Competition)")
            print("=" * 100)
            robust_scenarios = []
            for r in gated_sorted:
                print(f" - {r['scenario']} | P/MaxDD: {r['mean_profit_over_maxdd']:.4f} | Net: {r['mean_net_profit']:.2f}")
                robust_scenarios.append(str(r["scenario"]).strip().upper())

            if not robust_scenarios:
                print("\n" + "=" * 100)
                print(f"[WINNER][GATE] {pair}: NO TRADE FOR PREPAPER — no candidate passed TRAIN robustness.")
                print("=" * 100)
                return

            all_results = []
            all_summary_rows = []
            best_per_candidate = []
            
            trade_cycle = [params_by_scen[scen] for scen in robust_scenarios]
            
            #######################
            # TRADE - ALL FINALISTS
            #######################

            print_section("TRADE - SURVIVORS (FIB VERIFIER)")
            trade_rows = []
            best_per_candidate = []

            for f in trade_cycle:
                scen = str(f["possibility"]).strip().upper()

                parsed = {}
                try:
                    parsed = _parse_possibility(scen)
                except Exception:
                    parsed = {}

                close_gate = f.get("close", parsed.get("close", "ALL"))
                vol_gate = f.get("vol", parsed.get("vol", "ALL"))
                vol_rule = f.get("vol_rule", "ALL")

                if vol_rule in (None, "", "None", "null"):
                    r_op = parsed.get("r_op")
                    if r_op == "GE":
                        rv = pd.to_numeric(parsed.get("r_value", np.nan), errors="coerce")
                        vol_rule = f">={float(rv)}" if np.isfinite(rv) else "ALL"
                    elif r_op == "LT":
                        rv = pd.to_numeric(parsed.get("r_value", np.nan), errors="coerce")
                        vol_rule = f"<{float(rv)}" if np.isfinite(rv) else "ALL"
                    elif r_op == "BIN":
                        lo = float(parsed.get("r_low"))
                        hi = float(parsed.get("r_high"))
                        vol_rule = f"{lo}_{hi}"
                    else:
                        vol_rule = "ALL"

                print("\n" + "-" * 100)
                print(f"[AUTO_CYCLE][TRADE][FIB] Running finalist: {scen} | close={close_gate} vol={vol_gate} vol_rule={vol_rule}")

                fib_result = verify_symbol_fib_train(
                    pair=pair,
                    interval=INTERVAL,
                    train_start=trade_start,
                    train_end=trade_end,
                    initial_capital=initial_capital,
                    trade_size=trade_size,
                    close_gate=close_gate,
                    vol_gate=vol_gate,
                    vol_rule=vol_rule,
                    verbose=True,
                )

                row = {
                    "scenario": scen,
                    "trades_closed": int(fib_result.get("trades_closed", 0)),
                    "clusters_completed": int(fib_result.get("clusters_completed", 0)),
                    "net_profit_usdt": float(fib_result.get("net_profit_usdt", 0.0)),
                    "net_profit_pct": float(fib_result.get("net_profit_pct", 0.0)),
                    "max_dd_usdt": float(fib_result.get("max_dd_usdt", 0.0)),
                    "max_dd_pct": float(fib_result.get("max_dd_pct", 0.0)),
                }
                trade_rows.append(row)
                best_per_candidate.append(row)

            summary_trade = pd.DataFrame(trade_rows)
            print_section("TRADE SUMMARY (FIB)")
            print(summary_trade.to_string(index=False))
            
            ###############
            # TRADE SUMMARY
            ###############

            print_section("TRADE SUMMARY")
            print("ALL CANDIDATES SUMMARY (TRADE WINDOW)")
            print(summary_trade.to_string(index=False))
                        
            # Restrict finalists to ALL robust scenarios that survived TRAIN
            best_per_candidate =[
                x for x in best_per_candidate
                if str(x["scenario"]).strip().upper() in robust_scenarios
            ]

        # --- DYNAMIC MIN TRADES LOGIC FOR ALL SURVIVORS ---
        dynamic_min_trades = 80
        if 'gated_sorted' in locals() and len(gated_sorted) > 0:
            min_events_list =[]
            for r in gated_sorted:
                dfw = r.get("df_slices")
                if dfw is not None and not dfw.empty:
                    min_events_list.append(int(dfw["trades_closed"].min()))
            
            if min_events_list:
                dynamic_min_trades = min(min_events_list)
                print(f"\n[DYNAMIC GATE] Setting TRADE min_trades to {dynamic_min_trades} based on min(trades_closed) across TRAIN weeks.")
        # --------------------------------------------------

        # Winner across candidates (Scores the survivors based on 1-week TRADE performance)
        winner = choose_winner_across_candidates(best_per_candidate, pair_label=pair_label, min_trades=dynamic_min_trades)

        print_section("WINNER / NO TRADE")
        if not winner:
            print("[WINNER] Result: NO TRADE (see rejection details above).")
            return

        win_scenario = str(winner["scenario"]).strip().upper()
        print(f"[WINNER] Result: WINNER selected = {win_scenario}")
        win_params = params_by_scen[win_scenario]
        
        # Winner gates (native types per contract: bool or 'ALL', vol_rule string)
        win_close_gate = win_params.get("close", "ALL")
        win_vol_gate = win_params.get("vol", "ALL")
        win_vol_rule = win_params.get("vol_rule", "ALL")

        if win_close_gate in (None, "", "None", "null"):
            win_close_gate = "ALL"
        if win_vol_gate in (None, "", "None", "null"):
            win_vol_gate = "ALL"
        if win_vol_rule in (None, "", "None", "null"):
            win_vol_rule = "ALL"

        # 1. Unpack the parameters ONCE
        win_k, win_t, win_x = get_exit_params_from_finalist(win_params)

        # ===============================================================
        # 1. PLAYBACK THE TRAIN WINDOW LOGS (4 WEEKS)
        # ===============================================================
        PRINT_PLAY_BY_PLAY = True  # Turn the speakers back on!
        print("\n" + "=" * 100)
        print(f"TRAIN WINDOW PLAY-BY-PLAY LOGS (UTC): {train_start} -> {trade_start} | Scenario={win_scenario}")
        print("=" * 100)
        
        train_replay_results = verify_symbol_fib_train(
            pair=pair,
            interval=INTERVAL,
            train_start=train_start,
            train_end=trade_start,
            initial_capital=initial_capital,
            trade_size=trade_size,
            close_gate=win_close_gate,
            vol_gate=win_vol_gate,
            vol_rule=win_vol_rule,
        )

        print("\n" + "=" * 100)
        print(f"[TRAIN REPLAY — FIB] {train_start} -> {trade_start} | Scenario={win_scenario}")
        print("=" * 100)
        print(f"  -> Net Profit      : ${float(train_replay_results.get('net_profit_usdt', 0.0)):.2f} ({float(train_replay_results.get('net_profit_pct', 0.0)):.2f}%)")
        print(f"  -> Max Drawdown    : ${float(train_replay_results.get('max_dd_usdt', 0.0)):.2f} ({float(train_replay_results.get('max_dd_pct', 0.0)):.2f}%)")
        print(f"  -> Clusters Closed : {int(train_replay_results.get('clusters_completed', 0))}")
        print(f"  -> Trades Closed   : {int(train_replay_results.get('trades_closed', 0))}")
        print("=" * 100)

        # ===============================================================
        # 2. PLAYBACK THE TRADE WINDOW LOGS (1 WEEK)
        # ===============================================================
        print("\n" + "=" * 100)
        print(f"TRADE WINDOW PLAY-BY-PLAY LOGS (UTC): {trade_start} -> {trade_end} | Scenario={win_scenario}")
        print("=" * 100)

        # Normalize winner gates with safe ALL defaults (Option A)
        parsed = {}
        try:
            parsed = _parse_possibility(win_scenario)
        except Exception:
            parsed = {}

        win_close_gate = win_params.get("close", parsed.get("close", "ALL"))
        win_vol_gate = win_params.get("vol", parsed.get("vol", "ALL"))
        win_vol_rule = win_params.get("vol_rule", "ALL")

        if win_vol_rule in (None, "", "None", "null"):
            r_op = parsed.get("r_op")
            r_value = pd.to_numeric(parsed.get("r_value", np.nan), errors="coerce")
            if r_op == "GE":
                win_vol_rule = f">={float(r_value)}" if np.isfinite(r_value) else "ALL"
            elif r_op == "LT":
                win_vol_rule = f"<{float(r_value)}" if np.isfinite(r_value) else "ALL"
            elif r_op == "BIN":
                lo = float(parsed.get("r_low"))
                hi = float(parsed.get("r_high"))
                win_vol_rule = f"{lo}_{hi}"
            else:
                win_vol_rule = "ALL"

        win_results = verify_symbol_fib_train(
            pair=pair,
            interval=INTERVAL,
            train_start=trade_start,
            train_end=trade_end,
            initial_capital=initial_capital,
            trade_size=trade_size,
            close_gate=win_close_gate,
            vol_gate=win_vol_gate,
            vol_rule=win_vol_rule,
            verbose=True,
        )
        
        print("\n" + "=" * 100)
        print(f"[WINNER SCORECARD] Selected Scenario: {win_scenario}")
        print("=" * 100)

        net_usdt = float(win_results.get("net_profit_usdt", 0.0))
        net_pct = float(win_results.get("net_profit_pct", 0.0))
        mdd_usdt = float(win_results.get("max_dd_usdt", 0.0))
        mdd_pct = float(win_results.get("max_dd_pct", 0.0))
        clusters = int(win_results.get("clusters_completed", 0))
        trades_closed = int(win_results.get("trades_closed", 0))

        print(f"  -> Total Net Profit : ${net_usdt:.2f} ({net_pct:.2f}%)")
        print(f"  -> Max Drawdown     : ${mdd_usdt:.2f} ({mdd_pct:.2f}%)")
        print(f"  -> Clusters Closed  : {clusters}")
        print(f"  -> Individual Trades: {trades_closed}")
        print("=" * 100)                                  
           
        # --- save trade window workbook ---
        ident = f"{trade_start.strftime('%Y-%m-%d_%H%M')}_to_{trade_end.strftime('%Y-%m-%d_%H%M')}"
        out_trade = os.path.join(OUT_DIR, f"forwardtest_TRADEWINDOW_7d_ALLCANDS_{ident}_{pair}.xlsx")
        with pd.ExcelWriter(out_trade, engine="openpyxl") as w:
            summary_trade.to_excel(w, sheet_name="summary_all_candidates", index=False)

        print(f"\nSaved: {out_trade}")
            
        # -----------------------------
        # PREPAPER (winner only): user-provided Monday 08:00 UTC for 7 days
        # -----------------------------

        print("\n" + "=" * 100)

        # -----------------------------
        # PREPAPER (winner only): pure Fibonacci verifier (Option A aligned)
        # -----------------------------
        print(f"PREPAPER WINDOW (UTC): {pre_start} -> {pre_end} | Pair={pair} | Scenario={win_scenario}")
        PRINT_PLAY_BY_PLAY = True  # Always verbose for PREPAPER

        # Retrieve the winning candidate's specific gate configurations
        win_close_gate = win_params.get("close", "ALL")
        win_vol_gate = win_params.get("vol", "ALL")
        win_rule_gate = win_params.get("vol_rule", "ALL")

        # Execute the actual Fibonacci Trade window for PREPAPER
        res_pre = verify_symbol_fib_train(
            pair=pair,
            interval=INTERVAL,
            train_start=pre_start,
            train_end=pre_end,
            initial_capital=initial_capital,
            trade_size=trade_size,
            close_gate=win_close_gate,
            vol_gate=win_vol_gate,
            vol_rule=win_rule_gate,
            verbose=True,  # Triggers play-by-play console prints
        )

        # ==============================================================================
        # PREPAPER SUMMARY & EXCEL EXPORT (Pure Fib Cluster - Aligned with Option A)
        # ==============================================================================
        summary_pre = pd.DataFrame([{
            "scenario": win_scenario,
            "trades": int(res_pre.get("trades_closed", 0)),
            "net_profit_usdt": float(res_pre.get("net_profit_usdt", 0.0)),
            "net_profit_pct": float(res_pre.get("net_profit_pct", 0.0)),
            "max_dd_usdt": float(res_pre.get("max_dd_usdt", 0.0)),
            "max_dd_pct": float(res_pre.get("max_dd_pct", 0.0)),
            "clusters_completed": int(res_pre.get("clusters_completed", 0)),
        }])

        print("\n" + "=" * 100)
        print("PREPAPER SUMMARY (PURE FIB CLUSTER WINNER)")
        print("=" * 100)
        print(summary_pre.to_string(index=False))
        print("=" * 100)

        ident_pre = f"{pre_start.strftime('%Y-%m-%d_%H%M')}_to_{pre_end.strftime('%Y-%m-%d_%H%M')}"
        out_pre = os.path.join(OUT_DIR, f"forwardtest_PREPAPER_7d_WINNER_{ident_pre}_{win_scenario}_{pair}.xlsx")

        # Write our clean, crash-proof summary workbook to disk
        with pd.ExcelWriter(out_pre, engine="openpyxl") as w:
            summary_pre.to_excel(w, sheet_name="summary", index=False)

        print(f"\nSaved PrePaper workbook: {out_pre}")

        # IMPORTANT: stop here so we don't continue into MANUAL mode simulation below
        return

if __name__ == "__main__":
    main()