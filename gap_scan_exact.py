from __future__ import annotations

import os, time, json, re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

# ============ CONFIG ============
CANDIDATE_TRADE_JSON = os.getenv("CANDIDATE_TRADE_JSON", "candidate_for_TRADE.json")

PAIR = (os.getenv("PAIR") or "TNSRUSDT").strip().upper()

# Scan window (edit these)
START_UTC = "2025-10-25 00:00"  # e.g., train_start
END_UTC   = "2025-12-01 00:00"  # e.g., trade_end

SCENARIO = "C_ALL__V_ALL__R_LT_1.5"
K = 2.610559999999999
T = 21.33468999999991

ATR_LEN = 14
LIMIT = 1000
BINANCE_BASE = "https://api.binance.com"

# ============ Binance OHLCV ============
def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": LIMIT}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_ohlcv_binance(symbol: str, interval: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    start_utc = pd.to_datetime(start_utc, utc=True)
    end_utc = pd.to_datetime(end_utc, utc=True)

    start_ms = int(start_utc.value // 10**6)
    end_ms = int(end_utc.value // 10**6)

    rows = []
    cur = start_ms

    mins = int(interval[:-1]) if interval.endswith("m") else 1
    step_ms = mins * 60_000

    while cur < end_ms:
        data = fetch_klines(symbol, interval, cur, end_ms)
        if not data:
            break

        rows.extend(data)
        last_open = data[-1][0]
        cur = last_open + step_ms

        time.sleep(0.25)
        if len(data) < LIMIT:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["open_time_ms","open","high","low","close","volume","close_time_ms","qav","num_trades","tb","tq","ignore"],
    )
    df["time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df = df[["time","open","high","low","close","volume"]].copy()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("time").reset_index(drop=True)
    return df

# ============ Indicators ============
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

def compute_entry_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    d = ohlcv.copy()
    d["rsi"] = rsi_wilder(d["close"], 14)
    d["rsi_sma"] = sma(d["rsi"], 14)

    d["smma_200"] = wilders_rma(d["close"], 200)
    d["close_gt_smma_200"] = d["close"] > d["smma_200"]

    d["vol_sma_20"] = sma(d["volume"], 20)
    d["vol_gt_vol_sma"] = d["volume"] > d["vol_sma_20"]
    d["vol_ratio"] = d["volume"] / d["vol_sma_20"]
    return d.dropna().reset_index(drop=True)

# ============ Possibility parsing (same format as your core) ============
def _parse_possibility(possibility: str) -> dict:
    s = possibility.strip().upper()

    m_all = re.fullmatch(r"C_(TRUE|FALSE|ALL)__V_(TRUE|FALSE|ALL)__R_ALL", s)
    if m_all:
        close_s, vol_s = m_all.group(1), m_all.group(2)
        return {"close": close_s, "vol": vol_s, "r_op": "ALL", "r_value": None}

    m = re.fullmatch(r"C_(TRUE|FALSE|ALL)__V_(TRUE|FALSE|ALL)__R_(LT|GE)_([0-9]+(?:\.[0-9]+)?)", s)
    if m:
        close_s, vol_s, r_op, r_val = m.group(1), m.group(2), m.group(3), float(m.group(4))
        return {"close": close_s, "vol": vol_s, "r_op": r_op, "r_value": r_val}

    m_bin = re.fullmatch(r"C_(TRUE|FALSE|ALL)__V_(TRUE|FALSE|ALL)__R_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)", s)
    if m_bin:
        close_s, vol_s = m_bin.group(1), m_bin.group(2)
        lo, hi = float(m_bin.group(3)), float(m_bin.group(4))
        if not (hi > lo):
            raise ValueError(f"Invalid R bin (high must be > low): {possibility}")
        return {"close": close_s, "vol": vol_s, "r_op": "BIN", "r_low": lo, "r_high": hi, "r_value": None}

    raise ValueError(f"Invalid possibility format: {possibility}")

def build_events(d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, scenario: str) -> pd.DataFrame:
    scenario = (scenario or "").strip().upper()
    if not scenario:
        raise ValueError("scenario must be non-empty")

    prev = d["rsi_sma"].shift(1)
    curr = d["rsi_sma"]
    cross_up_51 = (prev < 51.0) & (curr >= 51.0)

    ev = d.loc[cross_up_51].copy()
    ev = ev[(ev["time"] >= start) & (ev["time"] < end)].copy()

    if scenario == "A1":
        ev = ev[(ev["close_gt_smma_200"] == True) & (ev["vol_gt_vol_sma"] == True) & (ev["vol_ratio"] >= 1.5)].copy()
    elif scenario == "C0":
        ev = ev[(ev["close_gt_smma_200"] == False) & (ev["vol_gt_vol_sma"] == True)].copy()
    elif scenario.startswith("C_"):
        p = _parse_possibility(scenario)

        if p["close"] == "TRUE":
            ev = ev[ev["close_gt_smma_200"] == True].copy()
        elif p["close"] == "FALSE":
            ev = ev[ev["close_gt_smma_200"] == False].copy()

        if p["vol"] == "TRUE":
            ev = ev[ev["vol_gt_vol_sma"] == True].copy()
        elif p["vol"] == "FALSE":
            ev = ev[ev["vol_gt_vol_sma"] == False].copy()

        if p["r_op"] == "ALL":
            pass
        elif p["r_op"] == "LT":
            ev = ev[ev["vol_ratio"] < float(p["r_value"])].copy()
        elif p["r_op"] == "GE":
            ev = ev[ev["vol_ratio"] >= float(p["r_value"])].copy()
        elif p["r_op"] == "BIN":
            lo = float(p["r_low"]); hi = float(p["r_high"])
            ev = ev[(ev["vol_ratio"] >= lo) & (ev["vol_ratio"] < hi)].copy()
        else:
            raise ValueError(f"Unknown r_op={p['r_op']}")
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    out = ev[["time"]].rename(columns={"time": "event_time"}).reset_index(drop=True)
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    return out

# ============ Gap Scan logic (no pyramiding; trailing immediate) ============
@dataclass
class Pos:
    entry_time: pd.Timestamp
    entry_price: float
    fixed_stop: float
    trail_dist: float
    peak_high: float
    bars: int = 0

def gap_scan_exact(
    *,
    pair: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    scenario: str,
    k: float,
    t: float,
) -> pd.DataFrame:
    ohlcv = get_ohlcv_binance(pair, interval, start, end)
    if ohlcv.empty:
        return pd.DataFrame()

    ohlcv["atr"] = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=ATR_LEN)
    ohlcv = ohlcv.dropna(subset=["atr"]).reset_index(drop=True)

    d = compute_entry_features(ohlcv)
    ev_df = build_events(d, start, end, scenario)
    event_times = set(pd.to_datetime(ev_df["event_time"], utc=True).tolist())

    window = ohlcv[(ohlcv["time"] >= start) & (ohlcv["time"] <= end)].copy()

    positions: Dict[str, Pos] = {}
    next_id = 1
    flagged: List[Dict[str, Any]] = []

    for _, bar in window.iterrows():
        ts = bar["time"]
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        atr = float(bar["atr"])

        # exits first (matches your core order)
        for pid, pos in list(positions.items()):
            pos.bars += 1
            pos.peak_high = max(pos.peak_high, h)

            trail_stop = pos.peak_high - pos.trail_dist  # immediate trailing
            stop_level = max(pos.fixed_stop, trail_stop)

            if l <= stop_level:
                if o < stop_level:
                    flagged.append({
                        "exit_time": ts,
                        "pid": pid,
                        "entry_time": pos.entry_time,
                        "entry_price": pos.entry_price,
                        "fixed_stop": pos.fixed_stop,
                        "trail_stop": trail_stop,
                        "stop_level": stop_level,
                        "bar_open": o,
                        "bar_low": l,
                        "bar_high": h,
                        "bar_close": c,
                        "bars_held": pos.bars,
                        "gap_amount": float(stop_level - o),
                    })
                del positions[pid]

        # entries
        if ts in event_times:
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry = c
            fixed_stop = entry - (k * atr)
            trail_dist = t * atr

            pid = f"{pair}_{scenario}_scan_{next_id}"
            positions[pid] = Pos(
                entry_time=ts,
                entry_price=entry,
                fixed_stop=fixed_stop,
                trail_dist=trail_dist,
                peak_high=entry,
                bars=0,
            )
            next_id += 1

    return pd.DataFrame(flagged)

def _load_interval_from_candidate_json(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    mins = d.get("metadata", {}).get("timeframe_minutes")
    if mins not in (1, 3):
        raise ValueError(f"metadata.timeframe_minutes must be 1 or 3, got {mins!r}")
    return f"{mins}m"

if __name__ == "__main__":
    interval = _load_interval_from_candidate_json(CANDIDATE_TRADE_JSON)
    start = pd.to_datetime(START_UTC, utc=True)
    end = pd.to_datetime(END_UTC, utc=True)

    print("\n" + "="*100)
    print(f"[GAP_SCAN_EXACT] pair={PAIR} interval={interval} scenario={SCENARIO}")
    print(f"[GAP_SCAN_EXACT] start={start} end={end}")
    print(f"[GAP_SCAN_EXACT] k={K} t={T}")
    print("="*100)

    df = gap_scan_exact(pair=PAIR, interval=interval, start=start, end=end, scenario=SCENARIO, k=K, t=T)

    print("\n" + "="*100)
    print(f"[GAP_SCAN_EXACT] flagged gap-through-stop events: {len(df)}")
    print("="*100)

    if df.empty:
        print("(none found)")
    else:
        df = df.sort_values("gap_amount", ascending=False).reset_index(drop=True)
        print(df.head(50).to_string(index=False))
        out = f"gap_through_stop_{PAIR}_{SCENARIO}_{interval}.csv".replace("/", "_")
        df.to_csv(out, index=False)
        print(f"\nSaved: {out}")