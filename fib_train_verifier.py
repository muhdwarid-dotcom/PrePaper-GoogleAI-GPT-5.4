"""fib_train_verifier.py

Helper module for Stage 4B Darwinian pruning.

Goals (audited requirements):
- No scenario logic; use MtfFibClusterEngine defaults only.
- Support LTF execution on either 1m or 3m as assigned by Stage 2.
- If HTF data is needed, fetch 1h candles directly from Binance.
- Return metrics needed for gates:
    - net_profit_pct (strictly >== 0.0 default gate)
    - clusters_completed (>= 1 default gate)
    - max_dd_pct (<= 10% default gate)

This verifier is intentionally self-contained and does not import the large
7_day_trade_window script (which has CLI/logging side effects).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from binance_fetch import fetch_klines_1m
from mtf_fib_cluster_engine import MtfFibClusterEngine


# ----------------------------
# Utilities
# ----------------------------

def _to_utc_dt(x: Any) -> datetime:
    ts = pd.to_datetime(x, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def _compute_max_drawdown_from_equity(equity: pd.Series) -> Tuple[float, float]:
    """Return (max_dd_pct, max_dd_abs) where dd values are positive magnitudes.

    max_dd_pct is e.g. 0.10 for 10%.
    max_dd_abs is e.g. 350.0 USDT.
    """
    s = pd.to_numeric(equity, errors="coerce").dropna()
    if s.empty:
        return (0.0, 0.0)
    peak = s.cummax()
    dd_abs = (peak - s)
    dd_pct = (peak - s) / peak.replace(0, np.nan)
    return (float(dd_pct.max() or 0.0), float(dd_abs.max() or 0.0))


# ----------------------------
# Positions / trades (minimal)
# ----------------------------

@dataclass
class _Position:
    pid: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    cluster_id: str


def _close_trade(
    *,
    pos: _Position,
    ts: pd.Timestamp,
    exit_price: float,
    reason: str,
    fee_rate: float,
    trade_size: float,
) -> Dict[str, Any]:
    entry_val = float(trade_size)
    exit_val = float(pos.qty * exit_price)
    buy_fee = entry_val * fee_rate
    sell_fee = exit_val * fee_rate
    pnl = (exit_val - entry_val) - (buy_fee + sell_fee)
    return {
        "position_id": pos.pid,
        "cluster_id": pos.cluster_id,
        "entry_time": pos.entry_time,
        "exit_time": ts,
        "entry_price": float(pos.entry_price),
        "exit_price": float(exit_price),
        "qty": float(pos.qty),
        "reason": reason,
        "buy_fee_usdt": float(buy_fee),
        "sell_fee_usdt": float(sell_fee),
        "net_pnl_usdt": float(pnl),
    }


# ----------------------------
# Core verifier
# ----------------------------

DEFAULT_FEE_RATE = 0.001
DEFAULT_WARMUP_DAYS = 10  # needs to be >= 200 bars for 3m + some safety


def verify_symbol_fib_train(
    *,
    pair: str,
    interval: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    initial_capital: float,
    trade_size: float,
    fee_rate: float = DEFAULT_FEE_RATE,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
) -> Dict[str, Any]:
    """Run a fib-mode verification simulation for a single pair.

    IMPORTANT:
    - No scenario selection here.
    - Uses the fib engine's natural trigger flow (spearhead detection is assumed to be handled
      by the engine through on_spearhead calls we define as a minimal trigger.

    Returns metrics used by Stage4B gating.
    """

    pair = str(pair).strip().upper()
    interval = str(interval).strip()
    if interval not in {"1m", "3m"}:
        raise ValueError(f"interval must be '1m' or '3m', got {interval!r}")

    train_start = pd.to_datetime(train_start, utc=True)
    train_end = pd.to_datetime(train_end, utc=True)
    if train_end <= train_start:
        raise ValueError("train_end must be after train_start")

    # Warmup range
    warmup_start = train_start - pd.Timedelta(days=int(warmup_days))

    # Fetch LTF bars natively (audited requirement: do not force 1m + resample)
    ltf = fetch_klines_1m(pair, _to_utc_dt(warmup_start), _to_utc_dt(train_end), interval=interval)
    ltf = ltf.rename(columns={"open_time": "time"}).copy()
    ltf["time"] = pd.to_datetime(ltf["time"], utc=True)

    # Fetch 1h bars directly for HTF pivot calculations (audited requirement)
    htf = fetch_klines_1m(pair, _to_utc_dt(warmup_start), _to_utc_dt(train_end), interval="1h")
    htf = htf.rename(columns={"open_time": "time"}).copy()
    htf["time"] = pd.to_datetime(htf["time"], utc=True)

    # Build EMA50 on LTF for bounce validation
    ltf = ltf.sort_values("time").reset_index(drop=True)
    ltf["ema50"] = ltf["close"].ewm(span=50, adjust=False).mean()

    # Slice execution window (TRAIN week)
    window = ltf[(ltf["time"] >= train_start) & (ltf["time"] < train_end)].copy()
    if window.empty:
        return {
            "pair": pair,
            "interval": interval,
            "train_start": train_start,
            "train_end": train_end,
            "bars": 0,
            "net_profit_usdt": 0.0,
            "net_profit_pct": 0.0,
            "max_dd_pct": 0.0,
            "max_dd_usdt": 0.0,
            "clusters_completed": 0,
            "trades_closed": 0,
            "error": "no_bars_in_window",
        }

    # Instantiate fib engine with HTF override
    fib = MtfFibClusterEngine(symbol=pair, ohlcv_1m=ltf[["time", "open", "high", "low", "close", "volume"]])
    # Override the engine's internally built 1h with true 1h candles fetched from Binance.
    # This preserves engine behavior while meeting the audited spec.
    fib.htf_1h = htf[["time", "open", "high", "low", "close", "volume"]].copy()
    fib._htf_opens = pd.to_datetime(fib.htf_1h["time"], utc=True).to_numpy()

    capital = float(initial_capital)
    positions: Dict[str, _Position] = {}
    trades: List[Dict[str, Any]] = []
    equity: List[float] = []

    next_id = 1

    for _, bar in window.iterrows():
        ts = pd.to_datetime(bar["time"], utc=True)
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        ema50 = float(bar["ema50"]) if np.isfinite(bar["ema50"]) else np.nan

        # Update stop if in a cluster
        if positions:
            cluster_sl = fib.update_cluster_sl(ts=ts, bar_high=h, ltf_ema50=ema50)
            if np.isfinite(cluster_sl) and l <= cluster_sl:
                exit_price = max(o, float(cluster_sl))
                for pid, pos in list(positions.items()):
                    tr = _close_trade(
                        pos=pos,
                        ts=ts,
                        exit_price=exit_price,
                        reason="FIB_CLUSTER_SL",
                        fee_rate=fee_rate,
                        trade_size=trade_size,
                    )
                    trades.append(tr)
                    capital += float(tr["net_pnl_usdt"])
                    del positions[pid]
                fib.trigger_cooldown(ts=ts)

        # Cooldown logic
        if fib.cooldown_active:
            fib.maybe_release_cooldown(ts=ts, ltf_price=c)
        else:
            fib.apply_pre_entry_wipes(ts=ts, ltf_high=h, ltf_low=l, ltf_price=c)

        # Minimal spearhead: feed every bar as "potential" spearhead.
        # The engine itself will decide whether a valid grid is drawn.
        immediate_entry = False
        if not fib.cooldown_active and not positions:
            route = fib.on_spearhead(
                ts=ts,
                ltf_open=o,
                ltf_high=h,
                ltf_low=l,
                ltf_close=c,
                ltf_ema50=ema50,
            )
            immediate_entry = bool(route.get("immediate_entry", False))

        # Entry check
        if (not fib.cooldown_active) and (not positions):
            entry_window_open = True  # since no scenario gating in verifier
            if entry_window_open and (immediate_entry or fib.should_enter(ltf_low=l, ltf_close=c, ltf_ema50=ema50)):
                tickets = int(fib.pending_triggers)
                if tickets <= 0:
                    pass
                else:
                    free_slots = int(np.floor(capital / trade_size))
                    if free_slots >= tickets:
                        cluster_id = f"{pair}_FIBCL_{next_id}"
                        fib.lock_cluster(cluster_id=cluster_id, ts=ts, entry_price=c, ltf_ema50=ema50)

                        for _ in range(tickets):
                            pid = f"{pair}_LTF_{next_id}"
                            next_id += 1
                            qty = float(trade_size / c)
                            positions[pid] = _Position(
                                pid=pid,
                                entry_time=ts,
                                entry_price=c,
                                qty=qty,
                                cluster_id=cluster_id,
                            )
                    else:
                        # Not enough capital for full cluster; do nothing
                        pass

        equity.append(float(capital))

    # End of window: force close any open positions for deterministic accounting
    if positions:
        final_ts = pd.to_datetime(window.iloc[-1]["time"], utc=True)
        final_close = float(window.iloc[-1]["close"])
        for pid, pos in list(positions.items()):
            tr = _close_trade(
                pos=pos,
                ts=final_ts,
                exit_price=final_close,
                reason="WINDOW_END_MTM",
                fee_rate=fee_rate,
                trade_size=trade_size,
            )
            trades.append(tr)
            capital += float(tr["net_pnl_usdt"])
            del positions[pid]
        equity.append(float(capital))

    trades_df = pd.DataFrame(trades)
    net_profit = float(capital - float(initial_capital))
    net_profit_pct = float(net_profit / float(initial_capital)) if float(initial_capital) else 0.0

    max_dd_pct, max_dd_usdt = _compute_max_drawdown_from_equity(pd.Series(equity))

    clusters_completed = 0
    if not trades_df.empty and "cluster_id" in trades_df.columns:
        clusters_completed = int(trades_df["cluster_id"].dropna().astype(str).nunique())

    return {
        "pair": pair,
        "interval": interval,
        "train_start": train_start,
        "train_end": train_end,
        "bars": int(len(window)),
        "net_profit_usdt": float(net_profit),
        "net_profit_pct": float(net_profit_pct),
        "max_dd_pct": float(max_dd_pct),
        "max_dd_usdt": float(max_dd_usdt),
        "clusters_completed": int(clusters_completed),
        "trades_closed": int(len(trades_df)),
    }
