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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import contextlib
import io

import numpy as np
import pandas as pd

from binance_fetch import fetch_klines_1m
from mtf_fib_cluster_engine import MtfFibClusterEngine, wilders_rma

# ----------------------------
# Utilities
# ----------------------------

# ANSI colors (match legacy runner)
COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_RESET = "\033[0m"


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

def _rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = wilders_rma(delta.clip(lower=0), length)
    avg_loss = wilders_rma(-delta.clip(upper=0), length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _normalize_bool_gate(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"", "ALL", "ANY", "NONE", "NULL", "NAN"}:
        return None
    if text in {"TRUE", "T", "1"}:
        return True
    if text in {"FALSE", "F", "0"}:
        return False
    return None


def parse_vol_rule(vol_rule_str: str) -> dict:
    vol_rule_str = str(vol_rule_str).strip()
    if vol_rule_str == "ALL":
        return {"type": "ALL"}
    if vol_rule_str.startswith(">="):
        threshold = float(vol_rule_str[2:])
        return {"type": "gte", "threshold": threshold}
    if vol_rule_str.startswith("<"):
        threshold = float(vol_rule_str[1:])
        return {"type": "lt", "threshold": threshold}
    if "_" in vol_rule_str:
        parts = vol_rule_str.split("_")
        if len(parts) == 2:
            low = float(parts[0])
            high = float(parts[1])
            return {"type": "bin", "low": low, "high": high}
    raise ValueError(f"Unsupported vol_rule format: {vol_rule_str}")


def apply_vol_rule_filter(events: pd.DataFrame, vol_rule: dict) -> pd.DataFrame:
    if vol_rule["type"] == "ALL":
        return events
    if vol_rule["type"] == "gte":
        return events[events["vol_ratio"] >= vol_rule["threshold"]].copy()
    if vol_rule["type"] == "lt":
        return events[events["vol_ratio"] < vol_rule["threshold"]].copy()
    if vol_rule["type"] == "bin":
        return events[
            (events["vol_ratio"] >= vol_rule["low"]) & (events["vol_ratio"] < vol_rule["high"])
        ].copy()
    return events


def _vprint(verbose: bool, *args, **kwargs) -> None:
    """Verbose print helper (safe no-op when verbose=False)."""
    if verbose:
        print(*args, **kwargs)


def _colorize(line: str, color: str | None) -> str:
    if not color:
        return line
    return f"{color}{line}{COLOR_RESET}"


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
RSI_SMA_CROSS_THRESHOLD = 51.0


def verify_symbol_fib_train(
    *,
    pair: str,
    interval: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    initial_capital: float,
    trade_size: float,
    close_gate: Any = "ALL",
    vol_gate: Any = "ALL",
    vol_rule: str = "ALL",
    fee_rate: float = DEFAULT_FEE_RATE,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a fib-mode verification simulation for a single pair.

    IMPORTANT:
    - No scenario selection here.
    - Uses the fib engine's natural trigger flow.

    Returns metrics used by Stage4B gating.
    """

    pair = str(pair).strip().upper()
    interval = str(interval).strip()
    if interval not in {"1m", "3m"}:
        raise ValueError(f"interval must be '1m' or '3m', got {interval!r}")

    train_start = pd.to_datetime(train_start).tz_localize(None)
    train_end = pd.to_datetime(train_end).tz_localize(None)
    if train_end <= train_start:
        raise ValueError("train_end must be after train_start")

    warmup_start = train_start - pd.Timedelta(days=int(warmup_days))

    # Fetch LTF bars natively
    ltf = fetch_klines_1m(pair, _to_utc_dt(warmup_start), _to_utc_dt(train_end), interval=interval)
    ltf = ltf.rename(columns={"open_time": "time"}).copy()
    ltf["time"] = pd.to_datetime(ltf["time"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        ltf[col] = pd.to_numeric(ltf[col], errors="coerce")
    ltf = ltf.dropna(subset=["time", "open", "high", "low", "close", "volume"]).copy()

    # Fetch 1h bars directly for HTF pivots
    htf = fetch_klines_1m(pair, _to_utc_dt(warmup_start), _to_utc_dt(train_end), interval="1h")
    htf = htf.rename(columns={"open_time": "time"}).copy()
    htf["time"] = pd.to_datetime(htf["time"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        htf[col] = pd.to_numeric(htf[col], errors="coerce")
    htf = htf.dropna(subset=["time", "open", "high", "low", "close", "volume"]).copy()

    # Indicators
    ltf = ltf.sort_values("time").reset_index(drop=True)
    ltf["rsi"] = _rsi_wilder(ltf["close"], 14)
    ltf["rsi_sma"] = ltf["rsi"].rolling(window=14).mean()
    ltf["cross_up_51"] = (ltf["rsi_sma"].shift(1) < RSI_SMA_CROSS_THRESHOLD) & (ltf["rsi_sma"] >= RSI_SMA_CROSS_THRESHOLD)
    ltf["smma_200"] = ltf["close"].ewm(span=200, adjust=False).mean()
    ltf["vol_sma"] = ltf["volume"].rolling(window=20).mean()
    ltf["ema20"] = ltf["close"].ewm(span=20, adjust=False).mean()
    ltf["ema50"] = ltf["close"].ewm(span=50, adjust=False).mean()

    gate_close = _normalize_bool_gate(close_gate)
    gate_vol = _normalize_bool_gate(vol_gate)
    vol_rule_str = str(vol_rule).strip() if vol_rule is not None else "ALL"
    if not vol_rule_str or vol_rule_str.upper() in {"NONE", "NULL", "NAN"}:
        vol_rule_str = "ALL"
    try:
        parsed_vol_rule = parse_vol_rule(vol_rule_str)
    except Exception:
        parsed_vol_rule = {"type": "ALL"}

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

    fib = MtfFibClusterEngine(symbol=pair, ohlcv_1m=ltf[["time", "open", "high", "low", "close", "volume"]])
    fib.htf_1h = htf[["time", "open", "high", "low", "close", "volume"]].copy()
    fib._htf_opens = pd.to_datetime(fib.htf_1h["time"]).dt.tz_localize(None).to_numpy()

    capital = float(initial_capital)
    positions: Dict[str, _Position] = {}
    trades: List[Dict[str, Any]] = []
    equity: List[float] = []

    setup_idx = 0

    for _, bar in window.iterrows():
        ts = pd.to_datetime(bar["time"]).tz_localize(None)
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        volume = float(bar["volume"])
        rsi_sma = float(bar["rsi_sma"]) if np.isfinite(bar["rsi_sma"]) else np.nan
        smma_200 = float(bar["smma_200"]) if np.isfinite(bar["smma_200"]) else np.nan
        vol_sma = float(bar["vol_sma"]) if np.isfinite(bar["vol_sma"]) else np.nan
        ema20 = float(bar["ema20"]) if np.isfinite(bar["ema20"]) else np.nan
        ema50 = float(bar["ema50"]) if np.isfinite(bar["ema50"]) else np.nan
        cross_up_51 = bool(bar["cross_up_51"]) if pd.notna(bar["cross_up_51"]) else False

        vol_ratio = np.nan
        if pd.notna(vol_sma) and np.isfinite(vol_sma) and (vol_sma > 0):
            vol_ratio = float(volume / float(vol_sma))

        bar_df = pd.DataFrame([{
            "close_gt_smma_200": bool(pd.notna(smma_200) and (c > float(smma_200))),
            "vol_gt_vol_sma": bool(pd.notna(vol_sma) and (volume > float(vol_sma))),
            "vol_ratio": vol_ratio,
        }])

        gate_ok = True
        if gate_close is not None:
            gate_ok = bool(bar_df.at[0, "close_gt_smma_200"] == gate_close)
        if gate_ok and gate_vol is not None:
            gate_ok = bool(bar_df.at[0, "vol_gt_vol_sma"] == gate_vol)
        if gate_ok and parsed_vol_rule.get("type") != "ALL":
            filtered_bar = apply_vol_rule_filter(bar_df, parsed_vol_rule)
            gate_ok = not filtered_bar.empty

        # Trigger routing: cross-up event -> legacy SIGNAL line
        if verbose and cross_up_51:
            line = (
                f"{ts.strftime('%Y-%m-%d %H:%M')} | SIGNAL     | {pair:<10} | Price {c:.6f}   | "
                f"RSI_SMA {rsi_sma:.2f} | SMMA {smma_200:.5f} | C>SMMA {str(c > smma_200):<6} | "
                f"V>VSMA {str(volume > vol_sma):<6} | VR  {vol_ratio:.2f}"
            )
            print(_colorize(line, COLOR_BLUE), flush=True)

        # Update stop if in a cluster
        if positions:
            # Mute noisy engine debug prints for clean legacy-like output.
            with contextlib.redirect_stdout(io.StringIO()):
                locked_000 = float(fib.locked_fib_000)
                locked_100 = float(fib.locked_fib_100)
                fib_0618 = (
                    locked_000 - (locked_000 - locked_100) * 0.618
                    if (np.isfinite(locked_000) and np.isfinite(locked_100))
                    else np.nan
                )

                if np.isfinite(fib_0618) and np.isfinite(smma_200) and np.isfinite(ema50) and np.isfinite(ema20):
                    if c < fib_0618 and c < smma_200 and c < ema50 and c < ema20:
                        for pid, pos in list(positions.items()):
                            tr = _close_trade(
                                pos=pos,
                                ts=ts,
                                exit_price=c,
                                reason="FIB_FORCE_STOP",
                                fee_rate=fee_rate,
                                trade_size=trade_size,
                            )
                            pnl = float(tr["net_pnl_usdt"])
                            capital += pnl
                            del positions[pid]

                            if verbose:
                                max_ports = int(capital // trade_size) if trade_size > 0 else 10
                                line = (
                                    f"{ts.strftime('%Y-%m-%d %H:%M')} | STOP       | {pair:<10} | Price {c:.6f}   | "
                                    f"ID {pos.pid:<10} | Trigger FORCE-STOP (Capitulation) | "
                                    f"P/L $ {pnl:>7.2f} | Cap $ {capital:>9.2f} | "
                                    f"Port {len(positions):02d}/{max_ports:02d}"
                                )
                                print(_colorize(line, COLOR_RED), flush=True)

                            trades.append(tr)

                        with contextlib.redirect_stdout(io.StringIO()):
                            fib.trigger_cooldown(ts=ts)
                        equity.append(float(capital))
                        continue
                    
                # Mute noisy engine debug prints for clean legacy-like output.
                with contextlib.redirect_stdout(io.StringIO()):
                    cluster_sl = fib.update_cluster_sl(ts=ts, bar_high=h, ltf_ema50=ema50)

            if np.isfinite(cluster_sl) and l <= cluster_sl:
                trail_sl = float(cluster_sl)
                exit_price = max(o, trail_sl)

                locked_000 = float(fib.locked_fib_000)
                locked_100 = float(fib.locked_fib_100)
                locked_050 = (
                    locked_000 - (locked_000 - locked_100) * 0.5
                    if (np.isfinite(locked_000) and np.isfinite(locked_100))
                    else np.nan
                )
                highest = float(fib.highest_price_since_entry)

                if np.isfinite(locked_050) and np.isfinite(highest) and highest >= locked_050:
                    trigger_reason = f"TTP (FIB_0500 @ {locked_050:.6f} Breached | SL Locked @ {trail_sl:.6f})"
                else:
                    initial_sl = (
                        (locked_000 - (locked_000 - locked_100) * 0.786) * 0.99
                        if (np.isfinite(locked_000) and np.isfinite(locked_100))
                        else np.nan
                    )
                    trigger_reason = f"SL (FIB_0786_Wipe @ {initial_sl:.6f})"

                for pid, pos in list(positions.items()):
                    tr = _close_trade(
                        pos=pos,
                        ts=ts,
                        exit_price=exit_price,
                        reason="FIB_CLUSTER_SL",
                        fee_rate=fee_rate,
                        trade_size=trade_size,
                    )
                    pnl = float(tr["net_pnl_usdt"])
                    capital += pnl
                    del positions[pid]

                    if verbose:
                        max_ports = int(capital // trade_size) if trade_size > 0 else 10
                        line = (
                            f"{ts.strftime('%Y-%m-%d %H:%M')} | STOP       | {pair:<10} | Price {exit_price:.6f}   | "
                            f"ID {pos.pid:<10} | {trigger_reason} | P/L $ {pnl:>7.2f} | Cap $ {capital:>9.2f} | "
                            f"Port {len(positions):02d}/{max_ports:02d}"
                        )
                        color = COLOR_GREEN if pnl >= 0 else COLOR_RED
                        print(_colorize(line, color), flush=True)

                    trades.append(tr)

                # Mute noisy engine debug prints for clean legacy-like output.
                with contextlib.redirect_stdout(io.StringIO()):
                    fib.trigger_cooldown(ts=ts)

        # Cooldown logic
        if fib.cooldown_active:
            with contextlib.redirect_stdout(io.StringIO()):
                fib.maybe_release_cooldown(ts=ts, ltf_price=c)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                fib.apply_pre_entry_wipes(ts=ts, ltf_high=h, ltf_low=l, ltf_price=c)

        # Candidate-gated event trigger: only feed cross-up bars that pass gates.
        immediate_entry = False
        if cross_up_51 and gate_ok and (not fib.cooldown_active) and (not positions):
            with contextlib.redirect_stdout(io.StringIO()):
                route = fib.on_spearhead(
                    ts=ts,
                    ltf_open=o,
                    ltf_high=h,
                    ltf_low=l,
                    ltf_close=c,
                    ltf_ema50=ema50,
                )

            fib.verbose = bool(verbose)
            fib.flush_pending_grid_log()

            immediate_entry = bool(route.get("immediate_entry", False))

        # Entry check
        if gate_ok and (not fib.cooldown_active) and (not positions):
            entry_window_open = True
            if entry_window_open and (immediate_entry or fib.should_enter(ltf_low=l, ltf_close=c, ltf_ema50=ema50)):
                tickets = int(fib.pending_triggers)
                if tickets > 0:
                    free_slots = int(np.floor(capital / trade_size))
                    if free_slots >= tickets:
                        setup_idx += 1
                        cluster_id = f"fib_{setup_idx}"
                        with contextlib.redirect_stdout(io.StringIO()):
                            fib.lock_cluster(
                                cluster_id=cluster_id,
                                ts=ts,
                                entry_price=c,
                                ltf_ema50=ema50,
                            )

                        for ticket_index in range(tickets):
                            pid = f"fib_{setup_idx}_{ticket_index}"
                            qty = float(trade_size / c)
                            positions[pid] = _Position(
                                pid=pid,
                                entry_time=ts,
                                entry_price=c,
                                qty=qty,
                                cluster_id=cluster_id,
                            )

                            if verbose:
                                max_ports = int(capital // trade_size) if trade_size > 0 else 10
                                price_gt_ema50 = bool(np.isfinite(ema50) and c > ema50)
                                line = (
                                    f"{ts.strftime('%Y-%m-%d %H:%M')} | OPEN       | {pair:<10} | Price {c:.6f}   | "
                                    f"ID {pid:<10} | C>SMMA {str(c > smma_200):<5} | V>VSMA {str(volume > vol_sma):<5} | "
                                    f"VR {vol_ratio:>5.2f} | Price>EMA50 {str(price_gt_ema50):<5} | "
                                    f"Port {len(positions):02d}/{max_ports:02d}"
                                )
                                print(_colorize(line, COLOR_BLUE), flush=True)

        equity.append(float(capital))

    # End of window: force close any open positions for deterministic accounting
    if positions:
        final_ts = pd.to_datetime(window.iloc[-1]["time"]).tz_localize(None)
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
