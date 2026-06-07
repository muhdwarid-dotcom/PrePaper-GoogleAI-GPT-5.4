from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _to_utc_ts(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def wilders_rma(series: pd.Series, length: int) -> pd.Series:
    """True Wilder's RMA (SMMA) matching TradingView and Binance charts,
    robustly handling leading NaNs (such as those from close.diff()).
    """
    vals = series.values
    out = np.empty_like(vals, dtype=float)
    out[:] = np.nan

    if len(vals) < length:
        return pd.Series(out, index=series.index)

    sma = series.rolling(window=length, min_periods=length).mean()

    first_valid_idx = -1
    for i in range(len(sma)):
        if not np.isnan(sma.values[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return pd.Series(out, index=series.index)

    out[first_valid_idx] = sma.values[first_valid_idx]

    alpha = 1.0 / length
    for i in range(first_valid_idx + 1, len(vals)):
        val = vals[i]
        if np.isnan(val):
            out[i] = out[i - 1]
        else:
            out[i] = val * alpha + out[i - 1] * (1.0 - alpha)

    return pd.Series(out, index=series.index)


def build_binance_aligned_1h(ohlcv_ltf: pd.DataFrame) -> pd.DataFrame:
    d = ohlcv_ltf[["time", "open", "high", "low", "close", "volume"]].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.sort_values("time").set_index("time")
    h1 = d.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    h1 = h1.reset_index()
    return h1


@dataclass
class GridState:
    fib_000: float
    fib_100: float
    swing_high_time: pd.Timestamp
    swing_low_time: pd.Timestamp
    dynamic_ceiling: bool = False


class MtfFibClusterEngine:
    def __init__(self, symbol: str, ohlcv_1m: pd.DataFrame) -> None:
        self.symbol = symbol
        self.verbose: bool = False
        self._pending_grid_log: Optional[dict] = None

        # store raw 1m for lookahead-free intra-hour comparisons
        self.ohlcv_1m = ohlcv_1m.copy()
        self.ohlcv_1m["time"] = pd.to_datetime(self.ohlcv_1m["time"], utc=True)
        self.ohlcv_1m = self.ohlcv_1m.sort_values("time").set_index("time")

        self.htf_1h = build_binance_aligned_1h(ohlcv_1m)
        self._htf_opens = pd.to_datetime(self.htf_1h["time"], utc=True).to_numpy()
        self.pending_triggers = 0
        self.pre_entry_grid: Optional[GridState] = None

        self.cluster_id: str = ""
        self.locked_fib_000: float = np.nan
        self.locked_fib_100: float = np.nan
        self.current_cluster_sl: float = np.nan
        self.highest_price_since_entry: float = np.nan

        self.cooldown_active = False
        self.cooldown_locked_fib_000: float = np.nan
        self.cooldown_locked_fib_100: float = np.nan

    @staticmethod
    def _fib_price(fib_000: float, fib_100: float, ratio: float) -> float:
        return fib_000 - ((fib_000 - fib_100) * ratio)

    @staticmethod
    def _ext_price(fib_000: float, fib_100: float, ext_ratio: float) -> float:
        return fib_000 + ((fib_000 - fib_100) * ext_ratio)

    def _find_containing_1h_index(self, spearhead_ts: pd.Timestamp) -> int:
        ts = _to_utc_ts(spearhead_ts)
        opens = self._htf_opens
        idx = int(opens.searchsorted(ts.to_datetime64(), side="right") - 1)
        if idx < 0:
            return -1
        if idx >= len(self.htf_1h):
            return len(self.htf_1h) - 1
        return idx

    def _walkback_grid(self, spearhead_ts: pd.Timestamp) -> Optional[GridState]:
        if self.htf_1h.empty:
            return None

        head_idx = self._find_containing_1h_index(spearhead_ts)
        if head_idx < 2:
            return None

        highs = self.htf_1h["high"].astype(float).to_numpy()
        lows = self.htf_1h["low"].astype(float).to_numpy()
        times = pd.to_datetime(self.htf_1h["time"], utc=True).to_list()

        pivot_high_idx = -1
        ts = _to_utc_ts(spearhead_ts)
        hour_start = ts.floor("1h")
        sub_df = self.ohlcv_1m.loc[hour_start:ts]
        current_hour_high_so_far = float(sub_df["high"].max()) if not sub_df.empty else -np.inf

        for i in range(head_idx - 1, 0, -1):
            compare_right_high = current_hour_high_so_far if i == head_idx - 1 else highs[i + 1]
            if highs[i] > highs[i - 1] and highs[i] > compare_right_high:
                pivot_high_idx = i
                break
        if pivot_high_idx < 1:
            return None

        pivot_low_idx = -1
        for j in range(pivot_high_idx - 1, 0, -1):
            if lows[j] < lows[j - 1] and lows[j] < lows[j + 1]:
                pivot_low_idx = j
                break
        if pivot_low_idx < 1:
            return None

        grid = GridState(
            fib_000=float(highs[pivot_high_idx]),
            fib_100=float(lows[pivot_low_idx]),
            swing_high_time=times[pivot_high_idx],
            swing_low_time=times[pivot_low_idx],
            dynamic_ceiling=False,
        )
        # Prepare a deferred verbose “ACTIVE GRID” block for the caller to print after SIGNAL.
        if self.verbose:
            fib_0382 = self._fib_price(grid.fib_000, grid.fib_100, 0.382)
            fib_0500 = self._fib_price(grid.fib_000, grid.fib_100, 0.5)
            fib_0618 = self._fib_price(grid.fib_000, grid.fib_100, 0.618)
            fib_0786 = self._fib_price(grid.fib_000, grid.fib_100, 0.786)

            self._pending_grid_log = {
                "ts": _to_utc_ts(spearhead_ts),
                "fib_0000": float(grid.fib_000),
                "fib_0382": float(fib_0382),
                "fib_0500": float(fib_0500),
                "fib_0618": float(fib_0618),
                "fib_0786": float(fib_0786),
                "fib_1000": float(grid.fib_100),
            }
        return grid

    def flush_pending_grid_log(self) -> None:
        if not self.verbose or not self._pending_grid_log:
            return

        d = self._pending_grid_log
        self._pending_grid_log = None

        print("=" * 60)
        print(f"  MTF FIB CLUSTER ENGINE - ACTIVE GRID FOR {self.symbol}")
        print(f"  Signal Timestamp: {d['ts']}")
        print("-" * 60)
        print(f"  FIB_0000 (HH / High):  {d['fib_0000']:.6f}")
        print(f"  FIB_0382:              {d['fib_0382']:.6f}")
        print(f"  FIB_0500:              {d['fib_0500']:.6f}")
        print(f"  FIB_0618:              {d['fib_0618']:.6f}")
        print(f"  FIB_0786:              {d['fib_0786']:.6f}")
        print(f"  FIB_1000 (LL / Low):   {d['fib_1000']:.6f}")
        print("=" * 60)
    
    def on_spearhead(
        self,
        *,
        ts: pd.Timestamp,
        ltf_open: float,
        ltf_high: float,
        ltf_low: float,
        ltf_close: float,
        ltf_ema50: float,
    ) -> Dict[str, bool]:
        if self.cooldown_active:
            print(f"[FIB_MTF][{self.symbol}] cooldown_ignore_spearhead ts={_to_utc_ts(ts)}", flush=True)
            return {"immediate_entry": False}

        grid = self._walkback_grid(ts)
        if (
            grid is None
            or (not np.isfinite(grid.fib_000))
            or (not np.isfinite(grid.fib_100))
            or grid.fib_000 <= grid.fib_100
        ):
            print(f"[FIB_MTF][{self.symbol}] grid_draw_failed ts={_to_utc_ts(ts)}", flush=True)
            return {"immediate_entry": False}

        self.pre_entry_grid = grid
        fib_0382 = self._fib_price(grid.fib_000, grid.fib_100, 0.382)
        fib_0500 = self._fib_price(grid.fib_000, grid.fib_100, 0.5)
        fib_0618 = self._fib_price(grid.fib_000, grid.fib_100, 0.618)

        immediate_entry = False
        route = "none"

        if ltf_close > grid.fib_000:
            route = "scenario_i_breakout"
            grid.dynamic_ceiling = True
            grid.fib_000 = float(ltf_high)
            self.pending_triggers += 1
        elif ltf_close <= grid.fib_000 and ltf_close > fib_0382:
            route = "scenario_ii_ideal_wait"
            self.pending_triggers += 1
        elif ltf_low <= fib_0500 and ltf_low >= (fib_0618 * 0.99):
            route = "scenario_iii_instant_touch"
            self.pending_triggers += 1
            immediate_entry = bool(np.isfinite(ltf_ema50) and ltf_close > ltf_ema50)
        elif ltf_low < (fib_0618 * 0.99):
            route = "scenario_iv_dead_setup"
            self.pending_triggers = 0
            self.pre_entry_grid = None

        print(
            f"[FIB_MTF][{self.symbol}] scenario_route ts={_to_utc_ts(ts)} route={route} "
            f"pending={self.pending_triggers} ohlc=({ltf_open:.8f},{ltf_high:.8f},{ltf_low:.8f},{ltf_close:.8f})",
            flush=True,
        )
        return {"immediate_entry": immediate_entry}

    def _kill_zone(self) -> Optional[Dict[str, float]]:
        if self.pre_entry_grid is None or self.pending_triggers <= 0:
            return None
        g = self.pre_entry_grid
        top_ratio = 0.5 if self.pending_triggers == 1 else 0.382
        top = self._fib_price(g.fib_000, g.fib_100, top_ratio)
        bottom = self._fib_price(g.fib_000, g.fib_100, 0.618) * 0.99
        return {"top": top, "bottom": bottom}

    def apply_pre_entry_wipes(self, *, ts: pd.Timestamp, ltf_high: float, ltf_low: float, ltf_price: float) -> None:
        """Pre-entry invalidation rules.

        As per latest spec, there is **no upside wipe**.
        The engine should preserve tickets + grid even if price trades above Fib_000 before entry.
        In breakout mode, Fib_000 may continue to stretch upward on new highs.

        Downside wipe remains: ltf_low < Fib_0786 * 0.99.
        """
        if self.pre_entry_grid is None or self.pending_triggers <= 0:
            return

        g = self.pre_entry_grid
        if g.dynamic_ceiling and ltf_high > g.fib_000:
            g.fib_000 = float(ltf_high)
            print(
                f"[FIB_MTF][{self.symbol}] dynamic_ceiling_stretch ts={_to_utc_ts(ts)} fib000={g.fib_000:.8f}",
                flush=True,
            )

        fib_0786 = self._fib_price(g.fib_000, g.fib_100, 0.786)
        if ltf_low < (fib_0786 * 0.99):
            self.pending_triggers = 0
            self.pre_entry_grid = None
            print(f"[FIB_MTF][{self.symbol}] downside_wipe ts={_to_utc_ts(ts)}", flush=True)

    def should_enter(self, *, ltf_low: float, ltf_close: float, ltf_ema50: float) -> bool:
        kz = self._kill_zone()
        if kz is None:
            return False
        if not np.isfinite(ltf_ema50):
            return False
        touch_ok = kz["bottom"] <= ltf_low <= kz["top"]
        bounce_ok = ltf_close > ltf_ema50
        return bool(self.pending_triggers > 0 and touch_ok and bounce_ok)

    def lock_cluster(self, *, cluster_id: str, ts: pd.Timestamp, entry_price: float, ltf_ema50: float) -> None:        
        if self.pre_entry_grid is None:
            return
        g = self.pre_entry_grid
        self.cluster_id = cluster_id
        self.locked_fib_000 = float(g.fib_000)
        self.locked_fib_100 = float(g.fib_100)
        self.highest_price_since_entry = float(entry_price)
        self.current_cluster_sl = self._fib_price(self.locked_fib_000, self.locked_fib_100, 0.786) * 0.99
        self.pre_entry_grid = None
        self.pending_triggers = 0
        print(
            f"[FIB_MTF][{self.symbol}] lock_cluster ts={_to_utc_ts(ts)} cluster={cluster_id} "
            f"fib000_locked={self.locked_fib_000:.8f} fib100_locked={self.locked_fib_100:.8f} "
            f"sl={self.current_cluster_sl:.8f} ema50={ltf_ema50:.8f}",
            flush=True,
        )

    def _cycle0_sl(self, highest_price: float, ltf_ema50: float) -> float:
        """Cycle-0 trailing SL logic (prior to breaching locked_fib_000)."""
        fib_000 = self.locked_fib_000
        fib_100 = self.locked_fib_100
        if not np.isfinite(fib_000) or not np.isfinite(fib_100) or fib_000 <= fib_100:
            return np.nan

        fib_0382 = self._fib_price(fib_000, fib_100, 0.382)
        fib_0500 = self._fib_price(fib_000, fib_100, 0.5)

        sl = self._fib_price(fib_000, fib_100, 0.786) * 0.99

        if highest_price > fib_0500 and highest_price <= fib_0382:
            sl = min(fib_0500 * 0.98, ltf_ema50 * 0.99)

        elif highest_price > fib_0382 and highest_price <= fib_000:
            sl = min(fib_0382 * 0.98, ltf_ema50 * 0.99)

        return max(sl, self.current_cluster_sl)

    def _strict_cycle_sl(self, highest: float) -> float:
        fib_000 = self.locked_fib_000
        fib_100 = self.locked_fib_100
        if not np.isfinite(fib_000) or not np.isfinite(fib_100) or fib_000 <= fib_100:
            return np.nan
        span = fib_000 - fib_100
        ext_progress = (highest - fib_000) / span
        if ext_progress <= 1.0:
            return self._cycle0_sl(highest, np.nan)

        rung_seq: List[float] = [0.0, 0.382, 0.618, 0.786, 1.0]
        max_cycle = int(np.ceil(ext_progress)) + 2
        for cyc in range(1, max_cycle + 1):
            rung_seq.extend([cyc + 0.382, cyc + 0.618, cyc + 0.786, cyc + 1.0])
        rung_seq = sorted(set(rung_seq))

        idx = 0
        for i, r in enumerate(rung_seq):
            if ext_progress >= r:
                idx = i
            else:
                break
        prev_ratio = rung_seq[max(0, idx - 1)]
        prev_price = self._ext_price(fib_000, fib_100, prev_ratio)
        return prev_price * 0.99

    def update_cluster_sl(self, *, ts: pd.Timestamp, bar_high: float, ltf_ema50: float) -> float:
        if not np.isfinite(self.locked_fib_000) or not np.isfinite(self.locked_fib_100):
            return np.nan
        if not np.isfinite(self.highest_price_since_entry):
            self.highest_price_since_entry = float(bar_high)
        else:
            self.highest_price_since_entry = max(float(self.highest_price_since_entry), float(bar_high))

        highest = self.highest_price_since_entry
        locked_000 = self.locked_fib_000
        locked_100 = self.locked_fib_100

        ext_0382 = self._ext_price(locked_000, locked_100, 0.382)
        ext_0618 = self._ext_price(locked_000, locked_100, 0.618)
        ext_0786 = self._ext_price(locked_000, locked_100, 0.786)
        ext_1000 = self._ext_price(locked_000, locked_100, 1.0)

        if highest <= locked_000:
            new_sl = self._cycle0_sl(highest, ltf_ema50)
            if np.isfinite(new_sl):
                if not np.isfinite(self.current_cluster_sl):
                    self.current_cluster_sl = new_sl
                else:
                    self.current_cluster_sl = max(float(self.current_cluster_sl), float(new_sl))
        else:
            if highest > locked_000 and highest <= ext_0382:
                sl_val = min(locked_000 * 0.98, ltf_ema50 * 0.99)
                self.current_cluster_sl = max(self.current_cluster_sl, sl_val)

            elif highest > ext_0382 and highest <= ext_0618:
                self.current_cluster_sl = max(self.current_cluster_sl, locked_000 * 0.98)

            elif highest > ext_0618 and highest <= ext_0786:
                self.current_cluster_sl = max(self.current_cluster_sl, ext_0382 * 0.99)

            elif highest > ext_0786 and highest <= ext_1000:
                self.current_cluster_sl = max(self.current_cluster_sl, ext_0618 * 0.99)

            elif highest > ext_1000:
                self.current_cluster_sl = max(self.current_cluster_sl, ext_0786 * 0.99)

        print(
            f"[FIB_MTF][{self.symbol}] sl_update ts={_to_utc_ts(ts)} highest={self.highest_price_since_entry:.8f} "
            f"sl={self.current_cluster_sl:.8f}",
            flush=True,
        )
        return self.current_cluster_sl

    def trigger_cooldown(self, *, ts: pd.Timestamp) -> None:
        self.cooldown_active = True
        self.cooldown_locked_fib_000 = self.locked_fib_000
        self.cooldown_locked_fib_100 = self.locked_fib_100
        self.cluster_id = ""
        self.locked_fib_000 = np.nan
        self.locked_fib_100 = np.nan
        self.current_cluster_sl = np.nan
        self.highest_price_since_entry = np.nan
        self.pending_triggers = 0
        self.pre_entry_grid = None
        print(
            f"[FIB_MTF][{self.symbol}] cooldown_start ts={_to_utc_ts(ts)} "
            f"fib000={self.cooldown_locked_fib_000:.8f} fib100={self.cooldown_locked_fib_100:.8f}",
            flush=True,
        )

    def maybe_release_cooldown(self, *, ts: pd.Timestamp, ltf_price: float) -> bool:
        if not self.cooldown_active:
            return False
        if ltf_price > self.cooldown_locked_fib_000 or ltf_price < self.cooldown_locked_fib_100:
            self.cooldown_active = False
            print(
                f"[FIB_MTF][{self.symbol}] cooldown_end ts={_to_utc_ts(ts)} price={ltf_price:.8f}",
                flush=True,
            )
            self.cooldown_locked_fib_000 = np.nan
            self.cooldown_locked_fib_100 = np.nan
            return True
        return False
