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
ROBUST_WORST_WEEK_NET_MIN = -200.0   # R2 gate: worst week net_profit must be > -X (tune)
ROBUST_RANK_PRIMARY = "median_profit_over_maxdd"  # for display; we will rank by this after gates
ROBUST_WEEK_NET_MIN = 500.0      # R1: each slice/week must earn at least this net profit
ROBUST_REQUIRE_ALL_WEEKS = True  # enforce 4/4 passing R1_week
ROBUST_MEAN_POMDD_MIN = 0.8    # R3: median_profit_over_maxdd must be >= this

# ----------------------------
#  OFF because x<60
# ----------------------------
#X_BARS_MIN_DELAY = 60  # below this, treat as OFF (immediate trailing)
X_BARS_MIN_DELAY = 0  # disable minimum delay; any x_bars (even 0) triggers immediate trailing

# ... (content truncated in chat upload)
