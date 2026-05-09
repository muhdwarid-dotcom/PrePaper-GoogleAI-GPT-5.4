import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from time import sleep
from binance.client import Client
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Import tenacity and yaml after installing them
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import yaml

# Define RESULTS_DIR before using it in logging configuration
RESULTS_DIR = "results_v29R_30d"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    UNIVERSE_LIMIT = config["universe_limit"]
    TOP_N_STAGE1A = config["top_n_stage1A"]
    STAGE1A_INTERVAL = config["stage1A_interval"]
    STAGE1B_INTERVAL = config["stage1B_interval"]
    RESULTS_DIR = config.get("results_dir", RESULTS_DIR)  # Use the config value if it exists, otherwise default
except FileNotFoundError:
    logger.warning("Config file not found, using default values.")
    UNIVERSE_LIMIT = 440      # How many symbols to scan at Stage1A
    TOP_N_STAGE1A = 60        # How many symbols to keep for Stage1B
    STAGE1A_INTERVAL = "4h"
    STAGE1B_INTERVAL = "15m"

# -----------------------------------------------------------
# Insert helper function HERE:
# -----------------------------------------------------------

def get_windows_from_manual_monday():
    logger.info("\n[WARN] AutoRun not detected.")
    monday_str = input("Please enter the reference Monday (YYYY-MM-DD): ").strip()

    try:
        M = datetime.strptime(monday_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.error(f"Invalid date format: {monday_str}")
        raise RuntimeError(f"Invalid date format: {monday_str}")

    trade_end = M
    trade_start = M - timedelta(days=7)
    train_end = trade_start
    train_start = train_end - timedelta(days=30)

    logger.info(f"Train window: {train_start} to {train_end}")
    logger.info(f"Trade window: {trade_start} to {trade_end}")

    return {
        "train_start": train_start,
        "train_end": train_end,
        "trade_start": trade_start,
        "trade_end": trade_end
    }

# ===============================
# CONFIG
# ===============================

TRAIN_START = None   # AUTORUN will override
TRAIN_END = None     # AUTORUN will override

RESULTS_DIR = "results_v29R_30d"
os.makedirs(RESULTS_DIR, exist_ok=True)

UNIVERSE_LIMIT = 440      # How many symbols to scan at Stage1A
TOP_N_STAGE1A = 60        # How many symbols to keep for Stage1B
STAGE1A_INTERVAL = "4h"
STAGE1B_INTERVAL = "15m"

# --- SURGICAL ADDITION: LINE 44 ---
def _wilders_rma(series, length):
    """Aligns Stage 1 with v30 SMMA 200 logic."""
    return series.ewm(alpha=1/length, adjust=False).mean()

# ===============================
# Fetch Historical Candles
# ===============================
# --- SURGICAL REPAIR: fetch_klines (Stage 1) ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_klines(client, symbol, interval, start, end):
    try:
        start_str = start.strftime("%d %b %Y %H:%M:%S")
        end_str = end.strftime("%d %b %Y %H:%M:%S")

        logger.info(f"Fetching data for {symbol} from {start_str} to {end_str}")

        raw = client.get_historical_klines(symbol, interval, start_str, end_str)
        if not raw:
            logger.warning(f"No data returned for {symbol}")
            return None

        df = pd.DataFrame(raw).iloc[:, :6]
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        df["time"] = pd.to_datetime(df["time"], unit='ms', utc=True)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        logger.info(f"Successfully fetched data for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        sleep(1)  # Add delay to avoid hitting rate limits
        return None

# ===============================
# Stage 1A — Macro Scan
# ===============================
# --- SURGICAL REPLACEMENT: Stage 1A Scoring ---
def compute_stage1A_score(df):
    if df is None or len(df) < 150:
        logger.warning("Insufficient data for scoring")
        return None

    close = df["close"]
    smma_macro = _wilders_rma(close, 100)

    if close.iloc[-1] < smma_macro.iloc[-1]:
        logger.debug(f"Filtered out due to macro downtrend: {close.iloc[-1]} < {smma_macro.iloc[-1]}")
        return 0.0

    df["ret"] = close.pct_change()
    vol = df["ret"].std()
    trend = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    volume_trend = (df["volume"].iloc[-1] - df["volume"].iloc[0]) / df["volume"].iloc[0]

    # Enhanced score calculation
    score = vol * abs(trend) * (1 + volume_trend)
    logger.debug(f"Score calculated: {score}")
    return float(score)

def stage1A(client, train_start, train_end):
    logger.info("========== STAGE 1A — MACRO SCAN ==========")
    logger.info(f"Train: {train_start} to {train_end}")

    exchange_info = client.get_exchange_info()
    symbols = [s["symbol"] for s in exchange_info["symbols"]
               if s["symbol"].endswith("USDT") and s["status"] == "TRADING"]

    if len(symbols) > UNIVERSE_LIMIT:
        symbols = symbols[:UNIVERSE_LIMIT]

    logger.info(f"Found {len(symbols)} valid symbols")

    rows = []
    for sym in symbols:
        df = fetch_klines(client, sym, STAGE1A_INTERVAL, train_start, train_end)
        if df is not None and len(df) >= 150:  # Ensure sufficient data
            score = compute_stage1A_score(df)
            if score is not None and score > 0:
                rows.append((sym, score))

    dfA = pd.DataFrame(rows, columns=["symbol", "score_1A"])
    dfA = dfA.sort_values("score_1A", ascending=False)

    out_path = os.path.join(RESULTS_DIR, "stage1A_snapshot.csv")
    dfA.to_csv(out_path, index=False)

    logger.info(f"[1A] Saved: {out_path}")
    logger.info(f"Top Stage1A Symbols: {len(dfA)}")  # Report the actual number of valid symbols

    return dfA

# ===============================
# Stage 1B — Micro Behavior
# ===============================
def compute_stage1B_behavior(df):
    """Score micro behavior: liquidity + micro volatility."""
    if df is None or len(df) < 150:  # Ensure sufficient data
        return None

    vol = df["close"].pct_change().std()
    liq = df["volume"].mean()

    # Apply stricter criteria for micro behavior
    if vol < 0.01 or liq < 1000:  # Example thresholds, adjust as needed
        return 0.0

    return float(vol * liq)

def stage1B(client, df1A_top, train_start, train_end):
    logger.info("========== STAGE 1B — MICRO BEHAVIOR ==========")

    micro_start = train_end - timedelta(days=7)
    micro_end = train_end

    logger.info(f"Window: {micro_start} to {micro_end}")
    logger.info(f"Subset size: {len(df1A_top)}")

    rows = []
    for sym in df1A_top["symbol"].tolist():
        df = fetch_klines(client, sym, STAGE1B_INTERVAL, micro_start, micro_end)
        if df is not None and len(df) >= 150:  # Ensure sufficient data
            score = compute_stage1B_behavior(df)
            if score is not None and score > 0:
                rows.append((sym, score))

    dfB = pd.DataFrame(rows, columns=["symbol", "beh_score"])
    dfB = dfB.sort_values("beh_score", ascending=False)

    out_path = os.path.join(RESULTS_DIR, "stage1B_behavior.csv")
    dfB.to_csv(out_path, index=False)

    logger.info(f"[1B] Saved: {out_path}")
    logger.info(f"Top Stage1B Symbols: {len(dfB)}")

    return dfB

# ===============================
# MAIN (manual run or AutoRun)
# ===============================
def main():
    global TRAIN_START, TRAIN_END

    logger.info("=============================================")
    logger.info(" Running Stage1A + Stage1B CLEAN Version ")
    logger.info("=============================================")

    # Load environment variables
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        logger.error("API key or secret not found in environment variables.")
        raise ValueError("API key or secret not found in environment variables.")

    # Load train window
    w = get_windows_from_manual_monday()
    train_start = pd.to_datetime(w["train_start"], utc=True)
    train_end = pd.to_datetime(w["train_end"], utc=True)

    TRAIN_START = train_start
    TRAIN_END = train_end

    logger.info(f"[1A] TRAIN_START = {TRAIN_START}")
    logger.info(f"[1A] TRAIN_END   = {TRAIN_END}")

    # Initialize Binance client
    client = Client(api_key=api_key, api_secret=api_secret)

    # Run Stage 1A
    df1A = stage1A(client, TRAIN_START, TRAIN_END)
    logger.info(f"Top Stage1A Symbols: {len(df1A)}")

    # Run Stage 1B
    df1B = stage1B(client, df1A, TRAIN_START, TRAIN_END)
    logger.info(f"Top Stage1B Symbols: {len(df1B)}")

    logger.info("========== COMPLETE ==========")

if __name__ == "__main__":
    main()
