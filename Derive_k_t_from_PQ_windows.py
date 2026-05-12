#!/usr/bin/env python3
"""
Derive k, t, and x_bars exit parameters from event study data — one-pair-at-a-time.

This script:
1. Loads top candidates from eventstudy_list_summary_{PAIR}_prepaper_{DATE}.csv
2. Filters events by candidate rules (close, vol, vol_rule)
3. Calculates exit parameters: k, t, and barrier delay x_bars
4. Outputs results in structured JSON format

Key concept: x_bars is a GATE/ACTIVATION DELAY for k and t stops.
Stops (k, t) only activate after x_bars bars have elapsed from entry.

USAGE (Step 3 in the 3-step pipeline):
  python Derive_k_t_from_PQ_windows.py --pair ACTUSDT --prepaper-start 2025-12-01

  When --pair and --prepaper-start are supplied, the script auto-locates:
    candidates CSV : forwardtest/eventstudy_list_summary_{PAIR}_prepaper_{DATE}.csv
    events CSV     : forwardtest/v30_eventstudy_{PAIR}_1m_*_prepaper_{DATE}.csv

  Output:
    candidate_exit_params_{PAIR}_prepaper_{DATE}.json

  Then copy/rename to candidate_for_TRADE.json and run:
    python 7_day_trade_window_forward_livefetch_v6+PrePaper.py
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np


# Legacy possibility mappings (from eventstudy_metrics.py)
# Note: Some possibilities have close=False and vol=False but still filter by vol_rule
# (e.g., B2) - these represent events where both gates failed but vol_ratio is in a specific range
LEGACY_POSSIBILITIES = {
    'G0': {'close': 'ALL', 'vol': 'ALL', 'vol_rule': 'ALL'},
    'A0': {'close': True, 'vol': True, 'vol_rule': 'ALL'},
    'B0': {'close': False, 'vol': False, 'vol_rule': 'ALL'},
    'C0': {'close': False, 'vol': True, 'vol_rule': 'ALL'},
    'D0': {'close': True, 'vol': False, 'vol_rule': 'ALL'},
    'E0': {'close': 'ALL', 'vol': True, 'vol_rule': 'ALL'},
    'F0': {'close': 'ALL', 'vol': False, 'vol_rule': 'ALL'},
    'H0': {'close': True, 'vol': 'ALL', 'vol_rule': 'ALL'},
    'I0': {'close': False, 'vol': 'ALL', 'vol_rule': 'ALL'},
    'A1': {'close': True, 'vol': True, 'vol_rule': '>=1.5'},
    'A2': {'close': True, 'vol': True, 'vol_rule': '>=3'},
    'A3': {'close': True, 'vol': True, 'vol_rule': '3_4'},
    'A4': {'close': True, 'vol': True, 'vol_rule': '5_10'},
    'B1': {'close': False, 'vol': True, 'vol_rule': '>=1.5'},
    'B2': {'close': False, 'vol': False, 'vol_rule': '1.5_2'},
    'C1': {'close': False, 'vol': True, 'vol_rule': '>=2'},
    'C2': {'close': False, 'vol': True, 'vol_rule': '2_3'},
    'D1': {'close': True, 'vol': False, 'vol_rule': '<1.5'},
    'E1': {'close': 'ALL', 'vol': True, 'vol_rule': '>=4'},
    'E2': {'close': 'ALL', 'vol': True, 'vol_rule': '4_5'},
}


def parse_vol_rule(vol_rule_str: str) -> Dict[str, Any]:
    """
    Parse vol_rule string into a structured filter.
    
    Supported formats:
    - 'ALL': No filtering
    - '>=X': vol_ratio >= X
    - '<X': vol_ratio < X
    - 'X_Y': X <= vol_ratio < Y (bin)
    
    Args:
        vol_rule_str: Vol rule string from CSV
        
    Returns:
        Dictionary with 'type' and filter parameters
    """
    vol_rule_str = str(vol_rule_str).strip()
    
    if vol_rule_str == 'ALL':
        return {'type': 'ALL'}
    
    # Greater than or equal: >=X
    if vol_rule_str.startswith('>='):
        try:
            threshold = float(vol_rule_str[2:])
            return {'type': 'gte', 'threshold': threshold}
        except ValueError:
            raise ValueError(f"Invalid vol_rule format: {vol_rule_str}")
    
    # Less than: <X
    if vol_rule_str.startswith('<'):
        try:
            threshold = float(vol_rule_str[1:])
            return {'type': 'lt', 'threshold': threshold}
        except ValueError:
            raise ValueError(f"Invalid vol_rule format: {vol_rule_str}")
    
    # Bin: X_Y
    if '_' in vol_rule_str:
        parts = vol_rule_str.split('_')
        if len(parts) == 2:
            try:
                low = float(parts[0])
                high = float(parts[1])
                return {'type': 'bin', 'low': low, 'high': high}
            except ValueError:
                raise ValueError(f"Invalid vol_rule bin format: {vol_rule_str}")
    
    raise ValueError(f"Unsupported vol_rule format: {vol_rule_str}")


def apply_vol_rule_filter(events: pd.DataFrame, vol_rule: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply vol_rule filter to events DataFrame.
    
    Args:
        events: DataFrame with vol_ratio column
        vol_rule: Parsed vol_rule dictionary
        
    Returns:
        Filtered DataFrame
    """
    if vol_rule['type'] == 'ALL':
        return events
    
    elif vol_rule['type'] == 'gte':
        return events[events['vol_ratio'] >= vol_rule['threshold']].copy()
    
    elif vol_rule['type'] == 'lt':
        return events[events['vol_ratio'] < vol_rule['threshold']].copy()
    
    elif vol_rule['type'] == 'bin':
        return events[
            (events['vol_ratio'] >= vol_rule['low']) &
            (events['vol_ratio'] < vol_rule['high'])
        ].copy()
    
    return events

# Helpers for Winsorie
def _winsorize_series(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    """
    Clip a series to [q_low, q_high] quantiles (winsorization).
    Returns a float series with NaNs preserved.
    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return s
    lo = float(s.quantile(lower_q))
    hi = float(s.quantile(upper_q))
    if hi < lo:
        lo, hi = hi, lo
    return s.clip(lower=lo, upper=hi)


def _clip_value(x: Optional[float], cap: Optional[float]) -> Optional[float]:
    if x is None or pd.isna(x) or cap is None:
        return x
    return float(min(float(x), float(cap)))
# --------------------------------------------

def load_candidates_from_csv(
    csv_path: str,
    top_n: Optional[int] = None,
    min_rank: Optional[float] = None
) -> pd.DataFrame:
    """
    Load candidates from eventstudy_list_summary.csv.
    
    Args:
        csv_path: Path to eventstudy_list_summary.csv
        top_n: Load only top N candidates by rank
        min_rank: Load only candidates with rank <= min_rank
        
    Returns:
        DataFrame with candidate information
    """
    df = pd.read_csv(csv_path)
    
    # Filter to eligible candidates (those with a Rank value)
    eligible = df[df['Rank'].notna()].copy()
    
    if len(eligible) == 0:
        print("Warning: No eligible candidates found in CSV", file=sys.stderr)
        return eligible
    
    # Sort by rank (ascending - lower is better)
    eligible = eligible.sort_values('Rank')
    
    # Apply filters
    if min_rank is not None:
        eligible = eligible[eligible['Rank'] <= min_rank]
    
    if top_n is not None:
        eligible = eligible.head(top_n)
    
    return eligible


def parse_candidate_possibility(possibility: str) -> Dict[str, Any]:
    """
    Parse possibility ID to extract close, vol, vol_rule.
    
    Handles both legacy format (e.g., 'A1', 'C0') and grid format
    (e.g., 'C_TRUE__V_TRUE__R_GE_1.5').
    
    Args:
        possibility: Possibility ID string
        
    Returns:
        Dictionary with 'close', 'vol', 'vol_rule' keys
    """
    # Grid format: C_<close>__V_<vol>__R_<vol_rule>
    if possibility.startswith('C_') and '__V_' in possibility and '__R_' in possibility:
        parts = possibility.split('__')
        
        # Parse close
        close_str = parts[0].replace('C_', '')
        if close_str == 'ALL':
            close = 'ALL'
        elif close_str == 'TRUE':
            close = True
        elif close_str == 'FALSE':
            close = False
        else:
            raise ValueError(f"Invalid close value in possibility: {possibility}")
        
        # Parse vol
        vol_str = parts[1].replace('V_', '')
        if vol_str == 'ALL':
            vol = 'ALL'
        elif vol_str == 'TRUE':
            vol = True
        elif vol_str == 'FALSE':
            vol = False
        else:
            raise ValueError(f"Invalid vol value in possibility: {possibility}")
        
        # Parse vol_rule
        vol_rule_str = parts[2].replace('R_', '')
        # Convert from grid format to standard format
        if vol_rule_str == 'ALL':
            vol_rule = 'ALL'
        elif vol_rule_str.startswith('GE_'):
            vol_rule = '>=' + vol_rule_str[3:].replace('_', '.')
        elif vol_rule_str.startswith('LT_'):
            vol_rule = '<' + vol_rule_str[3:].replace('_', '.')
        else:
            # Bin format: already in correct format (X_Y)
            vol_rule = vol_rule_str
        
        return {'close': close, 'vol': vol, 'vol_rule': vol_rule}
    
    # Legacy format - look up in known mappings
    if possibility in LEGACY_POSSIBILITIES:
        return LEGACY_POSSIBILITIES[possibility]
    
    # Unknown format
    return None

def load_event_data(csv_path: str, pnl_column: str) -> pd.DataFrame:
    """
    Load event study data from CSV.
    
    Args:
        csv_path: Path to event study CSV file
        
    Returns:
        DataFrame with event data
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop extra unnamed columns that come from CSV export artifacts
    df = df.loc[:, ~df.columns.str.startswith("unnamed:")].copy()

    # Ensure pnl column is numeric (prevents pandas treating it as string/object)
    col = pnl_column.lower()
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[col])
    else:
        raise KeyError(
            f"PnL column '{pnl_column}' not found in events CSV columns: {list(df.columns)}"
        )

    # Convert timestamps (these should NOT be under else)
    if 'event_time' in df.columns:
        df['event_time'] = pd.to_datetime(df['event_time'], utc=True, errors='coerce')
    if 'max_high_time' in df.columns:
        df['max_high_time'] = pd.to_datetime(df['max_high_time'], utc=True, errors='coerce')
    if 'stop_time' in df.columns:
        df['stop_time'] = pd.to_datetime(df['stop_time'], utc=True, errors='coerce')
    
    def _to_bool_series(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s
        x = s.astype(str).str.strip().str.lower()
        true_set = {"true", "1", "yes", "y", "t"}
        false_set = {"false", "0", "no", "n", "f", ""}
        out = pd.Series(pd.NA, index=s.index, dtype="boolean")
        out[x.isin(true_set)] = True
        out[x.isin(false_set)] = False
        return out.fillna(False)
    
    # Convert booleans
    for col in ["close_gt_smma_200", "vol_gt_vol_sma"]:
        if col in df.columns:
            df[col] = _to_bool_series(df[col])
    
    # Convert numerics
    numeric_cols = ['vol_ratio', 'time_to_stop_min', 'time_to_max_high_min', 
                    'atr_multiple_to_min', 'atr_multiple_to_max']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def filter_events_by_candidate(
    events: pd.DataFrame,
    candidate: Dict[str, Any]
) -> pd.DataFrame:
    """
    Filter events DataFrame by candidate rules.
    
    Args:
        events: Full events DataFrame
        candidate: Dictionary with 'close', 'vol', 'vol_rule' keys
        
    Returns:
        Filtered events DataFrame
    """
    filtered = events.copy()
    
    # Filter by close gate
    close = candidate['close']
    if close != 'ALL':
        filtered = filtered[filtered['close_gt_smma_200'] == close]
    
    # Filter by vol gate
    vol = candidate['vol']
    if vol != 'ALL':
        filtered = filtered[filtered['vol_gt_vol_sma'] == vol]
    
    # Filter by vol_rule
    vol_rule_str = candidate['vol_rule']
    vol_rule = parse_vol_rule(vol_rule_str)
    filtered = apply_vol_rule_filter(filtered, vol_rule)
    
    return filtered


def apply_policy_c_selection(
    candidates: List[Dict[str, Any]],
    x_min_tail: int,
    policyc_margin: float,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Apply Policy C logic to select the best x candidate.
    
    Policy C:
    - If fixed60.n_tail >= x_min_tail:
      - Let best_quantile be the eligible quantile candidate with max pnl_avg_tail
      - Choose best_quantile only if: best_quantile.pnl_avg_tail >= fixed60.pnl_avg_tail * (1 + policyc_margin)
      - Else choose fixed60
    - If fixed60.n_tail < x_min_tail:
      - Choose the eligible candidate with max pnl_avg_tail
      - If none eligible: fallback to fixed60 but emit WARNING
    
    Args:
        candidates: List of x candidate dictionaries
        x_min_tail: Minimum n_tail for eligibility
        policyc_margin: Margin for quantile vs fixed60 comparison (e.g., 0.10 for 10%)
        debug: Enable debug output
        
    Returns:
        Dictionary with selected candidate and decision info
    """
    # Find fixed60 candidate
    fixed60 = None
    for c in candidates:
        if c['source'] == 'fixed' and c['x_bars'] == 60:
            fixed60 = c
            break
    
    if fixed60 is None:
        # Fallback to old behavior if fixed60 not found
        if debug:
            print("  WARNING: fixed60 candidate not found, using simple max pnl_avg_tail selection")
        valid_candidates = [c for c in candidates if c['pnl_avg_tail'] is not None]
        if len(valid_candidates) == 0:
            return {
                'best_x_bars': None,
                'best_x_minutes': None,
                'best_source': None,
                'best_quantile': None,
                'best_n_tail': 0,
                'best_pnl_avg_tail': None,
                'best_pnl_sum_tail': None,
                'policy_decision': 'no_valid_candidates',
                'candidates': candidates
            }
        best = max(valid_candidates, key=lambda c: c['pnl_avg_tail'])
        return {
            'best_x_bars': best['x_bars'],
            'best_x_minutes': best['x_minutes'],
            'best_source': best['source'],
            'best_quantile': best.get('quantile'),
            'best_n_tail': best['n_tail'],
            'best_pnl_avg_tail': best['pnl_avg_tail'],
            'best_pnl_sum_tail': best['pnl_sum_tail'],
            'policy_decision': 'fixed60_not_found_fallback',
            'candidates': candidates
        }
    
    # Get eligible candidates (n_tail >= x_min_tail)
    eligible_candidates = [c for c in candidates if c['n_tail'] >= x_min_tail and c['pnl_avg_tail'] is not None]
    
    # Get eligible quantile candidates
    eligible_quantile_candidates = [c for c in eligible_candidates if c['source'] == 'quantile']
    
    if debug:
        print(f"  Policy C: fixed60 n_tail={fixed60['n_tail']}, x_min_tail={x_min_tail}")
        print(f"  Eligible candidates: {len(eligible_candidates)}, Eligible quantile candidates: {len(eligible_quantile_candidates)}")
    
    # Apply Policy C logic
    if fixed60['n_tail'] >= x_min_tail:
        # fixed60 is eligible
        if len(eligible_quantile_candidates) > 0:
            # Find best quantile candidate
            best_quantile = max(eligible_quantile_candidates, key=lambda c: c['pnl_avg_tail'])
            
            # Compare with threshold
            threshold = fixed60['pnl_avg_tail'] * (1 + policyc_margin)
            
            if debug:
                print(f"  Best quantile: pnl_avg_tail={best_quantile['pnl_avg_tail']:.2f}, threshold={threshold:.2f}")
            
            if best_quantile['pnl_avg_tail'] >= threshold:
                # Choose best_quantile
                return {
                    'best_x_bars': best_quantile['x_bars'],
                    'best_x_minutes': best_quantile['x_minutes'],
                    'best_source': best_quantile['source'],
                    'best_quantile': best_quantile.get('quantile'),
                    'best_n_tail': best_quantile['n_tail'],
                    'best_pnl_avg_tail': best_quantile['pnl_avg_tail'],
                    'best_pnl_sum_tail': best_quantile['pnl_sum_tail'],
                    'policy_decision': f'quantile_beats_fixed60_by_margin',
                    'fixed60_pnl_avg_tail': fixed60['pnl_avg_tail'],
                    'threshold': threshold,
                    'candidates': candidates
                }
            else:
                # Choose fixed60
                return {
                    'best_x_bars': fixed60['x_bars'],
                    'best_x_minutes': fixed60['x_minutes'],
                    'best_source': fixed60['source'],
                    'best_quantile': fixed60.get('quantile'),
                    'best_n_tail': fixed60['n_tail'],
                    'best_pnl_avg_tail': fixed60['pnl_avg_tail'],
                    'best_pnl_sum_tail': fixed60['pnl_sum_tail'],
                    'policy_decision': f'fixed60_within_margin',
                    'best_quantile_pnl_avg_tail': best_quantile['pnl_avg_tail'],
                    'threshold': threshold,
                    'candidates': candidates
                }
        else:
            # No eligible quantile candidates, choose fixed60
            return {
                'best_x_bars': fixed60['x_bars'],
                'best_x_minutes': fixed60['x_minutes'],
                'best_source': fixed60['source'],
                'best_quantile': fixed60.get('quantile'),
                'best_n_tail': fixed60['n_tail'],
                'best_pnl_avg_tail': fixed60['pnl_avg_tail'],
                'best_pnl_sum_tail': fixed60['pnl_sum_tail'],
                'policy_decision': 'fixed60_no_eligible_quantile',
                'candidates': candidates
            }
    else:
        # fixed60 not eligible (n_tail < x_min_tail)
        if len(eligible_candidates) > 0:
            # Choose best eligible candidate
            best = max(eligible_candidates, key=lambda c: c['pnl_avg_tail'])
            return {
                'best_x_bars': best['x_bars'],
                'best_x_minutes': best['x_minutes'],
                'best_source': best['source'],
                'best_quantile': best.get('quantile'),
                'best_n_tail': best['n_tail'],
                'best_pnl_avg_tail': best['pnl_avg_tail'],
                'best_pnl_sum_tail': best['pnl_sum_tail'],
                'policy_decision': 'fixed60_ineligible_best_eligible_chosen',
                'candidates': candidates
            }
        else:
            # No eligible candidates, fallback to fixed60 with WARNING
            print(f"  WARNING: No eligible x candidates (n_tail >= {x_min_tail}), falling back to fixed60 with n_tail={fixed60['n_tail']}")
            return {
                'best_x_bars': fixed60['x_bars'],
                'best_x_minutes': fixed60['x_minutes'],
                'best_source': fixed60['source'],
                'best_quantile': fixed60.get('quantile'),
                'best_n_tail': fixed60['n_tail'],
                'best_pnl_avg_tail': fixed60['pnl_avg_tail'],
                'best_pnl_sum_tail': fixed60['pnl_sum_tail'],
                'policy_decision': 'WARNING_fixed60_fallback_no_eligible',
                'candidates': candidates
            }


def evaluate_x_candidates(
    events: pd.DataFrame,
    x_quantiles: List[float],
    x_fixed: List[int],
    timeframe_minutes: int,
    pnl_column: str,
    x_min_tail: int = 50,
    policyc_margin: float = 0.10,
    debug: bool = False,
    tail_direction: str = ">="
) -> Dict[str, Any]:
    """
    Evaluate multiple x_bars candidates and select the best one using Policy C.
    
    For each candidate x, compute tail metrics over events where time_to_stop_min >= x.
    Select x using Policy C logic with x_min_tail eligibility and policyc_margin.
    
    Args:
        events: Filtered events DataFrame
        x_quantiles: List of quantiles to evaluate (e.g., [0.95, 0.90, 0.85, 0.80, 0.75])
        x_fixed: List of fixed x values to evaluate (e.g., [60])
        timeframe_minutes: Timeframe in minutes for converting time to bars
        pnl_column: Column name for PnL (default: "net_pnl_usdt")
        x_min_tail: Minimum n_tail for x-candidate eligibility (default: 50)
        policyc_margin: Policy C margin for comparison (default: 0.10)
        debug: Enable debug output (default: False)
        tail_direction: Direction for tail cohort (default: ">=")
        
    Returns:
        Dictionary with best_x, best_quantile, diagnostics, and evaluation table
    """
    # Check if required columns exist
    if 'time_to_stop_min' not in events.columns:
        raise ValueError("Column 'time_to_stop_min' not found in events DataFrame")
    if pnl_column not in events.columns:
        raise ValueError(f"Column '{pnl_column}' not found in events DataFrame")
    
    time_to_stop_values = events['time_to_stop_min'].dropna()
    
    candidates = []
    
    # Evaluate quantile-based x candidates
    for quantile in x_quantiles:
        if len(time_to_stop_values) > 0:
            # Step 1: Get raw quantile value in minutes (may be fractional)
            raw_quantile_minutes = time_to_stop_values.quantile(quantile)
            
            # Step 2: Convert to bars (ceiling to ensure we round up)
            x_bars = int(np.ceil(raw_quantile_minutes / timeframe_minutes))
            
            # Step 3: Convert back to bar-aligned minutes for filtering
            # This matches Excel logic where we use the bar boundary, not fractional quantile
            x_minutes = x_bars * timeframe_minutes
            
            # Self-check: Verify x_minutes is properly bar-aligned
            assert x_minutes == x_bars * timeframe_minutes, \
                f"x_minutes ({x_minutes}) != x_bars ({x_bars}) * timeframe_minutes ({timeframe_minutes})"
            
            # Filter tail cohort using bar-aligned threshold
            tail = events[events['time_to_stop_min'] >= x_minutes].copy()
            
            n_tail = len(tail)
            if n_tail > 0:
                pnl_avg_tail = tail[pnl_column].mean()
                pnl_sum_tail = tail[pnl_column].sum()
            else:
                pnl_avg_tail = None
                pnl_sum_tail = None
            
            candidates.append({
                'source': 'quantile',
                'quantile': quantile,
                'raw_quantile_minutes': float(raw_quantile_minutes),  # Store for debugging
                'x_minutes': float(x_minutes),  # Bar-aligned threshold used for filtering
                'x_bars': x_bars,
                'n_tail': n_tail,
                'pnl_avg_tail': float(pnl_avg_tail) if pnl_avg_tail is not None else None,
                'pnl_sum_tail': float(pnl_sum_tail) if pnl_sum_tail is not None else None
            })
    
    # Evaluate fixed x candidates
    for x_bars_fixed in x_fixed:
        x_minutes = x_bars_fixed * timeframe_minutes
        
        # Filter tail cohort
        tail = events[events['time_to_stop_min'] >= x_minutes].copy()
        
        n_tail = len(tail)
        if n_tail > 0:
            pnl_avg_tail = tail[pnl_column].mean()
            pnl_sum_tail = tail[pnl_column].sum()
        else:
            pnl_avg_tail = None
            pnl_sum_tail = None
        
        candidates.append({
            'source': 'fixed',
            'quantile': None,
            'x_minutes': float(x_minutes),
            'x_bars': x_bars_fixed,
            'n_tail': n_tail,
            'pnl_avg_tail': float(pnl_avg_tail) if pnl_avg_tail is not None else None,
            'pnl_sum_tail': float(pnl_sum_tail) if pnl_sum_tail is not None else None
        })
    
    # Apply Policy C logic to select best x candidate
    return apply_policy_c_selection(candidates, x_min_tail, policyc_margin, debug)

def calculate_exit_params(
    events: pd.DataFrame,
    kt_quantile: float = 0.95,
    x_quantiles: Optional[List[float]] = None,
    x_fixed: Optional[List[int]] = None,
    timeframe_minutes: int = 1,
    pnl_column: str = "net_pnl_usdt",
    x_min_tail: int = 50,
    policyc_margin: float = 0.10,
    debug: bool = False,
    enable_x_selection: bool = True,
    x_bars_min_delay: int = 60,
    kt_winsorize: bool = False,
    kt_winsorize_lower_q: float = 0.01,
    kt_winsorize_upper_q: float = 0.99,
    k_cap: Optional[float] = None,
    t_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate exit parameters from filtered events.
    
    Args:
        events: Filtered events DataFrame
        kt_quantile: Quantile for k and t calculation (default: 0.95)
        x_quantiles: List of quantiles for x_bars selection (default: [0.95])
        x_fixed: List of fixed x values to evaluate (default: None)
        timeframe_minutes: Timeframe in minutes for x_bars conversion
        pnl_column: Column name for PnL (default: "net_pnl_usdt")
        x_min_tail: Minimum n_tail for x-candidate eligibility (default: 50)
        policyc_margin: Policy C margin (default: 0.10)
        debug: Enable debug output (default: False)
        enable_x_selection: Enable automated x_bars selection (default: True)
        x_bars_min_delay: Minimum x_bars to treat as active trailing; values below
            this threshold are clamped to 0 (OFF). Default: 60.
        
    Returns:
        Dictionary with 'k', 't', 'x_bars', 'x_selection_diagnostics', 'events_count', 'events_used'
    """
    if x_quantiles is None:
        x_quantiles = [0.95]
    if x_fixed is None:
        x_fixed = []
    
    if len(events) == 0:
        return {
            'k': None,
            't': None,
            'x_bars': None,
            'x_selection_diagnostics': None,
            'events_count': 0,
            'events_used': 0
        }
    
    # Calculate k and t from atr_multiple columns using kt_quantile
    # k = atr_multiple_to_min (stop loss distance in ATR multiples)
    # t = atr_multiple_to_max (profit target distance in ATR multiples)
    
    k_raw = events["atr_multiple_to_min"]
    t_raw = events["atr_multiple_to_max"]

    if kt_winsorize:
        k_values = _winsorize_series(k_raw, kt_winsorize_lower_q, kt_winsorize_upper_q)
        t_values = _winsorize_series(t_raw, kt_winsorize_lower_q, kt_winsorize_upper_q)
    else:
        k_values = pd.to_numeric(k_raw, errors="coerce").dropna()
        t_values = pd.to_numeric(t_raw, errors="coerce").dropna()

    k = k_values.quantile(kt_quantile) if len(k_values) > 0 else None
    t = t_values.quantile(kt_quantile) if len(t_values) > 0 else None

    # Optional caps (opt-in)
    k = _clip_value(k, k_cap)
    t = _clip_value(t, t_cap)
    # ---------------------------------------------------------------------
    
    # Calculate x_bars using automated selection or simple quantile
    x_selection_diagnostics = None
    x_bars = None
    
    if enable_x_selection and len(x_quantiles) > 0:
        # Use automated x_bars selection with Policy C
        try:
            x_selection_result = evaluate_x_candidates(
                events, x_quantiles, x_fixed, timeframe_minutes, pnl_column,
                x_min_tail, policyc_margin, debug
            )
            x_bars = x_selection_result['best_x_bars']
            x_selection_diagnostics = x_selection_result
        except ValueError as e:
            # Fall back to simple quantile if columns are missing
            print(f"Warning: x_bars selection failed ({e}), falling back to kt_quantile={kt_quantile}", file=sys.stderr)
            time_to_stop_values = events['time_to_stop_min'].dropna()
            if len(time_to_stop_values) > 0:
                time_to_stop_minutes = time_to_stop_values.quantile(kt_quantile)
                x_bars = int(np.ceil(time_to_stop_minutes / timeframe_minutes))
    else:
        # Use simple quantile calculation (backward compatibility)
        time_to_stop_values = events['time_to_stop_min'].dropna()
        if len(time_to_stop_values) > 0:
            time_to_stop_minutes = time_to_stop_values.quantile(kt_quantile)
            x_bars = int(np.ceil(time_to_stop_minutes / timeframe_minutes))
    
    # x_bars below x_bars_min_delay bars treated as OFF (immediate trailing, effective_x=0).
    if x_bars is not None and not pd.isna(x_bars) and int(x_bars) < x_bars_min_delay:
        x_bars = 0

    # Diagnostic: estimate whether trailing distance is mechanically too wide
    price_proxy = None
    atr_proxy = None
    trail_dist_ratio_estimate = None
    t_mechanical_flag = None

    for price_col in ["entry_close", "close", "event_close"]:
        if price_col in events.columns:
            price_series = pd.to_numeric(events[price_col], errors="coerce").dropna()
            if len(price_series) > 0:
                price_proxy = float(price_series.median())
                break

    for atr_col in ["atr", "atr_entry", "entry_atr"]:
        if atr_col in events.columns:
            atr_series = pd.to_numeric(events[atr_col], errors="coerce").dropna()
            if len(atr_series) > 0:
                atr_proxy = float(atr_series.median())
                break

    if (
        t is not None and not pd.isna(t)
        and price_proxy is not None and price_proxy > 0
        and atr_proxy is not None and atr_proxy > 0
    ):
        trail_dist_ratio_estimate = float((float(t) * atr_proxy) / price_proxy)

        # NOTE:
        # trail_dist_ratio_estimate is diagnostic only.
        # Do NOT hard-gate candidates from this proxy estimate.
        # A hard gate should only be considered later using exact event/live entry conditions.
        if trail_dist_ratio_estimate < 0.25:
            t_mechanical_flag = "normal"
        elif trail_dist_ratio_estimate < 0.50:
            t_mechanical_flag = "wide"
        elif trail_dist_ratio_estimate < 1.00:
            t_mechanical_flag = "very_wide"
        else:
            t_mechanical_flag = "invisible_risk"
            
    events_used_k = int(len(k_values))
    events_used_t = int(len(t_values))
    events_used = int(min(events_used_k, events_used_t))

    return {
        'k': float(k) if k is not None and not pd.isna(k) else None,
        't': float(t) if t is not None and not pd.isna(t) else None,
        'x_bars': int(x_bars) if x_bars is not None and not pd.isna(x_bars) else None,
        'x_selection_diagnostics': x_selection_diagnostics,
        'events_count': len(events),
        'events_used': events_used,
        'events_used_k': events_used_k,
        'events_used_t': events_used_t,
        'price_proxy': price_proxy,
        'atr_proxy': atr_proxy,
        'trail_dist_ratio_estimate': trail_dist_ratio_estimate,
        't_mechanical_flag': t_mechanical_flag,
    }


def classify_vol_rule(vol_rule_str: str) -> str:
    """
    Classify vol_rule into categories: 'low', 'elevated', or 'ALL'.
    
    Low: <X rules (e.g., <1.5, <5)
    Elevated: >=X rules or bins (e.g., >=1.5, 3_4)
    ALL: No filtering
    
    Args:
        vol_rule_str: Vol rule string
        
    Returns:
        Category: 'low', 'elevated', or 'ALL'
    """
    vol_rule_str = str(vol_rule_str).strip()
    
    if vol_rule_str == 'ALL':
        return 'ALL'
    
    if vol_rule_str.startswith('<'):
        return 'low'
    
    if vol_rule_str.startswith('>='):
        return 'elevated'
    
    if '_' in vol_rule_str:
        # Bin format - consider as elevated
        return 'elevated'
    
    return 'ALL'


def get_candidate_category(candidate: Dict[str, Any]) -> Tuple[Tuple[str, str, str], str]:
    """
    Get category tuple and category_id for a candidate.
    
    Returns (close, vol, vol_rule_class) tuple and category_id string
    for category classification and manual Excel verification.
    
    Args:
        candidate: Dictionary with 'close', 'vol', 'vol_rule' keys
        
    Returns:
        Tuple of (category_tuple, category_id) where:
        - category_tuple: (close_str, vol_str, vol_rule_class)
        - category_id: formatted string like "CAT_C_True__V_ALL__RCLASS_low"
    """
    close = candidate['close']
    vol = candidate['vol']
    vol_rule = candidate['vol_rule']
    
    close_str = str(close) if close != 'ALL' else 'ALL'
    vol_str = str(vol) if vol != 'ALL' else 'ALL'
    vol_rule_class = classify_vol_rule(vol_rule)
    
    category_tuple = (close_str, vol_str, vol_rule_class)
    category_id = f"CAT_C_{close_str}__V_{vol_str}__RCLASS_{vol_rule_class}"
    
    return category_tuple, category_id


def phase_a_select_finalists(
    candidates_df: pd.DataFrame,
    min_trades: int,
    finalists_n: int,
    debug: bool = False
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Phase A: Apply eligibility filter and select category winners.
    
    Steps:
    1. Apply eligibility gate: keep only candidates with trades > min_trades
    2. Group by category key (close, vol, vol_rule)
    3. Select highest score per category → category winners
    4. Reduce to finalists_n by sorting by score desc if needed
    
    Args:
        candidates_df: DataFrame with all candidates
        min_trades: Minimum trades for eligibility
        finalists_n: Maximum number of finalists
        debug: Enable debug output
        
    Returns:
        Tuple of (finalists_df, category_info_list)
    """
    if debug:
        print(f"\nPhase A: Eligibility and Category Winners")
        print(f"  Total candidates loaded: {len(candidates_df)}")
    
    # Apply eligibility gate: trades > min_trades
    eligible_df = candidates_df[candidates_df['Trades'] > min_trades].copy()
    
    if debug:
        print(f"  Eligible candidates (trades > {min_trades}): {len(eligible_df)}")
    
    if len(eligible_df) == 0:
        print(f"WARNING: No eligible candidates with trades > {min_trades}")
        return pd.DataFrame(), []
    
    # Parse candidates to get category keys
    category_map = {}  # category_tuple -> list of candidates
    
    for idx, row in eligible_df.iterrows():
        possibility = row['Possibility']
        
        # Parse possibility to get close, vol, vol_rule
        parsed = parse_candidate_possibility(possibility)
        if parsed is None:
            if 'close' in row and 'vol' in row and 'vol_rule' in row:
                parsed = {
                    'close': row['close'],
                    'vol': row['vol'],
                    'vol_rule': row['vol_rule']
                }
            else:
                if debug:
                    print(f"  WARNING: Cannot parse {possibility}, skipping")
                continue
        
        # Get category key
        close_str = str(parsed['close']) if parsed['close'] != 'ALL' else 'ALL'
        vol_str = str(parsed['vol']) if parsed['vol'] != 'ALL' else 'ALL'
        vol_rule_class = classify_vol_rule(parsed['vol_rule'])
        category_key = (close_str, vol_str, vol_rule_class)
        
        if category_key not in category_map:
            category_map[category_key] = []
        
        category_map[category_key].append({
            'idx': idx,
            'row': row,
            'parsed': parsed,
            'category_key': category_key
        })
    
    # Select best (highest score) from each category
    category_winners = []
    for category_key, candidates in category_map.items():
        # Sort by score descending
        candidates_sorted = sorted(candidates, key=lambda c: c['row']['Score'], reverse=True)
        winner = candidates_sorted[0]
        category_winners.append(winner)
        
        if debug:
            print(f"  Category {category_key}: {len(candidates)} candidates, winner: {winner['row']['Possibility']} (score={winner['row']['Score']:.2f})")
    
    if debug:
        print(f"  Category winners: {len(category_winners)}")
    
    # Sort category winners by score descending
    category_winners_sorted = sorted(category_winners, key=lambda c: c['row']['Score'], reverse=True)
    
    # Reduce to finalists_n if needed
    if len(category_winners_sorted) > finalists_n:
        finalists = category_winners_sorted[:finalists_n]
        if debug:
            print(f"  Reducing {len(category_winners_sorted)} category winners to {finalists_n} finalists")
    else:
        finalists = category_winners_sorted
    
    # Build finalist DataFrame
    finalist_indices = [f['idx'] for f in finalists]
    finalists_df = eligible_df.loc[finalist_indices].copy()
    
    # Build category info list
    category_info = []
    for f in finalists:
        category_info.append({
            'possibility': f['row']['Possibility'],
            'category_key': f['category_key'],
            'score': f['row']['Score'],
            'trades': f['row']['Trades']
        })
    
    if debug:
        print(f"  Final finalists: {len(finalists_df)}")
    
    return finalists_df, category_info


def select_finalists_per_category(
    candidates_df: pd.DataFrame,
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Select finalist candidates using category-based selection.
    
    Excludes the no-gate candidate (close=ALL, vol=ALL, vol_rule=ALL).
    Selects top candidate per category based on rank.
    
    Categories:
    - 5 close/vol categories: (ALL, True), (True, ALL), (True, True), (True, False), (False, True)
    - 4 vol_rule categories: (ALL, ALL, low), (ALL, ALL, elevated), (False, ALL, low), (False, False, low)
    
    Args:
        candidates_df: DataFrame with candidate information
        results: List of processed candidate results
        
    Returns:
        Filtered list of finalist candidates
    """
    # Create mapping of possibility to result
    result_map = {r['possibility']: r for r in results}
    
    # Filter out no-gate candidate
    filtered_results = []
    for result in results:
        if (result['close'] == 'ALL' and 
            result['vol'] == 'ALL' and 
            result['vol_rule'] == 'ALL'):
            print(f"Excluding no-gate candidate: {result['possibility']}")
            continue
        filtered_results.append(result)
    
    # Define target categories
    # 5 close/vol categories
    close_vol_categories = [
        ('ALL', 'True', 'ALL'),
        ('True', 'ALL', 'ALL'),
        ('True', 'True', 'ALL'),
        ('True', 'False', 'ALL'),
        ('False', 'True', 'ALL'),
    ]
    
    # 4 vol_rule categories
    vol_rule_categories = [
        ('ALL', 'ALL', 'low'),
        ('ALL', 'ALL', 'elevated'),
        ('False', 'ALL', 'low'),
        ('False', 'False', 'low'),
    ]
    
    all_categories = close_vol_categories + vol_rule_categories
    
    # Group candidates by category
    category_candidates = {}
    for result in filtered_results:
        category_tuple, category_id = get_candidate_category(result)
        if category_tuple not in category_candidates:
            category_candidates[category_tuple] = []
        category_candidates[category_tuple].append(result)
    
    # Select best candidate per category (lowest rank)
    finalists = []
    selected_categories = []
    
    for category_tuple in all_categories:
        if category_tuple in category_candidates:
            candidates = category_candidates[category_tuple]
            # Sort by rank (ascending - lower is better)
            candidates = sorted(candidates, key=lambda c: c['rank'] if c['rank'] is not None else float('inf'))
            if len(candidates) > 0:
                best = candidates[0]
                # Add category information to the result
                _, category_id = get_candidate_category(best)
                best['category_tuple'] = category_tuple
                best['category_id'] = category_id
                finalists.append(best)
                selected_categories.append(category_tuple)
                print(f"Selected finalist for category {category_id}: {best['possibility']} (Rank {best['rank']})")
    
    print(f"\nSelected {len(finalists)} finalists from {len(selected_categories)} categories")
    
    return finalists


def process_candidates(
    candidates_df: pd.DataFrame,
    events_df: pd.DataFrame,
    kt_quantile: float = 0.95,
    x_quantiles: Optional[List[float]] = None,
    x_fixed: Optional[List[int]] = None,
    timeframe_minutes: int = 1,
    pnl_column: str = "net_pnl_usdt",
    x_min_tail: int = 50,
    policyc_margin: float = 0.10,
    debug: bool = False,
    enable_x_selection: bool = True,
    x_bars_min_delay: int = 60,

    # NEW: stabilize k/t (opt-in via CLI)
    kt_winsorize: bool = False,
    kt_winsorize_lower_q: float = 0.01,
    kt_winsorize_upper_q: float = 0.99,
    k_cap: Optional[float] = None,
    t_cap: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Process all candidates and calculate exit parameters.
    
    Args:
        candidates_df: DataFrame with candidate information
        events_df: DataFrame with event study data
        kt_quantile: Quantile for k and t calculation
        x_quantiles: List of quantiles for x_bars selection
        x_fixed: List of fixed x values to evaluate
        timeframe_minutes: Timeframe in minutes for x_bars conversion
        pnl_column: Column name for PnL
        x_min_tail: Minimum n_tail for x-candidate eligibility
        policyc_margin: Policy C margin
        debug: Enable debug output
        enable_x_selection: Enable automated x_bars selection
        x_bars_min_delay: Minimum x_bars to treat as active trailing; values below
            this threshold are clamped to 0 (OFF). Default: 60.
        
    Returns:
        List of dictionaries with candidate results
    """
    if x_quantiles is None:
        x_quantiles = [0.95]
    if x_fixed is None:
        x_fixed = []
    
    results = []
    
    for _, cand_row in candidates_df.iterrows():
        possibility = cand_row['Possibility']
        
        # Try to parse possibility ID
        parsed = parse_candidate_possibility(possibility)
        
        # If parsing failed, try to use columns from CSV
        if parsed is None:
            # Check if CSV has close/vol/vol_rule columns (grid mode)
            if 'close' in cand_row and 'vol' in cand_row and 'vol_rule' in cand_row:
                parsed = {
                    'close': cand_row['close'],
                    'vol': cand_row['vol'],
                    'vol_rule': cand_row['vol_rule']
                }
            else:
                print(f"Warning: Cannot parse possibility {possibility} and no close/vol/vol_rule columns found", 
                      file=sys.stderr)
                continue
        
        # Filter events
        filtered_events = filter_events_by_candidate(events_df, parsed)
        
        # Calculate exit parameters
        exit_params = calculate_exit_params(
            filtered_events,
            kt_quantile=kt_quantile,
            x_quantiles=x_quantiles,
            x_fixed=x_fixed,
            timeframe_minutes=timeframe_minutes,
            pnl_column=pnl_column,
            x_min_tail=x_min_tail,
            policyc_margin=policyc_margin,
            debug=debug,
            enable_x_selection=enable_x_selection,
            x_bars_min_delay=x_bars_min_delay,

            # NEW: stabilize k/t (opt-in via CLI)
            kt_winsorize=kt_winsorize,
            kt_winsorize_lower_q=kt_winsorize_lower_q,
            kt_winsorize_upper_q=kt_winsorize_upper_q,
            k_cap=k_cap,
            t_cap=t_cap,
        )
        
        # Build result
        result = {
            'possibility': possibility,
            'rank': float(cand_row['Rank']) if pd.notna(cand_row['Rank']) else None,
            'trades': int(cand_row['Trades']) if pd.notna(cand_row['Trades']) else 0,
            'total_net_pnl': float(cand_row['Total_Net_PnL']) if pd.notna(cand_row['Total_Net_PnL']) else None,
            'score': float(cand_row['Score']) if pd.notna(cand_row['Score']) else None,
            'close': parsed['close'],
            'vol': parsed['vol'],
            'vol_rule': parsed['vol_rule'],
            'exit_params': exit_params,
            'kt_quantile': kt_quantile,
            'timeframe_minutes': timeframe_minutes
        }
        
        results.append(result)
    
    return results


def _auto_find_csv(pattern: str) -> Optional[str]:
    """
    Glob for files matching *pattern* and return the most recently modified one,
    or None if no match is found.
    """
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=lambda p: Path(p).stat().st_mtime)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Derive k, t, and x_bars exit parameters from event study data'
    )
    parser.add_argument(
        '--pair',
        default=None,
        help='Trading pair symbol (e.g. ACTUSDT). Used to auto-locate input CSVs and name output JSON.'
    )
    parser.add_argument(
        '--prepaper-start',
        default='2025-12-01',
        metavar='YYYY-MM-DD',
        help='PrePaper window start date (default: 2025-12-01). Used to auto-locate input CSVs and name output JSON.'
    )
    parser.add_argument(
        '--interval',
        default='1m',
        choices=['1m', '3m'],
        help=(
            'Kline interval used when generating the events CSV (default: 1m). '
            'Used to build the auto-locate glob pattern and to derive --timeframe-minutes '
            'when that flag is not explicitly provided.'
        ),
    )
    parser.add_argument(
        '--candidates-csv',
        default=None,
        help=(
            'Path to eventstudy_list_summary CSV. '
            'Auto-selected from forwardtest/eventstudy_list_summary_{PAIR}_{INTERVAL}_prepaper_{DATE}.csv '
            'when --pair is supplied and this arg is omitted.'
        )
    )
    parser.add_argument(
        '--events-csv',
        default=None,
        help=(
            'Path to event study CSV file. '
            'Auto-selected from forwardtest/v30_eventstudy_{PAIR}_{INTERVAL}_*_prepaper_{DATE}.csv '
            'when --pair is supplied and this arg is omitted.'
        )
    )
    parser.add_argument(
        '--output',
        default=None,
        help=(
            'Output JSON file path. '
            'Default: candidate_exit_params_{PAIR}_prepaper_{DATE}.json '
            '(falls back to candidate_exit_params.json when pair/date unknown)'
        )
    )
    parser.add_argument(
        '--top-n',
        type=int,
        help='Process only top N candidates by rank'
    )
    parser.add_argument(
        '--min-rank',
        type=float,
        help='Process only candidates with rank <= min_rank'
    )
    parser.add_argument(
        '--kt-quantile',
        type=float,
        default=0.95,
        help='Quantile for k and t calculation (default: 0.95)'
    )
    parser.add_argument(
        '--x-quantiles',
        type=str,
        default='0.95,0.90,0.85,0.80,0.75',
        help='Comma-separated quantiles for x_bars selection (default: 0.95,0.90,0.85,0.80,0.75)'
    )
    parser.add_argument(
        '--x-fixed',
        type=str,
        default='60',
        help='Comma-separated fixed x values in bars to evaluate (default: 60)'
    )
    parser.add_argument(
        '--x-tail-direction',
        type=str,
        default='>=',
        help='Tail cohort direction (default: >=)'
    )
    parser.add_argument(
        '--pnl-column',
        type=str,
        default='net_pnl_usdt',
        help='Column name for PnL (default: net_pnl_usdt)'
    )
    parser.add_argument(
        '--timeframe-minutes',
        type=int,
        default=None,
        help=(
            'Timeframe in minutes for x_bars conversion. '
            'Defaults to the value derived from --interval (1m->1, 3m->3).'
        ),
    )
    parser.add_argument(
        '--finalists-per-category',
        action='store_true',
        default=True,
        help='Enable finalist selection mode (select 1 best per category) (default: True)'
    )
    parser.add_argument(
        '--finalists-n',
        type=int,
        default=3,
        help='Maximum number of finalists to process after category selection (default: 3)'
    )
    parser.add_argument(
        '--min-trades',
        type=int,
        default=50,
        help='Minimum trades for candidate eligibility (default: 50)'
    )
    parser.add_argument(
        '--x-min-tail',
        type=int,
        default=50,
        help='Minimum n_tail for x-candidate eligibility (default: 50)'
    )
    parser.add_argument(
        '--policyc-margin',
        type=float,
        default=0.10,
        help='Policy C margin for quantile vs fixed60 comparison (default: 0.10)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debug output showing all candidates (default: False)'
    )
    parser.add_argument(
        '--disable-x-selection',
        action='store_true',
        help='Disable automated x_bars selection, use simple quantile instead'
    )
    parser.add_argument(
        '--x-bars-min-delay',
        type=int,
        default=60,
        help='Minimum x_bars to treat as active trailing delay; values below this are clamped to 0 (OFF). Default: 60'
    )
    parser.add_argument("--kt-winsorize", action="store_true",
                    help="Winsorize atr_multiple distributions before computing k/t quantiles (default: off).")
    parser.add_argument("--kt-winsorize-lower-q", type=float, default=0.01,
                        help="Lower quantile for winsorization (default: 0.01).")
    parser.add_argument("--kt-winsorize-upper-q", type=float, default=0.99,
                        help="Upper quantile for winsorization (default: 0.99).")

    parser.add_argument("--k-cap", type=float, default=None,
                        help="Optional max cap for k in ATR multiples (default: none).")
    parser.add_argument("--t-cap", type=float, default=None,
                        help="Optional max cap for t in ATR multiples (default: none).")  
    
    args = parser.parse_args()
    args.pnl_column = args.pnl_column.lower()

    pair = args.pair.upper() if args.pair and args.pair.strip() else None
    prepaper_date = args.prepaper_start
    import re

    # 1. Fallback to args if provided, but default to extracting from filename later
    interval = args.interval 
    timeframe_minutes = args.timeframe_minutes

    _INTERVAL_TO_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240}

    # -----------------------------------------------------------------------
    # Auto-locate input CSVs when --pair is given but paths not explicitly set
    # -----------------------------------------------------------------------
    candidates_csv_arg = args.candidates_csv
    events_csv_arg = args.events_csv

    if pair and not candidates_csv_arg:
        pattern = f"forwardtest/eventstudy_list_summary_{pair}_{interval}_prepaper_{prepaper_date}.csv"
        found = _auto_find_csv(pattern)
        if found:
            candidates_csv_arg = found
        else:
            print(
                f"Error: Could not auto-locate candidates CSV matching: {pattern}\n"
                "Run eventstudy_analysis.py first, or supply --candidates-csv explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)

    if pair and not events_csv_arg:
        pattern = f"forwardtest/v30_eventstudy_{pair}_{interval}_*_prepaper_{prepaper_date}.csv"
        found = _auto_find_csv(pattern)
        if found:
            events_csv_arg = found
        else:
            print(
                f"Error: Could not auto-locate events CSV matching: {pattern}\n"
                "Run Funnel_Data_Test_V30_EventStudy.py first, or supply --events-csv explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Fallback to legacy defaults when neither --pair nor explicit paths are given
    if not candidates_csv_arg:
        candidates_csv_arg = 'forwardtest/eventstudy_list_summary.csv'
    if not events_csv_arg:
        events_csv_arg = 'forwardtest/v30_eventstudy_TNSRUSDT_3m_rsi_sma_cross_gt51_prepaper_2025-12-01.csv'

    # Build default output path
    if args.output:
        output_path_str = args.output
    elif pair and prepaper_date:
        output_path_str = f"candidate_exit_params_{pair}_prepaper_{prepaper_date}.json"
    else:
        output_path_str = 'candidate_exit_params.json'

    # Parse x_quantiles
    try:
        x_quantiles = [float(q.strip()) for q in args.x_quantiles.split(',') if q.strip()]
    except ValueError:
        print(f"Error: Invalid x-quantiles format: {args.x_quantiles}", file=sys.stderr)
        sys.exit(1)
    
    # Parse x_fixed
    try:
        x_fixed = [int(x.strip()) for x in args.x_fixed.split(',') if x.strip()]
    except ValueError:
        print(f"Error: Invalid x-fixed format: {args.x_fixed}", file=sys.stderr)
        sys.exit(1)
    
    # Validate x-tail-direction
    if args.x_tail_direction != '>=':
        print(f"Error: Only '>=' is supported for --x-tail-direction. Got: '{args.x_tail_direction}'", file=sys.stderr)
        print("       Other directions are not yet implemented.", file=sys.stderr)
        sys.exit(1)
    
    # Validate input files
    candidates_path = Path(candidates_csv_arg)
    if not candidates_path.exists():
        print(f"Error: Candidates CSV not found: {candidates_path}", file=sys.stderr)
        sys.exit(1)
    
    events_path = Path(events_csv_arg)
    if not events_path.exists():
        print(f"Error: Events CSV not found: {events_path}", file=sys.stderr)
        sys.exit(1)
    
    # --- AUTO-DETECT TIMEFRAME FROM FILENAME ---
    import re
    if timeframe_minutes is None:
        match = re.search(r'_([1-5][mh]?[0-9]*[mh]?)_', str(events_path))
        if match:
            interval = match.group(1)
        timeframe_minutes = _INTERVAL_TO_MINUTES.get(interval, 1)
        print(f"[AUTO-DETECT] Found interval '{interval}' from filename -> timeframe_minutes = {timeframe_minutes}")
    # -------------------------------------------

    print(f"Loading candidates from: {candidates_path}")
    
    # Load candidates
    try:
        candidates_df = load_candidates_from_csv(
            str(candidates_path),
            top_n=args.top_n,
            min_rank=args.min_rank
        )
        print(f"Loaded {len(candidates_df)} candidates")
    except Exception as e:
        print(f"Error loading candidates: {e}", file=sys.stderr)
        sys.exit(1)
    
    if len(candidates_df) == 0:
        print("No candidates to process", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading event data from: {events_path}")

    
    # Load events
    try:
        events_df = load_event_data(str(events_path), args.pnl_column)
        print(f"Loaded {len(events_df)} events")
    except Exception as e:
        print(f"Error loading events: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check for required columns
    if args.pnl_column not in events_df.columns:
        print(f"Warning: PnL column '{args.pnl_column}' not found in events data. X-bars selection may fail.", 
              file=sys.stderr)
    
    enable_x_selection = not args.disable_x_selection
    
    # ===========================================================================
    # PHASE A: Eligibility + Category Winners
    # ===========================================================================
    print(f"\n{'='*80}")
    print("PHASE A: Eligibility and Category Winner Selection")
    print('='*80)
    
    finalists_df, category_info = phase_a_select_finalists(
        candidates_df,
        min_trades=args.min_trades,
        finalists_n=args.finalists_n,
        debug=args.debug
    )
    
    if len(finalists_df) == 0:
        print("No finalists selected. Exiting.")
        sys.exit(1)
    
    # Print Phase A summary
    print(f"\nPhase A Summary:")
    print(f"  Total candidates loaded: {len(candidates_df)}")
    print(f"  Eligible candidates (trades > {args.min_trades}): {len(candidates_df[candidates_df['Trades'] > args.min_trades])}")
    print(f"  Category winners: {len(category_info)}")
    print(f"  Finalists (max {args.finalists_n}): {len(finalists_df)}")
    
    if not args.debug:
        print(f"\n  Finalists:")
        for info in category_info:
            print(f"    {info['possibility']}: trades={info['trades']}, score={info['score']:.2f}, category={info['category_key']}")
    
    # ===========================================================================
    # PHASE B: Compute Exit Params for Finalists Only
    # ===========================================================================
    print(f"\n{'='*80}")
    print("PHASE B: Computing Exit Parameters for Finalists")
    print('='*80)
    
    if not args.debug:
        print(f"  k/t quantile: {args.kt_quantile}")
        print(f"  x quantiles: {x_quantiles}")
        print(f"  x fixed: {x_fixed}")
        print(f"  x_min_tail: {args.x_min_tail}")
        print(f"  policyc_margin: {args.policyc_margin}")
        print(f"  timeframe: {timeframe_minutes}min ({interval})")
        print(f"  pnl column: {args.pnl_column}")
        print(f"  x_bars_min_delay: {args.x_bars_min_delay} (x_bars below this clamped to 0/OFF)")
    
    # Process only finalists
    try:
        results = process_candidates(
            finalists_df,
            events_df,
            kt_quantile=args.kt_quantile,
            x_quantiles=x_quantiles,
            x_fixed=x_fixed,
            timeframe_minutes=timeframe_minutes,
            pnl_column=args.pnl_column,
            x_min_tail=args.x_min_tail,
            policyc_margin=args.policyc_margin,
            debug=args.debug,
            enable_x_selection=(not args.disable_x_selection),
            x_bars_min_delay=args.x_bars_min_delay,

            kt_winsorize=args.kt_winsorize,
            kt_winsorize_lower_q=args.kt_winsorize_lower_q,
            kt_winsorize_upper_q=args.kt_winsorize_upper_q,
            k_cap=args.k_cap,
            t_cap=args.t_cap,
        )
    except Exception as e:
        print(f"Error processing finalists: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Finalists Exit Parameters")
    print('='*80)
    for result in results:
        ep = result['exit_params']
        x_diag = ep.get('x_selection_diagnostics')
        
        print(f"\n{result['possibility']} (Score {result['score']:.2f}, Trades {result['trades']})")
        print(f"  Filters: Close={result['close']}, Vol={result['vol']}, Vol_Rule={result['vol_rule']}")
        print(f"  Events: {ep['events_count']} total, {ep['events_used']} used for k/t")
        
        # Print x-candidate table if diagnostics available
        if x_diag and 'candidates' in x_diag:
            print(f"\n  X-Candidate Evaluation:")
            print(f"    {'Source':<10} {'Quantile':<10} {'x_bars':<8} {'n_tail':<8} {'pnl_avg_tail':<12}")
            print(f"    {'-'*60}")
            for c in x_diag['candidates']:
                source = c['source']
                q_str = f"Q{int(c['quantile']*100)}" if c['quantile'] else 'fixed'
                x_bars = c['x_bars']
                n_tail = c['n_tail']
                pnl_avg = c['pnl_avg_tail']
                pnl_str = f"{pnl_avg:.2f}" if pnl_avg is not None else "N/A"
                eligible = "✓" if n_tail >= args.x_min_tail else "✗"
                print(f"    {source:<10} {q_str:<10} {x_bars:<8} {n_tail:<8} {pnl_str:<12} {eligible}")
            
            # Print decision
            print(f"\n  Policy C Decision: {x_diag.get('policy_decision', 'N/A')}")
            print(f"  Selected: x_bars={ep['x_bars']} bars", end="")
            if x_diag.get('best_source') == 'quantile' and x_diag.get('best_quantile'):
                print(f" (Q{int(x_diag['best_quantile']*100)})")
            elif x_diag.get('best_source') == 'fixed':
                print(f" (fixed)")
            else:
                print()
            
            if x_diag.get('best_pnl_avg_tail') is not None:
                print(f"    n_tail={x_diag['best_n_tail']}, pnl_avg_tail={x_diag['best_pnl_avg_tail']:.2f}")
        else:
            print(f"  x_bars: {ep['x_bars']} bars" if ep['x_bars'] is not None else "  x_bars: N/A")
        
        # Print k and t
        print(f"\n  Final Parameters:")
        print(f"    k (P{int(args.kt_quantile*100)}): {ep['k']:.4f}" if ep['k'] is not None else "    k: N/A")
        print(f"    t (P{int(args.kt_quantile*100)}): {ep['t']:.4f}" if ep['t'] is not None else "    t: N/A")
        if ep.get('trail_dist_ratio_estimate') is not None:
            print(f"    t_ratio_est: {ep['trail_dist_ratio_estimate']:.4f} ({ep.get('t_mechanical_flag', 'N/A')})")
        print(f"    x_bars: {ep['x_bars']}" if ep['x_bars'] is not None else "    x_bars: N/A")
    print('='*80)
    
    # Write JSON output
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        output_data = {
            'metadata': {
                'pair': pair,
                'prepaper_start': prepaper_date,
                'candidates_csv': str(candidates_path),
                'events_csv': str(events_path),
                'kt_quantile': args.kt_quantile,
                'x_quantiles': x_quantiles,
                'x_fixed': x_fixed,
                'timeframe_minutes': timeframe_minutes,
                'engine_trailing_mode': 'immediate',
                'x_bars_record_only': True,
                'pnl_column': args.pnl_column,
                'x_selection_enabled': enable_x_selection,
                'x_bars_min_delay': args.x_bars_min_delay,
                'finalists_per_category': args.finalists_per_category,
                'finalists_n': args.finalists_n,
                'min_trades': args.min_trades,
                'x_min_tail': args.x_min_tail,
                'policyc_margin': args.policyc_margin,
                'total_candidates_loaded': len(candidates_df),
                'eligible_candidates': len(candidates_df[candidates_df['Trades'] > args.min_trades]),
                'finalists_processed': len(results),
                'kt_winsorize': args.kt_winsorize,
                'kt_winsorize_lower_q': args.kt_winsorize_lower_q,
                'kt_winsorize_upper_q': args.kt_winsorize_upper_q,
                'k_cap': args.k_cap,
                't_cap': args.t_cap
            },
            'finalists': results  # Changed from 'candidates' to 'finalists' for clarity
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults written to: {output_path}")
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
