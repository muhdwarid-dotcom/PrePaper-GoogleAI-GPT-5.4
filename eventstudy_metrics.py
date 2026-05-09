"""
Metrics computation module for event study analysis.
Reproduces Excel 'List' sheet calculations in Python.
"""

import pandas as pd
import numpy as np
import itertools


# Define possibilities with their gating rules
# Using ASCII hyphen (-) instead of Unicode en-dash (–) to avoid encoding issues
POSSIBILITIES = {
    'G0': {'close': 'ALL', 'vol': 'ALL', 'vol_rule': 'ALL'},
    'A0': {'close': True, 'vol': True, 'vol_rule': 'ALL'},
    'B0': {'close': False, 'vol': False, 'vol_rule': 'ALL'},
    'C0': {'close': False, 'vol': True, 'vol_rule': 'ALL'},
    'D0': {'close': True, 'vol': False, 'vol_rule': 'ALL'},
    # New base possibilities for missing close×vol combinations
    'E0': {'close': 'ALL', 'vol': True, 'vol_rule': 'ALL'},
    'F0': {'close': 'ALL', 'vol': False, 'vol_rule': 'ALL'},
    'H0': {'close': True, 'vol': 'ALL', 'vol_rule': 'ALL'},
    'I0': {'close': False, 'vol': 'ALL', 'vol_rule': 'ALL'},
    # Existing vol_rule variants
    'A1': {'close': True, 'vol': True, 'vol_rule': '>=1.5'},
    'A2': {'close': True, 'vol': True, 'vol_rule': '>=3'},
    'A3': {'close': True, 'vol': True, 'vol_rule': '3_4'},
    'A4': {'close': True, 'vol': True, 'vol_rule': '5_10'},
    'B1': {'close': False, 'vol': True, 'vol_rule': '>=1.5'},
    'B2': {'close': False, 'vol': False, 'vol_rule': '1.5_2'},
    # Additional vol_rule examples demonstrating new thresholds and bins
    'C1': {'close': False, 'vol': True, 'vol_rule': '>=2'},
    'C2': {'close': False, 'vol': True, 'vol_rule': '2_3'},
    'D1': {'close': True, 'vol': False, 'vol_rule': '<1.5'},
    'E1': {'close': 'ALL', 'vol': True, 'vol_rule': '>=4'},
    'E2': {'close': 'ALL', 'vol': True, 'vol_rule': '4_5'},
}


def generate_grid_possibilities():
    """
    Generate all combinations of close, vol, and vol_rule for grid mode.
    
    Returns:
        Dictionary mapping generated IDs to possibility rules
    """
    close_values = ['ALL', True, False]
    vol_values = ['ALL', True, False]
    # Using ASCII hyphen (-) instead of Unicode en-dash (–) to avoid encoding issues
    # Using underscore (_) for bins to prevent Excel date auto-conversion
    vol_rule_values = [
        'ALL',
        # Less-than thresholds (cumulative low-vol)
        '<1.5', '<2', '<3', '<4', '<5',
        # Greater-than thresholds (cumulative high-vol)
        '>=1.5', '>=2', '>=3', '>=4', '>=5', '>=10',
        # Bins (discrete ranges)
        '1.5_2', '2_3', '3_4', '4_5', '5_10'
    ]
    
    grid_possibilities = {}
    
    for close, vol, vol_rule in itertools.product(close_values, vol_values, vol_rule_values):
        # Generate descriptive ID
        possibility_id = _generate_possibility_id(close, vol, vol_rule)
        
        grid_possibilities[possibility_id] = {
            'close': close,
            'vol': vol,
            'vol_rule': vol_rule
        }
    
    return grid_possibilities


def _generate_possibility_id(close, vol, vol_rule):
    """
    Generate a descriptive ID for a possibility.
    
    Format: C_<close>__V_<vol>__R_<vol_rule>
    
    Args:
        close: Close gate value (ALL, True, False)
        vol: Vol gate value (ALL, True, False)
        vol_rule: Vol rule value
        
    Returns:
        String ID for the possibility
    """
    # Format close value
    if close == 'ALL':
        close_str = 'ALL'
    elif close is True:
        close_str = 'TRUE'
    else:
        close_str = 'FALSE'
    
    # Format vol value
    if vol == 'ALL':
        vol_str = 'ALL'
    elif vol is True:
        vol_str = 'TRUE'
    else:
        vol_str = 'FALSE'
    
    # Format vol_rule value
    # Normalize both Unicode en-dash (–) and ASCII hyphen (-) to underscore for ID
    vol_rule_str = vol_rule.replace('>=', 'GE_').replace('<', 'LT_').replace('–', '_').replace('-', '_')
    
    return f"C_{close_str}__V_{vol_str}__R_{vol_rule_str}"


def filter_by_possibility(df, possibility_id, possibilities_dict=None):
    """
    Filter dataframe by possibility gating rules.
    
    Args:
        df: Transformed dataframe
        possibility_id: Possibility identifier (e.g., 'G0', 'A0', etc.)
        possibilities_dict: Optional custom possibilities dict (for grid mode)
        
    Returns:
        Filtered dataframe
    """
    if possibilities_dict is None:
        possibilities_dict = POSSIBILITIES
    
    rules = possibilities_dict[possibility_id]
    filtered = df.copy()
    
    # Apply close_gt_smma_200 filter
    if rules['close'] != 'ALL':
        filtered = filtered[filtered['close_gt_smma_200'] == rules['close']]
    
    # Apply vol_gt_vol_sma filter
    if rules['vol'] != 'ALL':
        filtered = filtered[filtered['vol_gt_vol_sma'] == rules['vol']]
    
    # Apply vol_rule filter
    if rules['vol_rule'] != 'ALL':
        # Greater/equal thresholds
        if rules['vol_rule'] == '>=1.5':
            filtered = filtered[filtered['vol_ratio_ge_1_5']]
        elif rules['vol_rule'] == '>=2':
            filtered = filtered[filtered['vol_ratio_ge_2']]
        elif rules['vol_rule'] == '>=3':
            filtered = filtered[filtered['vol_ratio_ge_3']]
        elif rules['vol_rule'] == '>=4':
            filtered = filtered[filtered['vol_ratio_ge_4']]
        elif rules['vol_rule'] == '>=5':
            filtered = filtered[filtered['vol_ratio_ge_5']]
        elif rules['vol_rule'] == '>=10':
            filtered = filtered[filtered['vol_ratio_ge_10']]
        # Less-than thresholds (cumulative low-vol) - use vol_ratio column directly
        elif rules['vol_rule'] == '<2':
            filtered = filtered[filtered['vol_ratio'] < 2.0]
        elif rules['vol_rule'] == '<3':
            filtered = filtered[filtered['vol_ratio'] < 3.0]
        elif rules['vol_rule'] == '<4':
            filtered = filtered[filtered['vol_ratio'] < 4.0]
        elif rules['vol_rule'] == '<5':
            filtered = filtered[filtered['vol_ratio'] < 5.0]
        # Bins and <1.5 threshold - use vol_ratio_bin column (backward compatibility)
        elif rules['vol_rule'] in ['<1.5', '1.5-2', '1.5_2', '1.5–2', '2-3', '2_3', '2–3', '3-4', '3_4', '3–4', '4-5', '4_5', '4–5', '5-10', '5_10', '5–10', '>=10']:
            # Normalize the rule to match what's in the dataframe (now using underscore)
            normalized_rule = rules['vol_rule'].replace('–', '-').replace('-', '_')
            filtered = filtered[filtered['vol_ratio_bin'] == normalized_rule]
    
    return filtered


def compute_worst_day(df):
    """
    Compute worst day metric: minimum of daily summed PnL.
    Groups by UTC date of event_time.
    
    Args:
        df: Filtered dataframe
        
    Returns:
        Worst day value (most negative daily PnL)
    """
    if len(df) == 0:
        return np.nan
    
    # Extract UTC date
    df = df.copy()
    df['date'] = df['event_time'].dt.date
    
    # Sum PnL by date
    daily_pnl = df.groupby('date')['net_pnl_usdt'].sum()
    
    # Return minimum (worst day)
    return daily_pnl.min()


def compute_max_drawdown(df):
    """
    Compute max drawdown metric.
    
    Process:
    1. Compute daily PnL series (grouped by UTC date)
    2. Calculate cumulative equity
    3. Compute drawdown = peak - equity
    4. Return max drawdown
    
    Args:
        df: Filtered dataframe
        
    Returns:
        Max drawdown value
    """
    if len(df) == 0:
        return np.nan
    
    # Extract UTC date
    df = df.copy()
    df['date'] = df['event_time'].dt.date
    
    # Sum PnL by date and sort by date
    daily_pnl = df.groupby('date')['net_pnl_usdt'].sum().sort_index()
    
    # Calculate cumulative equity
    cumulative_equity = daily_pnl.cumsum()
    
    # Calculate running peak
    running_peak = cumulative_equity.expanding().max()
    
    # Calculate drawdown
    drawdown = running_peak - cumulative_equity
    
    # Return max drawdown
    return drawdown.max()


def compute_metrics_for_possibility(df, possibility_id, possibilities_dict=None, timeframe_minutes=1, timing_bars=60):
    """
    Compute all metrics for a given possibility.
    
    Metrics:
    - Trades (count)
    - Total Net PnL (sum of net_pnl_usdt)
    - Avg Net PnL (mean)
    - Worst Trade (min)
    - Worst Day (minimum of daily summed pnl)
    - Max Drawdown
    
    Args:
        df: Transformed dataframe
        possibility_id: Possibility identifier
        possibilities_dict: Optional custom possibilities dict (for grid mode)
        
    Returns:
        Dictionary with metrics
    """
    if possibilities_dict is None:
        possibilities_dict = POSSIBILITIES
    
    # Filter by possibility
    filtered = filter_by_possibility(df, possibility_id, possibilities_dict)
    
    # FIX: net_pnl_usdt is coming in as strings (object); convert to numeric
    if 'net_pnl_usdt' in filtered.columns:
        filtered = filtered.copy()
        filtered['net_pnl_usdt'] = pd.to_numeric(filtered['net_pnl_usdt'], errors='coerce')
    
    # --- Timing threshold (bars -> minutes) ---
    threshold_min = float(timing_bars) * float(timeframe_minutes)

    # Defaults
    p_peak_within_tb = np.nan
    median_time_to_peak_min = np.nan

    p_dip_below_entry_within_tb = np.nan
    median_time_to_dip_min = np.nan
    median_dip_below_entry_pct = np.nan

    if len(filtered) > 0:
        # Peak within threshold
        if 'time_to_max_high_min' in filtered.columns:
            tpeak = pd.to_numeric(filtered['time_to_max_high_min'], errors='coerce')
            if tpeak.notna().any():
                p_peak_within_tb = float((tpeak <= threshold_min).mean())
                median_time_to_peak_min = float(tpeak.median())

        # Dip below entry within threshold
        # Need: dip condition (min_low_before_stop < entry_close)
        # and dip timing: (min_low_time - event_time) in minutes
        if all(c in filtered.columns for c in ['min_low_before_stop', 'entry_close', 'min_low_time', 'event_time']):
            tmp = filtered[['min_low_before_stop', 'entry_close', 'min_low_time', 'event_time']].copy()

            # Ensure times are datetime
            tmp['event_time'] = pd.to_datetime(tmp['event_time'], errors='coerce', utc=True)
            tmp['min_low_time'] = pd.to_datetime(tmp['min_low_time'], errors='coerce', utc=True)

            min_low = pd.to_numeric(tmp['min_low_before_stop'], errors='coerce')
            entry = pd.to_numeric(tmp['entry_close'], errors='coerce')

            dip_below_entry = (min_low < entry) & min_low.notna() & entry.notna()

            time_to_dip_min = (tmp['min_low_time'] - tmp['event_time']).dt.total_seconds() / 60.0
            # Some rows could be NaT -> NaN
            time_to_dip_min = pd.to_numeric(time_to_dip_min, errors='coerce')

            dip_within = dip_below_entry & time_to_dip_min.notna() & (time_to_dip_min <= threshold_min)

            # Fraction of *all events* that dip below entry within the timing threshold
            p_dip_below_entry_within_tb = float(dip_within.mean())

            if time_to_dip_min.notna().any():
                median_time_to_dip_min = float(time_to_dip_min.median())

            # Depth below entry for dip events (not limited to within threshold; purely diagnostic)
            dip_mask = dip_below_entry & (entry != 0)
            if dip_mask.any():
                dip_pct = (min_low[dip_mask] / entry[dip_mask]) - 1.0  # negative number
                median_dip_below_entry_pct = float(dip_pct.median())
    
    # Compute metrics
    metrics = {
        'Possibility': possibility_id,
        'Trades': len(filtered),
        'Total_Net_PnL': filtered['net_pnl_usdt'].sum(skipna=True) if len(filtered) > 0 else 0,
        'Avg_Net_PnL': filtered['net_pnl_usdt'].mean(skipna=True) if len(filtered) > 0 else np.nan,
        'Worst_Trade': filtered['net_pnl_usdt'].min(skipna=True) if len(filtered) > 0 else np.nan,
        'Worst_Day': compute_worst_day(filtered),
        'Max_Drawdown': compute_max_drawdown(filtered),
        'p_peak_within_tb': p_peak_within_tb,
        'median_time_to_peak_min': median_time_to_peak_min,
        'p_dip_below_entry_within_tb': p_dip_below_entry_within_tb,
        'median_time_to_dip_min': median_time_to_dip_min,
        'median_dip_below_entry_pct': median_dip_below_entry_pct,
    }
    
    return metrics


def compute_score(metrics, possibilities_dict=None, grid_mode=False, vol_rule_as_gate=False, timeframe_minutes=1, timing_bars=60):
    """
    Compute score for a possibility using Excel LET formula logic.
    
    Formula:
    - minTrades = 50
    - hasAtLeastOneGate:
      * Default: (close == True OR vol == True)
      * With vol_rule_as_gate: (close == True OR vol == True OR vol_rule != 'ALL')
    - eligible = AND(trades>=minTrades, hasAtLeastOneGate)
    - If not eligible => NA
    - Else score = total / (maxDD + ABS(worstDay) + 1)
    
    Args:
        metrics: Dictionary with metrics including Possibility ID
        possibilities_dict: Optional custom possibilities dict (for grid mode)
        grid_mode: If True, apply grid mode rules (currently unused but kept for compatibility)
        vol_rule_as_gate: If True, count vol_rule != 'ALL' as a gate for eligibility
        
    Returns:
        Score value or np.nan if not eligible
    """
    if possibilities_dict is None:
        possibilities_dict = POSSIBILITIES
    
    min_trades = 50
    possibility_id = metrics['Possibility']
    rules = possibilities_dict[possibility_id]
    
    # Check if has at least one gate (depends on vol_rule_as_gate flag)
    if vol_rule_as_gate:
        has_at_least_one_gate = (
            rules['close'] is True or 
            rules['vol'] is True or 
            rules['vol_rule'] != 'ALL'
        )
    else:
        has_at_least_one_gate = rules['close'] is True or rules['vol'] is True
    
    # Check eligibility: requires Trades >= 50 AND at least one gate
    eligible = metrics['Trades'] >= min_trades and has_at_least_one_gate
    
    if not eligible:
        return np.nan
    
    # Calculate score
    total = metrics['Total_Net_PnL']
    max_dd = metrics['Max_Drawdown']
    worst_day = metrics['Worst_Day']
    
    base_score = total / (max_dd + abs(worst_day) + 1)

    # Pull timing metrics (neutral if missing)
    p_peak = metrics.get('p_peak_within_tb', np.nan)
    p_dip_within = metrics.get('p_dip_below_entry_within_tb', np.nan)

    if pd.isna(p_peak):
        p_peak = 0.0
    if pd.isna(p_dip_within):
        p_dip_within = 0.0

    # Multiplier 1: reward fast peaks (bounded, never < 0.5)
    peak_multiplier = 0.5 + 0.5 * float(np.clip(p_peak, 0.0, 1.0))

    # Multiplier 2: penalize dips below entry within threshold (bounded, never < 0.25)
    # You can tune 0.25 floor later.
    dip_multiplier = max(0.25, 1.0 - float(np.clip(p_dip_within, 0.0, 1.0)))

    score = base_score * peak_multiplier * dip_multiplier
    return score


def compute_all_metrics(df, grid_mode=False, vol_rule_as_gate=False, timeframe_minutes=1, timing_bars=60):
    """
    Compute metrics for all possibilities and compute scores/ranks.
    
    Args:
        df: Transformed dataframe
        grid_mode: If True, generate grid of all possible combinations
        vol_rule_as_gate: If True, count vol_rule != 'ALL' as a gate for eligibility
        
    Returns:
        DataFrame with all metrics, scores, and ranks
    """
    # Determine which possibilities to use
    if grid_mode:
        possibilities_dict = generate_grid_possibilities()
    else:
        possibilities_dict = POSSIBILITIES
    
    results = []
    
    # Compute metrics for each possibility
    for possibility_id in possibilities_dict.keys():
        metrics = compute_metrics_for_possibility(
            df,
            possibility_id,
            possibilities_dict,
            timeframe_minutes=timeframe_minutes,
            timing_bars=timing_bars
        )
        
        # Add close, vol, vol_rule columns for interpretability
        if grid_mode:
            rules = possibilities_dict[possibility_id]
            metrics['close'] = rules['close']
            metrics['vol'] = rules['vol']
            metrics['vol_rule'] = rules['vol_rule']
        
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute scores
    results_df['Score'] = results_df.apply(
        lambda row: compute_score(row, possibilities_dict, grid_mode, vol_rule_as_gate, timeframe_minutes=timeframe_minutes, timing_bars=timing_bars),
        axis=1
    )
    
    # Compute ranks (1 = best, only for eligible possibilities)
    # Filter out NaN scores for ranking
    valid_scores = results_df['Score'].dropna()
    if len(valid_scores) > 0:
        # Rank in descending order (higher score = better rank)
        ranks = valid_scores.rank(ascending=False, method='min')
        results_df.loc[ranks.index, 'Rank'] = ranks
    else:
        results_df['Rank'] = np.nan
    
    return results_df


def format_summary_table(results_df, grid_mode=False):
    """
    Format results for display similar to Excel List sheet.
    
    Args:
        results_df: DataFrame with metrics and scores
        grid_mode: If True, include close/vol/vol_rule columns
        
    Returns:
        Formatted DataFrame
    """
    # Reorder columns to match Excel layout
    if grid_mode:
        column_order = [
            'Possibility',
            'close',
            'vol',
            'vol_rule',
            'Trades',
            'Total_Net_PnL',
            'Avg_Net_PnL',
            
             # ADD THESE:
            'p_peak_within_tb',
            'median_time_to_peak_min',
            'p_dip_below_entry_within_tb',
            'median_time_to_dip_min',
            'median_dip_below_entry_pct',
            
            'Rank',
            'Worst_Trade',
            'Worst_Day',
            'Max_Drawdown',
            'Score'
        ]
    else:
        column_order = [
            'Possibility',
            'Trades',
            'Total_Net_PnL',
            'Avg_Net_PnL',
            'Rank',
            'Worst_Trade',
            'Worst_Day',
            'Max_Drawdown',
            'Score'
        ]
    
    formatted_df = results_df[column_order].copy()
    
    # Sort by Rank (NaN last)
    formatted_df = formatted_df.sort_values('Rank', na_position='last')
    
    return formatted_df


def get_top_per_category(results_df, vol_rule_as_gate=False):
    """
    Select top candidate per close/vol category by Score.
    
    When vol_rule_as_gate is False (default):
    - 5 categories based on (close, vol)
    
    When vol_rule_as_gate is True:
    - 9 categories: 5 original + 4 vol_rule-based categories
    
    Original 5 categories (close, vol):
    - (ALL, True): All close values, volume above SMA
    - (True, ALL): Close above SMA, all volume values
    - (True, True): Both close and volume above SMA
    - (True, False): Close above SMA, volume below SMA
    - (False, True): Close below SMA, volume above SMA
    
    Additional 4 vol_rule-based categories (when vol_rule_as_gate=True):
    - (ALL, ALL, low): Any close/vol, low vol_ratio (<1.5 to <5)
    - (ALL, ALL, elevated): Any close/vol, elevated vol_ratio (bins or >=thresholds)
    - (False, ALL, low): Close below SMA, any vol, low vol_ratio (<1.5 to <5)
    - (False, False, low): Both below SMA, low vol_ratio (<1.5 to <5)
    
    Args:
        results_df: DataFrame with metrics and scores
        vol_rule_as_gate: If True, include vol_rule-based categories
        
    Returns:
        DataFrame with top candidate per category
    """
    # Define low volatility thresholds
    LOW_VOL_THRESHOLDS = ['<1.5', '<2', '<3', '<4', '<5']
    
    # Define the target categories
    if vol_rule_as_gate:
        # 9 categories: 5 original + 4 vol_rule-based
        target_categories = [
            # Original 5 (close, vol) categories
            {'close': 'ALL', 'vol': True, 'vol_rule': None, 'name': 'close=ALL, vol=True'},
            {'close': True, 'vol': 'ALL', 'vol_rule': None, 'name': 'close=True, vol=ALL'},
            {'close': True, 'vol': True, 'vol_rule': None, 'name': 'close=True, vol=True'},
            {'close': True, 'vol': False, 'vol_rule': None, 'name': 'close=True, vol=False'},
            {'close': False, 'vol': True, 'vol_rule': None, 'name': 'close=False, vol=True'},
            # New 4 vol_rule-based categories (grouped by low/elevated)
            {'close': 'ALL', 'vol': 'ALL', 'vol_rule': 'low', 'name': 'close=ALL, vol=ALL, vol_rule=low (<1.5 to <5)'},
            {'close': 'ALL', 'vol': 'ALL', 'vol_rule': 'elevated', 'name': 'close=ALL, vol=ALL, vol_rule=elevated (bins or >=thresholds)'},
            {'close': False, 'vol': 'ALL', 'vol_rule': 'low', 'name': 'close=False, vol=ALL, vol_rule=low (<1.5 to <5)'},
            {'close': False, 'vol': False, 'vol_rule': 'low', 'name': 'close=False, vol=False, vol_rule=low (<1.5 to <5)'},
        ]
    else:
        # Original 5 categories
        target_categories = [
            {'close': 'ALL', 'vol': True, 'vol_rule': None, 'name': 'close=ALL, vol=True'},
            {'close': True, 'vol': 'ALL', 'vol_rule': None, 'name': 'close=True, vol=ALL'},
            {'close': True, 'vol': True, 'vol_rule': None, 'name': 'close=True, vol=True'},
            {'close': True, 'vol': False, 'vol_rule': None, 'name': 'close=True, vol=False'},
            {'close': False, 'vol': True, 'vol_rule': None, 'name': 'close=False, vol=True'},
        ]
    
    # Filter to eligible possibilities only (non-NaN score)
    eligible = results_df[results_df['Score'].notna()].copy()
    
    # Check if close/vol/vol_rule columns exist (grid mode) or need to use POSSIBILITIES dict
    if 'close' in eligible.columns and 'vol' in eligible.columns:
        # Grid mode: use explicit close/vol/vol_rule columns directly
        eligible['close_category'] = eligible['close']
        eligible['vol_category'] = eligible['vol']
        if 'vol_rule' in eligible.columns:
            eligible['vol_rule_category'] = eligible['vol_rule']
        else:
            eligible['vol_rule_category'] = None
    else:
        # Regular mode: use POSSIBILITIES dict lookup
        close_map = {p: POSSIBILITIES[p]['close'] for p in POSSIBILITIES}
        vol_map = {p: POSSIBILITIES[p]['vol'] for p in POSSIBILITIES}
        vol_rule_map = {p: POSSIBILITIES[p]['vol_rule'] for p in POSSIBILITIES}
        eligible['close_category'] = eligible['Possibility'].map(close_map)
        eligible['vol_category'] = eligible['Possibility'].map(vol_map)
        eligible['vol_rule_category'] = eligible['Possibility'].map(vol_rule_map)
    
    # Select top candidate per category
    top_per_category = []
    for category in target_categories:
        close_cat = category['close']
        vol_cat = category['vol']
        vol_rule_cat = category['vol_rule']
        category_name = category['name']
        
        # Build filter mask based on category requirements
        category_mask = (
            (eligible['close_category'] == close_cat) &
            (eligible['vol_category'] == vol_cat)
        )
        
        # Add vol_rule filter if specified
        if vol_rule_cat is not None:
            if vol_rule_cat == 'low':
                # "low" means any vol_rule in the low threshold list
                category_mask = category_mask & eligible['vol_rule_category'].isin(LOW_VOL_THRESHOLDS)
            elif vol_rule_cat == 'elevated':
                # "elevated" means any vol_rule that is not 'ALL' and not in low thresholds
                category_mask = category_mask & (
                    (eligible['vol_rule_category'] != 'ALL') &
                    (~eligible['vol_rule_category'].isin(LOW_VOL_THRESHOLDS))
                )
            else:
                # Specific vol_rule value
                category_mask = category_mask & (eligible['vol_rule_category'] == vol_rule_cat)
        
        category_candidates = eligible[category_mask]
        
        if len(category_candidates) > 0:
            # Get the candidate with highest Score
            top_candidate = category_candidates.nlargest(1, 'Score')
            top_per_category.append({
                'Category': category_name,
                'Possibility': top_candidate['Possibility'].values[0],
                'Trades': top_candidate['Trades'].values[0],
                'Total_Net_PnL': top_candidate['Total_Net_PnL'].values[0],
                'Score': top_candidate['Score'].values[0],
                'Rank': top_candidate['Rank'].values[0],
            })
    
    return pd.DataFrame(top_per_category)