#!/usr/bin/env python3
"""
Event Study Analysis CLI — one-pair-at-a-time.

Reads a V30 event-study CSV, applies transformations, computes metrics,
and writes pair-identified, window-stamped output CSVs.

USAGE (Step 2 in the 3-step pipeline):
  python eventstudy_analysis.py \\
      forwardtest/v30_eventstudy_ACTUSDT_1m_rsi_sma_cross_gt51_prepaper_2025-12-01.csv \\
      --pair ACTUSDT --prepaper-start 2025-12-01 --grid

  Outputs:
    forwardtest/eventstudy_list_summary_ACTUSDT_prepaper_2025-12-01.csv
    forwardtest/top20_view_ACTUSDT_prepaper_2025-12-01.csv

  (--pair and --prepaper-start are inferred from the input filename when possible.)
"""

import argparse
import re
import sys
from pathlib import Path
import pandas as pd

from eventstudy_transform import get_transformed_dataframe
from eventstudy_metrics import compute_all_metrics, format_summary_table, get_top_per_category, generate_grid_possibilities

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 40)
pd.set_option('display.float_format', lambda x: f'{x:,.6f}')

SUPPORTED_INTERVALS = ("1m", "3m")


def _interval_to_minutes(interval: str) -> int:
    """Map an interval string (e.g. '3m') to its duration in minutes."""
    _MAP = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
    return _MAP.get(interval, 1)


def _infer_pair_date_interval(filename: str):
    """
    Try to extract pair, prepaper date, and interval from an eventstudy filename.

    Expected pattern:
        v30_eventstudy_{PAIR}_{INTERVAL}_rsi_sma_cross_gt51_prepaper_{YYYY-MM-DD}.csv
    Returns (pair, date_str, interval) or (None, None, None) if not matched.
    """
    m = re.search(
        r'v30_eventstudy_([A-Z0-9]+)_(1m|3m)_.*?_prepaper_(\d{4}-\d{2}-\d{2})',
        Path(filename).name,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper(), m.group(3), m.group(2).lower()
    return None, None, None

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Event Study Analysis - Reproduce Excel study logic in Python'
    )
    parser.add_argument(
        'csv_path',
        help='Path to the source CSV file (e.g., forwardtest/v30_eventstudy_ACTUSDT_1m_rsi_sma_cross_gt51_prepaper_2025-12-01.csv)'
    )
    parser.add_argument(
        '--pair',
        default=None,
        help='Trading pair symbol (e.g. ACTUSDT). Inferred from filename when not supplied.'
    )
    parser.add_argument(
        '--prepaper-start',
        default=None,
        metavar='YYYY-MM-DD',
        help='PrePaper window start date. Inferred from filename when not supplied.'
    )
    parser.add_argument(
        '--interval',
        default=None,
        choices=list(SUPPORTED_INTERVALS),
        help=(
            'Kline interval used when generating the events CSV (default: inferred from '
            'filename, or "1m" if not determinable). Affects timeframe-minutes and output filenames.'
        ),
    )
    parser.add_argument(
        '--output',
        default=None,
        help=(
            'Output CSV path for the full summary. '
            'Default: forwardtest/eventstudy_list_summary_{PAIR}_{INTERVAL}_prepaper_{DATE}.csv '
            '(falls back to forwardtest/eventstudy_list_summary.csv when pair/date unknown)'
        )
    )
    parser.add_argument(
        '--no-print',
        action='store_true',
        help='Do not print table to console'
    )
    parser.add_argument(
        '--top-per-category',
        action='store_true',
        help='Display top candidate per close/vol category for trading-window evaluation'
    )
    parser.add_argument(
        '--grid',
        action='store_true',
        help='Generate and evaluate all possible combinations of close/vol/vol_rule gates'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='In grid mode, print only top N eligible candidates to console (default: 20)'
    )
    parser.add_argument(
        '--vol-rule-as-gate',
        action='store_true',
        help='Count vol_rule as a gate for eligibility (allows ALL/ALL/vol_rule possibilities to be eligible)'
    )
    parser.add_argument(
        '--timeframe-minutes',
        type=int,
        default=None,
        help=(
            'Timeframe in minutes (used to convert bars -> minutes for timing thresholds). '
            'Defaults to the value derived from --interval (1m->1, 3m->3).'
        ),
    )
    parser.add_argument(
        '--timing-bars',
        type=int,
        default=60,
        help='Timing threshold in bars for peak/dip metrics (converted to minutes via timeframe). Default: 60'
    )
    parser.add_argument(
    '--top-gate-families',
    action='store_true',
    help='Pick best candidates in Top-20 by gate_count family: 1-gate, 2-gate, 3-gate.'
    )
    parser.add_argument(
        '--family-top-k',
        type=int,
        default=20,
        help='Only consider candidates with Rank <= K for gate-family selection (default: 20).'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve pair, prepaper-start, and interval (explicit args override filename inference)
    inferred_pair, inferred_date, inferred_interval = _infer_pair_date_interval(str(csv_path))
    if args.pair and args.pair.strip():
        pair = args.pair.strip().upper()
    elif inferred_pair:
        pair = inferred_pair.upper()
    else:
        pair = None
    prepaper_date = args.prepaper_start or inferred_date or None

    # Resolve interval: CLI flag > filename inference > default "1m"
    interval = args.interval or inferred_interval or "1m"

    # Derive timeframe_minutes from interval unless explicitly overridden
    if args.timeframe_minutes is not None:
        timeframe_minutes = args.timeframe_minutes
    else:
        timeframe_minutes = _interval_to_minutes(interval)

    # Build default output paths (include interval so 1m and 3m runs don't collide)
    if args.output:
        output_path = Path(args.output)
    elif pair and prepaper_date:
        output_path = Path("forwardtest") / f"eventstudy_list_summary_{pair}_{interval}_prepaper_{prepaper_date}.csv"
    else:
        output_path = Path("forwardtest/eventstudy_list_summary.csv")

    # Derive top20_view path from output path
    if pair and prepaper_date:
        top20_path = Path("forwardtest") / f"top20_view_{pair}_{interval}_prepaper_{prepaper_date}.csv"
    else:
        top20_path = Path("forwardtest/top20_view.csv")

    print(f"Loading data from: {csv_path}")
    if pair:
        print(f"Pair: {pair}")
    if prepaper_date:
        print(f"PrePaper start: {prepaper_date}")
    print(f"Interval: {interval} ({timeframe_minutes} min/bar)")
    
    # Load and transform data
    try:
        df = get_transformed_dataframe(str(csv_path))
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading/transforming data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print eligibility mode
    if args.vol_rule_as_gate:
        print("Eligibility mode: vol_rule counts as gate")
    else:
        print("Eligibility mode: close or vol must be TRUE")
    
    # Compute metrics
    if args.grid:
        grid_count = len(generate_grid_possibilities())
        print(f"Computing metrics for all grid possibilities ({grid_count} combinations)...")
    else:
        print("Computing metrics for all possibilities...")
    try:
        results_df = compute_all_metrics(
            df,
            grid_mode=args.grid,
            vol_rule_as_gate=args.vol_rule_as_gate,
            timeframe_minutes=timeframe_minutes,
            timing_bars=args.timing_bars
        )
        formatted_df = format_summary_table(results_df, grid_mode=args.grid)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print table to console
    if not args.no_print:
        print("\n" + "="*80)
        print("Event Study Summary")
        print("="*80)
        
        # In grid mode with --top, show only top N eligible candidates
        if args.grid and args.top is not None:
            # Filter to eligible candidates (non-NaN Score)
            eligible = formatted_df[formatted_df['Score'].notna()]
            if len(eligible) > 0:
                top_n = eligible.head(args.top)
                print(f"Showing top {len(top_n)} of {len(eligible)} eligible candidates")
                print(top_n.to_string(index=False))
                if len(eligible) > args.top:
                    print(f"\n... and {len(eligible) - args.top} more eligible candidates")
                    print(f"(Full results with all {len(formatted_df)} possibilities in CSV)")
            else:
                print("No eligible candidates found")
                print(formatted_df.to_string(index=False))
        else:
            print(formatted_df.to_string(index=False))
        
        print("="*80 + "\n")
    
    # Print top per category if requested
    if args.top_per_category:
        print("\n" + "="*80)
        print("Top Candidates Per Category (for Trading-Window Evaluation)")
        print("="*80)
        top_per_cat = get_top_per_category(formatted_df, vol_rule_as_gate=args.vol_rule_as_gate)
        if len(top_per_cat) > 0:
            print(top_per_cat.to_string(index=False))
            print("\nCategories:")
            if args.vol_rule_as_gate:
                print("  Original 5 (close, vol) categories:")
                print("    - (ALL, True): All close values, volume above SMA")
                print("    - (True, ALL): Close above SMA, all volume values")
                print("    - (True, True): Both close and volume above SMA")
                print("    - (True, False): Close above SMA, volume below SMA")
                print("    - (False, True): Close below SMA, volume above SMA")
                print("  Additional 4 vol_rule-based categories:")
                print("    - (ALL, ALL, low): Any close/vol, low vol_ratio (<1.5 to <5)")
                print("    - (ALL, ALL, elevated): Any close/vol, elevated vol_ratio (bins or >=thresholds)")
                print("    - (False, ALL, low): Close below SMA, any vol, low vol_ratio (<1.5 to <5)")
                print("    - (False, False, low): Both close and vol below SMA, low vol_ratio (<1.5 to <5)")
            else:
                print("  - (ALL, True): All close values, volume above SMA")
                print("  - (True, ALL): Close above SMA, all volume values")
                print("  - (True, True): Both close and volume above SMA")
                print("  - (True, False): Close above SMA, volume below SMA")
                print("  - (False, True): Close below SMA, volume above SMA")
        else:
            print("No eligible candidates found for categories.")
        print("="*80 + "\n")
    
    # Write full summary CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        formatted_df.to_csv(output_path, index=False)
        print(f"Results written to: {output_path}")

        # Write top-20 view CSV
        eligible = formatted_df[formatted_df['Score'].notna()]
        top20 = eligible.head(20)
        top20_path.parent.mkdir(parents=True, exist_ok=True)
        top20.to_csv(top20_path, index=False)
        print(f"Top-20 view written to: {top20_path}")
        
        if args.top_gate_families:
            df_sel = formatted_df.copy()

            # Primary eligibility (keep aligned with your pipeline)
            df_sel = df_sel[df_sel['Score'].notna()]
            df_sel = df_sel[df_sel['Trades'] >= 50]

            # Only Top-K ranks
            df_sel = df_sel[df_sel['Rank'] <= args.family_top_k]

            # Gate counting
            close_gate = df_sel['close'].astype(str).eq('True')
            vol_gate = df_sel['vol'].astype(str).eq('True')

            if args.vol_rule_as_gate:
                vol_rule_gate = ~df_sel['vol_rule'].astype(str).eq('ALL')
            else:
                vol_rule_gate = close_gate.map(lambda _: False)  # all False

            df_sel = df_sel.assign(
                gate_count=(close_gate.astype(int) + vol_gate.astype(int) + vol_rule_gate.astype(int))
            )

            winners = []
            for k in (1, 2, 3):
                fam = df_sel[df_sel['gate_count'] == k].sort_values(['Rank', 'Score'], ascending=[True, False])
                if fam.empty:
                    continue
                winners.append(fam.iloc[0])

            print("\n" + "="*80)
            print(f"Best candidates by gate_count family within Top {args.family_top_k} ranks")
            print("="*80)
            if not winners:
                print("No gate-family winners found in Top-K (check eligibility filters).")
            else:
                out = pd.DataFrame(winners)[
                    ['gate_count', 'Possibility', 'close', 'vol', 'vol_rule', 'Trades', 'Total_Net_PnL', 'Score', 'Rank']
                ].sort_values(['gate_count'])
                print(out.to_string(index=False))
            print("="*80 + "\n")
        
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()