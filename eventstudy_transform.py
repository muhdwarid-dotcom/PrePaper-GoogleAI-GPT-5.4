"""
Data transformation module for event study analysis.
Reproduces Excel Power Query transformations in Python.
"""

import pandas as pd
import numpy as np


def load_and_transform_csv(csv_path):
    """
    Load CSV and apply Power Query equivalent transformations.
    
    Args:
        csv_path: Path to the source CSV file
        
    Returns:
        DataFrame with transformed data including vol_ratio_bin and boolean flags
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Convert event_time to UTC datetime
    df['event_time'] = pd.to_datetime(df['event_time'], utc=True)
    
    # Cast column types (most should already be correct from CSV)
    df['close_gt_smma_200'] = df['close_gt_smma_200'].astype(bool)
    df['vol_gt_vol_sma'] = df['vol_gt_vol_sma'].astype(bool)
    df['open_ended'] = df['open_ended'].astype(bool)
    
    # Add vol_ratio_bin column with 7 bins
    # Using underscore (_) for bins to avoid Excel date auto-conversion (e.g., 2-3 → 2-Mar)
    df['vol_ratio_bin'] = pd.cut(
        df['vol_ratio'],
        bins=[0, 1.5, 2, 3, 4, 5, 10, float('inf')],
        labels=['<1.5', '1.5_2', '2_3', '3_4', '4_5', '5_10', '>=10'],
        right=False,
        include_lowest=True
    )
    
    # Add boolean columns for vol_ratio thresholds
    # Null vol_ratio yields False for these flags
    df['vol_ratio_ge_1_5'] = df['vol_ratio'].fillna(0) >= 1.5
    df['vol_ratio_ge_2'] = df['vol_ratio'].fillna(0) >= 2.0
    df['vol_ratio_ge_3'] = df['vol_ratio'].fillna(0) >= 3.0
    df['vol_ratio_ge_4'] = df['vol_ratio'].fillna(0) >= 4.0
    df['vol_ratio_ge_5'] = df['vol_ratio'].fillna(0) >= 5.0
    df['vol_ratio_ge_10'] = df['vol_ratio'].fillna(0) >= 10.0
    
    return df


def get_transformed_dataframe(csv_path):
    """
    Main entry point for data transformation.
    
    Args:
        csv_path: Path to the source CSV file
        
    Returns:
        Transformed DataFrame ready for analysis
    """
    return load_and_transform_csv(csv_path)