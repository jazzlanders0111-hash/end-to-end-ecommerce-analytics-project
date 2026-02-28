"""
n3b_data_loader.py - Data Loading and Preparation

Loads and prepares data for customer segmentation analysis,
ensuring data quality and integrity.

Automatically calculates RFM scores if they're missing from the input data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from n3a_utils import setup_logger, get_project_root, load_config

logger = setup_logger(__name__)


def load_data_for_segmentation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load transaction data and RFM data for segmentation.

    Returns:
        Tuple of (transaction_df, rfm_df)

    Raises:
        FileNotFoundError: If required data files don't exist
        ValueError: If data validation fails
    """

    try:
        config = load_config()
        project_root = get_project_root()

        # Load transaction data
        enhanced_path  = project_root / config['paths']['enhanced_df']
        logger.info(f"Loading enhanced data from: {enhanced_path}")

        if not enhanced_path.exists():
            raise FileNotFoundError(f"Enhanced data not found: {enhanced_path}")

        df = pd.read_parquet(enhanced_path)
    
        # Convert date column to datetime for proper chronological operations
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'])
            logger.info(f"Loaded {len(df):,} transactions")
            logger.info(f"Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
        else:
            logger.info(f"Loaded {len(df):,} transactions")
            logger.warning("No 'order_date' column found in transaction data")

        # Load RFM data (should be generated in Notebook 01)
        enhanced_path = project_root / config['paths']['enhanced_df']
        rfm_path = project_root / config['paths']['rfm_df']

        if not rfm_path.exists():
            raise FileNotFoundError(f"RFM data not found: {rfm_path}")

        rfm_df = pd.read_parquet(rfm_path)
        logger.info(f"Loaded RFM data for {len(rfm_df):,} customers")

        # Calculate missing scores if needed
        rfm_df = _ensure_rfm_scores(rfm_df)

        # Validate data
        _validate_segmentation_data(df, rfm_df, config)

        return df, rfm_df
    except Exception as e:
        logger.error(f"load_data_for_segmentation failed: {e}")
        raise


def _ensure_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure RFM score columns exist. Calculate them if missing.
    
    Uses percentile rank-based scoring (1-5 scale) which is robust to 
    duplicate values and edge cases.
    
    Args:
        rfm_df: RFM DataFrame
        
    Returns:
        RFM DataFrame with score columns
    """
    score_cols = ['recency_score', 'frequency_score', 'monetary_score']
    missing_scores = [col for col in score_cols if col not in rfm_df.columns]
    
    if not missing_scores:
        logger.info("RFM scores already present")
        return rfm_df
    
    logger.info(f"Missing RFM score columns: {missing_scores}")
    logger.info("Auto-calculating scores using percentile rank method (1-5 scale)")
    logger.info("Consider generating these scores in the RFM creation step (Notebook 01)")
    
    # Calculate missing scores using robust rank-based method
    if 'recency_score' not in rfm_df.columns:
        if 'recency_days' in rfm_df.columns:
            # Recency: Lower is better, so we use ascending=False
            rfm_df['recency_score'] = _score_rfm_metric(
                rfm_df['recency_days'], 
                ascending=False
            )
            logger.info("Calculated recency_score (percentile ranks, inverted)")
        else:
            raise ValueError("Cannot calculate recency_score: 'recency_days' column missing")
    
    if 'frequency_score' not in rfm_df.columns:
        if 'frequency' in rfm_df.columns:
            # Frequency: Higher is better
            rfm_df['frequency_score'] = _score_rfm_metric(
                rfm_df['frequency'], 
                ascending=True
            )
            logger.info("Calculated frequency_score (percentile ranks)")
        else:
            raise ValueError("Cannot calculate frequency_score: 'frequency' column missing")
    
    if 'monetary_score' not in rfm_df.columns:
        if 'monetary' in rfm_df.columns:
            # Monetary: Higher is better
            rfm_df['monetary_score'] = _score_rfm_metric(
                rfm_df['monetary'], 
                ascending=True
            )
            logger.info("Calculated monetary_score (percentile ranks)")
        else:
            raise ValueError("Cannot calculate monetary_score: 'monetary' column missing")
    
    return rfm_df


def _score_rfm_metric(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Score an RFM metric using percentile ranks.
    
    This method is robust to duplicates, NaN values, and edge cases. 
    It converts percentile ranks (0-1) to scores (1-5).
    
    Args:
        series: The metric to score
        ascending: True if higher values should get higher scores,
                  False if lower values should get higher scores (e.g., recency)
    
    Returns:
        Integer scores from 1 to 5
    """
    # Check for and handle non-finite values (NaN, inf, -inf)
    non_finite_count = (~np.isfinite(series)).sum()
    if non_finite_count > 0:
        logger.warning(
            f"Found {non_finite_count} non-finite values in {series.name}. "
            f"Imputing with median."
        )
        # Create a copy to avoid modifying original
        series = series.copy()
        # Replace inf/-inf with NaN first
        series = series.replace([np.inf, -np.inf], np.nan)
        # Fill NaN with median of finite values
        median_val = series.median()
        series = series.fillna(median_val)
    
    # Get percentile rank (0-1 scale)
    # The pct=True parameter handles ties and returns values in [0, 1]
    pct_rank = series.rank(method='average', ascending=ascending, pct=True)
    
    # Convert to 1-5 score using ceiling function
    # This ensures values are distributed across all 5 scores
    # Use numpy arrays directly to avoid pandas conversion issues
    score_values = np.ceil(pct_rank.values * 5)
    
    # Clip to ensure bounds [1, 5] and convert to int
    score_values = np.clip(score_values, 1, 5).astype(int)
    
    # Return as pandas Series with original index
    return pd.Series(score_values, index=series.index)


def _validate_segmentation_data(
    df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    config: dict
) -> None:
    """
    Validate data integrity and required columns.

    Args:
        df: Transaction DataFrame
        rfm_df: RFM DataFrame
        config: Configuration dictionary

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating data integrity...")

    # Check row count
    expected_rows = config['validation']['expected_row_count']
    if len(df) != expected_rows:
        logger.warning(f"Row count mismatch: expected {expected_rows}, got {len(df)}")

    # Check RFM columns
    required_rfm_cols = [
        'customer_id', 'recency_days', 'frequency', 'monetary',
        'recency_score', 'frequency_score', 'monetary_score',
        'loyalty_score'
    ]

    missing_cols = [col for col in required_rfm_cols if col not in rfm_df.columns]
    if missing_cols:
        raise ValueError(f"Missing RFM columns: {missing_cols}")

    # Validate score ranges (should be 1-5)
    for score_col in ['recency_score', 'frequency_score', 'monetary_score']:
        min_score = rfm_df[score_col].min()
        max_score = rfm_df[score_col].max()
        if min_score < 1 or max_score > 5:
            logger.warning(
                f"{score_col} outside expected range [1-5]: "
                f"min={min_score}, max={max_score}"
            )

    logger.info("[OK] Data validation passed")


__all__ = [
    'load_data_for_segmentation',
]

if __name__ == "__main__":
    try:
        df, rfm_df = load_data_for_segmentation()
        print(f"[OK] Loaded {len(df):,} transactions and {len(rfm_df):,} customers")
        
        # Display score distributions
        print("\n=== RFM Score Distributions ===")
        for col in ['recency_score', 'frequency_score', 'monetary_score']:
            print(f"\n{col}:")
            print(rfm_df[col].value_counts().sort_index())
            
    except Exception as e:
        print(f"[ERROR] {e}")
        raise