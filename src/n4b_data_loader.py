# src/n4b_data_loader.py
"""
n4b_data_loader.py — Data Loading (UNUSED IN NB04 — see note below)

STATUS: This module is NOT imported anywhere in NB04.
The notebook uses n4b_time_split.py instead.

WHY IT EXISTS:
This file was written alongside n4b_time_split.py during development.
Two data approaches were drafted; n4b_time_split won because it correctly
handles the time-based split from a single enhanced_df source.

KNOWN BUG (would fail if ever called):
prepare_modeling_dataset() tries to read segments_df['churn'], but
Notebook 03 exports customer_segments.csv with columns:
    customer_id, cluster, segment_name
There is no 'churn' column in that file. This would raise a KeyError.
The churn label must be derived from enhanced_df transaction timestamps
(no purchase after cutoff = churned), which is what n4b_time_split does.

SAFE TO DELETE — nothing imports from this module.
Kept here as documentation of the design decision and to prevent confusion
if someone tries to use it directly.

For the actual data loading logic, see: n4b_time_split.py
For NB03 segment data used in business insights (not modeling), see:
    n4h_business_insights.load_segment_names()

Still kept for reference, but not called from NB04.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from n4a_utils import setup_logger, get_project_root, load_config

logger = setup_logger(__name__)


def load_churn_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from all sources for churn prediction.

    NOTE: This function is not called from NB04. Use n4b_time_split.py.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (rfm_df, segments_df, enhanced_df)
    """
    try:
        logger.warning("n4b_data_loader.load_churn_data() is not used in NB04.")
        logger.warning("Use n4b_time_split.load_transaction_data() + create_time_based_dataset() instead.")

        logger.info("=" * 60)
        logger.info("Loading Data for Churn Prediction")
        logger.info("=" * 60)

        project_root = get_project_root()
        processed_path = project_root / config['paths']['processed_data']

        # Load RFM data
        rfm_path = processed_path / 'rfm_df.parquet'
        logger.info(f"Loading RFM data: {rfm_path}")
        rfm_df = pd.read_parquet(rfm_path)
        logger.info(f"Loaded {len(rfm_df):,} customers with {len(rfm_df.columns)} features")

        # Load customer segments from NB03
        # KNOWN BUG: segments_df does NOT have a 'churn' column.
        # NB03 exports: customer_id, cluster, segment_name
        # The 'churn' column must be derived from transactions (see n4b_time_split).
        segments_path = processed_path / 'customer_segments.csv'
        logger.info(f"Loading customer segments: {segments_path}")
        segments_df = pd.read_csv(segments_path)
        logger.info(f"Loaded {len(segments_df):,} customer segment assignments")
        logger.info(f"Columns in segments_df: {list(segments_df.columns)}")
        if 'churn' not in segments_df.columns:
            logger.error("KNOWN BUG: 'churn' column not found in segments_df.")
            logger.error("NB03 does not define churn. Use n4b_time_split for churn label derivation.")

        # Load enhanced transaction data
        enhanced_path = project_root / config['paths']['enhanced_df']
        logger.info(f"Loading enhanced transactions: {enhanced_path}")
        enhanced_df = pd.read_parquet(enhanced_path)
        logger.info(f"Loaded {len(enhanced_df):,} transactions")

        logger.info("=" * 60)

        return rfm_df, segments_df, enhanced_df
    except Exception as e:
        logger.error(f"load_churn_data failed: {e}")
        raise


def prepare_modeling_dataset(
    rfm_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    enhanced_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge data sources into a modeling dataset.

    NOTE: This function is not called from NB04 and contains a known bug.
    If called, it will raise KeyError because segments_df has no 'churn' column.

    The correct approach is in n4b_time_split.create_time_based_dataset().

    Args:
        rfm_df: RFM metrics
        segments_df: Customer segments from NB03 (no 'churn' column — BUG)
        enhanced_df: Transaction data

    Returns:
        Would return merged DataFrame, but will fail on churn column access.

    Raises:
        KeyError: segments_df['churn'] does not exist
    """
    try:
        logger.warning("prepare_modeling_dataset() is not used in NB04 and has a known bug.")
        logger.warning("segments_df has no 'churn' column — NB03 does not define churn.")
        raise NotImplementedError(
            "This function is disabled. Use n4b_time_split.create_time_based_dataset() instead.\n"
            "See module docstring for details."
        )
    except Exception as e:
        logger.error(f"prepare_modeling_dataset failed: {e}")
        raise

__all__ = [
    'load_churn_data',
    'prepare_modeling_dataset',
]
