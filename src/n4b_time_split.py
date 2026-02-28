"""
n4b_time_split.py - Time-Based Dataset Creation for Churn Prediction

This module implements the logic for creating a time-based train/test split for churn prediction modeling. It includes functions to:
- Load enhanced transaction data and pre-computed RFM features.
- Calculate a cutoff date to prevent data leakage.
- Create a train/test split with time-based churn labels.

- Exclude the `all_features` since it inflates VIF.

FEATURE EDITING GUIDE
---------------------
To change which features belong to a strategy, edit the module-level constants
below: BASE_FEATURES, COMPOSITE_FEATURES, ALL_FEATURES.

  - recency_days is ALWAYS prepended by create_time_based_dataset() regardless
    of strategy. Do NOT add it to any feature list here.
  - All features listed here must be computable by compute_cutoff_features()
    or already present in rfm_df from Notebook 01.

To add a new strategy:
  1. Define a new <n>_FEATURES list here.
  2. Add a branch in get_feature_set() below.
  3. Add its display-name → key mapping to strategy_map in the notebook cell.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from n4a_utils import setup_logger, load_config, get_project_root

logger = setup_logger(__name__)
_config = load_config()

# ==============================================================================
# FEATURE SET DEFINITIONS
# ==============================================================================
# Edit the lists below to change which features each strategy uses.
# recency_days is prepended automatically — do not add it here.
# All names must be columns produced by compute_cutoff_features() or rfm_df.

BASE_FEATURES = [
    'frequency',
    'monetary',
    'tenure_days',
    'category_diversity',
    'last_order_was_return',
    'return_rate',
    'discount_usage_rate',
]

COMPOSITE_FEATURES = [
    'loyalty_score',
    'tenure_days',
    'category_diversity',
    'last_order_was_return',
    'return_rate',
    'discount_usage_rate',
]

ALL_FEATURES = [
    'frequency',
    'monetary',
    'net_monetary',
    'avg_order_value',
    'tenure_days',
    'loyalty_score',
    'discount_usage_rate',
    'category_diversity',
    'return_rate',
    'last_order_was_return',
]

# Columns always attached to full_dataset for downstream business reporting,
# regardless of which feature strategy is selected.
# These are NOT model features — they are context columns only.
_CONTEXT_COLS = ['monetary']


def get_feature_set(strategy: str = 'base') -> List[str]:
    """
    Get feature list based on strategy.

    Returns a copy of the module-level constant for the given strategy.
    recency_days is NOT included here — it is prepended in create_time_based_dataset().

    Args:
        strategy: 'base', 'composite', or 'all'

    Returns:
        List of feature column names

    Raises:
        ValueError: If strategy is not recognised
    """
    try:
        if strategy == 'base':
            return BASE_FEATURES.copy()
        elif strategy == 'composite':
            return COMPOSITE_FEATURES.copy()
        elif strategy == 'all':
            return ALL_FEATURES.copy()
        else:
            raise ValueError(
                f"Unknown feature strategy: '{strategy}'. "
                f"Options: 'base', 'composite', 'all'"
            )
    except Exception as e:
        logger.error(f"get_feature_set failed: {e}")
        raise


def validate_feature_set(feature_cols: List[str], rfm_df: pd.DataFrame) -> List[str]:
    """Validate and filter feature set based on what's available in rfm_df."""
    try:
        available = [f for f in feature_cols if f in rfm_df.columns or f == 'recency_days']
        missing = [f for f in feature_cols if f not in rfm_df.columns and f != 'recency_days']

        if missing:
            logger.warning(f"Features not in rfm_df (will be skipped): {missing}")

        return available
    except Exception as e:
        logger.error(f"validate_feature_set failed: {e}")
        raise


def load_transaction_data(config: Dict) -> pd.DataFrame:
    """Load enhanced transaction data from Notebook 01."""
    try:
        logger.info("=" * 70)
        logger.info("LOADING TRANSACTION DATA")
        logger.info("=" * 70)

        project_root = get_project_root()
        enhanced_path = project_root / config['paths']['enhanced_df']

        if not enhanced_path.exists():
            raise FileNotFoundError(f"Enhanced data not found: {enhanced_path}")

        logger.info(f"Loading: {enhanced_path}")
        enhanced_df = pd.read_parquet(enhanced_path)

        date_col = _detect_date_col(enhanced_df)
        enhanced_df[date_col] = pd.to_datetime(enhanced_df[date_col])

        logger.info(f"Loaded: {len(enhanced_df):,} transactions")
        logger.info(f"Customers: {enhanced_df['customer_id'].nunique():,}")
        logger.info(f"Date range: {enhanced_df[date_col].min().date()} to {enhanced_df[date_col].max().date()}")
        logger.info("=" * 70)

        return enhanced_df
    except Exception as e:
        logger.error(f"load_transaction_data failed: {e}")
        raise


def load_rfm_data(config: Dict) -> pd.DataFrame:
    """Load pre-computed RFM data from Notebook 01."""
    try:
        logger.info("Loading RFM data from Notebook 01")

        project_root = get_project_root()
        processed_path = project_root / config['paths']['processed_data']
        rfm_path = processed_path / 'rfm_df.parquet'

        if not rfm_path.exists():
            raise FileNotFoundError(
                f"rfm_df.parquet not found: {rfm_path}\n"
                f"Run Notebook 01 first to generate RFM features."
            )

        rfm_df = pd.read_parquet(rfm_path)
        logger.info(f"Loaded rfm_df: {len(rfm_df):,} customers, {len(rfm_df.columns)} columns")

        return rfm_df
    except Exception as e:
        logger.error(f"load_rfm_data failed: {e}")
        raise


def _detect_date_col(df: pd.DataFrame) -> str:
    """Detect the date column name in a DataFrame."""
    for col in ['order_date', 'transaction_date', 'date', 'purchase_date', 'invoice_date']:
        if col in df.columns:
            return col
    raise ValueError(f"No date column found. Available: {list(df.columns)}")


def _detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column name that exists in df."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_cutoff_features(
    enhanced_df: pd.DataFrame,
    customer_ids,
    cutoff_date: pd.Timestamp,
    date_col: str,
    feature_lookback_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute all behavioral features using only pre-cutoff transactions.

    Args:
        enhanced_df: Full transaction DataFrame
        customer_ids: Customer IDs to compute features for
        cutoff_date: Upper bound — no transactions after this date are used
        date_col: Name of the date column
        feature_lookback_days: If set, only transactions within this many days
                               BEFORE the cutoff are used for feature computation.
                               Implements config.yaml → notebook4.churn.feature_lookback.
                               Example: 365 means only the last 365 days before cutoff.
                               None means all pre-cutoff history is used.

    FIXED: AttributeError resolved by converting Index to Series before .replace()
    """
    try:
        logger.info(f"Computing cutoff features at {cutoff_date.date()} (leakage prevention)")

        pre_cutoff = enhanced_df[enhanced_df[date_col] <= cutoff_date].copy()

        # Apply feature lookback window if configured.
        # This restricts behavioral features to recent history only,
        # preventing very old transactions from inflating frequency/monetary
        # in ways that don't reflect the customer's current behaviour.
        if feature_lookback_days is not None:
            lookback_start = cutoff_date - timedelta(days=feature_lookback_days)
            pre_cutoff = pre_cutoff[pre_cutoff[date_col] >= lookback_start]
            logger.info(
                f"Feature lookback applied: {feature_lookback_days}d "
                f"(transactions from {lookback_start.date()} to {cutoff_date.date()})"
            )

        result = pd.DataFrame({'customer_id': list(customer_ids)}).set_index('customer_id')

        # Frequency
        freq = pre_cutoff.groupby('customer_id').size()
        result['frequency'] = result.index.map(freq).fillna(0).astype(int)

        # Tenure — measured from first purchase to cutoff, not affected by lookback
        # (tenure reflects how long the customer has been with us, not recent activity)
        pre_cutoff_full = enhanced_df[enhanced_df[date_col] <= cutoff_date]
        first_purchase = pre_cutoff_full.groupby('customer_id')[date_col].min()
        result['tenure_days'] = (cutoff_date - result.index.map(first_purchase)).days

        # Monetary
        monetary_col = _detect_column(pre_cutoff, [
            'total_price', 'order_value', 'amount', 'revenue',
            'sales', 'price', 'net_amount', 'total_amount', 'value'
        ])
        if monetary_col:
            monetary = pre_cutoff.groupby('customer_id')[monetary_col].sum()
            result['monetary'] = result.index.map(monetary).fillna(0)
            logger.info(f"Monetary column: '{monetary_col}'")
        else:
            logger.warning("Monetary column not detected - 'monetary' will retain rfm_df value")
            result['monetary'] = np.nan

        # Category diversity
        cat_col = _detect_column(pre_cutoff, [
            'category', 'product_category', 'item_category',
            'department', 'product_type', 'category_name'
        ])
        if cat_col:
            diversity = pre_cutoff.groupby('customer_id')[cat_col].nunique()
            result['category_diversity'] = result.index.map(diversity).fillna(0).astype(int)
            logger.info(f"Category column: '{cat_col}'")
        else:
            logger.warning("Category column not detected - 'category_diversity' will retain rfm_df value")
            result['category_diversity'] = np.nan

        # Return features
        return_col = _detect_column(pre_cutoff, [
            'is_return', 'returned', 'return_flag', 'is_returned', 'return'
        ])
        if return_col:
            returns_per_cust = pre_cutoff.groupby('customer_id')[return_col].sum()
            total_per_cust = pre_cutoff.groupby('customer_id').size()

            # Convert Index to Series before replace()
            total_mapped = pd.Series(
                result.index.map(total_per_cust).fillna(0),
                index=result.index
            )
            result['return_rate'] = (
                result.index.map(returns_per_cust).fillna(0)
                / total_mapped.replace(0, np.nan)
            ).fillna(0)

            last_order_return = (
                pre_cutoff.sort_values(date_col)
                .groupby('customer_id')[return_col]
                .last()
            )
            result['last_order_was_return'] = (
                result.index.map(last_order_return).fillna(0).astype(int)
            )
            logger.info(f"Return column:   '{return_col}'")
        else:
            logger.warning("Return column not detected - return features will retain rfm_df values")
            result['return_rate'] = np.nan
            result['last_order_was_return'] = np.nan

        # Discount usage rate
        discount_col = _detect_column(pre_cutoff, [
            'discount_amount', 'discount', 'has_discount',
            'discount_applied', 'is_discounted', 'coupon_used'
        ])
        if discount_col:
            discounted = (
                pre_cutoff[pre_cutoff[discount_col] > 0]
                .groupby('customer_id').size()
            )
            total_per_cust = pre_cutoff.groupby('customer_id').size()

            # Convert Index to Series before replace()
            total_mapped = pd.Series(
                result.index.map(total_per_cust).fillna(0),
                index=result.index
            )
            result['discount_usage_rate'] = (
                result.index.map(discounted).fillna(0)
                / total_mapped.replace(0, np.nan)
            ).fillna(0)
            logger.info(f"Discount column: '{discount_col}'")
        else:
            logger.warning("Discount column not detected - 'discount_usage_rate' will retain rfm_df value")
            result['discount_usage_rate'] = np.nan

        # Derived features
        if monetary_col and return_col:
            return_monetary = (
                pre_cutoff[pre_cutoff[return_col] == 1]
                .groupby('customer_id')[monetary_col]
                .sum()
            )
            result['net_monetary'] = (
                result['monetary'] - result.index.map(return_monetary).fillna(0)
            )
        elif not result['monetary'].isna().all():
            result['net_monetary'] = result['monetary']
        else:
            result['net_monetary'] = np.nan

        # avg_order_value: monetary / frequency
        freq_nonzero = result['frequency'].replace(0, np.nan)
        result['avg_order_value'] = result['monetary'] / freq_nonzero

        # Loyalty score
        if not result['monetary'].isna().all():
            recency_at_cutoff = (
                cutoff_date - pre_cutoff.groupby('customer_id')[date_col].max()
            ).dt.days
            recency_score = 1.0 / (result.index.map(recency_at_cutoff).fillna(999) + 1)

            rfs = pd.DataFrame({
                'r': recency_score,
                'f': result['frequency'],
                'm': result['monetary'].fillna(0)
            }, index=result.index)

            scaler_tmp = MinMaxScaler()
            rfs_scaled = pd.DataFrame(
                scaler_tmp.fit_transform(rfs),
                index=rfs.index,
                columns=['r', 'f', 'm']
            )
            result['loyalty_score'] = rfs_scaled.mean(axis=1)
            logger.info("loyalty_score: computed as normalised RFM composite at cutoff")
        else:
            logger.warning("loyalty_score cannot be computed without monetary - will retain rfm_df value")
            result['loyalty_score'] = np.nan

        logger.info(
            f"Cutoff features computed for {len(result):,} customers. "
            f"Features with fallback (NaN): "
            f"{[c for c in result.columns if result[c].isna().all()]}"
        )

        return result.reset_index()
    except Exception as e:
        logger.error(f"compute_cutoff_features failed: {e}")
        raise


def calculate_cutoff_date(
    max_date: pd.Timestamp,
    observation_window_days: int,
    config_cutoff: Optional[str] = None
) -> pd.Timestamp:
    """Calculate the cutoff date for time-based split."""
    try:
        if config_cutoff and config_cutoff.lower() not in ('null', 'none', ''):
            cutoff = pd.to_datetime(config_cutoff)
            actual_obs = (max_date - cutoff).days

            if actual_obs > observation_window_days * 2:
                logger.warning("=" * 70)
                logger.warning("CHURN LABEL WARNING")
                logger.warning("=" * 70)
                logger.warning(f"Fixed cutoff: {cutoff.date()}")
                logger.warning(f"Max date:     {max_date.date()}")
                logger.warning(f"Gap:          {actual_obs} days")
                logger.warning(f"Obs window:   {observation_window_days} days")
                logger.warning("Set cutoff_date: null in config.yaml to fix this.")
                logger.warning("=" * 70)

            logger.info(f"Using config cutoff date: {cutoff.date()} (obs window: {actual_obs}d)")
            return cutoff

        cutoff = max_date - timedelta(days=observation_window_days)
        logger.info(f"Auto-calculated cutoff: {cutoff.date()} ({observation_window_days}d before max)")
        return cutoff
    except Exception as e:
        logger.error(f"calculate_cutoff_date failed: {e}")
        raise


def recalculate_recency_at_cutoff(
    rfm_df: pd.DataFrame,
    enhanced_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    date_col: str = 'order_date'
) -> pd.DataFrame:
    """Recalculate recency_days at the cutoff date."""
    try:
        logger.info(f"Recalculating recency at cutoff: {cutoff_date.date()}")

        pre_cutoff = enhanced_df[enhanced_df[date_col] <= cutoff_date].copy()
        last_purchase = pre_cutoff.groupby('customer_id')[date_col].max()
        recency_days = (cutoff_date - last_purchase).dt.days

        rfm_df = rfm_df.copy()
        rfm_df['recency_days'] = rfm_df['customer_id'].map(recency_days)

        logger.info(f"Recency recalculated: mean={rfm_df['recency_days'].mean():.1f}d")

        return rfm_df
    except Exception as e:
        logger.error(f"recalculate_recency_at_cutoff failed: {e}")
        raise


def define_churn_target(
    customer_ids: pd.Series,
    enhanced_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    max_date: pd.Timestamp,
    date_col: str = 'order_date'
) -> pd.Series:
    """Define churn target based on observation window."""
    try:
        logger.info(f"Defining churn target from {cutoff_date.date()} to {max_date.date()}")

        obs_window = enhanced_df[
            (enhanced_df[date_col] > cutoff_date) &
            (enhanced_df[date_col] <= max_date)
        ]
        active_customers = set(obs_window['customer_id'].unique())

        churn = customer_ids.map(lambda cid: 0 if cid in active_customers else 1)

        logger.info(f"Churn rate: {churn.mean():.1%} ({churn.sum():,} / {len(churn):,})")

        return churn
    except Exception as e:
        logger.error(f"define_churn_target failed: {e}")
        raise


def create_time_based_dataset(
    enhanced_df: pd.DataFrame,
    config: Dict,
    feature_strategy: str = 'base'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Create train/test split with time-based churn labels.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
        Feature matrices and target vectors for modeling.
        These are NOT yet scaled — pass through n4c_feature_engineering.prepare_features().

    metadata : dict
        Keys include:
          'feature_cols'    : list of feature column names selected for this strategy
          'feature_strategy': strategy key used
          'full_dataset'    : customer-level DataFrame with customer_id, churn,
                              all feature_cols, AND any _CONTEXT_COLS (e.g. monetary)
                              regardless of strategy — used for downstream reporting.
          'cutoff_date'     : pd.Timestamp
          'observation_window_days', 'churn_rate', 'n_customers', 'max_date'
    """
    try:
        logger.info("=" * 70)
        logger.info("CREATING TIME-BASED DATASET")
        logger.info("=" * 70)

        nb_config = config['notebook4']
        obs_window = nb_config['observation_window_days']
        test_size = nb_config.get('test_size', 0.2)
        random_state = nb_config.get('random_state', 42)

        # cutoff_date lives under notebook4.churn.cutoff_date in config.yaml
        cutoff_config = nb_config.get('churn', {}).get('cutoff_date', None)

        # Read churn sub-config values that are now implemented
        # feature_lookback: restricts behavioral features to only the last N days before cutoff
        # min_orders: excludes customers with fewer than N pre-cutoff transactions
        feature_lookback = nb_config.get('churn', {}).get('feature_lookback', None)
        min_orders = nb_config.get('churn', {}).get('min_orders', 1)

        if feature_lookback is not None:
            logger.info(f"Feature lookback: {feature_lookback} days before cutoff")
        if min_orders > 1:
            logger.info(f"Minimum pre-cutoff orders required: {min_orders}")

        date_col = _detect_date_col(enhanced_df)
        enhanced_df = enhanced_df.copy()
        enhanced_df[date_col] = pd.to_datetime(enhanced_df[date_col])

        max_date = enhanced_df[date_col].max()
        logger.info(f"Data range: {enhanced_df[date_col].min().date()} to {max_date.date()}")
        logger.info(f"Observation window: {obs_window} days")

        # Calculate cutoff
        cutoff_date = calculate_cutoff_date(max_date, obs_window, cutoff_config)

        logger.info(f"Cutoff: {cutoff_date.date()}")
        logger.info(f"Feature window: up to {cutoff_date.date()}")
        logger.info(f"Churn observation: {cutoff_date.date()} to {max_date.date()}")

        # Load rfm_df features
        rfm_df = load_rfm_data(config)

        # Recalculate recency at cutoff date
        rfm_df = recalculate_recency_at_cutoff(rfm_df, enhanced_df, cutoff_date, date_col)

        # Recompute all behavioral features at cutoff (leakage prevention)
        logger.info("Replacing rfm_df behavioral features with cutoff-computed values")

        cutoff_feats = compute_cutoff_features(
            enhanced_df,
            rfm_df['customer_id'],
            cutoff_date,
            date_col,
            feature_lookback_days=feature_lookback  # pass through from config
        )
        cutoff_feats_idx = cutoff_feats.set_index('customer_id')

        replaceable = [
            'frequency', 'monetary', 'tenure_days', 'category_diversity',
            'return_rate', 'last_order_was_return', 'discount_usage_rate',
            'net_monetary', 'avg_order_value', 'loyalty_score'
        ]

        replaced, skipped = [], []
        for feat in replaceable:
            if feat not in cutoff_feats_idx.columns:
                continue
            col_vals = cutoff_feats_idx[feat]
            if col_vals.isna().all():
                skipped.append(feat)
                continue
            rfm_df[feat] = rfm_df['customer_id'].map(col_vals)
            replaced.append(feat)

        logger.info(f"Replaced with cutoff-computed values: {replaced}")
        if skipped:
            logger.warning(
                f"Column detection failed for: {skipped}. "
                f"These features retain rfm_df values — check enhanced_df column names."
            )

        # Keep only customers with pre-cutoff history.
        # Uses full pre-cutoff history (not lookback-restricted) so that customers
        # who purchased long ago are not incorrectly excluded.
        pre_cutoff_full = enhanced_df[enhanced_df[date_col] <= cutoff_date]
        customers_with_history = set(pre_cutoff_full['customer_id'].unique())
        all_customer_ids = set(rfm_df['customer_id'])
        dropped_no_history = len(all_customer_ids - customers_with_history)

        rfm_df = rfm_df[rfm_df['customer_id'].isin(customers_with_history)].copy()

        logger.info(f"Customers with pre-cutoff history: {len(rfm_df):,}")
        if dropped_no_history > 0:
            logger.info(f"Dropped {dropped_no_history:,} customers with no pre-cutoff transactions")

        # Apply min_orders filter — now reads from config and is actually enforced.
        # config.yaml: notebook4.churn.min_orders (default 1)
        # Customers with fewer than min_orders pre-cutoff transactions are excluded
        # because their behavioral features are too sparse to be meaningful.
        if min_orders > 1:
            order_counts = pre_cutoff_full.groupby('customer_id').size()
            customers_meeting_min = set(order_counts[order_counts >= min_orders].index)
            customers_before = len(rfm_df)
            rfm_df = rfm_df[rfm_df['customer_id'].isin(customers_meeting_min)].copy()
            dropped_min_orders = customers_before - len(rfm_df)
            if dropped_min_orders > 0:
                logger.info(
                    f"Dropped {dropped_min_orders:,} customers with fewer than "
                    f"{min_orders} pre-cutoff orders (min_orders filter)"
                )
            logger.info(f"Customers after min_orders={min_orders} filter: {len(rfm_df):,}")

        # Define churn target
        churn = define_churn_target(rfm_df['customer_id'], enhanced_df, cutoff_date, max_date, date_col)
        rfm_df['churn'] = churn.values

        # Select features based on strategy
        logger.info(f"Feature selection strategy: '{feature_strategy}'")
        requested_features = get_feature_set(feature_strategy)
        available_features = validate_feature_set(requested_features, rfm_df)

        feature_cols = ['recency_days'] + available_features
        feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate, preserve order

        logger.info(f"Selected {len(feature_cols)} features:")
        for i, feat in enumerate(feature_cols, 1):
            logger.info(f"{i:2d}. {feat}")

        X = rfm_df[feature_cols].copy()
        y = rfm_df['churn']

        # Create full dataset — features stored raw (NaN values retained).
        # generate_predictions() applies the train-only imputer at inference time
        # so that imputation statistics come exclusively from training data.
        context_cols = [
            c for c in _CONTEXT_COLS
            if c in rfm_df.columns and c not in feature_cols
        ]

        full_dataset_cols = ['customer_id', 'churn'] + feature_cols + context_cols
        full_dataset = rfm_df[full_dataset_cols].copy()
        # Do NOT impute here — full_dataset intentionally retains raw (possibly NaN)
        # feature values. Imputation is deferred to generate_predictions() where
        # the train-only imputer is applied, ensuring no all-data statistics leak
        # into inference.

        # Train/test split (raw X — scaling done by n4c_feature_engineering)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # ── Fix loyalty_score leakage ─────────────────────────────────────────
        # compute_cutoff_features() fitted its internal MinMaxScaler on all
        # customers (train + test combined), because the split hadn't happened
        # yet at that point.  We recompute loyalty_score here, after the split,
        # using a MinMaxScaler fitted on X_train customers only.  This replaces
        # the all-data loyalty_score values with train-only normalised ones,
        # eliminating the leakage before any model sees the data.
        if 'loyalty_score' in feature_cols:
            # Build R/F/M components aligned on rfm_df's integer row index
            # (same index as X, X_train, X_test — NOT on customer_id strings).
            rfs_all = pd.DataFrame({
                'r': 1.0 / (rfm_df['recency_days'] + 1),  # higher recency_days -> lower score
                'f': rfm_df['frequency'].values,
                'm': rfm_df['monetary'].fillna(0).values,
            }, index=rfm_df.index)

            # Fit on train customers only — X_train.index is a subset of rfm_df.index
            _loyalty_scaler = MinMaxScaler()
            _loyalty_scaler.fit(rfs_all.loc[X_train.index])

            # Transform all customers using train-only statistics
            rfs_scaled_all = pd.DataFrame(
                _loyalty_scaler.transform(rfs_all),
                index=rfs_all.index,
                columns=['r', 'f', 'm']
            )
            loyalty_score_corrected = rfs_scaled_all.mean(axis=1)  # index = rfm_df.index

            # Write back — all share rfm_df's integer row index so alignment is exact
            X.loc[:, 'loyalty_score']       = loyalty_score_corrected
            X_train.loc[:, 'loyalty_score'] = loyalty_score_corrected.loc[X_train.index]
            X_test.loc[:, 'loyalty_score']  = loyalty_score_corrected.loc[X_test.index]
            full_dataset.loc[:, 'loyalty_score'] = loyalty_score_corrected
            logger.info(
                "loyalty_score recomputed with train-only MinMaxScaler "
                "(replaces all-data normalisation from compute_cutoff_features)"
            )

        logger.info(f"Train set: {len(X_train):,} ({1 - test_size:.0%})")
        logger.info(f"Test set:  {len(X_test):,}  ({test_size:.0%})")
        logger.info(f"Train churn rate: {y_train.mean():.1%}")
        logger.info(f"Test churn rate:  {y_test.mean():.1%}")

        metadata = {
            'cutoff_date': cutoff_date,
            'max_date': max_date,
            'observation_window_days': obs_window,
            'feature_lookback_days': feature_lookback,
            'min_orders': min_orders,
            'churn_rate': y.mean(),
            'feature_cols': feature_cols,
            'feature_strategy': feature_strategy,
            'full_dataset': full_dataset,
            'churn_method': 'binary',
            'n_customers': len(rfm_df),
        }

        logger.info("=" * 70)
        logger.info("Time-based dataset created successfully")
        logger.info(f"Customers: {len(rfm_df):,}")
        logger.info(f"Features:  {len(feature_cols)} ({feature_strategy} strategy)")
        logger.info(f"Churn rate: {y.mean():.1%}")
        logger.info("=" * 70)

        return X_train, X_test, y_train, y_test, metadata
    except Exception as e:
        logger.error(f"create_time_based_dataset failed: {e}")
        raise


def validate_temporal_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metadata: Dict,
    config: Dict
) -> bool:
    """Validate the temporal split for correctness and data quality."""
    try:
        logger.info("=" * 70)
        logger.info("VALIDATING TEMPORAL SPLIT")
        logger.info("=" * 70)

        checks_passed = 0
        total_checks = 5

        # Check 1: No NaN in features after imputation
        train_nan = X_train.isnull().sum().sum()
        test_nan = X_test.isnull().sum().sum()
        if train_nan == 0 and test_nan == 0:
            logger.info("Check 1: No NaN values in features - PASSED")
            checks_passed += 1
        else:
            logger.warning(f"Check 1: NaN values (train={train_nan}, test={test_nan}) - FAILED")

        # Check 2: Churn rate is realistic
        train_churn = y_train.mean()
        test_churn = y_test.mean()
        if 0.05 <= train_churn <= 0.50:
            logger.info(f"Check 2: Churn rate realistic - train={train_churn:.1%}, test={test_churn:.1%} - PASSED")
            checks_passed += 1
        else:
            status = "TOO HIGH" if train_churn > 0.50 else "TOO LOW"
            logger.warning(f"Check 2: Churn rate {status} ({train_churn:.1%}) - FAILED")

        # Check 3: Stratification maintained
        churn_diff = abs(train_churn - test_churn)
        if churn_diff < 0.02:
            logger.info(f"Check 3: Stratification maintained (diff={churn_diff:.1%}) - PASSED")
            checks_passed += 1
        else:
            logger.warning(f"Check 3: Stratification issue (diff={churn_diff:.1%}) - FAILED")

        # Check 4: Observation window fully within data
        obs_window = metadata['observation_window_days']
        days_available = (metadata['max_date'] - metadata['cutoff_date']).days
        if days_available >= obs_window:
            logger.info(f"Check 4: Observation window fully observable ({obs_window}d <= {days_available}d) - PASSED")
            checks_passed += 1
        else:
            logger.error(f"Check 4: Insufficient observation window ({days_available}d < {obs_window}d) - FAILED")

        # Check 5: Feature columns present and consistent
        if len(metadata['feature_cols']) >= 5:
            logger.info(f"Check 5: Feature set adequate ({len(metadata['feature_cols'])} features) - PASSED")
            checks_passed += 1
        else:
            logger.warning(f"Check 5: Too few features ({len(metadata['feature_cols'])}) - FAILED")

        logger.info(f"Validation: {checks_passed}/{total_checks} checks passed")
        logger.info("=" * 70)

        if checks_passed == total_checks:
            logger.info("VALIDATION PASSED - Dataset ready for modeling")
            return True
        elif checks_passed >= total_checks - 1:
            logger.warning("VALIDATION PASSED WITH WARNINGS - Review issues above")
            return True
        else:
            logger.error("VALIDATION FAILED - Review issues above before proceeding")
            return False
    except Exception as e:
        logger.error(f"validate_temporal_split failed: {e}")
        raise


__all__ = [
    'BASE_FEATURES',
    'COMPOSITE_FEATURES',
    'ALL_FEATURES',
    'get_feature_set',
    'validate_feature_set',
    'load_transaction_data',
    'load_rfm_data',
    'compute_cutoff_features',
    'calculate_cutoff_date',
    'recalculate_recency_at_cutoff',
    'define_churn_target',
    'create_time_based_dataset',
    'validate_temporal_split',
]