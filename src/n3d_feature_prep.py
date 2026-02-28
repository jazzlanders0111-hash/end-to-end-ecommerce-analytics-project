"""
n3d_feature_prep.py - Feature Preparation for Clustering

Prepares and scales features for customer segmentation clustering.

FIXED: No winsorization - StandardScaler only for k=3 and k=4 optimal performance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

from n3a_utils import setup_logger, load_config

logger = setup_logger(__name__)


def prepare_clustering_features(
    rfm_df: pd.DataFrame,
    feature_list: List[str] = None,
    scaler_type: str = 'standard'
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Prepare and scale features for clustering.
    
    Uses StandardScaler only (no winsorization) as testing shows
    winsorization hurts performance at k=4 (-1.5% silhouette).
    
    CRITICAL: This function imputes missing values and returns the clean features
    in clust_df. Downstream modules (statistical tests, profiling) should use
    clust_df to access clean feature data instead of the original rfm_df.

    Args:
        rfm_df: RFM DataFrame with customer features
        feature_list: List of feature columns to use (default: config defaults)
        scaler_type: Type of scaler ('standard' only - robust removed)

    Returns:
        Tuple of (scaled_features, feature_names, clustering_dataframe_with_clean_features)
        - scaled_features: Standardized numpy array for clustering
        - feature_names: List of feature column names
        - clustering_dataframe: DataFrame with customer_id + imputed feature values

    Raises:
        ValueError: If required features are missing or imputation fails
    """
    # Logger removed - redundant with notebook section header
    try:
        config = load_config()
    
        # Use default features if not provided
        if feature_list is None:
            feature_list = config.get('notebook3', {}).get('rfm_analyzer', {}).get(
                'default_features',
                ['recency_days', 'frequency', 'monetary', 'avg_order_value', 'loyalty_score']
            )

        # Validate features
        missing_features = [f for f in feature_list if f not in rfm_df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        logger.info(f"Using {len(feature_list)} features: {feature_list}")
    
        # Verify we're using the correct 5 features
        expected_features = ['recency_days', 'frequency', 'monetary', 'avg_order_value', 'loyalty_score']
        if set(feature_list) != set(expected_features):
            logger.warning(f"Feature list differs from recommended: {expected_features}")

        # Extract features
        X = rfm_df[feature_list].copy()

        # Handle missing values BEFORE scaling
        nan_count_total = X.isnull().sum().sum()
        if nan_count_total > 0:
            logger.warning(f"Found {nan_count_total} missing values across all features")
            for col in feature_list:
                nan_count = X[col].isnull().sum()
                if nan_count > 0:
                    median_val = X[col].median()
                    # FIX: Avoid inplace on column - assign directly instead
                    X[col] = X[col].fillna(median_val)
                    logger.info(f"Imputed {nan_count} NaNs in '{col}' with median {median_val:.2f}")
    
        # Handle infinite values
        inf_count_total = np.isinf(X.values).sum()
        if inf_count_total > 0:
            logger.warning(f"Found {inf_count_total} infinite values, replacing with median")
            X = X.replace([np.inf, -np.inf], np.nan)
            for col in feature_list:
                nan_count = X[col].isnull().sum()
                if nan_count > 0:
                    median_val = X[col].median()
                    # FIX: Avoid inplace on column - assign directly instead
                    X[col] = X[col].fillna(median_val)
                    logger.info(f"Imputed {nan_count} Infs in '{col}' with median {median_val:.2f}")

        # Verify no NaNs or Infs remain after imputation
        remaining_nans = X.isnull().sum().sum()
        remaining_infs = np.isinf(X.values).sum()
    
        if remaining_nans > 0:
            raise ValueError(f"CRITICAL: {remaining_nans} NaN values remain after imputation!")
        if remaining_infs > 0:
            raise ValueError(f"CRITICAL: {remaining_infs} infinite values remain after imputation!")
    
        logger.info("Data quality checks passed: no NaNs or Infs in features")

        # Create clustering dataframe with customer IDs AND imputed features
        # This ensures downstream statistical tests receive clean data
        clust_df = pd.DataFrame({
            'customer_id': rfm_df['customer_id'].values
        })
    
        # Add imputed features to clust_df for downstream use
        for feature in feature_list:
            clust_df[feature] = X[feature].values
    
        logger.info(f"Created clust_df with {len(clust_df)} rows and {len(clust_df.columns)} columns")
        logger.info(f"Columns: {list(clust_df.columns)}")

        # Scale features (StandardScaler ONLY - no winsorization)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info("[OK] Features prepared and scaled")
        logger.info(f"Shape: {X_scaled.shape}")
        logger.info(f"Mean: {X_scaled.mean():.6f} (should be ~0)")
        logger.info(f"Std: {X_scaled.std():.6f} (should be ~1)")
        logger.info(f"NaN check: {np.isnan(X_scaled).sum()} (should be 0)")
        logger.info(f"Inf check: {np.isinf(X_scaled).sum()} (should be 0)")

        return X_scaled, feature_list, clust_df
    except Exception as e:
        logger.error(f"prepare_clustering_features failed: {e}")
        raise


def validate_feature_distribution(X: np.ndarray, feature_names: List[str]) -> dict:
    """
    Validate feature distributions for clustering assumptions.

    Args:
        X: Scaled feature matrix
        feature_names: List of feature names

    Returns:
        Dictionary of validation results
    """
    try:
        logger.info("Validating feature distributions...")

        validation_results = {}

        for i, feat in enumerate(feature_names):
            col_data = X[:, i]
            skewness = pd.Series(col_data).skew()
            kurtosis = pd.Series(col_data).kurtosis()

            validation_results[feat] = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'skewness': skewness,
                'kurtosis': kurtosis
            }

            logger.info(f"{feat}: mean={col_data.mean():.3f}, std={col_data.std():.3f}, skew={skewness:.3f}")

        # Check if standardization worked
        overall_mean = X.mean()
        overall_std = X.std()
    
        if abs(overall_mean) > 0.01:
            logger.warning(f"Standardization may have issues: overall mean = {overall_mean:.6f}")
        if abs(overall_std - 1.0) > 0.1:
            logger.warning(f"Standardization may have issues: overall std = {overall_std:.6f}")

        return validation_results
    except Exception as e:
        logger.error(f"validate_feature_distribution failed: {e}")
        raise


__all__ = [
    'prepare_clustering_features',
    'validate_feature_distribution',
]

if __name__ == "__main__":
    try:
        from n3b_data_loader import load_data_for_segmentation

        df, rfm_df = load_data_for_segmentation()

        # Use default 5 features
        feature_list = [
            'recency_days', 'frequency', 'monetary',
            'avg_order_value', 'loyalty_score'
        ]

        X_scaled, feature_names, clust_df = prepare_clustering_features(
            rfm_df, feature_list
        )

        print(f"\n[OK] Features prepared: {X_scaled.shape}")
        
        # Validate
        validation = validate_feature_distribution(X_scaled, feature_names)
        
        print(f"[OK] Feature validation test successful!")

    except Exception as e:
        print(f"[ERROR] {e}")
        raise