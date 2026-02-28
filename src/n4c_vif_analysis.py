# src/n4c_vif_analysis.py
"""
n4c_vif_analysis.py - Variance Inflation Factor (VIF) Analysis

Check for multicollinearity in feature sets using VIF.
VIF measures how much the variance of a coefficient is inflated due to
linear correlation with other features.

Rule of thumb:
- VIF < 5: Low multicollinearity (acceptable)
- VIF 5-10: Moderate multicollinearity (investigate)
- VIF > 10: High multicollinearity (problematic)
"""

import pandas as pd
import numpy as np
from typing import Tuple
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

from n4a_utils import setup_logger, load_config

logger = setup_logger(__name__)


def calculate_vif(X: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for each feature.
    
    Args:
        X: Feature DataFrame
        feature_cols: List of features to check (if None, uses all columns)
        
    Returns:
        DataFrame with features and their VIF scores, sorted by VIF descending
    """
    try:
        if feature_cols is None:
            feature_cols = X.columns.tolist()
    
        # Ensure we only use numeric columns
        X_numeric = X[feature_cols].select_dtypes(include=[np.number])
    
        # Remove any columns with zero variance (causes VIF=inf)
        zero_var_cols = X_numeric.columns[X_numeric.std() == 0].tolist()
        if zero_var_cols:
            logger.warning(f"Removing zero-variance columns: {zero_var_cols}")
            X_numeric = X_numeric.drop(columns=zero_var_cols)
    
        feature_cols_clean = X_numeric.columns.tolist()
    
        # Add a constant column so each auxiliary regression includes an intercept.
        # Standard VIF is defined as 1/(1-R²) where R² comes from regressing each
        # feature on all others WITH a constant. Without it, R² is biased downward
        # (model forced through origin), causing VIF to be underestimated and
        # genuinely multicollinear features to appear less correlated than they are.
        X_with_const = add_constant(X_numeric, has_constant='add')

        # Calculate VIF for each feature (index +1 to skip the constant column)
        vif_data = pd.DataFrame()
        vif_data["feature"] = feature_cols_clean
        vif_data["VIF"] = [
            variance_inflation_factor(X_with_const.values, i + 1)
            for i in range(len(feature_cols_clean))
        ]
    
        # Sort by VIF descending
        vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    
        return vif_data
    except Exception as e:
        logger.error(f"calculate_vif failed: {e}")
        raise


def check_multicollinearity(
    X: pd.DataFrame,
    feature_cols: list = None,
    threshold: float = 5.0
) -> Tuple[pd.DataFrame, bool]:
    """
    Check for multicollinearity and report findings.
    
    Args:
        X: Feature DataFrame
        feature_cols: List of features to check
        threshold: VIF threshold for flagging (default: 5.0)
        
    Returns:
        Tuple of (vif_dataframe, passes_check)
            - vif_dataframe: DataFrame with VIF scores
            - passes_check: True if all VIF < threshold
    """
    try:
        logger.info("=" * 70)
        logger.info("MULTICOLLINEARITY CHECK (VIF Analysis)")
        logger.info("=" * 70)
    
        vif_data = calculate_vif(X, feature_cols)
    
        logger.info(f"Variance Inflation Factor (threshold: {threshold}):")
        logger.info("-" * 70)
        for _, row in vif_data.iterrows():
            vif = row['VIF']
            feature = row['feature']
        
            if vif > 10:
                status = "HIGH (problematic)"
                logger.warning(f"  {feature:25s} VIF={vif:7.2f}  [{status}]")
            elif vif > threshold:
                status = "MODERATE (investigate)"
                logger.warning(f"  {feature:25s} VIF={vif:7.2f}  [{status}]")
            else:
                status = "LOW (acceptable)"
                logger.info(f"  {feature:25s} VIF={vif:7.2f}  [{status}]")
    
        # Check if any features exceed threshold
        high_vif = vif_data[vif_data['VIF'] > threshold]
        passes = len(high_vif) == 0
    
        logger.info("-" * 70)
        if passes:
            logger.info(f"✓ PASSED: All features have VIF < {threshold} (no multicollinearity)")
        else:
            logger.warning(f"✗ FAILED: {len(high_vif)} feature(s) have VIF > {threshold}")
            logger.warning("  Consider:")
            logger.warning("  1. Remove highly correlated features")
            logger.warning("  2. Use PCA/feature engineering")
            logger.warning("  3. Use regularized models (Ridge, Lasso)")
    
        logger.info("=" * 70)
    
        return vif_data, passes
    except Exception as e:
        logger.error(f"check_multicollinearity failed: {e}")
        raise


def compare_feature_sets_vif(
    X_train: pd.DataFrame,
    feature_sets: dict,
    threshold: float = 5.0
) -> pd.DataFrame:
    """
    Compare VIF scores across multiple feature sets.
    
    Args:
        X_train: Training data with all possible features
        feature_sets: Dict of {name: feature_list} to compare
        threshold: VIF threshold
        
    Returns:
        DataFrame summarizing VIF results for each feature set
    """
    try:
        logger.info("=" * 70)
        logger.info("COMPARING FEATURE SETS (VIF Analysis)")
        logger.info("=" * 70)
    
        results = []
    
        for name, features in feature_sets.items():
            logger.info(f"\n{name} feature set ({len(features)} features):")
        
            # Calculate VIF
            vif_data = calculate_vif(X_train, features)
            max_vif = vif_data['VIF'].max()
            mean_vif = vif_data['VIF'].mean()
            n_high = (vif_data['VIF'] > threshold).sum()
        
            # Log summary
            logger.info(f"  Max VIF:  {max_vif:.2f}")
            logger.info(f"  Mean VIF: {mean_vif:.2f}")
            logger.info(f"  Features with VIF > {threshold}: {n_high}")
        
            if n_high == 0:
                logger.info(f"  Status: ✓ PASS (no multicollinearity)")
            else:
                logger.warning(f"  Status: ✗ FAIL (multicollinearity detected)")
        
            results.append({
                'feature_set': name,
                'n_features': len(features),
                'max_vif': max_vif,
                'mean_vif': mean_vif,
                'n_high_vif': n_high,
                'passes': n_high == 0
            })
    
        comparison_df = pd.DataFrame(results)
    
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 70)
        logger.info("\n" + comparison_df.to_string(index=False))
        logger.info("=" * 70)
    
        return comparison_df
    except Exception as e:
        logger.error(f"compare_feature_sets_vif failed: {e}")
        raise


__all__ = [
    'calculate_vif',
    'check_multicollinearity',
    'compare_feature_sets_vif',
]

if __name__ == "__main__":
    print("Testing n4c_vif_analysis module...")
    
    # Create sample data with multicollinearity
    np.random.seed(42)
    n_samples = 1000
    
    # Create correlated features
    X1 = np.random.randn(n_samples)
    X2 = X1 + np.random.randn(n_samples) * 0.1  # Highly correlated with X1
    X3 = X1 + X2 + np.random.randn(n_samples) * 0.1  # Composite of X1, X2
    X4 = np.random.randn(n_samples)  # Independent
    X5 = np.random.randn(n_samples)  # Independent
    
    X = pd.DataFrame({
        'feature_1': X1,
        'feature_2': X2,
        'feature_3_composite': X3,
        'feature_4': X4,
        'feature_5': X5
    })
    
    print("\n1. Testing calculate_vif...")
    vif_data = calculate_vif(X)
    print(vif_data)
    
    print("\n2. Testing check_multicollinearity...")
    vif_data, passes = check_multicollinearity(X, threshold=5.0)
    
    print("\n3. Testing compare_feature_sets_vif...")
    feature_sets = {
        'with_composite': ['feature_1', 'feature_2', 'feature_3_composite', 'feature_4', 'feature_5'],
        'without_composite': ['feature_1', 'feature_2', 'feature_4', 'feature_5'],
        'minimal': ['feature_3_composite', 'feature_4', 'feature_5']
    }
    comparison = compare_feature_sets_vif(X, feature_sets)
    
    print("\nAll tests passed!")