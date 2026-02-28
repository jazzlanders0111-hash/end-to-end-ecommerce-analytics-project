# src/n4c_feature_engineering.py
"""
n4c_feature_engineering.py - Feature Engineering for Churn Prediction

Prepares features for modeling with proper scaling and imputation.
All parameters are configuration-driven from config.yaml.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from n4a_utils import setup_logger, load_config

logger = setup_logger(__name__)


def handle_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = 'median'
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[SimpleImputer]]:
    """
    Handle missing values using imputation.
    
    Args:
        X_train: Training features
        X_test: Test features
        strategy: Imputation strategy ('median', 'mean', 'most_frequent')
        
    Returns:
        Tuple of (X_train_imputed, X_test_imputed, imputer)
    """
    try:
        train_missing = X_train.isnull().sum().sum()
        test_missing = X_test.isnull().sum().sum()
    
        if train_missing == 0 and test_missing == 0:
            logger.info("No missing values found")
            return X_train.copy(), X_test.copy(), None
    
        logger.info(f"Missing values - Train: {train_missing}, Test: {test_missing}")
        logger.info(f"Using {strategy} imputation")
    
        imputer = SimpleImputer(strategy=strategy)
    
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
    
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    
        logger.info("Missing values imputed successfully")
    
        return X_train_imputed, X_test_imputed, imputer
    except Exception as e:
        logger.error(f"handle_missing_values failed: {e}")
        raise


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features for model training.
    
    Args:
        X_train: Training features
        X_test: Test features
        method: Scaling method ('standard' or 'minmax')
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    try:
        logger.info(f"Scaling features with {method} method")
    
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
    
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    
        logger.info("Features scaled successfully")
    
        return X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        logger.error(f"scale_features failed: {e}")
        raise


def prepare_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[SimpleImputer], Optional[StandardScaler]]:
    """
    Prepare features for modeling (imputation + scaling).
    
    Args:
        X_train: Training features
        X_test: Test features
        config: Configuration dictionary with feature_engineering section
        
    Returns:
        Tuple of (X_train_prepared, X_test_prepared, imputer, scaler)
    """
    try:
        logger.info("="*70)
        logger.info("PREPARING FEATURES FOR MODELING")
        logger.info("="*70)
    
        # Get config parameters
        fe_config = config['notebook4'].get('feature_engineering', {})
        impute_strategy = fe_config.get('imputation_strategy', 'median')
        scale_method = fe_config.get('scaling_method', 'standard')
    
        logger.info(f"Imputation strategy: {impute_strategy}")
        logger.info(f"Scaling method: {scale_method}")
    
        imputer = None
        scaler = None
    
        # Handle missing values
        X_train, X_test, imputer = handle_missing_values(
            X_train, X_test, impute_strategy
        )
    
        # Scale features
        X_train, X_test, scaler = scale_features(X_train, X_test, scale_method)
    
        logger.info("="*70)
        logger.info(f"Features prepared: {X_train.shape}")
        logger.info(f"Samples: {X_train.shape[0]:,}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info("="*70)
    
        return X_train, X_test, imputer, scaler
    except Exception as e:
        logger.error(f"prepare_features failed: {e}")
        raise


def get_feature_statistics(
    X: pd.DataFrame,
    y: pd.Series
) -> pd.DataFrame:
    """
    Calculate feature statistics for analysis.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        
    Returns:
        DataFrame with feature statistics
    """
    try:
        stats = pd.DataFrame({
            'mean': X.mean(),
            'std': X.std(),
            'min': X.min(),
            'max': X.max(),
            'missing': X.isnull().sum(),
            'missing_pct': X.isnull().sum() / len(X) * 100
        })
    
        # Calculate correlation with target
        stats['target_corr'] = X.apply(lambda col: col.corr(y))
    
        stats = stats.sort_values('target_corr', key=abs, ascending=False)
    
        return stats
    except Exception as e:
        logger.error(f"get_feature_statistics failed: {e}")
        raise


__all__ = [
    'handle_missing_values',
    'scale_features',
    'prepare_features',
    'get_feature_statistics',
]

if __name__ == "__main__":
    print("Testing n4c_feature_engineering module...")
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y, name='target')
    
    # Add some missing values
    X.iloc[::10, 0] = np.nan
    X.iloc[::15, 5] = np.nan
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Mock config
    config = {
        'notebook4': {
            'feature_engineering': {
                'imputation_strategy': 'median',
                'scaling_method': 'standard'
            }
        }
    }
    
    print("\n1. Testing prepare_features...")
    X_train_prep, X_test_prep, imp, scl = prepare_features(X_train, X_test, config)
    print(f"   Train shape: {X_train_prep.shape}")
    print(f"   Missing values: {X_train_prep.isnull().sum().sum()}")
    
    print("\n2. Testing get_feature_statistics...")
    stats = get_feature_statistics(X_train, y_train)
    print(f"   Stats shape: {stats.shape}")
    print("\n   Top 3 features by correlation:")
    print(stats.head(3)[['target_corr', 'mean']])
    
    print("\nAll tests passed!")