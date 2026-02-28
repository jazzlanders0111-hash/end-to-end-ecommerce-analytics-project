# src/n4d_model_training.py
"""
n4d_model_training.py - Model Training for Churn Prediction

Train multiple classification models with parameters from config.yaml.
All hyperparameters are configuration-driven.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from n4a_utils import setup_logger, load_config

logger = setup_logger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None
) -> LogisticRegression:
    """
    Train Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Model parameters from config
        
    Returns:
        Trained LogisticRegression model
    """
    try:
        logger.info("Training Logistic Regression")
    
        if params is None:
            params = {
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced',
                'solver': 'lbfgs'
            }
    
        logger.info(f"Parameters: {params}")
    
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
    
        train_score = model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
    
        return model
    except Exception as e:
        logger.error(f"train_logistic_regression failed: {e}")
        raise


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None
) -> RandomForestClassifier:
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Model parameters from config
        
    Returns:
        Trained RandomForestClassifier model
    """
    try:
        logger.info("Training Random Forest")
    
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            }
    
        logger.info(f"Parameters: {params}")
    
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
    
        train_score = model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
    
        return model
    except Exception as e:
        logger.error(f"train_random_forest failed: {e}")
        raise


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None
) -> XGBClassifier:
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Model parameters from config
        
    Returns:
        Trained XGBClassifier model
    """
    try:
        logger.info("Training XGBoost")
    
        # Calculate class weights
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
    
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
            }
        else:
            params = params.copy()
            params.pop('eval_metric', None)  # eval_metric has no effect without eval_set
    
        params['scale_pos_weight'] = float(scale_pos_weight)  # cast: avoid np.float64 in log
    
        logger.info(f"Parameters: {params}")
        logger.info(f"Class balance: {pos_count:,} positive / {neg_count:,} negative")
        logger.info(f"Scale pos weight: {float(scale_pos_weight):.2f}")
    
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
    
        train_score = model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
    
        return model
    except Exception as e:
        logger.error(f"train_xgboost failed: {e}")
        raise


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train all models with config parameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        config: Configuration dictionary from config['notebook4']['training']
        
    Returns:
        Dictionary of trained models
    """
    try:
        logger.info("="*70)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*70)
    
        models = {}
    
        # Get model configs
        if config is None:
            config = {}
    
        lr_params = config.get('logistic_regression', None)
        rf_params = config.get('random_forest', None)
        xgb_params = config.get('xgboost', None)
    
        # Train models
        models['Logistic Regression'] = train_logistic_regression(X_train, y_train, lr_params)
        logger.info("-"*70)
    
        models['Random Forest'] = train_random_forest(X_train, y_train, rf_params)
        logger.info("-"*70)
    
        models['XGBoost'] = train_xgboost(X_train, y_train, xgb_params)
    
        logger.info("="*70)
        logger.info(f"All {len(models)} models trained successfully")
        logger.info("="*70)
    
        return models
    except Exception as e:
        logger.error(f"train_all_models failed: {e}")
        raise


__all__ = [
    'train_logistic_regression',
    'train_random_forest',
    'train_xgboost',
    'train_all_models',
]

if __name__ == "__main__":
    print("Testing n4d_model_training module...")
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.5, 0.5],
        random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y, name='churn')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print("\n1. Testing train_all_models...")
    all_models = train_all_models(X_train, y_train)
    print(f"   Trained {len(all_models)} models: {list(all_models.keys())}")
    
    print("\nAll tests passed!")