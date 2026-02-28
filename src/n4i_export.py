# src/n4i_export.py
"""
n4i_export.py - Export Business Deliverables

Export predictions, insights, and model artifacts with proper UTF-8 encoding.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from n4a_utils import setup_logger, load_config

logger = setup_logger(__name__)


def write_text_utf8(path: Path, text: str) -> None:
    """Write text to file with UTF-8 encoding."""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def clean_for_json(obj: Any) -> Any:
    """Clean object for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    else:
        return obj


def export_predictions(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    filename: str = 'churn_predictions.csv'
) -> Path:
    """
    Export customer predictions to CSV.
    
    Args:
        predictions_df: DataFrame with predictions
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    try:
        logger.info("Exporting predictions")
    
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        path = output_dir / filename
        predictions_df.to_csv(path, index=False, encoding='utf-8')
    
        logger.info(f"  Saved: {path.name}")
        logger.info(f"  Total customers: {len(predictions_df):,}")
    
        if 'predicted_churn' in predictions_df.columns:
            churn_count = predictions_df['predicted_churn'].sum()
            logger.info(f"  Predicted churn: {churn_count:,} ({churn_count/len(predictions_df):.1%})")
    
        return path
    except Exception as e:
        logger.error(f"export_predictions failed: {e}")
        raise


def export_business_deliverables(
    importance_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    strategies: Optional[str],
    campaigns: Optional[str],
    output_dir: Path
) -> Dict[str, Path]:
    """
    Export business deliverables.
    
    Args:
        importance_df: Feature importance DataFrame
        risk_df: Risk stratification DataFrame
        strategies: Retention strategies text
        campaigns: Campaign recommendations text
        output_dir: Output directory
        
    Returns:
        Dictionary of saved file paths
    """
    try:
        logger.info("Exporting business deliverables")
    
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        paths = {}
    
        # Feature importance
        if importance_df is not None and not importance_df.empty:
            path = output_dir / 'feature_importance.csv'
            importance_df.to_csv(path, index=False, encoding='utf-8')
            paths['importance'] = path
            logger.info(f"  Saved: {path.name}")
    
        # Risk segments
        if risk_df is not None and not risk_df.empty:
            path = output_dir / 'customer_risk_segments.csv'
            risk_df.to_csv(path, index=False, encoding='utf-8')
            paths['risk_segments'] = path
            logger.info(f"  Saved: {path.name}")
    
        # Retention strategies
        if strategies:
            path = output_dir / 'retention_strategies.txt'
            write_text_utf8(path, strategies)
            paths['strategies'] = path
            logger.info(f"  Saved: {path.name}")
    
        # Campaign recommendations
        if campaigns:
            path = output_dir / 'campaign_recommendations.txt'
            write_text_utf8(path, campaigns)
            paths['campaigns'] = path
            logger.info(f"  Saved: {path.name}")
    
        logger.info(f"Exported {len(paths)} deliverables")
    
        return paths
    except Exception as e:
        logger.error(f"export_business_deliverables failed: {e}")
        raise


def export_model_artifacts(
    model,
    scaler,
    imputer,
    feature_cols: List[str],
    output_dir: Path,
    model_performance: Optional[Dict[str, Any]] = None
) -> Dict[str, Path]:
    """
    Export model artifacts for deployment.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        imputer: Feature imputer
        feature_cols: List of feature column names
        output_dir: Output directory
        model_performance: Model performance metrics
        
    Returns:
        Dictionary of saved file paths
    """
    try:
        logger.info("Exporting model artifacts")
    
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        paths = {}
    
        # Save model
        if model is not None:
            path = output_dir / 'churn_model.pkl'
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            paths['model'] = path
            logger.info(f"  Saved: {path.name}")
    
        # Save scaler
        if scaler is not None:
            path = output_dir / 'scaler.pkl'
            with open(path, 'wb') as f:
                pickle.dump(scaler, f)
            paths['scaler'] = path
            logger.info(f"  Saved: {path.name}")
    
        # Save imputer
        if imputer is not None:
            path = output_dir / 'imputer.pkl'
            with open(path, 'wb') as f:
                pickle.dump(imputer, f)
            paths['imputer'] = path
            logger.info(f"  Saved: {path.name}")
    
        # Save feature columns
        if feature_cols:
            path = output_dir / 'feature_columns.txt'
            write_text_utf8(path, "\n".join(feature_cols))
            paths['features'] = path
            logger.info(f"  Saved: {path.name}")
    
        # Save model performance
        if model_performance:
            path = output_dir / 'model_performance.json'
        
            # Clean for JSON
            clean_metrics = {}
            for key in ['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                if key in model_performance:
                    clean_metrics[key] = clean_for_json(model_performance[key])
        
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(clean_metrics, f, indent=2)
        
            paths['performance'] = path
            logger.info(f"  Saved: {path.name}")
    
        logger.info(f"Exported {len(paths)} model artifacts")
    
        return paths
    except Exception as e:
        logger.error(f"export_model_artifacts failed: {e}")
        raise


__all__ = [
    'write_text_utf8',
    'clean_for_json',
    'export_predictions',
    'export_business_deliverables',
    'export_model_artifacts',
]

if __name__ == "__main__":
    print("Testing n4i_export module...")
    
    # Create sample data
    predictions_df = pd.DataFrame({
        'customer_id': range(100),
        'churn_probability': np.random.uniform(0, 1, 100),
        'predicted_churn': np.random.binomial(1, 0.3, 100),
        'risk_level': np.random.choice(['Low', 'Medium', 'High'], 100)
    })
    
    output_dir = Path('/tmp/test_export')
    
    print("\n1. Testing export_predictions...")
    export_predictions(predictions_df, output_dir)
    
    print("\n2. Testing export_business_deliverables...")
    importance_df = pd.DataFrame({
        'feature': ['f1', 'f2', 'f3'],
        'importance': [0.5, 0.3, 0.2]
    })
    
    export_business_deliverables(
        importance_df,
        predictions_df,
        "Test strategies",
        "Test campaigns",
        output_dir
    )
    
    print("\nAll tests passed!")