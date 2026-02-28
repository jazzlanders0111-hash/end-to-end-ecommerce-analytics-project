# src/n4f_feature_importance.py
"""
n4f_feature_importance.py - Feature Importance Analysis

Extract and visualize feature importance from trained models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Any

from n4a_utils import setup_logger, load_config

logger = setup_logger(__name__)


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Supports tree-based (RandomForest, XGBoost) and linear (LogisticRegression) models.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        DataFrame with features and importance scores
    """
    try:
        logger.info(f"Extracting feature importance from {model_name}")
    
        # Check model type and extract importance
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
            logger.info("Using feature_importances_ (tree-based model)")
        
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0])
            logger.info("Using coefficient absolute values (linear model)")
        
        else:
            raise ValueError(f"Model {model_name} does not support feature importance")
    
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
    
        # Normalize to percentages — guard against degenerate model where no
        # feature contributed (all importances == 0), which would cause division
        # by zero and produce inf/NaN values.
        importance_sum = importance_df['importance'].sum()
        if importance_sum == 0:
            logger.warning(
                f"{model_name} has all-zero feature importances. "
                f"Model may be degenerate. Setting importance_pct to 0."
            )
            importance_df['importance_pct'] = 0.0
        else:
            importance_df['importance_pct'] = (
                importance_df['importance'] / importance_sum * 100
            )
    
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
        # Add rank
        importance_df['rank'] = range(1, len(importance_df) + 1)
    
        logger.info(f"Extracted importance for {len(importance_df)} features")
        logger.info(f"Top 3 features:")
        for idx, row in importance_df.head(3).iterrows():
            logger.info(f"{row['rank']}. {row['feature']}: {row['importance_pct']:.2f}%")
    
        return importance_df
    except Exception as e:
        logger.error(f"extract_feature_importance failed: {e}")
        raise


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20
) -> None:
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        importance_df: DataFrame with feature importance
        output_dir: Directory to save plot
        top_n: Number of top features to plot
    """
    try:
        logger.info(f"Plotting top {top_n} features")
    
        # Select top N features
        plot_data = importance_df.head(top_n).copy()
    
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 12))
    
        # Create color gradient
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(plot_data)))
    
        # Horizontal bar chart
        bars = ax.barh(
            range(len(plot_data)),
            plot_data['importance_pct'],
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )
    
        # Customize
        ax.set_yticks(range(len(plot_data)))
        ax.set_yticklabels(plot_data['feature'])
        ax.set_xlabel('Importance (%)', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
        # Add percentage labels
        for idx, (bar, pct) in enumerate(zip(bars, plot_data['importance_pct'])):
            width = bar.get_width()
            ax.text(
                width + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%',
                ha='left',
                va='center',
                fontsize=9
            )
    
        plt.tight_layout()
    
        output_path = Path(output_dir) / 'feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
        logger.info(f"Saved: {output_path.name}")
    except Exception as e:
        logger.error(f"plot_feature_importance failed: {e}")
        raise


__all__ = [
    'extract_feature_importance',
    'plot_feature_importance',
]

if __name__ == "__main__":
    print("Testing n4f_feature_importance module...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    feature_names = [f'feature_{i}' for i in range(20)]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    print("\n1. Testing extract_feature_importance...")
    importance_df = extract_feature_importance(model, feature_names, "Random Forest")
    print(f"   Extracted {len(importance_df)} features")
    
    print("\n2. Testing plot_feature_importance...")
    output_dir = Path('/tmp')
    plot_feature_importance(importance_df, output_dir, top_n=10)
    
    print("\nAll tests passed!")