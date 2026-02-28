# src/n4g_risk_stratification.py
"""
n4g_risk_stratification.py - Risk Stratification for Churn Prediction

Segment customers by churn risk level for targeted retention campaigns.
Uses quantile-based stratification for balanced distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, List, Dict

from n4a_utils import setup_logger, load_config

logger = setup_logger(__name__)


def generate_predictions(
    model: Any,
    X: pd.DataFrame,
    original_df: pd.DataFrame,
    feature_cols: List[str],
    scaler=None,
    imputer=None,
) -> pd.DataFrame:
    """
    Generate churn predictions for all customers.

    Args:
        model: Trained classifier
        X: Raw (unimputed, unscaled) full-dataset features from metadata['full_dataset']
        original_df: Original DataFrame with customer info
        feature_cols: List of feature column names
        scaler: The StandardScaler fitted on X_train in n4c.prepare_features().
                REQUIRED for Logistic Regression — LR was trained on scaled data,
                so passing unscaled X produces wrong probabilities.
                Tree-based models (RF, XGBoost) are scale-invariant but passing
                the scaler is still recommended for pipeline consistency.
                Pass None only if you have already manually scaled X externally.
        imputer: The SimpleImputer fitted on X_train in n4c.prepare_features().
                 REQUIRED for consistency — ensures full-dataset predictions use
                 the same imputation statistics (train-only medians) as model
                 training. Pass None only if X has no missing values.

    Returns:
        DataFrame with predictions and probabilities
    """
    try:
        logger.info("Generating predictions for all customers")

        X_input = X[feature_cols].copy()

        # Apply train-only imputer.
        # full_dataset features are stored raw in metadata (NaN values retained).
        # Imputation is deferred to here so that the train-only imputer statistics
        # (fitted on X_train in n4c.prepare_features) are used for the full
        # dataset, matching the model's training distribution exactly.
        if imputer is not None:
            X_input = pd.DataFrame(
                imputer.transform(X_input),
                columns=feature_cols,
                index=X_input.index
            )
            logger.info("Train-only imputer applied to full dataset")
        else:
            missing = X_input.isnull().sum().sum()
            if missing > 0:
                logger.warning(
                    f"No imputer passed to generate_predictions() — "
                    f"{missing} missing values remain. "
                    f"Fix: pass imputer=imputer (the object returned by n4c.prepare_features())."
                )

        # Apply scaler after imputation.
        # CRITICAL for Logistic Regression: the model was trained on StandardScaler-
        # transformed data. Passing raw features produces systematically wrong
        # probabilities. RF and XGBoost are scale-invariant but we scale for
        # pipeline consistency.
        if scaler is not None:
            X_input = pd.DataFrame(
                scaler.transform(X_input),
                columns=feature_cols,
                index=X_input.index
            )
            logger.info("Scaler applied to full dataset")
        else:
            logger.warning(
                "No scaler passed to generate_predictions(). "
                "If best_model is Logistic Regression, churn probabilities will be WRONG. "
                "Fix: pass scaler=scaler (the object returned by n4c.prepare_features())."
            )

        y_pred = model.predict(X_input)
        y_pred_proba = model.predict_proba(X_input)[:, 1]
    
        base_cols = ['customer_id', 'churn']
        predictions_df = original_df[base_cols].copy()
    
        predictions_df['predicted_churn'] = y_pred
        predictions_df['churn_probability'] = y_pred_proba
    
        # Add RFM if available
        if 'monetary' in original_df.columns:
            predictions_df['monetary'] = original_df['monetary']
        if 'recency_days' in original_df.columns:
            predictions_df['recency_days'] = original_df['recency_days']
        if 'frequency' in original_df.columns:
            predictions_df['frequency'] = original_df['frequency']
    
        logger.info(f"Generated predictions for {len(predictions_df):,} customers")
        logger.info(f"Predicted churn rate: {y_pred.mean():.1%}")
        logger.info(f"Actual churn rate: {predictions_df['churn'].mean():.1%}")
    
        return predictions_df
    except Exception as e:
        logger.error(f"generate_predictions failed: {e}")
        raise


def stratify_risk(
    predictions_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Stratify customers into risk levels.
    
    Supports two methods:
    - 'quantile': Equal-sized buckets (recommended)
    - 'threshold': Fixed probability cutoffs
    
    Args:
        predictions_df: DataFrame with churn predictions
        config: Configuration dictionary
        
    Returns:
        DataFrame with risk_level column
    """
    try:
        df = predictions_df.copy()
    
        method = config['notebook4']['risk_stratification'].get('method', 'quantile')
    
        if method == 'quantile':
            low_q = config['notebook4']['risk_stratification']['quantiles']['low']
            high_q = config['notebook4']['risk_stratification']['quantiles']['high']
        
            low_threshold = df['churn_probability'].quantile(low_q)
            high_threshold = df['churn_probability'].quantile(high_q)
        
            logger.info("Stratifying customers by risk level")
            logger.info(f"Method: QUANTILE-BASED")
            logger.info(f"Low < {low_threshold:.3f} (p{low_q:.0%})")
            logger.info(f"High >= {high_threshold:.3f} (p{high_q:.0%})")
        else:
            low_threshold = config['notebook4']['risk_stratification']['thresholds']['low']
            high_threshold = config['notebook4']['risk_stratification']['thresholds']['high']
        
            logger.info("Stratifying customers by risk level")
            logger.info(f"Method: THRESHOLD-BASED")
            logger.info(f"Low < {low_threshold:.1%}, High >= {high_threshold:.1%}")
    
        # Guard against duplicate bin edges — can happen when churn_probability
        # is nearly uniform (e.g. AUC ~0.5 model), making p33 ≈ p67.
        # pd.cut raises ValueError: "Bin edges must be unique" in that case.
        # Fix: nudge thresholds apart by a small epsilon so bins are always distinct.
        if low_threshold >= high_threshold:
            logger.warning(
                f"low_threshold ({low_threshold:.4f}) >= high_threshold ({high_threshold:.4f}) "
                f"— churn probability distribution is very uniform. "
                f"Nudging thresholds apart to avoid pd.cut ValueError."
            )
            mid = (low_threshold + high_threshold) / 2
            eps = max(1e-6, mid * 1e-4)
            low_threshold  = mid - eps
            high_threshold = mid + eps

        df['risk_level'] = pd.cut(
            df['churn_probability'],
            bins=[-np.inf, low_threshold, high_threshold, np.inf],
            labels=['Low', 'Medium', 'High']
        )
    
        logger.info("Risk distribution:")
        for level in ['Low', 'Medium', 'High']:
            count = (df['risk_level'] == level).sum()
            pct = count / len(df) * 100
            logger.info(f"{level:8s} {count:6,} ({pct:5.1f}%)")
    
        return df
    except Exception as e:
        logger.error(f"stratify_risk failed: {e}")
        raise


def plot_risk_distribution(risk_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot risk level distribution."""
    try:
        logger.info("Plotting risk distribution")
    
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # 1. Risk level counts
        ax1 = axes[0, 0]
        risk_counts = risk_df['risk_level'].value_counts()
        colors = {'Low': '#2E7D32', 'Medium': '#F57C00', 'High': '#C62828'}
        bars = ax1.bar(
            risk_counts.index,
            risk_counts.values,
            color=[colors.get(x, 'gray') for x in risk_counts.index]
        )
        ax1.set_title('Customer Count by Risk Level', fontweight='bold')
        ax1.set_ylabel('Number of Customers')
        ax1.grid(True, alpha=0.3, axis='y')
    
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{int(height):,}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
    
        # 2. Churn probability distribution
        ax2 = axes[0, 1]
        ax2.hist(risk_df['churn_probability'], bins=50, color='steelblue', edgecolor='black')
        ax2.set_title('Churn Probability Distribution', fontweight='bold')
        ax2.set_xlabel('Churn Probability')
        ax2.set_ylabel('Number of Customers')
        ax2.grid(True, alpha=0.3)
    
        # 3. Revenue at risk
        ax3 = axes[1, 0]
        if 'monetary' in risk_df.columns:
            revenue_by_risk = risk_df.groupby('risk_level', observed=True)['monetary'].sum()
            bars = ax3.bar(
                revenue_by_risk.index,
                revenue_by_risk.values,
                color=[colors.get(x, 'gray') for x in revenue_by_risk.index]
            )
            ax3.set_title('Revenue at Risk by Level', fontweight='bold')
            ax3.set_ylabel('Total Revenue ($)')
            ax3.grid(True, alpha=0.3, axis='y')
        
            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'${height/1000:.0f}K',
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )
        else:
            # monetary not in risk_df — annotate rather than leaving a blank panel
            ax3.text(
                0.5, 0.5,
                'Revenue data\nnot available\n(monetary not in risk_df)',
                ha='center', va='center',
                transform=ax3.transAxes,
                fontsize=11, color='gray', style='italic'
            )
            ax3.set_title('Revenue at Risk by Level', fontweight='bold')
            ax3.axis('off')
            logger.warning(
                "plot_risk_distribution: 'monetary' not in risk_df — "
                "revenue panel skipped. Pass monetary in original_df to enable."
            )
    
        # 4. Actual vs Predicted churn
        ax4 = axes[1, 1]
        if 'churn' in risk_df.columns:
            risk_summary = risk_df.groupby('risk_level', observed=True).agg({
                'churn': 'mean',
                'churn_probability': 'mean'
            }).reset_index()
        
            x = np.arange(len(risk_summary))
            width = 0.35
        
            ax4.bar(x - width/2, risk_summary['churn'] * 100, width,
                    label='Actual Churn %', color='#C62828')
            ax4.bar(x + width/2, risk_summary['churn_probability'] * 100, width,
                    label='Predicted Churn %', color='#2E86AB')
        
            ax4.set_title('Actual vs Predicted Churn Rate', fontweight='bold')
            ax4.set_ylabel('Churn Rate (%)')
            ax4.set_xlabel('Risk Level')
            ax4.set_xticks(x)
            ax4.set_xticklabels(risk_summary['risk_level'])
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
    
        plt.tight_layout()
    
        output_path = Path(output_dir) / 'risk_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
        logger.info(f"Saved: {output_path.name}")
    except Exception as e:
        logger.error(f"plot_risk_distribution failed: {e}")
        raise


__all__ = [
    'generate_predictions',
    'stratify_risk',
    'plot_risk_distribution',
]

if __name__ == "__main__":
    print("Testing n4g_risk_stratification module...")
    
    np.random.seed(42)
    
    predictions_df = pd.DataFrame({
        'customer_id': range(1000),
        'churn': np.random.binomial(1, 0.3, 1000),
        'predicted_churn': np.random.binomial(1, 0.3, 1000),
        'churn_probability': np.random.beta(2, 5, 1000),
        'monetary': np.random.uniform(100, 5000, 1000),
        'recency_days': np.random.uniform(0, 300, 1000),
        'frequency': np.random.poisson(3, 1000)
    })
    
    config = {
        'notebook4': {
            'risk_stratification': {
                'method': 'quantile',
                'quantiles': {'low': 0.33, 'high': 0.67},
                'thresholds': {'low': 0.30, 'high': 0.70}
            }
        }
    }
    
    print("\n1. Testing stratify_risk...")
    risk_df = stratify_risk(predictions_df, config)
    print(f"   Risk levels: {risk_df['risk_level'].value_counts().to_dict()}")
    
    print("\n2. Testing plot_risk_distribution...")
    output_dir = Path('/tmp')
    plot_risk_distribution(risk_df, output_dir)
    
    print("\nAll tests passed!")