# src/n4e_model_evaluation.py
"""
n4e_model_evaluation.py - Model Evaluation for Churn Prediction

Evaluate models using multiple metrics and create comparison visualizations.
All thresholds are configuration-driven from config.yaml.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score
)

from n4a_utils import setup_logger, load_config

logger = setup_logger(__name__)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluate a single model on test set.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        logger.info(f"Evaluating {model_name}")
    
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
        results = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
    
        logger.info(f"Accuracy:  {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall:    {results['recall']:.4f}")
        logger.info(f"F1-Score:  {results['f1']:.4f}")
        logger.info(f"ROC-AUC:   {results['roc_auc']:.4f}")
    
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        support_active  = tn + fp
        support_churned = fn + tp
        support_total   = len(y_test)

        def _prf(tp_, fp_, fn_):
            p = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
            r = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            return p, r, f

        p_act, r_act, f_act = _prf(tn, fn, fp)   # Active = negative class
        p_chu, r_chu, f_chu = _prf(tp, fp, fn)   # Churned = positive class

        header  = f"{'':20s}  {'precision':>10}  {'recall':>10}  {'f1-score':>10}  {'support':>10}"
        divider = "" + "-" * (len(header))
        logger.info(header)
        logger.info(divider)
        logger.info(f"{'Active':20s}  {p_act:10.2f}  {r_act:10.2f}  {f_act:10.2f}  {support_active:10d}")
        logger.info(f"{'Churned':20s}  {p_chu:10.2f}  {r_chu:10.2f}  {f_chu:10.2f}  {support_churned:10d}")
        logger.info(divider)
        logger.info(f"{'accuracy':20s}  {'':10}  {'':10}  {results['accuracy']:10.2f}  {support_total:10d}")
        logger.info(f"{'macro avg':20s}  {(p_act+p_chu)/2:10.2f}  {(r_act+r_chu)/2:10.2f}  {(f_act+f_chu)/2:10.2f}  {support_total:10d}")
        logger.info("*"*len(header))
        return results
    except Exception as e:
        logger.error(f"evaluate_model failed: {e}")
        raise


def compare_models(
    results_dict: Dict[str, Dict],
    config: Dict = None
) -> pd.DataFrame:
    """
    Compare multiple models and rank them.
    
    Args:
        results_dict: Dictionary of model results
        config: Configuration with evaluation thresholds (optional)
        
    Returns:
        DataFrame with model comparison
    """
    try:
        comparison_data = []
        for model_name, results in results_dict.items():
            comparison_data.append({
                'model': model_name,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'roc_auc': results['roc_auc']
            })
    
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('f1', ascending=False).reset_index(drop=True)
    
        logger.info("="*70)
        logger.info("MODEL COMPARISON (sorted by F1-Score)")
        logger.info("="*70)
        
        # Set colum
        col_w     = comparison_df['model'].str.len().max()
        col_space = 12
        header = f"{'model':<{col_w}}  {'accuracy':>{col_space}}  {'precision':>{col_space}}  {'recall':>{col_space}}  {'f1':>{col_space}}  {'roc_auc':>{col_space}}"
        divider = "-" * len(header)

        logger.info(divider)
        logger.info(header)
        logger.info(divider)
        for _, row in comparison_df.iterrows():
            line = (
                f"{row['model']:<{col_w}}  "
                f"{row['accuracy']:>{col_space}.6f}  "
                f"{row['precision']:>{col_space}.6f}  "
                f"{row['recall']:>{col_space}.6f}  "
                f"{row['f1']:>{col_space}.6f}  "
                f"{row['roc_auc']:>{col_space}.6f}"
            )
            logger.info(line)
        logger.info(divider)

        # Check thresholds if config provided
        if config and 'notebook4' in config:
            thresholds = config['notebook4'].get('evaluation', {}).get('thresholds', {})
            if thresholds:
                logger.info("Threshold Validation:")
                logger.info("-"*70)
            
                best_model = comparison_df.iloc[0]
                checks = {
                    'accuracy': (best_model['accuracy'], thresholds.get('min_accuracy', 0)),
                    'precision': (best_model['precision'], thresholds.get('min_precision', 0)),
                    'recall': (best_model['recall'], thresholds.get('min_recall', 0)),
                    'f1': (best_model['f1'], thresholds.get('min_f1', 0)),
                    'roc_auc': (best_model['roc_auc'], thresholds.get('min_roc_auc', 0))
                }
            
                all_pass = True
                for metric, (value, threshold) in checks.items():
                    status = "PASS" if value >= threshold else "FAIL"
                    logger.info(f"{metric:10s}: {value:.4f} >= {threshold:.4f} [{status}]")
                    if value < threshold:
                        all_pass = False
            
                if all_pass:
                    logger.info("All thresholds met - model ready for production")
                else:
                    logger.warning("Some thresholds not met - consider retraining or hyperparameter tuning")
    
        logger.info("="*70)
    
        return comparison_df
    except Exception as e:
        logger.error(f"compare_models failed: {e}")
        raise


def plot_roc_curves(results_dict: Dict[str, Dict], output_dir: Path) -> None:
    """Plot ROC curves for all models."""
    try:
        logger.info("Plotting ROC curves")
    
        plt.figure(figsize=(10, 8))
    
        for model_name, results in results_dict.items():
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
            auc = results['roc_auc']
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)
    
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
        output_path = Path(output_dir) / 'roc_curves_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
        logger.info(f"Saved: {output_path.name}")
    except Exception as e:
        logger.error(f"plot_roc_curves failed: {e}")
        raise


def plot_confusion_matrices(
    results_dict: Dict[str, Dict],
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """Plot confusion matrices for all models."""
    try:
        logger.info("Plotting confusion matrices")
    
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
        if n_models == 1:
            axes = [axes]
    
        for idx, (model_name, results) in enumerate(results_dict.items()):
            cm = confusion_matrix(results['y_test'], results['y_pred'])
        
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Active', 'Churned'],
                yticklabels=['Active', 'Churned'],
                ax=axes[idx],
                cbar=False
            )
        
            axes[idx].set_title(f'{model_name}\nF1={results["f1"]:.3f}', fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=11)
            axes[idx].set_xlabel('Predicted', fontsize=11)
    
        plt.tight_layout()
    
        output_path = Path(output_dir) / 'confusion_matrices.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
        logger.info(f"Saved: {output_path.name}")
    except Exception as e:
        logger.error(f"plot_confusion_matrices failed: {e}")
        raise


__all__ = [
    'evaluate_model',
    'compare_models',
    'plot_roc_curves',
    'plot_confusion_matrices',
]

if __name__ == "__main__":
    print("Testing n4e_model_evaluation module...")
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    
    print("\n1. Testing evaluate_model...")
    rf_results = evaluate_model(rf, X_test, y_test, "Random Forest")
    
    print("\n2. Testing compare_models...")
    results_dict = {'Random Forest': rf_results}
    comparison_df = compare_models(results_dict)
    
    print("\nAll tests passed!")