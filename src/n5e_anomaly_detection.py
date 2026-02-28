"""
n5e_anomaly_detection.py
========================
Unsupervised anomaly detection for Notebook 05.

Implements two complementary approaches:
  1. Isolation Forest — global anomalies via random path-length encoding
  2. Local Outlier Factor (LOF) — local density-based anomalies

Both models operate on the scaled fraud feature matrix produced by
n5c_feature_engineering. Results are expressed as anomaly scores (0–1,
higher = more anomalous) and binary anomaly flags.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_anomaly_detection(
    fraud_features: pd.DataFrame,
    feature_cols: list[str],
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run Isolation Forest and LOF anomaly detection.

    Parameters
    ----------
    fraud_features:
        Customer-level fraud feature matrix.
    feature_cols:
        Features to use as model input.
    config:
        Project configuration.
    run_id:
        Pipeline run identifier.
    verbose:
        Print detection summary if True.

    Returns
    -------
    dict with keys:
        'df'         — fraud_features augmented with anomaly scores/flags
        'scaler'     — fitted StandardScaler
        'iso_model'  — fitted IsolationForest
        'lof_model'  — fitted LocalOutlierFactor
        'metadata'   — run metadata dict
    """
    nb5 = config.get("notebook5", {})
    iso_cfg = nb5.get("models", {}).get("isolation_forest", {})
    lof_cfg = nb5.get("models", {}).get("lof", {})

    # Defaults
    iso_params = {
        "contamination": iso_cfg.get("contamination", 0.05),
        "n_estimators":  iso_cfg.get("n_estimators", 200),
        "random_state":  iso_cfg.get("random_state", 42),
        "n_jobs":        -1,
    }
    lof_params = {
        "n_neighbors":  lof_cfg.get("n_neighbors", 20),
        "contamination": lof_cfg.get("contamination", 0.05),
        "n_jobs":        -1,
    }

    df = fraud_features.copy()

    # Validate and select features
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("[%s] Missing features: %s — proceeding without them", run_id, missing)

    X = df[available].copy()

    # Impute remaining NaNs with column median
    X = X.fillna(X.median())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(
        "[%s] Running Isolation Forest (contamination=%.2f, n_estimators=%d)",
        run_id, iso_params["contamination"], iso_params["n_estimators"],
    )

    # --- Isolation Forest ---
    iso_model = IsolationForest(**iso_params)
    iso_pred  = iso_model.fit_predict(X_scaled)
    # decision_function: negative = anomaly; invert + scale to [0,1]
    iso_scores = iso_model.decision_function(X_scaled)
    iso_scores_norm = _minmax_invert(iso_scores)

    df["iso_anomaly_score"] = iso_scores_norm
    df["iso_anomaly_flag"]  = (iso_pred == -1).astype(int)

    logger.info(
        "[%s] Isolation Forest: %d anomalies flagged (%.1f%%)",
        run_id,
        df["iso_anomaly_flag"].sum(),
        df["iso_anomaly_flag"].mean() * 100,
    )

    # --- Local Outlier Factor ---
    logger.info(
        "[%s] Running LOF (n_neighbors=%d, contamination=%.2f)",
        run_id, lof_params["n_neighbors"], lof_params["contamination"],
    )

    lof_model = LocalOutlierFactor(**lof_params)
    lof_pred  = lof_model.fit_predict(X_scaled)
    # negative_outlier_factor_: more negative = more anomalous
    lof_scores = -lof_model.negative_outlier_factor_
    lof_scores_norm = _minmax_scale(lof_scores)

    df["lof_anomaly_score"] = lof_scores_norm
    df["lof_anomaly_flag"]  = (lof_pred == -1).astype(int)

    logger.info(
        "[%s] LOF: %d anomalies flagged (%.1f%%)",
        run_id,
        df["lof_anomaly_flag"].sum(),
        df["lof_anomaly_flag"].mean() * 100,
    )

    # --- Ensemble agreement score ---
    # Average of both normalised anomaly scores
    df["ensemble_anomaly_score"] = (
        df["iso_anomaly_score"] + df["lof_anomaly_score"]
    ) / 2.0

    # Both models agree
    df["both_models_flag"] = (
        (df["iso_anomaly_flag"] == 1) & (df["lof_anomaly_flag"] == 1)
    ).astype(int)

    metadata = {
        "n_customers":          len(df),
        "n_features":           len(available),
        "iso_anomalies":        int(df["iso_anomaly_flag"].sum()),
        "lof_anomalies":        int(df["lof_anomaly_flag"].sum()),
        "both_models_anomalies": int(df["both_models_flag"].sum()),
        "iso_contamination":    iso_params["contamination"],
        "lof_contamination":    lof_params["contamination"],
        "features_used":        available,
    }

    if verbose:
        _print_detection_summary(df, metadata)

    return {
        "df":        df,
        "scaler":    scaler,
        "iso_model": iso_model,
        "lof_model": lof_model,
        "metadata":  metadata,
    }


def compare_model_agreement(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute agreement statistics between Isolation Forest and LOF.

    Parameters
    ----------
    df:
        DataFrame with iso_anomaly_flag and lof_anomaly_flag columns.
    verbose:
        Print agreement table if True.

    Returns
    -------
    pd.DataFrame — 2x2 agreement table.
    """
    required = ["iso_anomaly_flag", "lof_anomaly_flag"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"DataFrame must contain: {required}")

    agreement = pd.crosstab(
        df["iso_anomaly_flag"].map({0: "IF: Normal", 1: "IF: Anomaly"}),
        df["lof_anomaly_flag"].map({0: "LOF: Normal", 1: "LOF: Anomaly"}),
        margins=True,
    )

    both_anomaly  = df["both_models_flag"].sum()
    either_anomaly = ((df["iso_anomaly_flag"] == 1) | (df["lof_anomaly_flag"] == 1)).sum()
    cohen_kappa    = _cohen_kappa(df["iso_anomaly_flag"], df["lof_anomaly_flag"])

    if verbose:
        print(f"\n{'='*80}")
        print("MODEL AGREEMENT ANALYSIS".center(80))
        print("=" * 80)
        print(f"\nAgreement Table:")
        print(agreement.to_string())
        print(f"\nBoth flag as anomaly:   {both_anomaly:,} customers")
        print(f"Either flags as anomaly: {either_anomaly:,} customers")
        print(f"Agreement rate:          {(df['iso_anomaly_flag'] == df['lof_anomaly_flag']).mean()*100:.1f}%")
        print(f"Cohen's Kappa:           {cohen_kappa:.4f}")
        _interpret_kappa(cohen_kappa)
        print("=" * 80)

    return agreement


def _minmax_invert(arr: np.ndarray) -> np.ndarray:
    """Invert and min-max scale so higher = more anomalous."""
    arr_inv = -arr
    mn, mx = arr_inv.min(), arr_inv.max()
    if mx == mn:
        return np.zeros_like(arr_inv)
    return (arr_inv - mn) / (mx - mn)


def _minmax_scale(arr: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Compute Cohen's Kappa for two binary classification arrays."""
    n = len(a)
    po = (a == b).mean()
    p_a = a.mean()
    p_b = b.mean()
    pe  = p_a * p_b + (1 - p_a) * (1 - p_b)
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0


def _interpret_kappa(kappa: float) -> None:
    """Print a textual interpretation of Cohen's Kappa."""
    if kappa >= 0.80:
        interp = "Almost perfect agreement"
    elif kappa >= 0.60:
        interp = "Substantial agreement"
    elif kappa >= 0.40:
        interp = "Moderate agreement"
    elif kappa >= 0.20:
        interp = "Fair agreement"
    else:
        interp = "Slight agreement — models detect different anomaly types"
    print(f"Interpretation:          {interp}")


def _print_detection_summary(df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Print anomaly detection run summary."""
    print(f"\n{'='*80}")
    print("ANOMALY DETECTION RESULTS".center(80))
    print("=" * 80)
    print(f"\nCustomers analysed:      {metadata['n_customers']:,}")
    print(f"Features used:           {metadata['n_features']}")
    print()
    print(f"{'Model':<30} {'Anomalies':>10} {'Rate':>10} {'Avg Score':>12}")
    print("-" * 65)
    print(
        f"{'Isolation Forest':<30} "
        f"{metadata['iso_anomalies']:>10,} "
        f"{metadata['iso_anomalies']/metadata['n_customers']*100:>9.1f}% "
        f"{df['iso_anomaly_score'].mean():>12.4f}"
    )
    print(
        f"{'Local Outlier Factor':<30} "
        f"{metadata['lof_anomalies']:>10,} "
        f"{metadata['lof_anomalies']/metadata['n_customers']*100:>9.1f}% "
        f"{df['lof_anomaly_score'].mean():>12.4f}"
    )
    print(
        f"{'Both Models (ensemble)':<30} "
        f"{metadata['both_models_anomalies']:>10,} "
        f"{metadata['both_models_anomalies']/metadata['n_customers']*100:>9.1f}% "
        f"{df['ensemble_anomaly_score'].mean():>12.4f}"
    )
    print("=" * 80)
