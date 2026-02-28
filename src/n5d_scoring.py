"""
n5d_scoring.py
==============
Rule-based fraud risk scoring for Notebook 05.

Computes a weighted composite fraud score (0–1) from behavioral indicators.
Each rule maps to a specific fraud hypothesis with documented business rationale.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score component definitions
# ---------------------------------------------------------------------------

# Each rule: (feature_column, threshold_key, weight, description)
# Score for each rule is min(raw_value / threshold, 1.0) * weight
RULE_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "return_abuse",
        "feature": "return_rate",
        "threshold_key": "return_rate_high",
        "weight_key": "return_abuse",
        "description": "High proportion of orders returned — potential return fraud",
    },
    {
        "name": "discount_exploitation",
        "feature": "max_discount_rate",
        "threshold_key": "discount_rate_high",
        "weight_key": "discount_exploitation",
        "description": "Consistently purchases at maximum discount level",
    },
    {
        "name": "high_discount_returns",
        "feature": "high_discount_return_rate",
        "threshold_key": "return_rate_high",
        "weight_key": "high_discount_returns",
        "description": "Returns concentrated on high-discount orders (discount + return abuse)",
    },
    {
        "name": "velocity_anomaly",
        "feature": "max_orders_7d",
        "threshold_key": "velocity_order_count",
        "weight_key": "velocity_anomaly",
        "description": "Unusually high order count in short time window",
        # floor=1: every customer has max_orders_7d >= 1 by definition, so the
        # standard raw/threshold formula adds a constant ~0.05 to every customer.
        # Scoring above the floor means only genuine burst behaviour contributes.
        "floor": 1,
    },
    {
        "name": "margin_exploitation",
        "feature": "negative_margin_rate",
        "threshold_key": "negative_margin_rate_high",
        "weight_key": "margin_exploitation",
        "description": "High proportion of orders generating negative profit margins",
    },
    {
        "name": "order_value_anomaly",
        "feature": "high_value_rate_p99",
        "threshold_key": "high_value_p99_rate",
        "weight_key": "order_value_anomaly",
        "description": "Elevated rate of extremely high-value orders",
    },
    {
        "name": "category_concentration",
        "feature": "category_hhi",
        "threshold_key": "category_hhi_threshold",
        "weight_key": "category_concentration",
        "description": "Extremely concentrated category purchasing (single-category targeting)",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rule_based_scores(
    fraud_features: pd.DataFrame,
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute rule-based fraud risk scores for all customers.

    Parameters
    ----------
    fraud_features:
        Customer-level feature matrix from n5c_feature_engineering.
    config:
        Project configuration dict.
    run_id:
        Pipeline run identifier.
    verbose:
        Print scoring summary if True.

    Returns
    -------
    pd.DataFrame — original features plus score component columns and
                   composite 'rule_fraud_score'.
    """
    nb5 = config.get("notebook5", {})
    thresholds = nb5.get("fraud", {}).get("rule_thresholds", {})
    weights    = nb5.get("fraud", {}).get("scoring", {}).get("weights", {})

    df = fraud_features.copy()

    # Default thresholds if not in config
    default_thresholds = {
        "return_rate_high": 0.20,
        "discount_rate_high": 0.80,
        "velocity_order_count": 3.0,
        "negative_margin_rate_high": 0.30,
        "high_value_p99_rate": 0.20,
        "category_hhi_threshold": 0.90,
    }
    for k, v in default_thresholds.items():
        thresholds.setdefault(k, v)

    # Default weights (must sum to 1.0)
    default_weights = {
        "return_abuse": 0.20,
        "discount_exploitation": 0.15,
        "high_discount_returns": 0.20,
        "velocity_anomaly": 0.15,
        "margin_exploitation": 0.15,
        "order_value_anomaly": 0.10,
        "category_concentration": 0.05,
    }
    for k, v in default_weights.items():
        weights.setdefault(k, v)

    logger.info("[%s] Computing rule-based fraud scores", run_id)

    component_scores: list[str] = []

    for rule in RULE_REGISTRY:
        feat       = rule["feature"]
        thresh_key = rule["threshold_key"]
        weight_key = rule["weight_key"]
        col_name   = f"score_{rule['name']}"

        if feat not in df.columns:
            logger.warning("[%s] Feature '%s' not found — skipping rule '%s'",
                           run_id, feat, rule['name'])
            df[col_name] = 0.0
        else:
            threshold = thresholds[thresh_key]
            weight    = weights.get(weight_key, 0.0)
            floor     = rule.get("floor", 0)
            # Normalise: (raw - floor) / (threshold - floor), capped at [0, 1].
            # floor > 0 means the feature has a structural non-zero minimum for all
            # customers (e.g. max_orders_7d is always >= 1), so we score only the
            # excess above that floor rather than giving partial credit to everyone.
            effective_range = threshold - floor
            if effective_range <= 0:
                normalised = (df[feat] / threshold).clip(upper=1.0)
            else:
                normalised = ((df[feat] - floor) / effective_range).clip(lower=0.0, upper=1.0)
            df[col_name] = normalised * weight

        component_scores.append(col_name)

    # Composite score
    df["rule_fraud_score"] = df[component_scores].sum(axis=1)

    # Scale to [0, 1] relative to observed maximum
    score_max = df["rule_fraud_score"].max()
    if score_max > 0:
        df["rule_fraud_score_norm"] = df["rule_fraud_score"] / score_max
    else:
        df["rule_fraud_score_norm"] = df["rule_fraud_score"]

    if verbose:
        _print_scoring_summary(df, component_scores, thresholds, weights)

    logger.info(
        "[%s] Scoring complete. Mean score: %.4f, Max: %.4f",
        run_id,
        df["rule_fraud_score_norm"].mean(),
        df["rule_fraud_score_norm"].max(),
    )

    return df


def get_rule_descriptions() -> dict[str, str]:
    """Return a mapping of rule name to business description."""
    return {rule["name"]: rule["description"] for rule in RULE_REGISTRY}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_scoring_summary(
    df: pd.DataFrame,
    component_scores: list[str],
    thresholds: dict[str, float],
    weights: dict[str, float],
) -> None:
    """Print rule-based scoring diagnostics."""
    print(f"\n{'='*80}")
    print("RULE-BASED FRAUD SCORING".center(80))
    print("=" * 80)

    print(f"\nCustomers scored:   {len(df):,}")
    print(f"Mean fraud score:   {df['rule_fraud_score_norm'].mean():.4f}")
    print(f"Median:             {df['rule_fraud_score_norm'].median():.4f}")
    print(f"90th percentile:    {df['rule_fraud_score_norm'].quantile(0.90):.4f}")
    print(f"95th percentile:    {df['rule_fraud_score_norm'].quantile(0.95):.4f}")

    print(f"\n{'Rule':<30} {'Weight':>8} {'Mean Contrib':>15} {'Max Contrib':>12}")
    print("-" * 70)
    for col in component_scores:
        rule_name = col.replace("score_", "")
        weight_key = rule_name
        w = weights.get(weight_key, 0.0)
        print(
            f"{rule_name:<30} {w:>8.2f} "
            f"{df[col].mean():>15.4f} {df[col].max():>12.4f}"
        )

    print("\nScore Distribution:")
    print("-" * 40)
    pct_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(pct_bins) - 1):
        lo, hi = pct_bins[i], pct_bins[i + 1]
        count = ((df["rule_fraud_score_norm"] >= lo) & (df["rule_fraud_score_norm"] < hi)).sum()
        pct   = count / len(df) * 100
        bar   = "#" * int(pct / 2)
        print(f"  [{lo:.1f}–{hi:.1f}): {count:5,} ({pct:5.1f}%) {bar}")

    print("=" * 80)