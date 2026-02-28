"""
n5c_feature_engineering.py
==========================
Fraud-oriented feature engineering for Notebook 05.

Derives customer-level behavioral indicators from transaction history.
All features are designed to capture patterns associated with:
  - Return abuse
  - Discount exploitation
  - Order velocity anomalies
  - High-value order irregularities
  - Profit margin exploitation
  - Payment method diversity (identity uncertainty)
  - Category concentration (targeted purchasing)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DISCOUNT_LEVEL: float = 0.30  # Highest available discount in this dataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_fraud_features(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a customer-level fraud feature matrix.

    Parameters
    ----------
    df:
        Transaction DataFrame (enhanced_df) with all orders.
    rfm:
        Customer-level RFM DataFrame.
    config:
        Project configuration dict.
    run_id:
        Pipeline run identifier.
    verbose:
        Print feature summary if True.

    Returns
    -------
    pd.DataFrame — one row per customer with fraud features.
                   Index is customer_id.
    """
    logger.info("[%s] Starting fraud feature engineering", run_id)

    # Ensure order_date is datetime
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["returned"] = pd.to_numeric(df["returned"], errors="coerce").fillna(0).astype(int)

    # Build each feature group
    base       = _base_aggregates(df)
    return_ft  = _return_features(df)
    discount_ft = _discount_features(df)
    velocity_ft = _velocity_features(df)
    value_ft   = _order_value_features(df)
    margin_ft  = _margin_features(df)
    diversity_ft = _diversity_features(df)

    # Merge all feature groups on customer_id
    frames = [base, return_ft, discount_ft, velocity_ft, value_ft, margin_ft, diversity_ft]
    fraud_features = frames[0]
    for frame in frames[1:]:
        fraud_features = fraud_features.merge(frame, on="customer_id", how="left")

    # Merge select RFM columns (behavioral context)
    rfm_cols = [
        "customer_id", "recency_days", "frequency",
        "monetary", "loyalty_score",
    ]
    available_rfm_cols = [c for c in rfm_cols if c in rfm.columns]
    fraud_features = fraud_features.merge(
        rfm[available_rfm_cols], on="customer_id", how="left"
    )

    # Z-score for avg_order_value (global anomaly signal)
    fraud_features["aov_zscore"] = stats.zscore(
        fraud_features["avg_order_value"].fillna(
            fraud_features["avg_order_value"].median()
        )
    )

    # Fill remaining NaNs conservatively
    num_cols = fraud_features.select_dtypes(include=[np.number]).columns
    fraud_features[num_cols] = fraud_features[num_cols].fillna(0)

    n_customers = len(fraud_features)
    n_features  = fraud_features.shape[1] - 1  # exclude customer_id

    logger.info(
        "[%s] Feature engineering complete: %d customers, %d features",
        run_id, n_customers, n_features,
    )

    if verbose:
        _print_feature_summary(fraud_features)

    return fraud_features


# ---------------------------------------------------------------------------
# Feature group builders
# ---------------------------------------------------------------------------

def _base_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic order-level aggregates per customer."""
    agg = (
        df.groupby("customer_id")
        .agg(
            order_count=("order_id", "count"),
            avg_order_value=("total_amount", "mean"),
            max_order_value=("total_amount", "max"),
            min_order_value=("total_amount", "min"),
            total_revenue=("total_amount", "sum"),
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
            avg_quantity=("quantity", "mean"),
            max_quantity=("quantity", "max"),
        )
        .reset_index()
    )

    # Tenure in days
    agg["tenure_days"] = (
        agg["last_order_date"] - agg["first_order_date"]
    ).dt.days.clip(lower=1)

    # Orders per active day
    agg["order_rate_per_day"] = agg["order_count"] / agg["tenure_days"]

    # Order value coefficient of variation (consistency check)
    cv_df = (
        df.groupby("customer_id")["total_amount"]
        .std()
        .div(
            df.groupby("customer_id")["total_amount"].mean()
        )
        .reset_index()
        .rename(columns={"total_amount": "order_value_cv"})
    )
    agg = agg.merge(cv_df, on="customer_id", how="left")
    agg["order_value_cv"] = agg["order_value_cv"].fillna(0)

    return agg.drop(columns=["first_order_date", "last_order_date"])


def _return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute return-related fraud indicators."""
    agg = (
        df.groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            total_returns=("returned", "sum"),
            returned_value=("total_amount", lambda x: x[df.loc[x.index, "returned"] == 1].sum()),
        )
        .reset_index()
    )
    agg["return_rate"] = agg["total_returns"] / agg["total_orders"]

    # High-discount orders that were returned
    high_disc = df[df["discount"] >= 0.15].copy()
    if len(high_disc) > 0:
        hd_return = (
            high_disc.groupby("customer_id")
            .agg(
                hd_orders=("order_id", "count"),
                hd_returns=("returned", "sum"),
            )
            .reset_index()
        )
        hd_return["high_discount_return_rate"] = (
            hd_return["hd_returns"] / hd_return["hd_orders"]
        )
        agg = agg.merge(
            hd_return[["customer_id", "high_discount_return_rate"]],
            on="customer_id", how="left",
        )
    else:
        agg["high_discount_return_rate"] = 0.0

    agg["high_discount_return_rate"] = agg["high_discount_return_rate"].fillna(0)
    agg["returned_value"] = agg["returned_value"].fillna(0)

    return agg[["customer_id", "return_rate", "returned_value", "high_discount_return_rate"]]


def _discount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute discount exploitation indicators."""
    agg = (
        df.groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            discount_orders=("discount", lambda x: (x > 0).sum()),
            max_discount_orders=("discount", lambda x: (x >= MAX_DISCOUNT_LEVEL).sum()),
            avg_discount=("discount", "mean"),
            max_discount=("discount", "max"),
        )
        .reset_index()
    )

    agg["discount_usage_rate"] = agg["discount_orders"] / agg["total_orders"]
    agg["max_discount_rate"]   = agg["max_discount_orders"] / agg["total_orders"]

    # Customers who ALWAYS buy at maximum discount
    agg["always_max_discount"] = (
        agg["max_discount_rate"] >= 0.80
    ).astype(int)

    return agg[[
        "customer_id",
        "discount_usage_rate",
        "max_discount_rate",
        "avg_discount",
        "max_discount",
        "always_max_discount",
    ]]


def _velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute order velocity and burst indicators.

    A 'burst' is defined as >= 3 orders within any 7-day rolling window.
    """
    results: list[dict] = []

    for cid, group in df.groupby("customer_id"):
        dates = group["order_date"].sort_values().reset_index(drop=True)
        n = len(dates)

        max_burst_7d  = 0
        max_burst_30d = 0

        for i in range(n):
            # Count orders within 7 days
            window_7d  = ((dates - dates[i]).dt.days.between(0, 7)).sum()
            window_30d = ((dates - dates[i]).dt.days.between(0, 30)).sum()
            max_burst_7d  = max(max_burst_7d, window_7d)
            max_burst_30d = max(max_burst_30d, window_30d)

        # Inter-order gap consistency
        if n > 1:
            gaps = dates.diff().dt.days.dropna()
            gap_cv = gaps.std() / gaps.mean() if gaps.mean() > 0 else 0.0
            min_gap_days = gaps.min()
        else:
            gap_cv       = 0.0
            min_gap_days = 0.0

        results.append({
            "customer_id": cid,
            "max_orders_7d": max_burst_7d,
            "max_orders_30d": max_burst_30d,
            "gap_cv": gap_cv,
            "min_order_gap_days": min_gap_days,
        })

    velocity_df = pd.DataFrame(results)
    velocity_df["velocity_burst_flag"] = (
        velocity_df["max_orders_7d"] >= 3
    ).astype(int)

    return velocity_df


def _order_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute high-value order anomaly indicators."""
    global_p95 = df["total_amount"].quantile(0.95)
    global_p99 = df["total_amount"].quantile(0.99)

    agg = (
        df.groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            high_value_orders_p95=("total_amount", lambda x: (x >= global_p95).sum()),
            high_value_orders_p99=("total_amount", lambda x: (x >= global_p99).sum()),
        )
        .reset_index()
    )

    agg["high_value_rate_p95"] = agg["high_value_orders_p95"] / agg["total_orders"]
    agg["high_value_rate_p99"] = agg["high_value_orders_p99"] / agg["total_orders"]

    return agg[[
        "customer_id",
        "high_value_orders_p99",
        "high_value_rate_p95",
        "high_value_rate_p99",
    ]]


def _margin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute negative profit margin exploitation indicators."""
    agg = (
        df.groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            negative_margin_orders=("profit_margin", lambda x: (x < 0).sum()),
            avg_profit_margin=("profit_margin", "mean"),
            min_profit_margin=("profit_margin", "min"),
        )
        .reset_index()
    )

    agg["negative_margin_rate"] = (
        agg["negative_margin_orders"] / agg["total_orders"]
    )

    # Flag customers with exclusively negative margins
    agg["all_negative_margins"] = (
        agg["negative_margin_rate"] >= 1.0
    ).astype(int)

    return agg[[
        "customer_id",
        "negative_margin_rate",
        "avg_profit_margin",
        "min_profit_margin",
        "all_negative_margins",
    ]]


def _diversity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute category concentration and payment diversity features.

    Category concentration is measured using the Herfindahl-Hirschman Index (HHI):
      HHI = sum(share^2) where share = orders in category / total orders.
      HHI = 1.0 means monopoly (single category); HHI near 0 means uniform spread.
    """
    results: list[dict] = []

    for cid, group in df.groupby("customer_id"):
        # Category HHI
        cat_counts = group["category"].value_counts(normalize=True)
        hhi = (cat_counts ** 2).sum()

        # Payment method diversity
        n_payment_methods = group["payment_method"].nunique()

        # Preferred payment method concentration
        pm_counts = group["payment_method"].value_counts(normalize=True)
        pm_concentration = pm_counts.iloc[0] if len(pm_counts) > 0 else 1.0

        results.append({
            "customer_id": cid,
            "category_hhi": hhi,
            "n_payment_methods": n_payment_methods,
            "payment_concentration": pm_concentration,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_feature_summary(fraud_features: pd.DataFrame) -> None:
    """Print a summary of the engineered feature matrix."""
    num_cols = fraud_features.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\n{'='*80}")
    print("FRAUD FEATURE MATRIX".center(80))
    print("=" * 80)
    print(f"\nCustomers:          {len(fraud_features):,}")
    print(f"Features:           {len(num_cols)}")
    print()
    print(f"{'Feature':<35} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print("-" * 70)

    highlight_features = [
        "return_rate", "discount_usage_rate", "max_discount_rate",
        "negative_margin_rate", "max_orders_7d", "velocity_burst_flag",
        "high_value_rate_p95", "category_hhi", "order_value_cv",
        "always_max_discount", "high_discount_return_rate",
    ]
    for col in highlight_features:
        if col in fraud_features.columns:
            s = fraud_features[col]
            print(
                f"{col:<35} {s.mean():>10.4f} {s.median():>10.4f} {s.std():>10.4f}"
            )

    print("=" * 80)


def get_feature_list(config: dict[str, Any]) -> list[str]:
    """
    Return the list of fraud features to pass to models.

    Uses config['notebook5']['features'] if defined, else returns defaults.
    """
    default_features = [
        "return_rate",
        "high_discount_return_rate",
        "discount_usage_rate",
        "max_discount_rate",
        "always_max_discount",
        "negative_margin_rate",
        "max_orders_7d",
        "velocity_burst_flag",
        "gap_cv",
        "high_value_rate_p95",
        "high_value_rate_p99",
        "category_hhi",
        "order_value_cv",
        "aov_zscore",
        "n_payment_methods",
        "payment_concentration",
    ]

    nb5_cfg = config.get("notebook5", {})
    return nb5_cfg.get("features", default_features)
