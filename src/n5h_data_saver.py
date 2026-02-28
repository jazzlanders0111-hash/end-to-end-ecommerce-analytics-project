"""
n5h_data_saver.py
=================
Output persistence for Notebook 05: Fraud Detection & Anomaly Analysis.

Saves the customer fraud risk profile as a parquet/CSV artefact for use
by downstream notebooks (NB06+) and business reporting pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Columns to include in the saved fraud risk profile
OUTPUT_COLUMNS: list[str] = [
    "customer_id",
    "composite_fraud_score",
    "rule_fraud_score_norm",
    "iso_anomaly_score",
    "lof_anomaly_score",
    "ensemble_anomaly_score",
    "risk_tier",
    "primary_typology",
    "return_rate",
    "max_discount_rate",
    "negative_margin_rate",
    "velocity_burst_flag",
    "max_orders_7d",
    "high_value_rate_p95",
    "category_hhi",
    "order_value_cv",
    "type_return_abuser",
    "type_discount_exploiter",
    "type_velocity",
    "type_margin_exploiter",
    "type_high_value",
    "type_combined",
    "typology_count",
    "iso_anomaly_flag",
    "lof_anomaly_flag",
    "both_models_flag",
]


def save_fraud_profile(
    df: pd.DataFrame,
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Persist the customer fraud risk profile.

    Saves two formats:
      - fraud_risk_profile.parquet — full feature set for downstream use
      - fraud_risk_summary.csv     — human-readable summary (high/critical only)

    Parameters
    ----------
    df:
        Customer-level DataFrame with fraud scores, tiers, and typologies.
    config:
        Project configuration.
    run_id:
        Pipeline run identifier.
    verbose:
        Print file info if True.

    Returns
    -------
    dict mapping file type to saved Path.
    """
    from n5a_utils import get_project_root

    root        = get_project_root()
    processed   = root / config.get("paths", {}).get("processed_data", "data/processed")
    processed.mkdir(parents=True, exist_ok=True)

    # Select available output columns
    available_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    missing_cols   = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(
            "[%s] Output columns not found (will be omitted): %s",
            run_id, missing_cols,
        )

    output_df = df[available_cols].copy()

    saved: dict[str, Path] = {}

    # Full parquet
    parquet_path = processed / "fraud_risk_profile.parquet"
    output_df.to_parquet(parquet_path, compression="snappy", index=False)
    saved["parquet"] = parquet_path
    logger.info("[%s] Saved fraud_risk_profile.parquet: %d rows", run_id, len(output_df))

    # High/Critical CSV summary
    summary_df = output_df[
        output_df["risk_tier"].isin(["Critical", "High"])
    ].sort_values("composite_fraud_score", ascending=False)

    summary_cols = [
        "customer_id", "risk_tier", "primary_typology",
        "composite_fraud_score", "return_rate", "max_discount_rate",
        "negative_margin_rate", "velocity_burst_flag",
    ]
    summary_cols = [c for c in summary_cols if c in summary_df.columns]

    csv_path = processed / "fraud_risk_summary.csv"
    summary_df[summary_cols].to_csv(csv_path, index=False)
    saved["csv"] = csv_path
    logger.info(
        "[%s] Saved fraud_risk_summary.csv: %d high/critical customers",
        run_id, len(summary_df),
    )

    if verbose:
        print(f"\n{'='*80}")
        print("PIPELINE OUTPUTS SAVED".center(80))
        print("=" * 80)
        print()
        for file_type, path in saved.items():
            size_kb = path.stat().st_size / 1024
            label   = path.name
            print(f"  {file_type.upper():<10}: {label:<45} {size_kb:>8.1f} KB")

        print()
        print(f"  Full profile:    {len(output_df):,} customers, {len(available_cols)} features")
        print(f"  High/Critical:   {len(summary_df):,} customers flagged for review")
        print("=" * 80)

    return saved
