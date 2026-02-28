"""
n6k_data_saver.py
-----------------
Output persistence for Notebook 06: Cohort Retention & Lifecycle Analysis.

Saves all cohort matrices and summary statistics to the processed data
directory in Parquet format, consistent with NB01–NB05 output conventions.

Functions
---------
save_cohort_outputs — Persist all NB06 outputs and return saved file paths
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from n6_utils import get_output_paths, get_project_root, get_run_id, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------

_FILE_MAP = {
    "cohort_retention":  "cohort_retention.parquet",
    "cohort_ltv":        "cohort_ltv.parquet",
    "cohort_summary":    "cohort_summary.parquet",
    "segment_ltv":       "cohort_segment_ltv.parquet",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_cohort_outputs(
    df_cohorted: pd.DataFrame,
    retention_matrix: pd.DataFrame,
    revenue_matrix: pd.DataFrame,
    ltv_matrix: pd.DataFrame,
    segment_ltv: dict[str, dict[int, float]],
    config: dict[str, Any],
    run_id: str | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Persist all NB06 outputs to disk.

    Parameters
    ----------
    df_cohorted:
        Output of ``assign_acquisition_cohorts`` — transaction-level with
        cohort metadata columns attached.
    retention_matrix:
        Output of ``compute_retention_rates`` — cohort x period offset
        retention rates.
    revenue_matrix:
        Output of ``build_revenue_matrix`` — cohort x period offset revenue.
    ltv_matrix:
        Output of ``compute_cumulative_ltv`` — cohort x period offset LTV.
    segment_ltv:
        Output of ``compute_segment_ltv`` — nested dict mapping
        segment → {window: ltv}.
    config:
        Loaded project configuration dictionary.
    run_id:
        Pipeline run identifier for log tracing.
    verbose:
        Print save confirmation to stdout when True.

    Returns
    -------
    dict[str, Path]
        Maps output label → resolved file path.
    """
    run_id    = run_id or get_run_id()
    root      = get_project_root()
    paths_cfg = config.get("paths", {})
    out_dir   = root / paths_cfg.get("processed_data", "data/processed/")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # ---- 1. Cohort-level retention matrix ----------------------------------
    ret_path = out_dir / _FILE_MAP["cohort_retention"]
    _save_matrix(retention_matrix, ret_path, run_id=run_id, label="cohort_retention")
    saved["cohort_retention"] = ret_path

    # ---- 2. Cumulative LTV matrix ------------------------------------------
    ltv_path = out_dir / _FILE_MAP["cohort_ltv"]
    _save_matrix(ltv_matrix, ltv_path, run_id=run_id, label="cohort_ltv")
    saved["cohort_ltv"] = ltv_path

    # ---- 3. Cohort summary statistics --------------------------------------
    summary = _build_cohort_summary(retention_matrix, revenue_matrix, ltv_matrix)
    sum_path = out_dir / _FILE_MAP["cohort_summary"]
    summary.to_parquet(sum_path, index=True)
    logger.info("[%s] Saved cohort_summary: %s", run_id, sum_path.name)
    saved["cohort_summary"] = sum_path

    # ---- 4. Segment LTV table ----------------------------------------------
    seg_df   = _segment_ltv_to_df(segment_ltv)
    seg_path = out_dir / _FILE_MAP["segment_ltv"]
    seg_df.to_parquet(seg_path, index=True)
    logger.info("[%s] Saved segment_ltv: %s", run_id, seg_path.name)
    saved["segment_ltv"] = seg_path

    if verbose:
        print(f"\n{'='*80}")
        print("SAVED OUTPUTS".center(80))
        print("=" * 80)
        for label, path in saved.items():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {label:<25}  {path.name:<35}  ({size_mb:.2f} MB)")
        print("=" * 80)

    return saved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_matrix(
    matrix: pd.DataFrame,
    path: Path,
    run_id: str,
    label: str,
) -> None:
    """Save a cohort matrix DataFrame to parquet.

    Index and column names are cast to string for cross-platform
    parquet compatibility (Period objects are not universally supported).
    """
    df_out = matrix.copy()
    df_out.index   = df_out.index.astype(str)
    df_out.columns = df_out.columns.astype(str)
    df_out.to_parquet(path, index=True)
    logger.info("[%s] Saved %s: %s", run_id, label, path.name)


def _build_cohort_summary(
    retention_matrix: pd.DataFrame,
    revenue_matrix: pd.DataFrame,
    ltv_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-cohort summary statistics across all period offsets.

    Returns
    -------
    pd.DataFrame
        One row per cohort with columns for key retention and LTV metrics.
    """
    records = []
    for cohort in retention_matrix.index:
        ret_row = retention_matrix.loc[cohort].dropna()
        rev_row = revenue_matrix.loc[cohort].dropna() if cohort in revenue_matrix.index else pd.Series(dtype=float)
        ltv_row = ltv_matrix.loc[cohort].dropna()     if cohort in ltv_matrix.index     else pd.Series(dtype=float)

        records.append({
            "cohort_month":        str(cohort),
            "max_period_observed": int(ret_row.index.max()) if len(ret_row) else 0,
            "m1_retention":        float(ret_row.get(1, float("nan"))),
            "m3_retention":        float(ret_row.get(3, float("nan"))),
            "m6_retention":        float(ret_row.get(6, float("nan"))),
            "total_revenue":       float(rev_row.sum()) if len(rev_row) else float("nan"),
            "ltv_3m":              float(ltv_row.get(3, float("nan"))),
            "ltv_6m":              float(ltv_row.get(6, float("nan"))),
            "ltv_12m":             float(ltv_row.get(12, float("nan"))),
        })

    return pd.DataFrame(records).set_index("cohort_month")


def _segment_ltv_to_df(segment_ltv: dict[str, dict[int, float]]) -> pd.DataFrame:
    """Convert the segment_ltv nested dict to a tidy DataFrame.

    Returns
    -------
    pd.DataFrame
        Index = segment; columns = ltv_3m, ltv_6m, ltv_12m.
    """
    records = []
    for seg, windows in segment_ltv.items():
        records.append({
            "segment": seg,
            "ltv_3m":  windows.get(3,  float("nan")),
            "ltv_6m":  windows.get(6,  float("nan")),
            "ltv_12m": windows.get(12, float("nan")),
        })

    if not records:
        return pd.DataFrame(columns=["segment", "ltv_3m", "ltv_6m", "ltv_12m"]).set_index("segment")

    return pd.DataFrame(records).set_index("segment")
