"""
n6j_validation.py
-----------------
Data quality validation for Notebook 06 cohort outputs.

Runs a systematic battery of checks against the constructed matrices and
DataFrames before they are persisted.  Mirrors the validation contract
used in n1f_sanity_check_wrapper and n5 inline validation.

Functions
---------
validate_cohort_outputs — Run all checks and return a structured results dict
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from n6_utils import get_run_id, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_cohort_outputs(
    df_cohorted: pd.DataFrame,
    retention_matrix: pd.DataFrame,
    revenue_matrix: pd.DataFrame,
    ltv_matrix: pd.DataFrame,
    cohort_sizes: pd.Series,
    config: dict[str, Any],
    run_id: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Validate all NB06 outputs before saving.

    Parameters
    ----------
    df_cohorted:
        Output of ``assign_acquisition_cohorts``.
    retention_matrix:
        Output of ``compute_retention_rates``.
    revenue_matrix:
        Output of ``build_revenue_matrix``.
    ltv_matrix:
        Output of ``compute_cumulative_ltv``.
    cohort_sizes:
        Series mapping cohort_month → n_customers.
    config:
        Loaded project configuration dictionary.
    run_id:
        Pipeline run identifier.
    verbose:
        Print each check result when True.

    Returns
    -------
    dict
        Keys: ``checks``, ``total_checks``, ``passed_checks``,
        ``failed_checks``, ``warning_checks``, ``all_passed``, ``messages``.
    """
    run_id   = run_id or get_run_id()
    val_cfg  = config.get("validation", {})
    checks   = []
    messages = []

    def _check(name: str, passed: bool, detail: str, severity: str = "error") -> None:
        status = "PASS" if passed else ("WARN" if severity == "warn" else "FAIL")
        checks.append({"name": name, "status": status, "detail": detail})
        if not passed:
            messages.append(f"[{status}] {name}: {detail}")
        log_fn = logger.info if passed else (logger.warning if severity == "warn" else logger.error)
        log_fn("[%s] %s — %s: %s", run_id, status, name, detail)
        if verbose:
            print(f"  [{status}] {name:<55} {detail}")

    # ---- Cohorted DataFrame ------------------------------------------------
    _check(
        "df_cohorted has cohort_month column",
        "cohort_month" in df_cohorted.columns,
        f"Columns: {list(df_cohorted.columns[:5])}...",
    )
    _check(
        "df_cohorted has period_offset column",
        "period_offset" in df_cohorted.columns,
        f"Columns: {list(df_cohorted.columns[:5])}...",
    )
    null_cohort = df_cohorted["cohort_month"].isna().sum() if "cohort_month" in df_cohorted.columns else -1
    _check(
        "No null cohort_month values",
        null_cohort == 0,
        f"{null_cohort} null cohort_month rows",
    )
    neg_offset = (df_cohorted["period_offset"] < 0).sum() if "period_offset" in df_cohorted.columns else -1
    _check(
        "No negative period_offset values",
        neg_offset == 0,
        f"{neg_offset} negative period offsets",
    )

    # ---- Retention matrix --------------------------------------------------
    _check(
        "Retention matrix is not empty",
        len(retention_matrix) > 0 and len(retention_matrix.columns) > 0,
        f"Shape: {retention_matrix.shape}",
    )
    m0_col   = retention_matrix[0] if 0 in retention_matrix.columns else pd.Series(dtype=float)
    non_one  = (m0_col.dropna() != 1.0).sum()
    _check(
        "Period-0 retention is 1.0 for all cohorts",
        non_one == 0,
        f"{non_one} cohorts with M0 retention != 1.0",
        severity="warn",
    )
    out_of_range = ((retention_matrix > 1.0) | (retention_matrix < 0.0)).sum().sum()
    _check(
        "Retention values in [0, 1]",
        out_of_range == 0,
        f"{out_of_range} cells outside [0, 1]",
    )

    # ---- Revenue matrix ----------------------------------------------------
    _check(
        "Revenue matrix is not empty",
        len(revenue_matrix) > 0 and len(revenue_matrix.columns) > 0,
        f"Shape: {revenue_matrix.shape}",
    )
    neg_rev = (revenue_matrix.fillna(0) < 0).sum().sum()
    _check(
        "No negative revenue values",
        neg_rev == 0,
        f"{neg_rev} negative revenue cells",
        severity="warn",
    )

    # ---- LTV matrix --------------------------------------------------------
    _check(
        "LTV matrix is not empty",
        len(ltv_matrix) > 0 and len(ltv_matrix.columns) > 0,
        f"Shape: {ltv_matrix.shape}",
    )
    neg_ltv = (ltv_matrix.fillna(0) < 0).sum().sum()
    _check(
        "No negative LTV values",
        neg_ltv == 0,
        f"{neg_ltv} negative LTV cells",
        severity="warn",
    )

    # ---- Cohort size alignment ---------------------------------------------
    matrix_cohorts = set(retention_matrix.index)
    size_cohorts   = set(cohort_sizes.index)
    extra_in_matrix = matrix_cohorts - size_cohorts
    _check(
        "All retention matrix cohorts have a size record",
        len(extra_in_matrix) == 0,
        f"{len(extra_in_matrix)} cohorts without size record",
    )

    # ---- Expected customer count -------------------------------------------
    expected_custs = val_cfg.get("expected_customers", 0)
    actual_custs   = df_cohorted["customer_id"].nunique()
    if expected_custs > 0:
        delta    = abs(actual_custs - expected_custs)
        tol      = max(1, int(expected_custs * 0.02))
        _check(
            "Cohorted customer count within 2% of expected",
            delta <= tol,
            f"Expected {expected_custs:,}, actual {actual_custs:,}",
            severity="warn",
        )

    # ---- Summary -----------------------------------------------------------
    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")

    return {
        "checks":         checks,
        "total_checks":   len(checks),
        "passed_checks":  n_pass,
        "warning_checks": n_warn,
        "failed_checks":  n_fail,
        "all_passed":     n_fail == 0,
        "messages":       messages,
    }
