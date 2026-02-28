"""
n6b_data_loader.py
------------------
Data loading and integrity validation for Notebook 06: Cohort Retention.

Loads enhanced_df.parquet and rfm_df.parquet produced by NB01, performs
cross-dataset integrity checks, and returns validated DataFrames ready
for cohort construction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from n6_utils import get_project_root, get_run_id, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

_TX_REQUIRED = [
    "order_id", "customer_id", "order_date", "total_amount",
    "returned", "category", "quantity",
]

_RFM_REQUIRED = [
    "customer_id", "recency_days", "frequency", "monetary",
    "loyalty_score", "tenure_days",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cohort_inputs(
    config: dict[str, Any],
    run_id: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load transaction and RFM datasets for cohort analysis.

    Parameters
    ----------
    config:
        Loaded project configuration dictionary.
    run_id:
        Pipeline run identifier for log tracing.
    verbose:
        Print loading progress to stdout when True.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_tx, rfm_df)`` — transaction-level and customer-level datasets.

    Raises
    ------
    FileNotFoundError
        If either parquet file cannot be located.
    ValueError
        If required columns are absent from either dataset.
    """
    run_id = run_id or get_run_id()
    root   = get_project_root()
    paths  = config.get("paths", {})

    tx_path  = root / paths.get("enhanced_df",  "data/processed/enhanced_df.parquet")
    rfm_path = root / paths.get("rfm_df",        "data/processed/rfm_df.parquet")

    # --- transactions -------------------------------------------------------
    if not tx_path.exists():
        raise FileNotFoundError(f"enhanced_df not found at {tx_path}")

    logger.info("[%s] Loading transactions from %s", run_id, tx_path.name)
    df_tx = pd.read_parquet(tx_path)

    # Ensure order_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_tx["order_date"]):
        df_tx["order_date"] = pd.to_datetime(df_tx["order_date"])

    _check_columns(df_tx, _TX_REQUIRED, "enhanced_df")
    logger.info("[%s] Transactions loaded: %d rows x %d cols", run_id, *df_tx.shape)

    # --- RFM ----------------------------------------------------------------
    if not rfm_path.exists():
        raise FileNotFoundError(f"rfm_df not found at {rfm_path}")

    logger.info("[%s] Loading RFM data from %s", run_id, rfm_path.name)
    rfm_df = pd.read_parquet(rfm_path)
    _check_columns(rfm_df, _RFM_REQUIRED, "rfm_df")
    logger.info("[%s] RFM data loaded: %d rows x %d cols", run_id, *rfm_df.shape)

    if verbose:
        _print_load_summary(df_tx, rfm_df)

    return df_tx, rfm_df


def validate_cohort_integrity(
    df_tx: pd.DataFrame,
    rfm_df: pd.DataFrame,
    config: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run cross-dataset integrity checks before cohort construction.

    Parameters
    ----------
    df_tx:
        Transaction-level DataFrame.
    rfm_df:
        Customer-level RFM DataFrame.
    config:
        Loaded project configuration dictionary.
    run_id:
        Pipeline run identifier for log tracing.

    Returns
    -------
    dict
        Keys: ``checks`` (list of check result dicts), ``passed``,
        ``warnings``, ``failed``, ``all_passed``.
    """
    run_id   = run_id or get_run_id()
    val_cfg  = config.get("validation", {})
    checks   = []

    def _add(name: str, passed: bool, detail: str, severity: str = "error") -> None:
        status = "PASS" if passed else ("WARN" if severity == "warn" else "FAIL")
        checks.append({"name": name, "status": status, "detail": detail})
        log_fn = logger.info if passed else (logger.warning if severity == "warn" else logger.error)
        log_fn("[%s] %s: %s — %s", run_id, status, name, detail)

    # 1. Customer count within tolerance of expected
    expected_custs = val_cfg.get("expected_customers", 0)
    actual_custs   = df_tx["customer_id"].nunique()
    delta          = abs(actual_custs - expected_custs)
    tolerance      = max(1, int(expected_custs * 0.02))          # 2% tolerance
    _add(
        "Customer count within 2% of expected",
        delta <= tolerance or expected_custs == 0,
        f"Expected {expected_custs:,}, actual {actual_custs:,}, delta {delta:+,}",
        severity="warn",
    )

    # 2. RFM customers subset of transaction customers
    tx_custs   = set(df_tx["customer_id"].unique())
    rfm_custs  = set(rfm_df["customer_id"].unique())
    extra_rfm  = rfm_custs - tx_custs
    _add(
        "RFM customers are subset of transaction customers",
        len(extra_rfm) == 0,
        f"{len(extra_rfm):,} RFM IDs not in transactions",
    )

    # 3. No future-dated orders
    max_date    = df_tx["order_date"].max()
    future_rows = (df_tx["order_date"] > pd.Timestamp.now()).sum()
    _add(
        "No future-dated orders",
        future_rows == 0,
        f"{future_rows:,} future-dated rows (max date: {max_date.date()})",
    )

    # 4. returned flag is binary
    returned_vals = set(df_tx["returned"].dropna().unique())
    valid_binary  = returned_vals.issubset({0, 1, True, False})
    _add(
        "returned flag is binary (0/1)",
        valid_binary,
        f"Unique values: {returned_vals}",
    )

    # 5. Date range covers expected window
    expected_start = pd.Timestamp(config.get("validation", {}).get("expected_date_range", {}).get("start", "2000-01-01"))
    expected_end   = pd.Timestamp(config.get("validation", {}).get("expected_date_range", {}).get("end",   "2100-01-01"))
    actual_start   = df_tx["order_date"].min()
    actual_end     = df_tx["order_date"].max()
    _add(
        "Date range within expected window",
        actual_start >= expected_start and actual_end <= expected_end,
        f"Actual: {actual_start.date()} to {actual_end.date()}",
        severity="warn",
    )

    # 6. No null customer IDs
    null_cust = df_tx["customer_id"].isna().sum()
    _add(
        "No null customer IDs in transactions",
        null_cust == 0,
        f"{null_cust:,} null customer_id values",
    )

    # 7. total_amount is non-negative (for non-returned orders)
    non_returned = df_tx[df_tx["returned"] == 0]
    neg_amount   = (non_returned["total_amount"] < 0).sum()
    _add(
        "Non-returned orders have non-negative total_amount",
        neg_amount == 0,
        f"{neg_amount:,} negative-amount non-returned orders",
        severity="warn",
    )

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")

    return {
        "checks":     checks,
        "passed":     n_pass,
        "warnings":   n_warn,
        "failed":     n_fail,
        "all_passed": n_fail == 0,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Raise ValueError if any required column is missing from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _print_load_summary(df_tx: pd.DataFrame, rfm_df: pd.DataFrame) -> None:
    """Print a concise load summary to stdout."""
    return_rate = df_tx["returned"].mean() * 100
    print(f"\n{'='*80}")
    print("DATA LOAD SUMMARY".center(80))
    print("=" * 80)
    print(f"  Transactions:      {len(df_tx):,} rows  x  {df_tx.shape[1]} columns")
    print(f"  Date range:        {df_tx['order_date'].min().date()} to {df_tx['order_date'].max().date()}")
    print(f"  Unique customers:  {df_tx['customer_id'].nunique():,}")
    print(f"  Returned orders:   {df_tx['returned'].sum():,}  ({return_rate:.1f}%)")
    print(f"  RFM customers:     {len(rfm_df):,}  x  {rfm_df.shape[1]} columns")
    print("=" * 80)
