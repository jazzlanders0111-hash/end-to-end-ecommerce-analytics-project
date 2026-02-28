"""
n5b_data_loader.py
==================
Data loading and validation for Notebook 05: Fraud Detection & Anomaly Analysis.

Loads enhanced_df.parquet and rfm_df.parquet produced by NB01, performs
integrity checks, and returns validated DataFrames ready for feature engineering.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns for fraud feature engineering
REQUIRED_ENHANCED_COLS: list[str] = [
    "order_id",
    "customer_id",
    "product_id",
    "category",
    "price",
    "discount",
    "quantity",
    "payment_method",
    "order_date",
    "delivery_time_days",
    "region",
    "returned",
    "total_amount",
    "shipping_cost",
    "profit_margin",
    "customer_age",
    "customer_gender",
]

REQUIRED_RFM_COLS: list[str] = [
    "customer_id",
    "recency_days",
    "frequency",
    "monetary",
    "loyalty_score",
]


def load_transaction_data(
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load enhanced transaction DataFrame from parquet.

    Validates schema, row count, and date range against config expectations.

    Parameters
    ----------
    config:
        Full project configuration dict.
    run_id:
        Pipeline run identifier for log traceability.
    verbose:
        Print summary statistics if True.

    Returns
    -------
    pd.DataFrame — validated enhanced_df.
    """
    from n5a_utils import get_project_root

    root = get_project_root()
    parquet_path = Path(config["paths"]["enhanced_df"])
    if not parquet_path.is_absolute():
        parquet_path = root / parquet_path

    logger.info("[%s] Loading enhanced_df from: %s", run_id, parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"enhanced_df.parquet not found at {parquet_path}. "
            "Run Notebook 01 first to generate this file."
        )

    df = pd.read_parquet(parquet_path)

    # Schema validation
    missing_cols = [c for c in REQUIRED_ENHANCED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"enhanced_df is missing required columns: {missing_cols}"
        )

    # Ensure order_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
        df["order_date"] = pd.to_datetime(df["order_date"])
        logger.info("[%s] Converted order_date to datetime", run_id)

    # Ensure returned is numeric (0/1)
    if df["returned"].dtype == object:
        df["returned"] = df["returned"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        logger.info("[%s] Converted 'returned' column to int", run_id)

    # Row count validation
    expected_rows = config["validation"].get("expected_row_count", 34500)
    delta = df.shape[0] - expected_rows
    if abs(delta) > 100:
        logger.warning(
            "[%s] Row count mismatch: expected %d, got %d (delta %+d)",
            run_id, expected_rows, df.shape[0], delta,
        )

    if verbose:
        print(f"\n{'='*80}")
        print("TRANSACTION DATA LOADED".center(80))
        print("=" * 80)
        print(f"\nSource:             {parquet_path.name}")
        print(f"Rows:               {df.shape[0]:,}")
        print(f"Columns:            {df.shape[1]}")
        print(
            f"Date range:         "
            f"{df['order_date'].min().date()} -> {df['order_date'].max().date()}"
        )
        print(f"Unique customers:   {df['customer_id'].nunique():,}")
        print(f"Unique products:    {df['product_id'].nunique():,}")
        print(f"Return rate:        {df['returned'].mean()*100:.2f}%")
        print(
            f"Negative margins:   "
            f"{(df['profit_margin'] < 0).sum():,} "
            f"({(df['profit_margin'] < 0).mean()*100:.1f}%)"
        )
        print("=" * 80)

    logger.info(
        "[%s] enhanced_df loaded: %d rows x %d cols",
        run_id, df.shape[0], df.shape[1],
    )

    return df


def load_rfm_data(
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load customer-level RFM DataFrame from parquet.

    Parameters
    ----------
    config:
        Full project configuration dict.
    run_id:
        Pipeline run identifier.
    verbose:
        Print summary statistics if True.

    Returns
    -------
    pd.DataFrame — validated rfm_df.
    """
    from n5a_utils import get_project_root

    root = get_project_root()
    parquet_path = Path(config["paths"]["rfm_df"])
    if not parquet_path.is_absolute():
        parquet_path = root / parquet_path

    logger.info("[%s] Loading rfm_df from: %s", run_id, parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"rfm_df.parquet not found at {parquet_path}. "
            "Run Notebook 01 first to generate this file."
        )

    rfm = pd.read_parquet(parquet_path)

    missing_cols = [c for c in REQUIRED_RFM_COLS if c not in rfm.columns]
    if missing_cols:
        raise ValueError(
            f"rfm_df is missing required columns: {missing_cols}"
        )

    if verbose:
        print(f"\n{'='*80}")
        print("RFM DATA LOADED".center(80))
        print("=" * 80)
        print(f"\nSource:             {parquet_path.name}")
        print(f"Customers:          {rfm.shape[0]:,}")
        print(f"Features:           {rfm.shape[1]}")
        print(f"Avg recency (days): {rfm['recency_days'].mean():.1f}")
        print(f"Avg frequency:      {rfm['frequency'].mean():.2f}")
        print(f"Median monetary:    ${rfm['monetary'].median():.2f}")
        print(f"Avg loyalty score:  {rfm['loyalty_score'].mean():.3f}")
        print("=" * 80)

    logger.info(
        "[%s] rfm_df loaded: %d customers x %d features",
        run_id, rfm.shape[0], rfm.shape[1],
    )

    return rfm


def load_segment_data(
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load NB03 customer segment assignments from customer_segments.csv.

    Parameters
    ----------
    config:
        Full project configuration dict.
    run_id:
        Pipeline run identifier.
    verbose:
        Print summary statistics if True.

    Returns
    -------
    pd.DataFrame — customer_id, segment_name, churn_risk, loyalty_score, cluster.
    """
    from n5a_utils import get_project_root

    root = get_project_root()
    seg_path = Path(
        config.get("paths", {}).get(
            "customer_segments",
            "data/processed/customer_segments.csv",
        )
    )
    if not seg_path.is_absolute():
        seg_path = root / seg_path

    logger.info("[%s] Loading customer segments from: %s", run_id, seg_path)

    if not seg_path.exists():
        raise FileNotFoundError(
            f"customer_segments.csv not found at {seg_path}. "
            "Run Notebook 03 first."
        )

    seg = pd.read_csv(seg_path)

    required = ["customer_id", "segment_name"]
    missing = [c for c in required if c not in seg.columns]
    if missing:
        raise ValueError(f"customer_segments.csv missing columns: {missing}")

    if verbose:
        print(f"\n{'='*80}")
        print("NB03 SEGMENT DATA LOADED".center(80))
        print("=" * 80)
        print(f"\nSource:    {seg_path.name}")
        print(f"Customers: {seg.shape[0]:,}")
        print()
        print(f"{'Segment':<25} {'Count':>8} {'Share':>8}")
        print("-" * 45)
        for seg_name, count in seg["segment_name"].value_counts().items():
            print(f"{seg_name:<25} {count:>8,} {count/len(seg)*100:>7.1f}%")
        print("=" * 80)

    logger.info(
        "[%s] Segments loaded: %d customers, %d segments",
        run_id, len(seg), seg["segment_name"].nunique(),
    )

    return seg


def load_churn_predictions(
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load NB04 churn predictions from customer_risk_segments.csv.

    Parameters
    ----------
    config:
        Full project configuration dict.
    run_id:
        Pipeline run identifier.
    verbose:
        Print summary if True.

    Returns
    -------
    pd.DataFrame — customer_id, churn_probability, risk_level, predicted_churn.
    """
    from n5a_utils import get_project_root

    root = get_project_root()

    # Try customer_risk_segments first, fall back to churn_predictions
    candidates = [
        config.get("paths", {}).get(
            "customer_risk_segments",
            "data/processed/customer_risk_segments.csv",
        ),
        "data/processed/churn_predictions.csv",
    ]

    churn_path = None
    for candidate in candidates:
        p = Path(candidate)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            churn_path = p
            break

    if churn_path is None:
        logger.warning(
            "[%s] Churn predictions file not found — cross-tab will be skipped",
            run_id,
        )
        return pd.DataFrame(columns=["customer_id", "churn_probability", "risk_level"])

    logger.info("[%s] Loading churn predictions from: %s", run_id, churn_path)

    churn = pd.read_csv(churn_path)

    # --- Dynamic model performance note from NB04 export -------------------
    _perf_note: str = ""
    try:
        _perf_candidates = [
            config.get("paths", {}).get(
                "model_performance",
                "data/models/model_performance.json",
            ),
            "data/models/model_performance.json",
            "models/model_performance.json",
        ]
        _perf_path: Path | None = None
        for _cand in _perf_candidates:
            _p = Path(_cand)
            if not _p.is_absolute():
                from n5a_utils import get_project_root as _get_root
                _p = _get_root() / _p
            if _p.exists():
                _perf_path = _p
                break

        if _perf_path is not None:
            import json as _json
            with open(_perf_path) as _fh:
                _mp = _json.load(_fh)
            _roc = float(_mp["roc_auc"])
            _model_label = _mp.get("model", "Model")
            if _roc < 0.60:
                _perf_note = (
                    f"Note: {_model_label} ROC-AUC={_roc:.3f} (near-chance). "
                    f"Probability scores have limited\n"
                    f"      discrimination power. Use as context, not ground truth."
                )
            elif _roc < 0.75:
                _perf_note = (
                    f"Note: {_model_label} ROC-AUC={_roc:.3f} (moderate). "
                    f"Probability scores have some discriminative value."
                )
            else:
                _perf_note = (
                    f"Note: {_model_label} ROC-AUC={_roc:.3f} (good). "
                    f"Probability scores are reliable for prioritisation."
                )
        else:
            _perf_note = (
                "Note: model_performance.json not found — run NB04 export to "
                "generate it.\n"
                "      Churn probability reliability cannot be assessed."
            )
    except Exception as _exc:
        logger.debug("Could not load model_performance.json: %s", _exc)
        _perf_note = (
            "Note: Churn probability discrimination power unknown "
            "(model_performance.json unavailable)."
        )
    # -------------------------------------------------------------------------

    if verbose:
        print(f"\n{'='*80}")
        print("NB04 CHURN PREDICTIONS LOADED".center(80))
        print("=" * 80)
        print(f"\nSource:    {churn_path.name}")
        print(f"Customers: {churn.shape[0]:,}")
        if "risk_level" in churn.columns:
            print(f"\nRisk level distribution:")
            for lvl, cnt in churn["risk_level"].value_counts().items():
                print(f"  {lvl:<10} {cnt:>6,} ({cnt/len(churn)*100:.1f}%)")
        if "churn_probability" in churn.columns:
            print(f"\nChurn probability — mean: {churn['churn_probability'].mean():.4f}, "
                  f"median: {churn['churn_probability'].median():.4f}")
        if _perf_note:
            print(f"\n{_perf_note}")
        print("=" * 80)

    logger.info(
        "[%s] Churn predictions loaded: %d customers",
        run_id, len(churn),
    )

    return churn


def validate_data_integrity(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    config: dict[str, Any],
    run_id: str = "",
) -> dict[str, Any]:
    """
    Cross-validate transaction and RFM datasets for consistency.

    Parameters
    ----------
    df:
        Transaction DataFrame.
    rfm:
        RFM customer DataFrame.
    config:
        Project configuration.
    run_id:
        Run identifier.

    Returns
    -------
    dict containing validation results and pass/fail status.
    """
    results: dict[str, Any] = {
        "checks": [],
        "passed": 0,
        "failed": 0,
        "warnings": 0,
        "all_passed": True,
    }

    def _check(
        name: str,
        passed: bool,
        detail: str,
        severity: str = "error",
    ) -> None:
        status = "PASS" if passed else ("WARN" if severity == "warn" else "FAIL")
        results["checks"].append(
            {"name": name, "status": status, "detail": detail}
        )
        if passed:
            results["passed"] += 1
        elif severity == "warn":
            results["warnings"] += 1
        else:
            results["failed"] += 1
            results["all_passed"] = False

    # Check 1: Customer ID overlap
    tx_customers = set(df["customer_id"].unique())
    rfm_customers = set(rfm["customer_id"].unique())
    overlap_pct = len(tx_customers & rfm_customers) / len(tx_customers) * 100
    _check(
        "Customer ID overlap",
        overlap_pct >= 95,
        f"{overlap_pct:.1f}% of transaction customers present in RFM",
    )

    # Check 2: No missing values in key fraud features
    critical_cols = ["total_amount", "discount", "returned", "profit_margin"]
    null_counts = df[critical_cols].isnull().sum().sum()
    _check(
        "Null values in critical columns",
        null_counts == 0,
        f"{null_counts} nulls found in {critical_cols}",
    )

    # Check 3: Returned column is binary
    returned_values = set(df["returned"].unique())
    _check(
        "Binary returned column",
        returned_values.issubset({0, 1}),
        f"Unique values: {returned_values}",
    )

    # Check 4: Discount values in valid range
    invalid_discount = ((df["discount"] < 0) | (df["discount"] > 1)).sum()
    _check(
        "Discount range [0, 1]",
        invalid_discount == 0,
        f"{invalid_discount} out-of-range discount values",
    )

    # Check 5: Positive total_amount
    neg_total = (df["total_amount"] <= 0).sum()
    _check(
        "Positive total_amount",
        neg_total == 0,
        f"{neg_total} non-positive total_amount values",
        severity="warn",
    )

    # Check 6: RFM has no duplicated customer IDs
    dupes = rfm["customer_id"].duplicated().sum()
    _check(
        "No duplicate customer IDs in RFM",
        dupes == 0,
        f"{dupes} duplicate customer_id entries",
    )

    logger.info(
        "[%s] Data integrity: %d passed, %d failed, %d warnings",
        run_id,
        results["passed"],
        results["failed"],
        results["warnings"],
    )

    return results
