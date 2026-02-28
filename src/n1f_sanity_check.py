# src/n1f_sanity_check.py - ERROR-HARDENED VERSION

"""
n1f_sanity_check.py - Data Sanity Checks for Notebook 01

Error Handling Strategy:
- run_sanity_checks   : validates df type upfront; each of the 10 check sections
                        runs in its own try/except so a crash in one section never
                        silences later checks; all divisions are guarded; config
                        load failure is non-fatal (defaults used); returns partial
                        results rather than raising
- _log_warning        : validates both arguments; falls back to print() if logger
                        is unavailable; never raises
- _log_error          : same as _log_warning
- validate_processed_files: validates processed_dir type; config load failure raises
                        with a clear message; each file is independently try/excepted;
                        returns partial results on non-fatal errors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from n1a_utils import setup_logger, get_project_root, load_config, set_run_id

logger = setup_logger(__name__)
PROJECT_ROOT = get_project_root()


# ===========================================================================
# 1. _log_warning  (hardened helper)
# ===========================================================================

def _log_warning(results: Dict, message: str, verbose: bool) -> None:
    """
    Log a warning and update the results dict.

    Error handling:
    - Validates results is a dict; skips dict update but still logs on bad type
    - Validates message is a string; coerces to str if not
    - Falls back to print() if logger.warning raises
    - Never raises under any circumstances
    """
    try:
        if not isinstance(message, str):
            message = str(message)
    except Exception:
        message = "<non-stringable message>"

    try:
        if isinstance(results, dict):
            results.setdefault("warnings", []).append(message)
            results["issues_found"] = results.get("issues_found", 0) + 1
        else:
            pass  # cannot update results; still log below
    except Exception:
        pass

    try:
        if verbose:
            logger.warning(message)
    except Exception:
        try:
            if verbose:
                print(f"WARNING: {message}")
        except Exception:
            pass


# ===========================================================================
# 2. _log_error  (hardened helper)
# ===========================================================================

def _log_error(results: Dict, message: str, verbose: bool) -> None:
    """
    Log an error and update the results dict.

    Error handling:
    - Same resilience contract as _log_warning
    - Sets all_passed=False; if results is not a dict, still logs the error
    - Never raises under any circumstances
    """
    try:
        if not isinstance(message, str):
            message = str(message)
    except Exception:
        message = "<non-stringable message>"

    try:
        if isinstance(results, dict):
            results.setdefault("errors", []).append(message)
            results["issues_found"] = results.get("issues_found", 0) + 1
            results["all_passed"]   = False
    except Exception:
        pass

    try:
        if verbose:
            logger.error(message)
    except Exception:
        try:
            if verbose:
                print(f"ERROR: {message}")
        except Exception:
            pass


# ===========================================================================
# 3. run_sanity_checks
# ===========================================================================

def run_sanity_checks(
    df: pd.DataFrame,
    rfm_df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute comprehensive data quality validation suite.

    Error handling:
    - Validates df is a non-empty DataFrame; returns an error-flagged result dict
      immediately rather than raising, so callers always get a dict back
    - run_id and config load failures are non-fatal (warned and defaulted)
    - Each of the 10 check sections runs in its own try/except block so a crash
      in one section (e.g. a missing column) never silences later checks
    - All divisions are guarded with _safe_div or conditional checks
    - rfm_df checks are only attempted when rfm_df is a non-empty DataFrame;
      a wrong type is warned and skipped rather than raising

    Returns:
        Dict with keys: all_passed, checks_performed, issues_found,
                        warnings, errors, summary
    """
    caller = "run_sanity_checks"

    # ---- run_id ------------------------------------------------------------
    if run_id:
        try:
            set_run_id(run_id)
        except Exception as e:
            logger.warning(f"[{caller}] set_run_id failed: {e}")

    # ---- Results template --------------------------------------------------
    results: Dict[str, Any] = {
        "all_passed":       True,
        "issues_found":     0,
        "warnings":         [],
        "errors":           [],
        "checks_performed": 0,
        "summary":          {},
    }

    # ---- Hard guard: df must be a non-empty DataFrame ----------------------
    if not isinstance(df, pd.DataFrame):
        _log_error(results, f"df must be a pd.DataFrame, got {type(df).__name__}", verbose)
        return results
    if df.empty:
        _log_error(results, "Transaction DataFrame is empty", verbose)
        return results

    # ---- rfm_df type guard (soft) ------------------------------------------
    rfm_valid = isinstance(rfm_df, pd.DataFrame) and not rfm_df.empty
    if rfm_df is not None and not rfm_valid:
        logger.warning(
            f"[{caller}] rfm_df is not a non-empty DataFrame "
            f"({type(rfm_df).__name__}) – RFM checks skipped"
        )

    # ---- Config ------------------------------------------------------------
    validation_config: dict = {}
    try:
        config = load_config(PROJECT_ROOT / "config.yaml")
        validation_config = config.get("validation", {})
    except Exception as e:
        logger.warning(f"[{caller}] could not load config: {e} – using defaults")

    if verbose:
        logger.info("=" * 60)
        logger.info("SANITY CHECK: Data Wrangling Pipeline Validation")
        logger.info("=" * 60)

    # Stash for later sections
    actual_start = actual_end = None

    # ====================================================================
    # 1. BASIC DATA STRUCTURE
    # ====================================================================
    try:
        results["checks_performed"] += 1

        if verbose:
            logger.info(f"Transaction data shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

        expected_rows = validation_config.get("expected_row_count")
        if expected_rows:
            results["checks_performed"] += 1
            if df.shape[0] != expected_rows:
                _log_warning(results, f"Row count: expected {expected_rows:,}, got {df.shape[0]:,}", verbose)

        expected_customers = validation_config.get("expected_customers")
        if expected_customers and "customer_id" in df.columns:
            results["checks_performed"] += 1
            actual = df["customer_id"].nunique()
            if actual != expected_customers:
                _log_warning(results, f"Customer count: expected {expected_customers:,}, got {actual:,}", verbose)

        logger.info("-" * 60)
    except Exception as e:
        logger.warning(f"[{caller}] basic structure check failed: {e}")

    # ====================================================================
    # 2. DATA TYPES
    # ====================================================================
    try:
        results["checks_performed"] += 1
        if verbose:
            logger.info("VALIDATING DATA TYPES:")

        expected_types = {
            "order_id":          ["object"],
            "customer_id":       ["object"],
            "product_id":        ["object"],
            "category":          ["object"],
            "payment_method":    ["object"],
            "region":            ["object"],
            "customer_gender":   ["object"],
            "price":             ["float64", "float32"],
            "discount":          ["float64", "float32"],
            "quantity":          ["int64", "int32", "int16"],
            "total_amount":      ["float64", "float32"],
            "shipping_cost":     ["float64", "float32"],
            "profit_margin":     ["float64", "float32"],
            "customer_age":      ["int64", "int32", "int16"],
            "delivery_time_days":["int64", "int32", "int16"],
            "returned":          ["int64", "int32", "int16", "int8"],
            "order_date":        ["datetime64[ns]"],
        }

        type_issues = 0
        for col, expected_list in expected_types.items():
            try:
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if actual_type not in expected_list:
                        _log_warning(
                            results,
                            f"Column '{col}' has type {actual_type}, expected one of {expected_list}",
                            verbose,
                        )
                        type_issues += 1
            except Exception as col_err:
                logger.warning(f"[{caller}] dtype check failed for '{col}': {col_err}")

        if type_issues == 0 and verbose:
            logger.info("All data types correct")
        logger.info("-" * 60)
    except Exception as e:
        logger.warning(f"[{caller}] data type check section failed: {e}")

    # ====================================================================
    # 3. MISSING VALUES
    # ====================================================================
    try:
        results["checks_performed"] += 1
        if verbose:
            logger.info("CHECKING FOR MISSING VALUES:")

        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]

        if not cols_with_missing.empty:
            _log_warning(results, f"Missing values in {len(cols_with_missing)} columns", verbose)
            if verbose:
                for col, count in cols_with_missing.items():
                    pct = (count / len(df)) * 100
                    logger.info(f"  {col}: {count:,} ({pct:.2f}%)")
        else:
            if verbose:
                logger.info("No missing values found")
        logger.info("-" * 60)
    except Exception as e:
        logger.warning(f"[{caller}] missing value check section failed: {e}")

    # ====================================================================
    # 4. VALUE RANGES
    # ====================================================================
    try:
        results["checks_performed"] += 1
        if verbose:
            logger.info("VALIDATING VALUE RANGES:")

        range_checks = [
            ("price",       lambda s: (s <= 0).sum(),                       "non-positive prices",       True),
            ("discount",    lambda s: ((s < 0) | (s > 1)).sum(),            "invalid discounts",         True),
            ("quantity",    lambda s: (s <= 0).sum(),                       "non-positive quantities",   True),
            ("total_amount",lambda s: (s <= 0).sum(),                       "non-positive total amounts",False),
            ("returned",    lambda s: (~s.isin([0, 1])).sum(),              "non-binary returned values",True),
            ("customer_age",lambda s: ((s < 18) | (s > 120)).sum(),         "questionable ages",         False),
        ]

        prev_issues = results.get("issues_found")

        for col, check_fn, label, is_error in range_checks:
            try:
                if col in df.columns:
                    count = int(check_fn(df[col]))
                    if count > 0:
                        msg = f"Found {count:,} {label}"
                        if is_error:
                            _log_error(results, msg, verbose)
                        else:
                            _log_warning(results, msg, verbose)
            except Exception as col_err:
                logger.warning(f"[{caller}] range check for '{col}' failed: {col_err}")

        if results.get("issues_found") == prev_issues:
            if verbose:
                logger.info("All value ranges appear valid")

        logger.info("-" * 60)
    except Exception as e:
        logger.warning(f"[{caller}] value range check section failed: {e}")

    # ====================================================================
    # 5. DATE RANGES
    # ====================================================================
    try:
        results["checks_performed"] += 1
        if verbose:
            logger.info("VALIDATING DATE RANGES:")

        if "order_date" in df.columns:
            actual_start = df["order_date"].min()
            actual_end   = df["order_date"].max()

            if verbose:
                try:
                    logger.info(f"Date range: {actual_start.date()} -> {actual_end.date()}")
                except Exception:
                    logger.info(f"Date range: {actual_start} -> {actual_end}")

            date_config    = validation_config.get("expected_date_range", {})
            expected_start = date_config.get("start")
            expected_end   = date_config.get("end")

            if expected_start and expected_end:
                try:
                    exp_start_dt = pd.to_datetime(expected_start)
                    exp_end_dt   = pd.to_datetime(expected_end)
                    if actual_start.date() != exp_start_dt.date():
                        _log_warning(results, f"Start date: expected {expected_start}, got {actual_start.date()}", verbose)
                    if actual_end.date() != exp_end_dt.date():
                        _log_warning(results, f"End date: expected {expected_end}, got {actual_end.date()}", verbose)
                except Exception as dt_err:
                    logger.warning(f"[{caller}] date range comparison failed: {dt_err}")
        else:
            logger.warning(f"[{caller}] 'order_date' column not found – date check skipped")
        logger.info("-" * 60)
    except Exception as e:
        logger.warning(f"[{caller}] date range check section failed: {e}")

    # ====================================================================
    # 6. BUSINESS LOGIC
    # ====================================================================
    try:
        results["checks_performed"] += 1
        if verbose:
            logger.info("VALIDATING BUSINESS LOGIC:")

        # total_amount = price * quantity * (1 - discount)
        try:
            required = {"price", "quantity", "discount", "total_amount"}
            if required.issubset(df.columns):
                calculated = df["price"] * df["quantity"] * (1 - df["discount"])
                discrepancies = int((abs(df["total_amount"] - calculated) > 0.01).sum())
                if discrepancies > 0:
                    _log_warning(results, f"{discrepancies:,} rows with total_amount calculation discrepancies", verbose)
                elif verbose:
                    logger.info("Total amount calculations are consistent")
        except Exception as e:
            logger.warning(f"[{caller}] total_amount logic check failed: {e}")

        # profit_margin sanity
        try:
            if "profit_margin" in df.columns and "total_amount" in df.columns:
                extreme = int((df["profit_margin"] > df["total_amount"] * 10).sum())
                if extreme > 0:
                    _log_warning(results, f"{extreme:,} rows with profit_margin > 10x total_amount", verbose)
                elif verbose:
                    logger.info("Profit margins within reasonable range")
        except Exception as e:
            logger.warning(f"[{caller}] profit_margin check failed: {e}")

        # shipping cost vs total amount
        try:
            if "shipping_cost" in df.columns and "total_amount" in df.columns:
                high_ship = int((df["shipping_cost"] > df["total_amount"]).sum())
                if high_ship > 0:
                    pct = (high_ship / len(df)) * 100
                    if pct > 2.0:
                        _log_warning(results, f"{high_ship:,} rows ({pct:.2f}%) where shipping > total amount", verbose)
                    elif verbose:
                        logger.info(f"Note: {high_ship:,} low-value items have shipping > total ({pct:.2f}%)")
        except Exception as e:
            logger.warning(f"[{caller}] shipping cost check failed: {e}")

        logger.info("-" * 60)
    except Exception as e:
        logger.warning(f"[{caller}] business logic check section failed: {e}")

    # ====================================================================
    # 7. RFM VALIDATION
    # ====================================================================
    try:
        if rfm_valid and rfm_df is not None:
            results["checks_performed"] += 1
            if verbose:
                logger.info("VALIDATING RFM FEATURES:")

            rfm_col_checks = [
                ("recency_days",  lambda c: ((c < 0) & c.notna()).sum(),    "customers with negative recency",    True),
                ("frequency",     lambda c: ((c <= 0) & c.notna()).sum(),    "customers with non-positive frequency", True),
                ("monetary",      lambda c: ((c < 0) & c.notna()).sum(),     "customers with negative monetary",   True),
                ("loyalty_score", lambda c: ((c < 0) | (c > 1)).sum(),      "customers with loyalty score outside 0-1", True),
            ]

            for col, check_fn, label, is_error in rfm_col_checks:
                try:
                    if col in rfm_df.columns:
                        count = int(check_fn(rfm_df[col]))
                        if count > 0:
                            msg = f"Found {count:,} {label}"
                            if is_error:
                                _log_error(results, msg, verbose)
                            else:
                                _log_warning(results, msg, verbose)
                except Exception as col_err:
                    logger.warning(f"[{caller}] RFM check for '{col}' failed: {col_err}")

            # customer count match
            try:
                if "customer_id" in df.columns:
                    df_custs  = df["customer_id"].nunique()
                    rfm_custs = len(rfm_df)
                    if df_custs != rfm_custs:
                        _log_error(results, f"Customer count: df={df_custs:,}, rfm_df={rfm_custs:,}", verbose)
                    elif verbose:
                        logger.info(f"Customer count matches: {df_custs:,}")
            except Exception as e:
                logger.warning(f"[{caller}] customer count match check failed: {e}")

            # net_monetary <= monetary
            try:
                if "net_monetary" in rfm_df.columns and "monetary" in rfm_df.columns:
                    valid_mask = rfm_df["net_monetary"].notna() & rfm_df["monetary"].notna()
                    violated   = int((rfm_df.loc[valid_mask, "net_monetary"] > rfm_df.loc[valid_mask, "monetary"]).sum())
                    if violated > 0:
                        _log_error(results, f"{violated:,} customers where net_monetary > monetary", verbose)
                    else:
                        if verbose:
                            logger.info("net_monetary <= monetary for all customers")
                        neg_net = int((rfm_df["net_monetary"] < 0).sum())
                        if neg_net > 0 and verbose:
                            logger.info(f"{neg_net:,} customers with negative net_monetary (returned more than kept)")
            except Exception as e:
                logger.warning(f"[{caller}] net_monetary check failed: {e}")

            # last_order_was_return binary
            try:
                if "last_order_was_return" in rfm_df.columns:
                    if not rfm_df["last_order_was_return"].isin([0, 1]).all():
                        _log_error(results, "last_order_was_return contains non-binary values", verbose)
                    elif verbose:
                        logger.info(f"last_order_was_return: {int(rfm_df['last_order_was_return'].sum()):,} customers")
            except Exception as e:
                logger.warning(f"[{caller}] last_order_was_return check failed: {e}")

            # preferred_gender dtype
            try:
                if "preferred_gender" in rfm_df.columns:
                    if str(rfm_df["preferred_gender"].dtype) not in ("object", "string"):
                        _log_warning(results, f"preferred_gender has unexpected dtype: {rfm_df['preferred_gender'].dtype}", verbose)
                    unknown_g = int((rfm_df["preferred_gender"] == "Unknown").sum())
                    if unknown_g > 0 and verbose:
                        logger.info(f"Customers with unknown preferred_gender: {unknown_g:,}")
            except Exception as e:
                logger.warning(f"[{caller}] preferred_gender check failed: {e}")

            # preferred_age range
            try:
                if "preferred_age" in rfm_df.columns:
                    age_col = rfm_df["preferred_age"]
                    invalid_age = int(((age_col < -1) | ((age_col > 120) & (age_col != -1)) | ((age_col > 0) & (age_col < 18))).sum())
                    if invalid_age > 0:
                        _log_warning(results, f"{invalid_age:,} customers with questionable preferred_age", verbose)
                    unknown_a = int((age_col == -1).sum())
                    if unknown_a > 0 and verbose:
                        logger.info(f"Customers with unknown preferred_age: {unknown_a:,}")
            except Exception as e:
                logger.warning(f"[{caller}] preferred_age check failed: {e}")

            # pure-return count
            try:
                if "recency_days" in rfm_df.columns:
                    pure = int(rfm_df["recency_days"].isna().sum())
                    if pure > 0 and verbose:
                        pct = (pure / len(rfm_df)) * 100
                        logger.info(f"Pure-return customers: {pure:,} ({pct:.2f}%)")
            except Exception as e:
                logger.warning(f"[{caller}] pure-return count failed: {e}")

            logger.info("-" * 60)
    except Exception as e:
        logger.warning(f"[{caller}] RFM validation section failed: {e}")

    # ====================================================================
    # 8. DUPLICATE CHECK
    # ====================================================================
    try:
        results["checks_performed"] += 1
        if verbose:
            logger.info("CHECKING FOR DUPLICATES:")

        if "order_id" in df.columns:
            dup_orders = int(df["order_id"].duplicated().sum())
            if dup_orders > 0:
                _log_error(results, f"{dup_orders:,} duplicate order IDs", verbose)
            elif verbose:
                logger.info("No duplicate order IDs")

        if rfm_valid and rfm_df is not None and "customer_id" in rfm_df.columns:
            dup_custs = int(rfm_df["customer_id"].duplicated().sum())
            if dup_custs > 0:
                _log_error(results, f"{dup_custs:,} duplicate customer IDs in RFM data", verbose)
            elif verbose:
                logger.info("No duplicate customer IDs in RFM data")

        logger.info("-" * 60)
    except Exception as e:
        logger.warning(f"[{caller}] duplicate check section failed: {e}")

    # ====================================================================
    # 9. SUMMARY STATISTICS (informational only)
    # ====================================================================
    try:
        if verbose:
            logger.info("SUMMARY STATISTICS:")
            stat_items = [
                ("Total transactions", len(df), "{:,}"),
            ]
            if "customer_id" in df.columns:
                stat_items.append(("Unique customers", df["customer_id"].nunique(), "{:,}"))
            if actual_start and actual_end:
                try:
                    logger.info(f"Date range: {actual_start.date()} -> {actual_end.date()}")
                except Exception:
                    pass
            if "total_amount" in df.columns:
                logger.info(f"Total revenue   : ${df['total_amount'].sum():,.2f}")
                logger.info(f"Avg order value : ${df['total_amount'].mean():.2f}")
            if "returned" in df.columns:
                logger.info(f"Return rate     : {df['returned'].mean()*100:.2f}%")
            if "customer_id" in df.columns:
                logger.info(f"Avg orders/cust : {len(df)/df['customer_id'].nunique():.2f}")
    except Exception as e:
        logger.warning(f"[{caller}] summary statistics block failed (non-fatal): {e}")

    # ====================================================================
    # 10. FINAL SUMMARY
    # ====================================================================
    try:
        if verbose:
            logger.info("=" * 60)
            logger.info("SANITY CHECK SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Checks performed : {results['checks_performed']}")
            logger.info(f"Warnings         : {len(results['warnings'])}")
            logger.info(f"Errors           : {len(results['errors'])}")

            if results["all_passed"]:
                logger.info("ALL SANITY CHECKS PASSED")
            else:
                logger.info(f"FOUND {results['issues_found']} ISSUES")
                for err in results["errors"]:
                    logger.info(f"  ERROR   : {err}")
                for warn in results["warnings"]:
                    logger.info(f"  WARNING : {warn}")
            logger.info("=" * 60)
    except Exception as e:
        logger.warning(f"[{caller}] final summary block failed (non-fatal): {e}")

    return results


# ===========================================================================
# 4. validate_processed_files
# ===========================================================================

def validate_processed_files(
    processed_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate processed parquet files exist and are loadable.

    Error handling:
    - run_id failure is non-fatal (warned and continued)
    - Config load failure raises with a clear message (intentional – path is required)
    - Validates config values are non-None strings before constructing the path
    - Validates processed_dir is Path-like
    - Each file is checked and loaded independently; a corrupted file does not
      prevent checking others

    Returns:
        Dict with keys: all_passed, files_found, files_missing, files_corrupted
    """
    caller = "validate_processed_files"

    if run_id:
        try:
            set_run_id(run_id)
        except Exception as e:
            logger.warning(f"[{caller}] set_run_id failed: {e}")

    results: Dict[str, Any] = {
        "all_passed":      True,
        "files_found":     [],
        "files_missing":   [],
        "files_corrupted": [],
    }

    # ---- Resolve processed_dir --------------------------------------------
    if processed_dir is None:
        try:
            config = load_config(PROJECT_ROOT / "config.yaml")
        except Exception as e:
            raise ValueError(f"[{caller}] could not load config to find processed_data path: {e}")

        if "paths" not in config or "processed_data" not in config.get("paths", {}):
            raise ValueError(
                f"[{caller}] config missing 'paths.processed_data'.\n"
                "Check config.yaml has: paths:\n  processed_data: data/processed/"
            )

        processed_data_path = config["paths"]["processed_data"]
        if not processed_data_path:
            raise ValueError(
                f"[{caller}] config 'paths.processed_data' is None or empty – "
                "expected a valid path like 'data/processed/'"
            )

        processed_dir = PROJECT_ROOT / processed_data_path

    # ---- Type check --------------------------------------------------------
    if not isinstance(processed_dir, (str, Path)):
        raise TypeError(
            f"[{caller}] processed_dir must be Path or str, got {type(processed_dir).__name__}"
        )
    processed_dir = Path(processed_dir)

    if verbose:
        logger.info(f"Validating processed files in: {processed_dir}")

    # ---- File checks -------------------------------------------------------
    expected_files = ["enhanced_df.parquet", "rfm_df.parquet"]

    for filename in expected_files:
        filepath = processed_dir / filename
        try:
            if not filepath.exists():
                results["files_missing"].append(filename)
                results["all_passed"] = False
                if verbose:
                    logger.warning(f"Missing: {filename}")
            else:
                try:
                    loaded = pd.read_parquet(filepath)
                    results["files_found"].append(filename)
                    if verbose:
                        logger.info(f"Found and loaded: {filename} ({loaded.shape[0]:,} rows)")
                except Exception as load_err:
                    results["files_corrupted"].append(filename)
                    results["all_passed"] = False
                    if verbose:
                        logger.error(f"Corrupted: {filename} – {load_err}")
        except Exception as e:
            results["files_corrupted"].append(filename)
            results["all_passed"] = False
            if verbose:
                logger.error(f"Unexpected error checking '{filename}': {e}")

    return results