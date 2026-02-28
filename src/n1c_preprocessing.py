# src/n1c_preprocessing.py - ERROR-HARDENED VERSION

"""
n1c_preprocessing.py - Data Cleaning and Validation

Error Handling Strategy:
- clean_data    : validates df type; config load failure raises with context;
                  each of the 7 cleaning steps is independently try/excepted so
                  one failed step does not abort later ones; validate_data call
                  is wrapped so a validation failure is surfaced cleanly
- validate_data : validates df type; each of the 3 validation checks (nulls,
                  monetary negatives, final summary) is independently guarded;
                  the original raise-on-error behaviour (auto_fix=False) is
                  preserved; never raises on internal logging failures
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal
from n1a_utils import setup_logger, set_run_id, get_project_root, load_config

logger = setup_logger(__name__)
PROJECT_ROOT = get_project_root()


# ===========================================================================
# clean_data
# ===========================================================================

def clean_data(
    df: pd.DataFrame,
    verbose: bool = True,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Clean and standardize transaction data with configurable imputation strategies.

    Error handling:
    - Raises TypeError immediately if df is not a DataFrame (intentional – callers
      must pass the right type)
    - run_id failure is non-fatal (warned and continued)
    - Config load failure raises ValueError with full context so the caller knows
      exactly what is misconfigured
    - Validates imputation strategy values; raises ValueError on bad config before
      any data is mutated
    - Each of the 7 cleaning steps is independently try/excepted:
        1. Duplicate removal
        2. Date handling
        3. Categorical standardisation
        4. Returned flag mapping
        5. Numeric missing value imputation
        6. Categorical missing value imputation
        7. Progress summary
    - A failed step logs an error and continues rather than aborting the pipeline
    - validate_data call is wrapped; errors from it propagate to the caller
      (intentional – if validation itself raises, the caller should know)
    """
    caller = "clean_data"

    # ---- Hard type guard ---------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{caller}] expected pd.DataFrame, got {type(df).__name__}")

    # ---- run_id (soft) -----------------------------------------------------
    if run_id:
        try:
            set_run_id(run_id)
        except Exception as e:
            logger.warning(f"[{caller}] set_run_id failed: {e} – continuing")

    # ---- Config load -------------------------------------------------------
    try:
        config       = load_config(PROJECT_ROOT / "config.yaml")
        nb1_cfg      = config.get("notebook1", {})
        cleaning_cfg = nb1_cfg.get("cleaning", {})
    except Exception as e:
        raise ValueError(f"[{caller}] failed to load cleaning configuration: {e}")

    auto_fix_negatives = cleaning_cfg.get("auto_fix_negatives", True)
    imputation_strategy = cleaning_cfg.get(
        "imputation_strategy", {"numeric": "median", "categorical": "mode"}
    )

    if not isinstance(imputation_strategy, dict):
        raise ValueError(
            f"[{caller}] config error: cleaning.imputation_strategy must be a dict, "
            f"got {type(imputation_strategy).__name__}"
        )

    numeric_strategy:     str = imputation_strategy.get("numeric",     "median")
    categorical_strategy: str = imputation_strategy.get("categorical", "mode")

    valid_numeric     = ["mean", "median", "zero"]
    valid_categorical = ["mode", "unknown"]

    if numeric_strategy not in valid_numeric:
        raise ValueError(
            f"[{caller}] invalid numeric strategy '{numeric_strategy}'. "
            f"Must be one of: {valid_numeric}"
        )
    if categorical_strategy not in valid_categorical:
        raise ValueError(
            f"[{caller}] invalid categorical strategy '{categorical_strategy}'. "
            f"Must be one of: {valid_categorical}"
        )

    logger.info("Starting data cleaning process")
    logger.info(f"Input shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"Numeric strategy   : {numeric_strategy}")
    logger.info(f"Categorical strategy: {categorical_strategy}")

    df = df.copy()
    initial_rows = len(df)

    # ---- Step 1: Remove duplicate orders -----------------------------------
    try:
        if "order_id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset="order_id", keep="first")
            removed = before - len(df)
            if verbose:
                if removed > 0:
                    logger.info(f"Removed {removed:,} duplicate order_id rows")
                else:
                    logger.info("No duplicate order_id rows found")
        else:
            logger.warning(f"[{caller}] 'order_id' column not found – duplicate removal skipped")
    except Exception as e:
        logger.error(f"[{caller}] duplicate removal failed: {e}")

    # ---- Step 2: Date handling ---------------------------------------------
    try:
        if "order_date" in df.columns:
            logger.info("Processing date columns:")
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
            invalid_mask = df["order_date"].isna()
            if invalid_mask.any():
                invalid_count = int(invalid_mask.sum())
                if verbose:
                    logger.info(f"Dropped {invalid_count:,} rows with invalid/missing dates")
                df = df[~invalid_mask]
            else:
                if verbose:
                    logger.info("All dates valid")
            df["order_date"] = df["order_date"].dt.normalize() #type: ignore
        else:
            logger.warning(f"[{caller}] 'order_date' column not found – date handling skipped")
    except Exception as e:
        logger.error(f"[{caller}] date handling failed: {e}")

    # ---- Step 3: Standardise categorical columns ---------------------------
    try:
        str_cols = ["category", "region", "customer_gender", "payment_method"]
        present  = [c for c in str_cols if c in df.columns]
        for col in present:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.title()
                .replace(["", "Nan", "None", "nan"], "Unknown")
            )
        if verbose:
            logger.info(f"Standardised {len(present)} categorical columns")
        skipped = [c for c in str_cols if c not in df.columns]
        if skipped:
            logger.warning(f"[{caller}] categorical columns not found (skipped): {skipped}")
    except Exception as e:
        logger.error(f"[{caller}] categorical standardisation failed: {e}")

    # ---- Step 4: Returned flag mapping ------------------------------------
    try:
        if "returned" in df.columns:
            logger.info("Processing returned flag:")
            mapping = {
                "yes": 1, "y": 1, "true": 1, "1": 1,
                "no":  0, "n": 0, "false": 0, "0": 0,
            }
            df["returned"] = (
                df["returned"]
                .astype(str)
                .str.lower()
                .str.strip()
                .map(mapping)
                .fillna(0)
                .astype("int8")
            )
            if verbose:
                logger.info("Converted returned flag to binary (0/1)")
        else:
            logger.warning(f"[{caller}] 'returned' column not found – flag mapping skipped")
    except Exception as e:
        logger.error(f"[{caller}] returned flag mapping failed: {e}")

    # ---- Step 5: Numeric missing value imputation --------------------------
    missing_imputed = 0
    try:
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            try:
                missing_count = int(df[col].isna().sum())
                if missing_count > 0:
                    original_dtype = df[col].dtype
                    if numeric_strategy == "median":
                        fill_value = df[col].median()
                    elif numeric_strategy == "mean":
                        fill_value = df[col].mean()
                    else:  # zero
                        fill_value = 0
                    df[col] = df[col].fillna(fill_value).astype(original_dtype)
                    missing_imputed += missing_count
            except Exception as col_err:
                logger.warning(f"[{caller}] numeric imputation failed for '{col}': {col_err}")
    except Exception as e:
        logger.error(f"[{caller}] numeric imputation loop failed: {e}")

    # ---- Step 6: Categorical missing value imputation ----------------------
    try:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            try:
                missing_count = int(df[col].isna().sum())
                if missing_count > 0:
                    if categorical_strategy == "mode":
                        mode_vals = df[col].mode()
                        fill_val  = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                    else:
                        fill_val = "Unknown"
                    df[col] = df[col].fillna(fill_val)
                    missing_imputed += missing_count
            except Exception as col_err:
                logger.warning(f"[{caller}] categorical imputation failed for '{col}': {col_err}")
    except Exception as e:
        logger.error(f"[{caller}] categorical imputation loop failed: {e}")

    if verbose:
        if missing_imputed > 0:
            logger.info(f"Imputed {missing_imputed:,} missing values")
        else:
            logger.info("No missing values to impute")

    # ---- Step 7: Progress summary before validation ------------------------
    try:
        if verbose:
            logger.info(f"Pre-validation shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
            if "order_date" in df.columns and not df["order_date"].empty:
                logger.info(
                    f"Date range: {df['order_date'].min().date()} -> {df['order_date'].max().date()}"
                )
    except Exception as e:
        logger.warning(f"[{caller}] pre-validation summary logging failed (non-fatal): {e}")

    # ---- Step 8: Validate -------------------------------------------------
    logger.info("Running data validation:")
    df = validate_data(df, verbose=verbose, auto_fix=auto_fix_negatives)

    # ---- Exit logging -------------------------------------------------------
    try:
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.info(
                f"Data cleaning complete: removed {rows_removed:,} rows "
                f"({rows_removed/initial_rows*100:.2f}%)"
            )
        else:
            logger.info("Data cleaning complete: no rows removed")
        logger.info(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    except Exception as e:
        logger.warning(f"[{caller}] exit logging failed (non-fatal): {e}")

    return df


# ===========================================================================
# validate_data
# ===========================================================================

def validate_data(
    df: pd.DataFrame,
    verbose: bool = True,
    auto_fix: bool = False,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate cleaned data and optionally fix issues.

    Error handling:
    - Raises TypeError immediately if df is not a DataFrame (intentional)
    - run_id failure is non-fatal (warned and continued)
    - Each of the 3 validation checks (null check, monetary negatives,
      final summary) runs in its own try/except so a crash in one check
      does not silently mask other issues
    - The null check logs a warning but never raises (nulls are surfaced
      without aborting)
    - The monetary negative check raises ValueError when auto_fix=False,
      preserving the original contract; when auto_fix=True it fixes and
      logs a warning; the fix itself is guarded against unexpected errors
    - The final summary block is wrapped to ensure logging failures cannot
      mask a valid return value
    - Always returns the (possibly modified) DataFrame; never returns None

    Args:
        df:       DataFrame to validate
        verbose:  Log detailed progress
        auto_fix: Automatically fix negatives (True) or raise ValueError (False)
        run_id:   Optional run_id

    Returns:
        Validated (and potentially fixed) DataFrame

    Raises:
        TypeError:  If df is not a pd.DataFrame
        ValueError: If monetary negatives found and auto_fix=False
    """
    caller = "validate_data"

    # ---- Hard type guard ---------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{caller}] expected pd.DataFrame, got {type(df).__name__}")

    # ---- run_id (soft) -----------------------------------------------------
    if run_id:
        try:
            set_run_id(run_id)
        except Exception as e:
            logger.warning(f"[{caller}] set_run_id failed: {e} – continuing")

    if not verbose:
        logger.info("Starting data validation")

    validation_issues = 0

    # ---- Check 1: Null values ----------------------------------------------
    try:
        null_counts = df.isnull().sum()
        total_nulls = int(null_counts.sum())
        if total_nulls > 0:
            validation_issues += total_nulls
            try:
                null_report = null_counts[null_counts > 0].to_string()
            except Exception:
                null_report = str(null_counts[null_counts > 0].to_dict())
            logger.warning(
                f"[{caller}] remaining null values after cleaning:\n{null_report}"
            )
        else:
            if verbose:
                logger.info("No null values found")
    except Exception as e:
        logger.error(f"[{caller}] null value check failed: {e}")

    # ---- Check 2: Negative monetary values ---------------------------------
    monetary_cols = ["total_amount", "price", "shipping_cost"]
    for col in monetary_cols:
        try:
            if col not in df.columns:
                continue

            neg_count = int((df[col] < 0).sum())
            if neg_count == 0:
                if verbose:
                    logger.info(f"No negative values in '{col}'")
                continue

            # Negatives found
            validation_issues += neg_count

            if auto_fix:
                try:
                    logger.warning(
                        f"[{caller}] {neg_count:,} negative values in '{col}' – "
                        "setting to 0 (auto_fix=True)"
                    )
                    df.loc[df[col] < 0, col] = 0
                except Exception as fix_err:
                    logger.error(
                        f"[{caller}] auto_fix failed for '{col}': {fix_err} – "
                        "negatives not corrected"
                    )
            else:
                logger.error(
                    f"[{caller}] found {neg_count:,} negative values in '{col}'. "
                    "Fix data or use auto_fix=True"
                )
                raise ValueError(
                    f"[{caller}] validation failed: {neg_count:,} negative values in '{col}'. "
                    "Fix data before proceeding or set auto_fix=True."
                )

        except ValueError:
            raise  # re-raise intentional validation failures
        except Exception as e:
            logger.error(f"[{caller}] negative value check failed for '{col}': {e}")

    # ---- Check 3: Final summary --------------------------------------------
    try:
        if validation_issues == 0:
            if verbose:
                logger.info("Data validation passed: no issues found")
        else:
            if auto_fix:
                logger.info(f"Data validation complete: fixed {validation_issues:,} issues")
            else:
                logger.warning(f"Data validation found {validation_issues:,} issues")
    except Exception as e:
        logger.warning(f"[{caller}] validation summary logging failed (non-fatal): {e}")

    return df