# src/n1f_sanity_check_wrapper.py - ERROR-HARDENED VERSION

"""
Compatibility wrapper for sanity check functions.

Error Handling Strategy:
- comprehensive_data_check: validates df_clean is a non-empty DataFrame before
                            delegating to run_sanity_checks; validates config and
                            rfm_df types (soft - non-fatal on bad type); catches
                            ALL exceptions from run_sanity_checks so a crash there
                            never propagates to the notebook; always returns a dict
                            with the expected keys even if everything fails

Usage:
    from n1f_sanity_check_wrapper import comprehensive_data_check

    results = comprehensive_data_check(
        df_clean=df_clean,
        rfm_df=rfm_df,
        config=config,
        run_id=run_id,
        verbose=True
    )
"""

import pandas as pd
from typing import Dict, Optional, Any

from n1f_sanity_check import run_sanity_checks, validate_processed_files


# ---------------------------------------------------------------------------
# Safe fallback result – returned whenever we cannot produce a real result
# ---------------------------------------------------------------------------
def _empty_result(reason: str = "check could not be performed") -> Dict[str, Any]:
    """
    Returns a safe fallback result for comprehensive_data_check.

    This function is called whenever comprehensive_data_check cannot produce a valid result.
    The returned dictionary has the same structure as a successful comprehensive_data_check,
    but all values are set to safe defaults.

    Parameters:
        reason (str): reason why comprehensive_data_check could not produce a result
            (default: "check could not be performed")

    Returns:
        Dict[str, Any]: safe fallback result
    """
    return {
        "all_passed":      False,
        "total_checks":    0,
        "passed_checks":   0,
        "failed_checks":   1,
        "warning_checks":  0,
        "messages":        [f"comprehensive_data_check: {reason}"],
    }


# ===========================================================================
# comprehensive_data_check
# ===========================================================================

def comprehensive_data_check(
    df_clean: pd.DataFrame,
    rfm_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive data check wrapper that delegates to run_sanity_checks.

    Error handling:
    - Validates df_clean is a non-empty DataFrame; returns _empty_result immediately
      on bad input so the caller always gets a dict with the expected keys
    - Validates rfm_df is a DataFrame or None; coerces bad type to None with a warning
    - Validates config is a dict or None; ignores bad type with a warning (config is
      not currently used by the underlying checks, so this is safe)
    - run_id is validated as a non-empty string; bad values are warned and ignored
    - The entire run_sanity_checks call is wrapped in try/except; any crash there
      is caught and returned as a failed _empty_result so the notebook never sees
      an unhandled exception from this function
    - Result key remapping is guarded; missing keys from run_sanity_checks fall
      back to 0 / [] so the returned dict always has all five expected keys

    Args:
        df_clean: Cleaned transaction DataFrame
        rfm_df:   RFM customer DataFrame (optional)
        config:   Configuration dictionary (optional, not used by underlying checks)
        run_id:   Run ID for logging correlation
        verbose:  Whether to print detailed output

    Returns:
        Dict with keys:
            all_passed      : bool
            total_checks    : int
            passed_checks   : int
            failed_checks   : int
            warning_checks  : int
            messages        : List[str]
    """
    import logging
    _log = logging.getLogger(__name__)
    caller = "comprehensive_data_check"

    # ---- Validate df_clean -------------------------------------------------
    if not isinstance(df_clean, pd.DataFrame):
        _log.error(
            f"[{caller}] df_clean must be a pd.DataFrame, got {type(df_clean).__name__}"
        )
        return _empty_result(f"df_clean must be a pd.DataFrame, got {type(df_clean).__name__}")

    if df_clean.empty:
        _log.error(f"[{caller}] df_clean is an empty DataFrame")
        return _empty_result("df_clean is an empty DataFrame")

    # ---- Validate rfm_df (soft) --------------------------------------------
    if rfm_df is not None and not isinstance(rfm_df, pd.DataFrame):
        _log.warning(
            f"[{caller}] rfm_df must be a pd.DataFrame or None, "
            f"got {type(rfm_df).__name__} – treating as None"
        )
        rfm_df = None

    # ---- Validate config (soft – informational only) -----------------------
    if config is not None and not isinstance(config, dict):
        _log.warning(
            f"[{caller}] config must be a dict or None, "
            f"got {type(config).__name__} – ignored"
        )
        # config is not forwarded to run_sanity_checks anyway; this is safe

    # ---- Validate run_id (soft) --------------------------------------------
    if run_id is not None:
        if not isinstance(run_id, str) or not run_id.strip():
            _log.warning(
                f"[{caller}] run_id={run_id!r} is not a non-empty string – ignored"
            )
            run_id = None

    # ---- Delegate to run_sanity_checks -------------------------------------
    try:
        raw = run_sanity_checks(
            df=df_clean,
            rfm_df=rfm_df,
            run_id=run_id,
            verbose=verbose,
        )
    except Exception as e:
        _log.error(f"[{caller}] run_sanity_checks raised an unexpected exception: {e}")
        return _empty_result(f"run_sanity_checks failed with: {e}")

    # ---- Validate shape of returned dict -----------------------------------
    if not isinstance(raw, dict):
        _log.error(
            f"[{caller}] run_sanity_checks returned {type(raw).__name__} instead of dict"
        )
        return _empty_result("run_sanity_checks returned a non-dict result")

    # ---- Remap keys (with safe .get() fallbacks) ---------------------------
    try:
        checks_performed = int(raw.get("checks_performed", 0))
        issues_found     = int(raw.get("issues_found", 0))
        errors           = raw.get("errors", [])
        warnings         = raw.get("warnings", [])

        if not isinstance(errors, list):
            errors = []
        if not isinstance(warnings, list):
            warnings = []

        passed_checks = max(0, checks_performed - issues_found)

        return {
            "all_passed":     bool(raw.get("all_passed", False)),
            "total_checks":   checks_performed,
            "passed_checks":  passed_checks,
            "failed_checks":  len(errors),
            "warning_checks": len(warnings),
            "messages":       errors + warnings,
        }

    except Exception as e:
        _log.error(f"[{caller}] result remapping failed: {e}")
        return _empty_result(f"result remapping failed: {e}")