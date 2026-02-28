# src/n1d_rfm_features.py - ERROR-HARDENED VERSION

"""
n1d_rfm_features.py - RFM Feature Engineering

Error Handling Strategy:
- clear_rfm_cache   : validates CACHE_DIR exists; catches OSError per file so one
                      undeletable file does not abort deletion of others
- clear_old_cache   : validates ttl_hours is a positive number; catches stat() and
                      unlink() errors per file independently; falls back to config
                      default on bad config value
- build_rfm_features: validates df type and required columns; each of the 10 feature
                      blocks is independently try/excepted; guards all divisions and
                      normalisation steps; cache read/write errors are non-fatal;
                      returns partial DataFrame on recoverable errors, raises only
                      when the input is completely unusable
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import Optional
from n1a_utils import setup_logger, get_project_root, load_config, set_run_id

logger = setup_logger(__name__)
PROJECT_ROOT = get_project_root()

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    logger.warning(f"Could not create cache directory '{CACHE_DIR}': {e} – caching will be disabled")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_div(series_or_val, divisor, fallback=np.nan):
    """Element-wise division that replaces ZeroDivisionError / NaN-inf with fallback."""
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = series_or_val / divisor
        if isinstance(result, pd.Series):
            return result.replace([np.inf, -np.inf], fallback)
        return fallback if (divisor == 0 or np.isnan(divisor)) else result
    except Exception:
        return fallback


def _mode_or_fallback(series: pd.Series, fallback=None):
    """Return mode of a series, or fallback if series is empty or mode fails."""
    try:
        m = series.mode()
        return m.iloc[0] if not m.empty else (fallback if fallback is not None else "Unknown")
    except Exception:
        return fallback if fallback is not None else "Unknown"


# ===========================================================================
# 1. clear_rfm_cache
# ===========================================================================

def clear_rfm_cache() -> None:
    """
    Clear all RFM cache files.

    Error handling:
    - Warns and returns early if CACHE_DIR does not exist (graceful no-op)
    - Each file deletion is independently try/excepted so one locked or
      permission-denied file does not abort deletion of the others
    - OSError per file is logged at warning level (not error) so the caller
      is not blocked
    """
    caller = "clear_rfm_cache"

    if not CACHE_DIR.exists():
        logger.warning(f"[{caller}] cache directory does not exist: {CACHE_DIR} – nothing to clear")
        return

    try:
        cache_files = list(CACHE_DIR.glob("rfm_cache_*.parquet"))
    except OSError as e:
        logger.error(f"[{caller}] cannot list cache directory '{CACHE_DIR}': {e}")
        return

    if not cache_files:
        logger.info("No RFM cache files to clear")
        return

    removed = 0
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            removed += 1
        except OSError as e:
            logger.warning(f"[{caller}] could not delete '{cache_file.name}': {e} – skipped")

    logger.info(f"Cleared {removed} of {len(cache_files)} RFM cache file(s)")


# ===========================================================================
# 2. clear_old_cache
# ===========================================================================

def clear_old_cache(ttl_hours: Optional[int] = None) -> None:
    """
    Clear cache files older than TTL (time-to-live).

    Error handling:
    - Validates ttl_hours is a positive number; falls back to config value on None,
      falls back to 24 if config is unavailable or malformed
    - Warns and returns early if CACHE_DIR does not exist
    - stat() and unlink() are individually guarded per file so one inaccessible
      file does not abort the rest
    - time.time() failure is caught (extremely unlikely, but guarded)

    Args:
        ttl_hours: Cache lifetime in hours. If None, reads from config.
    """
    caller = "clear_old_cache"

    # ---- Resolve TTL -------------------------------------------------------
    if ttl_hours is None:
        try:
            config = load_config(PROJECT_ROOT / "config.yaml")
            ttl_hours = config.get("rfm", {}).get("cache_ttl_hours", 24)
        except Exception as e:
            logger.warning(f"[{caller}] could not load config for ttl_hours: {e} – defaulting to 24h")
            ttl_hours = 24

    if not isinstance(ttl_hours, (int, float)) or ttl_hours <= 0:
        logger.warning(
            f"[{caller}] ttl_hours={ttl_hours!r} is not a positive number – defaulting to 24h"
        )
        ttl_hours = 24

    # ---- Cache directory check ---------------------------------------------
    if not CACHE_DIR.exists():
        logger.warning(f"[{caller}] cache directory does not exist: {CACHE_DIR} – nothing to clear")
        return

    try:
        cache_files = list(CACHE_DIR.glob("rfm_cache_*.parquet"))
    except OSError as e:
        logger.error(f"[{caller}] cannot list cache directory '{CACHE_DIR}': {e}")
        return

    # ---- Current time ------------------------------------------------------
    try:
        current_time = time.time()
    except Exception as e:
        logger.error(f"[{caller}] time.time() failed: {e} – cannot determine file ages")
        return

    ttl_seconds = ttl_hours * 3600
    removed = 0

    for cache_file in cache_files:
        try:
            file_age = current_time - cache_file.stat().st_mtime
        except OSError as e:
            logger.warning(f"[{caller}] cannot stat '{cache_file.name}': {e} – skipped")
            continue

        if file_age > ttl_seconds:
            try:
                cache_file.unlink()
                removed += 1
            except OSError as e:
                logger.warning(f"[{caller}] could not delete '{cache_file.name}': {e} – skipped")

    if removed > 0:
        logger.info(f"Cleared {removed} old cache file(s) (older than {ttl_hours}h)")
    else:
        logger.info(f"No cache files older than {ttl_hours}h")


# ===========================================================================
# 3. build_rfm_features
# ===========================================================================

def build_rfm_features(
    df: pd.DataFrame,
    config_hash: Optional[str] = None,
    verbose: bool = True,
    run_id: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build customer-level RFM and behavioural features.

    Error handling:
    - Validates df is a non-empty DataFrame with required columns before any work
    - run_id and config load failures are non-fatal (warned and continued)
    - Cache read errors are non-fatal (logged and re-computed)
    - Cache write errors are non-fatal (logged; result still returned)
    - Each of the 10 feature blocks (recency, frequency, monetary, net_monetary,
      avg_order_value, tenure, discount, category diversity, return rate,
      last_order_was_return, categorical features) is independently try/excepted
    - Normalisation guards against max == 0 (constant column)
    - Loyalty score weights are validated; missing keys fall back to 0.0
    - Churn flag guards against all-NaN recency_days
    - Validation section is fully wrapped and non-fatal
    - Returns whatever was successfully merged; raises only on unrecoverable input

    Returns:
        DataFrame with one row per customer and RFM features
    """
    caller = "build_rfm_features"

    # ---- run_id ------------------------------------------------------------
    if run_id:
        try:
            set_run_id(run_id)
        except Exception as e:
            logger.warning(f"[{caller}] set_run_id failed: {e} – continuing without run_id")

    # ---- Input validation --------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{caller}] expected pd.DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError(f"[{caller}] input DataFrame is empty")

    required_cols = {"customer_id", "order_date", "returned", "total_amount"}
    missing_req = required_cols - set(df.columns)
    if missing_req:
        raise ValueError(f"[{caller}] missing required columns: {missing_req}")

    logger.info("Starting RFM feature engineering")
    logger.info(f"Input shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    try:
        logger.info(f"Unique customers: {df['customer_id'].nunique():,}")
    except Exception:
        pass

    # ---- Load config -------------------------------------------------------
    rfm_config: dict = {}
    try:
        config    = load_config(PROJECT_ROOT / "config.yaml")
        rfm_config = config.get("rfm", {})
    except Exception as e:
        logger.warning(f"[{caller}] could not load config: {e} – using defaults")

    # ---- Cache check -------------------------------------------------------
    if use_cache and config_hash:
        if not isinstance(config_hash, str) or not config_hash.strip():
            logger.warning(f"[{caller}] config_hash is invalid ({config_hash!r}) – cache skipped")
        else:
            try:
                cache_file = CACHE_DIR / f"rfm_cache_{config_hash}.parquet"
                if cache_file.exists():
                    logger.info(f"Loading cached RFM features from {cache_file.name}")
                    rfm_cached = pd.read_parquet(cache_file)
                    logger.info(f"Loaded cached features: {rfm_cached.shape[0]:,} customers")
                    return rfm_cached
            except Exception as e:
                logger.warning(f"[{caller}] cache read failed: {e} – recomputing")

    # ---- Analysis date & churn threshold -----------------------------------
    try:
        analysis_date = pd.to_datetime(df["order_date"]).max()
        if pd.isna(analysis_date):
            raise ValueError("all order_date values are NaT")
    except Exception as e:
        raise ValueError(f"[{caller}] cannot determine analysis_date from order_date: {e}")

    churn_days = rfm_config.get("churn_threshold_days", 120)
    if not isinstance(churn_days, (int, float)) or churn_days <= 0:
        logger.warning(f"[{caller}] invalid churn_threshold_days={churn_days!r} – defaulting to 120")
        churn_days = 120

    logger.info(f"Analysis date: {analysis_date.date()}")
    logger.info(f"Churn threshold: {churn_days} days")

    # ---- Partition rows ----------------------------------------------------
    try:
        non_returned  = df[df["returned"] == 0].copy()
        returned_only = df[df["returned"] == 1].copy()
    except Exception as e:
        raise ValueError(f"[{caller}] cannot partition on 'returned' column: {e}")

    if verbose:
        try:
            total = len(df)
            nr    = len(non_returned)
            ret   = len(returned_only)
            logger.info(f"Total orders: {total:,}")
            logger.info(f"Non-returned : {nr:,} ({nr/total*100:.1f}%)")
            logger.info(f"Returned     : {ret:,} ({ret/total*100:.2f}%)")
            cust_ret = df.groupby("customer_id")["returned"].agg(["sum", "count"])
            pure     = (cust_ret["sum"] == cust_ret["count"]).sum()
            mixed    = ((cust_ret["sum"] > 0) & (cust_ret["sum"] < cust_ret["count"])).sum()
            never    = (cust_ret["sum"] == 0).sum()
            logger.info(f"Never returned : {never:,}")
            logger.info(f"Mixed returners: {mixed:,}")
            logger.info(f"Pure returners : {pure:,} (RFM set to NaN, flagged as churned)")
        except Exception as e:
            logger.warning(f"[{caller}] return summary logging failed: {e}")

    # ---- Collect computed series -------------------------------------------
    series_list: list = []

    # RECENCY
    try:
        recency = (
            non_returned.groupby("customer_id")["order_date"]
            .max()
            .apply(lambda x: (analysis_date - pd.to_datetime(x)).days)
            .rename("recency_days")
        )
        series_list.append(recency)
        if verbose:
            logger.info("Recency calculated (last non-returned order)")
    except Exception as e:
        logger.error(f"[{caller}] recency calculation failed: {e}")

    # FREQUENCY
    try:
        frequency = non_returned.groupby("customer_id").size().rename("frequency")
        series_list.append(frequency)
        if verbose:
            logger.info("Frequency calculated")
    except Exception as e:
        logger.error(f"[{caller}] frequency calculation failed: {e}")

    # MONETARY
    monetary: Optional[pd.Series] = None
    try:
        monetary = (
            non_returned.groupby("customer_id")["total_amount"]
            .sum()
            .rename("monetary")
        )
        series_list.append(monetary)
        if verbose:
            logger.info("Monetary calculated")
    except Exception as e:
        logger.error(f"[{caller}] monetary calculation failed: {e}")

    # NET MONETARY
    try:
        if monetary is not None:
            returned_rev = (
                returned_only.groupby("customer_id")["total_amount"]
                .sum()
                .rename("_returned_rev")
            )
            net_monetary = monetary.sub(returned_rev, fill_value=0).rename("net_monetary")
            series_list.append(net_monetary)
            if verbose:
                logger.info("Net monetary calculated")
        else:
            logger.warning(f"[{caller}] skipping net_monetary – monetary not available")
    except Exception as e:
        logger.error(f"[{caller}] net_monetary calculation failed: {e}")

    # AVG ORDER VALUE
    try:
        if monetary is not None and "frequency" in [s.name for s in series_list]:
            freq_s = next(s for s in series_list if s.name == "frequency")
            avg_order_value = (monetary / freq_s).replace([np.inf, -np.inf], np.nan).rename("avg_order_value")
            series_list.append(avg_order_value)
            if verbose:
                logger.info("Avg order value calculated")
    except Exception as e:
        logger.error(f"[{caller}] avg_order_value calculation failed: {e}")

    # TENURE
    try:
        tenure = (
            df.groupby("customer_id")["order_date"]
            .apply(lambda x: (pd.to_datetime(x).max() - pd.to_datetime(x).min()).days)
            .rename("tenure_days")
        )
        series_list.append(tenure)
        if verbose:
            logger.info("Tenure calculated")
    except Exception as e:
        logger.error(f"[{caller}] tenure calculation failed: {e}")

    # DISCOUNT USAGE RATE
    try:
        if "discount" in df.columns:
            discount_usage = (
                df.groupby("customer_id")
                .apply(lambda x: (x["discount"] > 0).mean())
                .rename("discount_usage_rate")
            )
            series_list.append(discount_usage)
            if verbose:
                logger.info("Discount usage rate calculated")
        else:
            logger.warning(f"[{caller}] 'discount' column not found – discount_usage_rate skipped")
    except Exception as e:
        logger.error(f"[{caller}] discount_usage_rate calculation failed: {e}")

    # CATEGORY DIVERSITY
    try:
        if "category" in non_returned.columns:
            category_diversity = (
                non_returned.groupby("customer_id")["category"]
                .nunique()
                .rename("category_diversity")
            )
            series_list.append(category_diversity)
            if verbose:
                logger.info("Category diversity calculated")
        else:
            logger.warning(f"[{caller}] 'category' column not found – category_diversity skipped")
    except Exception as e:
        logger.error(f"[{caller}] category_diversity calculation failed: {e}")

    # RETURN RATE
    try:
        return_rate = df.groupby("customer_id")["returned"].mean().rename("return_rate")
        series_list.append(return_rate)
        if verbose:
            logger.info("Return rate calculated")
    except Exception as e:
        logger.error(f"[{caller}] return_rate calculation failed: {e}")

    # LAST ORDER WAS RETURN
    try:
        last_order_any = df.sort_values("order_date").groupby("customer_id").last()
        last_order_was_return = (
            last_order_any["returned"].astype(int).rename("last_order_was_return")
        )
        series_list.append(last_order_was_return)
        if verbose:
            logger.info(f"last_order_was_return: {int(last_order_was_return.sum()):,} customers")
    except Exception as e:
        logger.error(f"[{caller}] last_order_was_return calculation failed: {e}")

    # CATEGORICAL FEATURES
    for col, out_name in [
        ("region",          "preferred_region"),
        ("payment_method",  "preferred_payment"),
        ("customer_gender", "preferred_gender"),
    ]:
        try:
            if col in df.columns:
                s = (
                    df.groupby("customer_id")[col]
                    .apply(lambda x: _mode_or_fallback(x, "Unknown"))
                    .rename(out_name)
                )
                series_list.append(s)
                if verbose:
                    logger.info(f"{out_name} calculated")
            else:
                logger.warning(f"[{caller}] column '{col}' not found – {out_name} skipped")
        except Exception as e:
            logger.error(f"[{caller}] {out_name} calculation failed: {e}")

    try:
        if "customer_age" in df.columns:
            def safe_age_mode(x):
                val = _mode_or_fallback(x, "-1")
                return int(val) if val != "-1" else -1
            preferred_age = (
                df.groupby("customer_id")["customer_age"]
                .apply(safe_age_mode)
                .rename("preferred_age")
            )
            series_list.append(preferred_age)
            if verbose:
                logger.info("preferred_age calculated")
        else:
            logger.warning(f"[{caller}] 'customer_age' column not found – preferred_age skipped")
    except Exception as e:
        logger.error(f"[{caller}] preferred_age calculation failed: {e}")

    # ---- Combine all features ----------------------------------------------
    try:
        rfm_full = pd.DataFrame({"customer_id": df["customer_id"].unique()})
        for s in series_list:
            try:
                rfm_full = rfm_full.merge(s.reset_index(), on="customer_id", how="left")
            except Exception as merge_err:
                logger.warning(
                    f"[{caller}] failed to merge feature '{s.name}': {merge_err} – skipped"
                )
        logger.info(f"Features combined: {rfm_full.shape[1]} columns")
    except Exception as e:
        raise RuntimeError(f"[{caller}] feature combination failed: {e}")

    # ---- Loyalty score -----------------------------------------------------
    try:
        rec_max  = rfm_full["recency_days"].max()  if "recency_days"  in rfm_full.columns else None
        freq_max = rfm_full["frequency"].max()     if "frequency"     in rfm_full.columns else None
        mon_max  = rfm_full["monetary"].max()      if "monetary"      in rfm_full.columns else None

        weights = rfm_config.get("loyalty_weights", {"recency": 0.3, "frequency": 0.4, "monetary": 0.3})
        if not isinstance(weights, dict):
            logger.warning(f"[{caller}] loyalty_weights is not a dict - using defaults")
            weights = {"recency": 0.3, "frequency": 0.4, "monetary": 0.3}

        w_r = float(weights.get("recency",   0.0))
        w_f = float(weights.get("frequency", 0.0))
        w_m = float(weights.get("monetary",  0.0))

        rec_norm  = 1 - _safe_div(rfm_full.get("recency_days",  pd.Series(dtype=float)), rec_max  or 1, 0.0) if rec_max  else 0.0
        freq_norm = _safe_div(rfm_full.get("frequency",  pd.Series(dtype=float)), freq_max or 1, 0.0) if freq_max else 0.0
        mon_norm  = _safe_div(rfm_full.get("monetary",   pd.Series(dtype=float)), mon_max  or 1, 0.0) if mon_max  else 0.0

        rfm_full["loyalty_score"] = w_r * rec_norm + w_f * freq_norm + w_m * mon_norm
        logger.info("Loyalty score calculated")
    except Exception as e:
        logger.error(f"[{caller}] loyalty score calculation failed: {e}")

    # ---- Churn flag --------------------------------------------------------
    try:
        if "recency_days" in rfm_full.columns:
            rfm_full["churn"] = (
                (rfm_full["recency_days"] > churn_days) | rfm_full["recency_days"].isna()
            ).astype("int8")
            logger.info(f"Churn flag applied (threshold: {churn_days} days)")
        else:
            logger.warning(f"[{caller}] recency_days not available – churn flag skipped")
    except Exception as e:
        logger.error(f"[{caller}] churn flag calculation failed: {e}")

    # ---- Pure-return count -------------------------------------------------
    pure_return_count = 0
    try:
        if "recency_days" in rfm_full.columns:
            pure_return_count = int(rfm_full["recency_days"].isna().sum())
            if pure_return_count > 0:
                logger.info(
                    f"Pure-return customers: {pure_return_count:,} — "
                    "recency/frequency/monetary=NaN, churn=1"
                )
    except Exception:
        pass

    # ---- Validation (non-fatal) -------------------------------------------
    try:
        nullable_cols = [
            "recency_days", "frequency", "monetary", "avg_order_value",
            "category_diversity", "net_monetary", "loyalty_score",
        ]
        other_cols = [
            c for c in rfm_full.columns
            if c not in nullable_cols + ["customer_id"]
        ]
        unexpected_nans = rfm_full[other_cols].isna().sum()
        bad = unexpected_nans[unexpected_nans > 0]
        if not bad.empty:
            for col, count in bad.items():
                logger.warning(f"[{caller}] unexpected NaN in '{col}': {count}")
        else:
            logger.info("No unexpected NaN values")

        if "net_monetary" in rfm_full.columns and "monetary" in rfm_full.columns:
            valid = rfm_full[rfm_full["monetary"].notna()]
            violated = (valid["net_monetary"] > valid["monetary"]).sum()
            if violated > 0:
                logger.warning(f"[{caller}] net_monetary > monetary for {violated:,} customers")
            else:
                logger.info("net_monetary <= monetary for all customers")
    except Exception as e:
        logger.warning(f"[{caller}] validation block failed (non-fatal): {e}")

    # ---- Cache write -------------------------------------------------------
    if use_cache and config_hash and isinstance(config_hash, str) and config_hash.strip():
        try:
            cache_file = CACHE_DIR / f"rfm_cache_{config_hash}.parquet"
            rfm_full.to_parquet(cache_file, index=False)
            logger.info(f"Cached RFM features to {cache_file.name}")
        except Exception as e:
            logger.warning(f"[{caller}] cache write failed (non-fatal): {e}")

    # ---- Summary -----------------------------------------------------------
    if verbose:
        try:
            logger.info(f"Total customers       : {len(rfm_full):,}")
            if "churn" in rfm_full.columns:
                logger.info(
                    f"Churned               : {rfm_full['churn'].sum():,} "
                    f"({rfm_full['churn'].mean()*100:.1f}%)"
                )
            logger.info(f"Pure-return customers : {pure_return_count:,}")
            for col, label, fmt in [
                ("recency_days",  "Avg recency",       "{:.1f} days"),
                ("frequency",     "Avg frequency",     "{:.1f} orders"),
                ("monetary",      "Avg monetary",      "${:.2f}"),
                ("net_monetary",  "Avg net_monetary",  "${:.2f}"),
                ("loyalty_score", "Avg loyalty score", "{:.3f}"),
            ]:
                if col in rfm_full.columns:
                    val = rfm_full[col].mean()
                    logger.info(f"{label:22s}: {fmt.format(val)}")
        except Exception as e:
            logger.warning(f"[{caller}] summary logging failed (non-fatal): {e}")

    logger.info("RFM feature engineering complete")
    logger.info(f"Output shape: {rfm_full.shape[0]:,} customers x {rfm_full.shape[1]} features")

    return rfm_full