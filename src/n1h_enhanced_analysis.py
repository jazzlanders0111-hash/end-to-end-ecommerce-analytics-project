# src/n1h_enhanced_analysis.py

"""
n1h_enhanced_analysis.py

Enhanced business analysis functions for e-commerce customer analytics.
Provides data quality scoring, temporal analysis, and customer retention insights.

Error Handling Strategy:
- calculate_data_quality_score  : validates df + optional int; guards zero-division on
                                  empty df and zero initial_rows; returns partial results
- analyze_temporal_distribution : validates df, column names, and date parseability;
                                  each metric block is independently guarded; returns
                                  whatever was successfully computed
- analyze_churn_and_retention   : validates rfm_df and required columns; guards all
                                  division operations; returns partial results on failure
- generate_business_summary     : validates all four input dicts/dfs and required columns;
                                  each metric is computed independently; returns whatever
                                  succeeded
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_dataframe(df: Any, caller: str, min_rows: int = 1) -> bool:
    """Return True if df is a non-empty DataFrame with at least min_rows rows."""
    if not isinstance(df, pd.DataFrame):
        logger.error(f"[{caller}] expected pd.DataFrame, got {type(df).__name__}")
        return False
    if len(df) < min_rows:
        logger.warning(f"[{caller}] DataFrame has {len(df)} rows (need >= {min_rows})")
        return False
    return True


def _require_columns(df: pd.DataFrame, cols: list, caller: str) -> list:
    """Return list of columns that are missing from df."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error(f"[{caller}] missing required columns: {missing}")
    return missing


def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that returns fallback instead of raising ZeroDivisionError."""
    try:
        if denominator == 0:
            return fallback
        return numerator / denominator
    except Exception:
        return fallback


# ===========================================================================
# 1. calculate_data_quality_score
# ===========================================================================

def calculate_data_quality_score(
    df_clean: pd.DataFrame,
    initial_row_count: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Calculate comprehensive data quality score.

    Scoring components:
    - Completeness (40%)             : percentage of non-null values
    - Business rule compliance (30%) : no negative prices, valid discounts, etc.
    - Data retention (30%)           : percentage of records kept after cleaning

    Error handling:
    - Validates df_clean is a non-empty DataFrame
    - Validates initial_row_count is a positive int when supplied
    - Guards every division against ZeroDivisionError
    - Each business-rule check runs independently; a missing column is silently skipped
    - Returns {} on unrecoverable input errors; partial dict on computation errors
    - verbose print block is fully wrapped so a formatting error cannot mask results

    Returns:
        Dictionary with quality metrics and score, or {} on fatal input error
    """
    caller = "calculate_data_quality_score"

    if not _require_dataframe(df_clean, caller):
        return {}

    if initial_row_count is not None:
        if not isinstance(initial_row_count, int) or initial_row_count <= 0:
            logger.warning(
                f"[{caller}] initial_row_count={initial_row_count!r} is not a positive int "
                "- treating as None (will use final row count)"
            )
            initial_row_count = None

    logger.info("=" * 60)
    logger.info("Calculating Data Quality Score")
    logger.info("=" * 60)

    final_rows = len(df_clean)
    initial_rows = initial_row_count if initial_row_count is not None else final_rows
    rows_removed = initial_rows - final_rows

    # ---- Completeness ------------------------------------------------------
    try:
        completeness = (1 - _safe_div(float(df_clean.isna().sum().sum()), float(df_clean.size), 0.0)) * 100
    except Exception as e:
        logger.error(f"[{caller}] completeness calculation failed: {e}")
        completeness = 0.0

    # ---- Business rule compliance ------------------------------------------
    violations = 0
    try:
        if "price" in df_clean.columns:
            try:
                violations += int((df_clean["price"] < 0).sum())
            except Exception as e:
                logger.warning(f"[{caller}] price violation check failed: {e}")

        if "discount" in df_clean.columns:
            try:
                violations += int(
                    ((df_clean["discount"] < 0) | (df_clean["discount"] > 1)).sum()
                )
            except Exception as e:
                logger.warning(f"[{caller}] discount violation check failed: {e}")

        if "quantity" in df_clean.columns:
            try:
                violations += int((df_clean["quantity"] <= 0).sum())
            except Exception as e:
                logger.warning(f"[{caller}] quantity violation check failed: {e}")
    except Exception as e:
        logger.error(f"[{caller}] business rule checks failed: {e}")

    compliance_rate = (1 - _safe_div(violations, final_rows, 0.0)) * 100
    retention_rate = _safe_div(final_rows, initial_rows, fallback=100.0) * 100

    # ---- Weighted quality score --------------------------------------------
    try:
        quality_score = (
            (completeness / 100 * 40)
            + (compliance_rate / 100 * 30)
            + (retention_rate / 100 * 30)
        )
    except Exception as e:
        logger.error(f"[{caller}] quality score aggregation failed: {e}")
        quality_score = 0.0

    results: Dict[str, float] = {
        "quality_score": quality_score,
        "completeness": completeness,
        "compliance_rate": compliance_rate,
        "retention_rate": retention_rate,
        "initial_rows": float(initial_rows),
        "final_rows": float(final_rows),
        "rows_removed": float(rows_removed),
        "violations": float(violations),
    }

    if verbose:
        try:
            rows_removed_pct = _safe_div(rows_removed, initial_rows, 0.0) * 100
            print("=" * 80)
            print("DATA QUALITY SCORECARD".center(80))
            print("=" * 80)
            print(f"Completeness              : {completeness:.1f}%")
            print(f"Records processed         : {initial_rows:,} -> {final_rows:,}")
            print(f"Records removed           : {rows_removed:,} ({rows_removed_pct:.2f}%)")
            print(f"Business rule compliance  : {compliance_rate:.1f}%")
            print(f"Overall data quality      : {quality_score:.1f}/100")
            print("\nData Quality Assessment:")
            if quality_score >= 90:
                print("EXCELLENT - Data is production-ready")
            elif quality_score >= 75:
                print("GOOD - Minor data quality issues addressed")
            elif quality_score >= 60:
                print("FAIR - Some data quality concerns remain")
            else:
                print("POOR - Significant data quality issues")
        except Exception as e:
            logger.warning(f"[{caller}] verbose output failed (results still valid): {e}")

    logger.info(f"Data quality score: {quality_score:.1f}/100")
    logger.info(f"Completeness: {completeness:.1f}%")
    logger.info(f"Compliance: {compliance_rate:.1f}%")

    return results


# ===========================================================================
# 2. analyze_temporal_distribution
# ===========================================================================

def analyze_temporal_distribution(
    df: pd.DataFrame,
    date_column: str = "order_date",
    amount_column: str = "total_amount",
    customer_column: str = "customer_id",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze temporal data distribution and patterns.

    Error handling:
    - Validates df is a non-empty DataFrame
    - Validates column name args are non-empty strings and exist in df
    - Coerces date column with errors='coerce'; warns if rows become NaT
    - Each metric block (monthly agg, gap detection, growth rate) is
      independently try/excepted so one failure does not abort others
    - Guards all divisions; returns {} on fatal input error, partial dict otherwise
    - verbose print block is fully wrapped

    Returns:
        Dictionary with temporal metrics, or {} on fatal input error
    """
    caller = "analyze_temporal_distribution"

    if not _require_dataframe(df, caller):
        return {}

    for param_name, param_val in [
        ("date_column", date_column),
        ("amount_column", amount_column),
        ("customer_column", customer_column),
    ]:
        if not isinstance(param_val, str) or not param_val.strip():
            logger.error(f"[{caller}] '{param_name}' must be a non-empty string")
            return {}

    if _require_columns(df, [date_column, amount_column, customer_column], caller):
        return {}

    logger.info("=" * 60)
    logger.info("Analyzing Temporal Distribution")
    logger.info("=" * 60)

    # ---- Date parsing ------------------------------------------------------
    try:
        df_temp = df.copy()
        df_temp[date_column] = pd.to_datetime(df_temp[date_column], errors="coerce")
        nat_count = int(df_temp[date_column].isna().sum())
        if nat_count > 0:
            logger.warning(
                f"[{caller}] {nat_count:,} rows have unparseable dates and will be excluded"
            )
        df_temp["year_month"] = df_temp[date_column].dt.strftime("%Y-%m") #type: ignore
    except Exception as e:
        logger.error(f"[{caller}] date parsing failed: {e}")
        return {}

    valid_temp = df_temp.dropna(subset=[date_column])
    if valid_temp.empty:
        logger.error(f"[{caller}] no valid dates remain after parsing")
        return {}

    results: Dict[str, Any] = {}

    # ---- Monthly aggregations ----------------------------------------------
    try:
        monthly_revenue = valid_temp.groupby("year_month")[amount_column].sum()
        monthly_orders = valid_temp.groupby("year_month").size()
        monthly_customers = valid_temp.groupby("year_month")[customer_column].nunique()
    except Exception as e:
        logger.error(f"[{caller}] monthly aggregation failed: {e}")
        return {}

    if monthly_revenue.empty:
        logger.warning(f"[{caller}] monthly revenue series is empty")
        return {}

    # ---- Core metrics ------------------------------------------------------
    try:
        avg_monthly_revenue = float(monthly_revenue.mean())
        revenue_volatility = _safe_div(monthly_revenue.std(), monthly_revenue.mean(), 0.0) * 100

        results.update({
            "active_months": len(monthly_revenue),
            "peak_month": str(monthly_revenue.idxmax()),
            "peak_revenue": float(monthly_revenue.max()),
            "trough_month": str(monthly_revenue.idxmin()),
            "trough_revenue": float(monthly_revenue.min()),
            "avg_monthly_revenue": avg_monthly_revenue,
            "revenue_volatility": revenue_volatility,
            "monthly_revenue": monthly_revenue.to_dict(),
            "monthly_orders": monthly_orders.to_dict(),
            "monthly_customers": monthly_customers.to_dict(),
        })
    except Exception as e:
        logger.error(f"[{caller}] core metric computation failed: {e}")

    # ---- Dataset span ------------------------------------------------------
    try:
        results["dataset_span_days"] = int(
            (valid_temp[date_column].max() - valid_temp[date_column].min()).days
        )
    except Exception as e:
        logger.warning(f"[{caller}] dataset span calculation failed: {e}")
        results["dataset_span_days"] = None

    # ---- Data collection gaps ----------------------------------------------
    missing_dates: set = set()
    try:
        date_range = pd.date_range(
            start=valid_temp[date_column].min(),
            end=valid_temp[date_column].max(),
            freq="D",
        )
        dates_with_txn = set(valid_temp[date_column].dt.normalize()) #type: ignore
        missing_dates = set(date_range.normalize()) - dates_with_txn
        results["missing_dates_count"] = len(missing_dates)
    except Exception as e:
        logger.warning(f"[{caller}] gap detection failed: {e}")
        results["missing_dates_count"] = None

    # ---- Growth rate -------------------------------------------------------
    growth_rate: Optional[float] = None
    try:
        if len(monthly_revenue) >= 3:
            first_3 = float(monthly_revenue.head(3).mean())
            last_3 = float(monthly_revenue.tail(3).mean())
            growth_rate = _safe_div(last_3 - first_3, first_3, 0.0) * 100
    except Exception as e:
        logger.warning(f"[{caller}] growth rate calculation failed: {e}")

    results["growth_rate"] = growth_rate

    # ---- Verbose output ----------------------------------------------------
    if verbose:
        try:
            print("=" * 80)
            print("TEMPORAL DATA DISTRIBUTION ANALYSIS")
            print("=" * 80)
            print(f"Dataset span        : {results.get('dataset_span_days', 'N/A')} days")
            print(f"Active months       : {results.get('active_months', 'N/A')}")
            if "peak_month" in results:
                print(f"Peak month          : {results['peak_month']} (${results['peak_revenue']:,.0f})")
                print(f"Trough month        : {results['trough_month']} (${results['trough_revenue']:,.0f})")
                print(f"Avg monthly revenue : ${results['avg_monthly_revenue']:,.0f}")
                print(f"Revenue volatility  : {results['revenue_volatility']:.1f}% (CoV)")

            gap_count = results.get("missing_dates_count")
            if gap_count is not None:
                if gap_count > 0:
                    print(f"\nData Collection Gaps: {gap_count} days with no transactions")
                else:
                    print("\nNo data collection gaps detected")

            if growth_rate is not None:
                first_3 = float(monthly_revenue.head(3).mean())
                last_3 = float(monthly_revenue.tail(3).mean())
                print(f"\nGrowth Trend:")
                print(f"First 3 months avg : ${first_3:,.0f}")
                print(f"Last 3 months avg  : ${last_3:,.0f}")
                print(f"Growth rate        : {growth_rate:+.1f}%")
                if growth_rate > 10:
                    print("Strong positive growth trajectory")
                elif growth_rate > 0:
                    print("Modest growth observed")
                elif growth_rate > -10:
                    print("Slight decline in recent months")
                else:
                    print("Significant revenue decline - investigate urgently")
        except Exception as e:
            logger.warning(f"[{caller}] verbose output failed (results still valid): {e}")

    try:
        logger.info(
            f"Monthly revenue: ${results.get('avg_monthly_revenue', 0):,.0f} "
            f"(sigma={results.get('revenue_volatility', 0):.1f}%)"
        )
        logger.info(f"Data gaps: {results.get('missing_dates_count', 'N/A')} days")
        logger.info(f"Growth rate: {growth_rate:+.1f}%" if growth_rate is not None else "Growth rate: N/A")
    except Exception:
        pass

    return results


# ===========================================================================
# 3. analyze_churn_and_retention
# ===========================================================================

def analyze_churn_and_retention(
    rfm_df: pd.DataFrame,
    churn_threshold_days: int = 120,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Deep-dive analysis of customer churn and retention patterns.

    Error handling:
    - Validates rfm_df is a non-empty DataFrame
    - Validates required columns: frequency, monetary, churn, recency_days
    - Validates churn_threshold_days is a positive int
    - Guards every percentage/division with _safe_div
    - Each insight block is independently try/excepted
    - Returns {} on fatal input error, partial dict on computation errors

    Returns:
        Dictionary with churn and retention metrics, or {} on fatal input error
    """
    caller = "analyze_churn_and_retention"

    if not _require_dataframe(rfm_df, caller, min_rows=1):
        return {}

    required = ["frequency", "monetary", "churn", "recency_days"]
    if _require_columns(rfm_df, required, caller):
        return {}

    if not isinstance(churn_threshold_days, int) or churn_threshold_days <= 0:
        logger.warning(
            f"[{caller}] churn_threshold_days={churn_threshold_days!r} is not a positive int "
            "- defaulting to 120"
        )
        churn_threshold_days = 120

    logger.info("=" * 60)
    logger.info("Analyzing Churn & Retention")
    logger.info("=" * 60)

    total = len(rfm_df)
    results: Dict[str, Any] = {
        "total_customers": total,
        "churn_threshold_days": churn_threshold_days,
    }

    # ---- Frequency segments ------------------------------------------------
    try:
        one_time_buyers  = int((rfm_df["frequency"] == 1).sum())
        repeat_customers = int((rfm_df["frequency"] >= 2).sum())
        loyal_customers  = int((rfm_df["frequency"] >= 5).sum())
        super_loyal      = int((rfm_df["frequency"] >= 10).sum())

        results.update({
            "one_time_buyers":      one_time_buyers,
            "one_time_buyer_pct":   _safe_div(one_time_buyers, total, 0.0) * 100,
            "repeat_customers":     repeat_customers,
            "repeat_customer_pct":  _safe_div(repeat_customers, total, 0.0) * 100,
            "loyal_customers":      loyal_customers,
            "super_loyal":          super_loyal,
        })
    except Exception as e:
        logger.error(f"[{caller}] frequency segmentation failed: {e}")
        one_time_buyers = repeat_customers = loyal_customers = super_loyal = 0

    # ---- Revenue contribution ----------------------------------------------
    total_revenue = 0.0
    try:
        total_revenue      = float(rfm_df["monetary"].sum())
        one_timer_revenue  = float(rfm_df[rfm_df["frequency"] == 1]["monetary"].sum())
        repeat_revenue     = float(rfm_df[rfm_df["frequency"] >= 2]["monetary"].sum())
        super_loyal_revenue = float(rfm_df[rfm_df["frequency"] >= 10]["monetary"].sum())

        results.update({
            "total_revenue":        total_revenue,
            "one_timer_revenue":    one_timer_revenue,
            "repeat_revenue":       repeat_revenue,
            "super_loyal_revenue":  super_loyal_revenue,
        })
    except Exception as e:
        logger.error(f"[{caller}] revenue contribution calculation failed: {e}")

    # ---- Churn analysis ----------------------------------------------------
    churned_customers = pd.DataFrame()
    active_customers  = pd.DataFrame()
    try:
        churned_customers = rfm_df[rfm_df["churn"] == 1]
        active_customers  = rfm_df[rfm_df["churn"] == 0]
        results.update({
            "churned_count": len(churned_customers),
            "churned_pct":   _safe_div(len(churned_customers), total, 0.0) * 100,
            "active_count":  len(active_customers),
        })
    except Exception as e:
        logger.error(f"[{caller}] churn analysis failed: {e}")

    # ---- At-risk customers -------------------------------------------------
    at_risk = pd.DataFrame()
    revenue_at_risk = 0.0
    try:
        at_risk = rfm_df[
            (rfm_df["churn"] == 0)
            & (rfm_df["recency_days"] > churn_threshold_days * 0.75)
        ]
        revenue_at_risk = float(at_risk["monetary"].sum())
        results.update({
            "at_risk_count":   len(at_risk),
            "revenue_at_risk": revenue_at_risk,
        })
    except Exception as e:
        logger.error(f"[{caller}] at-risk calculation failed: {e}")

    # ---- Verbose output ----------------------------------------------------
    if verbose:
        try:
            print("\n" + "=" * 80)
            print("CHURN & CUSTOMER RETENTION DEEP-DIVE")
            print("=" * 80)

            print("CHURN DEFINITION:")
            print(f"No purchase in {churn_threshold_days} days (~{churn_threshold_days/30:.0f} months)")

            print("\nCUSTOMER FREQUENCY BREAKDOWN:")
            for key, label in [
                ("one_time_buyers",  "One-time buyers"),
                ("repeat_customers", "Repeat customers"),
                ("loyal_customers",  "Loyal (5+ orders)"),
                ("super_loyal",      "Super loyal (10+ orders)"),
            ]:
                try:
                    val = results.get(key, 0)
                    pct = _safe_div(val, total, 0.0) * 100
                    print(f"  {label}: {val:,} ({pct:.1f}%)")
                except Exception:
                    pass

            print("\nREVENUE CONTRIBUTION:")
            try:
                ot_rev = results.get("one_timer_revenue", 0.0)
                rp_rev = results.get("repeat_revenue", 0.0)
                tot_rev = results.get("total_revenue", 0.0)
                rp_cust = results.get("repeat_customers", 0)
                print(f"  One-time buyers  : ${ot_rev:,.0f} ({_safe_div(ot_rev, tot_rev, 0)*100:.1f}%)")
                print(f"  Repeat customers : ${rp_rev:,.0f} ({_safe_div(rp_rev, tot_rev, 0)*100:.1f}%)")
                print(
                    f"  {_safe_div(rp_cust, total, 0)*100:.0f}% of customers drive "
                    f"{_safe_div(rp_rev, tot_rev, 0)*100:.0f}% of revenue"
                )
            except Exception:
                pass

            print("\nCHURN ANALYSIS:")
            try:
                print(f"  Churned : {results.get('churned_count', 'N/A'):,} ({results.get('churned_pct', 0):.1f}%)")
                print(f"  Active  : {results.get('active_count', 'N/A'):,} ({_safe_div(results.get('active_count', 0), total, 0)*100:.1f}%)")
            except Exception:
                pass

            if not at_risk.empty:
                try:
                    print(f"\nAT-RISK CUSTOMERS (near churn threshold):")
                    print(f"  Count             : {len(at_risk):,}")
                    print(f"  Revenue at risk   : ${revenue_at_risk:,.0f}")
                    print(f"  Avg lifetime value: ${at_risk['monetary'].mean():.2f}")
                except Exception:
                    pass

            print("\nKEY INSIGHTS:")
            try:
                ot_pct = results.get("one_time_buyer_pct", 0.0)
                if ot_pct > 40:
                    print(f"  {ot_pct:.0f}% are one-time buyers -> acquisition-to-retention gap")
                    print("  Focus on first-purchase experience")
                    print("  Implement post-purchase engagement campaigns")
            except Exception:
                pass

            try:
                churn_pct = results.get("churned_pct", 0.0)
                if churn_pct > 40 and not churned_customers.empty:
                    avg_churned = churned_customers["monetary"].mean()
                    print(f"  ${len(churned_customers) * avg_churned:,.0f} in lost customer LTV")
                    print("  Implement win-back campaigns")
            except Exception:
                pass

            try:
                sl = results.get("super_loyal", 0)
                sl_rev = results.get("super_loyal_revenue", 0.0)
                tot_rev = results.get("total_revenue", 0.0)
                if sl > 0:
                    print(
                        f"  Top {_safe_div(sl, total, 0)*100:.1f}% drive "
                        f"${sl_rev:,.0f} ({_safe_div(sl_rev, tot_rev, 0)*100:.0f}%)"
                    )
                    print("  Nurture with VIP program")
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"[{caller}] verbose output failed (results still valid): {e}")

    try:
        logger.info(f"One-time buyers : {results.get('one_time_buyer_pct', 0):.1f}%")
        logger.info(f"Churn rate      : {results.get('churned_pct', 0):.1f}%")
        logger.info(f"Revenue at risk : ${results.get('revenue_at_risk', 0):,.0f}")
    except Exception:
        pass

    return results


# ===========================================================================
# 4. generate_business_summary
# ===========================================================================

def generate_business_summary(
    df_clean: pd.DataFrame,
    rfm_df: pd.DataFrame,
    quality_metrics: Dict,
    churn_metrics: Dict,
    run_id: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate comprehensive business summary with actionable insights.

    Error handling:
    - Validates df_clean and rfm_df are non-empty DataFrames
    - Validates quality_metrics and churn_metrics are dicts
    - Validates run_id is a non-empty string; falls back to 'UNKNOWN'
    - Validates required columns in both DataFrames
    - Each metric is computed in its own try/except block
    - verbose print block is fully wrapped
    - Returns {} on fatal input error, partial summary otherwise

    Returns:
        Dictionary with summary metrics, or {} on fatal input error
    """
    caller = "generate_business_summary"

    if not _require_dataframe(df_clean, caller):
        return {}
    if not _require_dataframe(rfm_df, caller):
        return {}

    if not isinstance(quality_metrics, dict):
        logger.error(f"[{caller}] quality_metrics must be a dict, got {type(quality_metrics).__name__}")
        return {}
    if not isinstance(churn_metrics, dict):
        logger.error(f"[{caller}] churn_metrics must be a dict, got {type(churn_metrics).__name__}")
        return {}

    if not isinstance(run_id, str) or not run_id.strip():
        logger.warning(f"[{caller}] run_id={run_id!r} is not a non-empty string - using 'UNKNOWN'")
        run_id = "UNKNOWN"

    if _require_columns(df_clean, ["total_amount", "order_date", "returned"], caller):
        return {}
    if _require_columns(rfm_df, ["monetary", "churn", "loyalty_score"], caller):
        return {}

    logger.info("=" * 60)
    logger.info("Generating Business Summary")
    logger.info("=" * 60)

    summary: Dict[str, Any] = {"run_id": run_id}

    # Each metric block is independently guarded
    try:
        summary["total_transactions"] = len(df_clean)
        summary["total_customers"]    = len(rfm_df)
    except Exception as e:
        logger.error(f"[{caller}] row counts failed: {e}")

    try:
        summary["date_range_start"] = str(df_clean["order_date"].min())
        summary["date_range_end"]   = str(df_clean["order_date"].max())
    except Exception as e:
        logger.warning(f"[{caller}] date range failed: {e}")

    try:
        summary["total_revenue"] = float(df_clean["total_amount"].sum())
    except Exception as e:
        logger.error(f"[{caller}] total revenue failed: {e}")
        summary["total_revenue"] = None

    try:
        summary["churn_rate"] = float(rfm_df["churn"].mean() * 100)
    except Exception as e:
        logger.warning(f"[{caller}] churn_rate failed: {e}")
        summary["churn_rate"] = None

    try:
        summary["return_rate"] = float(df_clean["returned"].mean() * 100)
    except Exception as e:
        logger.warning(f"[{caller}] return_rate failed: {e}")
        summary["return_rate"] = None

    try:
        summary["avg_customer_value"] = float(rfm_df["monetary"].mean())
    except Exception as e:
        logger.warning(f"[{caller}] avg_customer_value failed: {e}")
        summary["avg_customer_value"] = None

    try:
        summary["avg_loyalty_score"] = float(rfm_df["loyalty_score"].mean())
    except Exception as e:
        logger.warning(f"[{caller}] avg_loyalty_score failed: {e}")
        summary["avg_loyalty_score"] = None

    try:
        summary["avg_orders_per_customer"] = float(
            _safe_div(len(df_clean), len(rfm_df), 0.0)
        )
    except Exception as e:
        logger.warning(f"[{caller}] avg_orders_per_customer failed: {e}")
        summary["avg_orders_per_customer"] = None

    try:
        summary["avg_transaction_value"] = float(
            _safe_div(float(df_clean["total_amount"].sum()), len(df_clean), 0.0)
        )
    except Exception as e:
        logger.warning(f"[{caller}] avg_transaction_value failed: {e}")
        summary["avg_transaction_value"] = None

    try:
        top20_n = max(1, int(len(rfm_df) * 0.2))
        top20_rev = rfm_df.nlargest(top20_n, "monetary")["monetary"].sum()
        rfm_total = rfm_df["monetary"].sum()
        summary["top20_contribution"] = float(_safe_div(top20_rev, rfm_total, 0.0) * 100)
    except Exception as e:
        logger.warning(f"[{caller}] top20 contribution failed: {e}")
        summary["top20_contribution"] = None

    # Pull from upstream dicts with .get() so missing keys return None cleanly
    summary["data_quality_score"]  = quality_metrics.get("quality_score")
    summary["one_time_buyer_pct"]  = churn_metrics.get("one_time_buyer_pct")
    summary["revenue_at_risk"]     = churn_metrics.get("revenue_at_risk")

    if verbose:
        try:
            print("=" * 80)
            print("NOTEBOOK 01 - EXECUTIVE SUMMARY")
            print("=" * 80)

            print("\nDATA PROCESSED:")
            print(f"  Transactions  : {summary.get('total_transactions', 'N/A'):,}")
            print(f"  Customers     : {summary.get('total_customers', 'N/A'):,}")
            print(
                f"  Date Range    : {summary.get('date_range_start', 'N/A')} -> "
                f"{summary.get('date_range_end', 'N/A')}"
            )
            rev = summary.get("total_revenue")
            dq  = summary.get("data_quality_score")
            print(f"  Total Revenue : {'${:,.2f}'.format(rev) if rev is not None else 'N/A'}")
            print(f"  Data Quality  : {f'{dq:.1f}/100' if dq is not None else 'N/A'}")

            print("\nKEY METRICS:")
            for label, key, fmt in [
                ("Churn Rate",     "churn_rate",       "{:.1f}%"),
                ("Return Rate",    "return_rate",      "{:.2f}%"),
                ("Avg Cust Value", "avg_customer_value","${:.2f}"),
                ("Avg Loyalty",    "avg_loyalty_score", "{:.3f}"),
            ]:
                val = summary.get(key)
                print(f"  {label:18s}: {fmt.format(val) if val is not None else 'N/A'}")

            print("\nBUSINESS INSIGHTS:")
            for label, key, fmt in [
                ("Avg orders/cust",  "avg_orders_per_customer", "{:.1f}"),
                ("Avg txn value",    "avg_transaction_value",   "${:.2f}"),
                ("One-time buyers",  "one_time_buyer_pct",      "{:.0f}%"),
            ]:
                val = summary.get(key)
                print(f"  {label:18s}: {fmt.format(val) if val is not None else 'N/A'}")

            top20 = summary.get("top20_contribution")
            if top20 is not None:
                print(f"  Top 20% customers drive {top20:.0f}% of revenue")

            print("\nACTION ITEMS (PRIORITISED):")
            try:
                churn_rate = summary.get("churn_rate") or 0
                if churn_rate > 50:
                    print(f"  HIGH PRIORITY: address {churn_rate:.0f}% churn rate")
                    at_risk_n = churn_metrics.get("at_risk_count", 0)
                    rar = summary.get("revenue_at_risk") or 0
                    if at_risk_n:
                        print(f"  {at_risk_n:,} customers at risk (${rar:,.0f})")
                    ot_pct = summary.get("one_time_buyer_pct") or 0
                    print(f"  Context: {ot_pct:.0f}% are one-time buyers")
                    print("  Action: launch win-back campaigns targeting 90+ day recency")
            except Exception:
                pass

            try:
                top20 = summary.get("top20_contribution") or 0
                if top20 > 70:
                    print(f"\n  MEDIUM PRIORITY: reduce revenue concentration")
                    print(f"  Top 20% drive {top20:.0f}% of revenue")
                    print("  Action: develop mid-tier customer growth programs")
            except Exception:
                pass

            try:
                ot_pct = summary.get("one_time_buyer_pct") or 0
                if ot_pct > 40:
                    print(f"\n  MEDIUM PRIORITY: convert one-time buyers")
                    print(f"  {ot_pct:.0f}% made only 1 purchase")
                    print("  Action: post-purchase nurture sequence")
                    print("  Action: second-purchase incentives")
            except Exception:
                pass

            print("\nNEXT STEPS:")
            for step in [
                "Notebook 02: Sales & performance analysis",
                "Notebook 03: Customer segmentation (K-Means)",
                "Notebook 04: Churn prediction modeling",
                "Notebook 05: Cohort & retention analysis",
                "Notebook 06: Fraud detection",
            ]:
                print(f"  {step}")

            print("=" * 80)
            print(f"Analysis Complete - Run ID: {run_id}")
            print("=" * 80)

        except Exception as e:
            logger.warning(f"[{caller}] verbose output failed (results still valid): {e}")

    try:
        logger.info(f"Summary generated for run: {run_id}")
        logger.info(f"Churn rate      : {summary.get('churn_rate', 'N/A')}")
        logger.info(f"Revenue at risk : ${summary.get('revenue_at_risk', 0):,.0f}")
    except Exception:
        pass

    return summary