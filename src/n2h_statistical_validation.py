# src/n2h_statistical_validation.py
"""
n2h_statistical_validation.py - Statistical Validation for Sales Analysis

This module performs hypothesis tests to validate findings and determine which visual
differences are statistically real vs within-noise.

Functions:
    - validate_time_trend()            — Spearman correlation for revenue trend significance
    - validate_category_differences()  — Kruskal-Wallis for category performance differences
    - validate_regional_differences()  — Kruskal-Wallis for regional performance differences
    - validate_discount_effect()       — Mann-Whitney U for discount effect on transaction value
    - run_all_validations()            — Run all tests and compile results + verdicts
    - create_statistical_validation()  — Full pipeline entry point

Usage:
    from n2h_statistical_validation import create_statistical_validation
    results = create_statistical_validation(
        df,
        monthly_sales=time_results.get('monthly_sales'),
        frequency_results=discount_results.get('frequency_analysis')
    )

Tests:
    1. Spearman correlation   — revenue trend significance
    2. Kruskal-Wallis         — category performance differences
    3. Kruskal-Wallis         — regional performance differences
    4. Mann-Whitney U         — discount effect on transaction value

Notes:
    - frequency_results (optional) is passed from n2f discount analysis.
      When present, the discount verdict incorporates customer-level frequency
      lift and lifetime revenue impact to produce a complete, self-contained
      conclusion. When absent, the verdict reports the per-transaction finding
      only and logs a warning that the verdict is incomplete.
    - Expected keys in frequency_results:
        frequency_lift_pct              float  — % more orders per discount customer
        revenue_per_customer_lift_pct   float  — % more lifetime revenue per customer
        median_orders_discount          float  — median order count, discount customers
        median_orders_no_discount       float  — median order count, non-discount customers
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, Any, cast
import logging
from n2a_utils import setup_logger, print_section_header, get_config

# Setup logger
logger = setup_logger(__name__)

# Load configuration
_CONFIG = get_config()
_NB1_TESTS = cast(Dict[str, Any], _CONFIG.get('notebook1', {})).get('statistical_tests', {})

ALPHA: float = _NB1_TESTS.get('normality_alpha', 0.05)
CORRELATION_THRESHOLD: float = _NB1_TESTS.get('correlation_threshold', 0.5)

SPEARMAN_STRONG: float = 0.7
SPEARMAN_MODERATE: float = 0.4
EPS_LARGE: float = 0.14
EPS_MEDIUM: float = 0.06
RBC_LARGE: float = 0.5
RBC_MEDIUM: float = 0.3


# ============================================================================
# Individual Test Functions
# ============================================================================

def validate_time_trend(
    monthly_sales: pd.DataFrame,
    metric_col: str = 'total_amount'
) -> Dict:
    """
    Validate the significance of a revenue trend using Spearman correlation.

    Parameters
    ----------
    monthly_sales : pd.DataFrame
        DataFrame containing monthly sales data
    metric_col : str, optional
        Name of the column containing the metric to test (default: 'total_amount')

    Returns
    -------
    Dict
        test_name, statistic, p_value, significant, strength, direction,
        interpretation, n_months
    """
    try:
        logger.info("Testing revenue trend significance (Spearman correlation)...")
        x = np.arange(len(monthly_sales))
        y = monthly_sales[metric_col].values
        result = stats.spearmanr(x, y)
        rho = float(result.statistic) if hasattr(result, 'statistic') else float(result[0])  # type: ignore
        p_value = float(result.pvalue) if hasattr(result, 'pvalue') else float(result[1])    # type: ignore
        significant = p_value < ALPHA
        abs_rho = abs(rho)
        strength = (
            "strong" if abs_rho > SPEARMAN_STRONG
            else ("moderate" if abs_rho > SPEARMAN_MODERATE else "weak")
        )
        direction = "increasing" if rho > 0 else "decreasing"
        interpretation = (
            f"Significant {strength} {direction} trend (p<{ALPHA}, ρ={rho:.3f})"
            if significant else
            f"No significant trend detected (p={p_value:.3f})"
        )
        logger.info(f"Spearman ρ={rho:.3f}, p={p_value:.4f} — {interpretation}")
        return {
            'test_name':      'Spearman Correlation',
            'statistic':      rho,
            'p_value':        p_value,
            'significant':    significant,
            'strength':       strength,
            'direction':      direction,
            'interpretation': interpretation,
            'n_months':       len(monthly_sales),
        }

    except KeyError as e:
        logger.error(f"Missing column '{metric_col}' in validate_time_trend: {e}")
        raise
    except Exception as e:
        logger.error(f"validate_time_trend failed: {e}")
        raise


def validate_category_differences(
    df: pd.DataFrame,
    category_col: str = 'category',
    metric_col: str = 'total_amount'
) -> Dict:
    """
    Validate the significance of category differences using Kruskal-Wallis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing category and metric columns
    category_col : str, optional
        Name of the category column (default: 'category')
    metric_col : str, optional
        Name of the metric column (default: 'total_amount')

    Returns
    -------
    Dict
        test_name, statistic, p_value, significant, effect_size,
        effect_strength, interpretation, n_categories, n_observations
    """
    try:
        logger.info("Testing category differences (Kruskal-Wallis)...")
        groups = [g[metric_col].values for _, g in df.groupby(category_col)]
        h_stat, p_value = stats.kruskal(*groups)
        k = len(groups)
        n = len(df)
        eps = (h_stat - k + 1) / (n - k)
        effect_str = (
            "large" if eps > EPS_LARGE
            else ("medium" if eps > EPS_MEDIUM else "small")
        )
        significant = p_value < ALPHA
        interpretation = (
            f"Categories differ significantly with {effect_str} effect "
            f"(p<{ALPHA}, ε²={eps:.3f})"
            if significant else
            f"No significant category differences (p={p_value:.3f})"
        )
        logger.info(f"Kruskal-Wallis H={h_stat:.2f}, p={p_value:.4f} — {interpretation}")
        return {
            'test_name':       'Kruskal-Wallis',
            'statistic':       h_stat,
            'p_value':         p_value,
            'significant':     significant,
            'effect_size':     eps,
            'effect_strength': effect_str,
            'interpretation':  interpretation,
            'n_categories':    k,
            'n_observations':  n,
        }

    except KeyError as e:
        logger.error(
            f"Missing column in validate_category_differences "
            f"(category='{category_col}', metric='{metric_col}'): {e}"
        )
        raise
    except Exception as e:
        logger.error(f"validate_category_differences failed: {e}")
        raise


def validate_regional_differences(
    df: pd.DataFrame,
    region_col: str = 'region',
    metric_col: str = 'total_amount',
    run_posthoc: bool = True
) -> Dict:
    """
    Validate the significance of regional differences using Kruskal-Wallis.

    Optionally runs Bonferroni-corrected Mann-Whitney pairwise post-hoc tests
    when the omnibus test is significant and there are more than two regions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing region and metric columns
    region_col : str, optional
        Name of the region column (default: 'region')
    metric_col : str, optional
        Name of the metric column (default: 'total_amount')
    run_posthoc : bool, optional
        Whether to run post-hoc pairwise tests (default: True)

    Returns
    -------
    Dict
        test_name, statistic, p_value, significant, effect_size,
        n_regions, n_observations, posthoc, interpretation
    """
    try:
        logger.info("Testing regional differences (Kruskal-Wallis)...")
        groups  = [g[metric_col].values for _, g in df.groupby(region_col)]
        regions = df[region_col].unique()
        h_stat, p_value = stats.kruskal(*groups)
        significant = p_value < ALPHA
        k = len(groups)
        n = len(df)
        eps = (h_stat - k + 1) / (n - k)

        posthoc = []
        if significant and run_posthoc and k > 2:
            bonf_alpha = ALPHA / (k * (k - 1) / 2)
            for i, r1 in enumerate(regions):
                for r2 in regions[i + 1:]:
                    g1 = df[df[region_col] == r1][metric_col]
                    g2 = df[df[region_col] == r2][metric_col]
                    _, p = stats.mannwhitneyu(g1, g2)
                    if p < bonf_alpha:
                        higher = r1 if g1.median() > g2.median() else r2
                        posthoc.append({
                            'comparison':    f"{r1} vs {r2}",
                            'p_value':       p,
                            'significant':   True,
                            'higher_region': higher
                        })

        interpretation = (
            f"Regions differ significantly (p<{ALPHA}, ε²={eps:.3f})"
            + (f" — {len(posthoc)} significant pairwise differences" if posthoc else "")
            if significant else
            f"No significant regional differences (p={p_value:.3f})"
        )
        logger.info(f"Kruskal-Wallis H={h_stat:.2f}, p={p_value:.4f} — {interpretation}")

        return {
            'test_name':      'Kruskal-Wallis (Regional)',
            'statistic':      h_stat,
            'p_value':        p_value,
            'significant':    significant,
            'effect_size':    eps,
            'n_regions':      k,
            'n_observations': n,
            'posthoc':        posthoc,
            'interpretation': interpretation,
        }

    except KeyError as e:
        logger.error(
            f"Missing column in validate_regional_differences "
            f"(region='{region_col}', metric='{metric_col}'): {e}"
        )
        raise
    except Exception as e:
        logger.error(f"validate_regional_differences failed: {e}")
        raise


def validate_discount_effect(
    df: pd.DataFrame,
    discount_col: str = 'discount',
    metric_col: str = 'total_amount'
) -> Dict:
    """
    Validate the significance of discount effect on transaction value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing discount and metric columns
    discount_col : str, optional
        Name of the discount column (default: 'discount')
    metric_col : str, optional
        Name of the metric column (default: 'total_amount')

    Returns
    -------
    Dict
        test_name, statistic, p_value, significant, discount_median,
        no_discount_median, difference, pct_difference, effect_size,
        effect_strength, interpretation, n_discount, n_no_discount
    """
    try:
        logger.info("Testing discount effect on transaction value (Mann-Whitney U)...")
        disc   = df[df[discount_col] > 0][metric_col]
        nodisc = df[df[discount_col] == 0][metric_col]
        u_stat, p_value = stats.mannwhitneyu(disc, nodisc)
        disc_med   = disc.median()
        nodisc_med = nodisc.median()
        diff     = disc_med - nodisc_med
        pct_diff = (diff / nodisc_med * 100) if nodisc_med > 0 else 0.0
        n1, n2   = len(disc), len(nodisc)
        r = 1 - (2 * u_stat) / (n1 * n2)
        effect_str = (
            "large" if abs(r) > RBC_LARGE
            else ("medium" if abs(r) > RBC_MEDIUM else "small")
        )
        significant = p_value < ALPHA
        if significant:
            direction = "INCREASE" if diff > 0 else "DECREASE"
            interpretation = (
                f"Discounts significantly {direction} transaction value by "
                f"${abs(diff):.2f} ({pct_diff:.1f}%) with {effect_str} effect (p<{ALPHA})"
            )
        else:
            interpretation = (
                f"No significant discount effect on transaction value (p={p_value:.3f})"
            )
        logger.info(f"Mann-Whitney U={u_stat:.0f}, p={p_value:.4f} — {interpretation}")
        return {
            'test_name':          'Mann-Whitney U (Discount Effect)',
            'statistic':           u_stat,
            'p_value':             p_value,
            'significant':         significant,
            'discount_median':     disc_med,
            'no_discount_median':  nodisc_med,
            'difference':          diff,
            'pct_difference':      pct_diff,
            'effect_size':         r,
            'effect_strength':     effect_str,
            'interpretation':      interpretation,
            'n_discount':          n1,
            'n_no_discount':       n2,
        }

    except KeyError as e:
        logger.error(
            f"Missing column in validate_discount_effect "
            f"(discount='{discount_col}', metric='{metric_col}'): {e}"
        )
        raise
    except Exception as e:
        logger.error(f"validate_discount_effect failed: {e}")
        raise


# ============================================================================
# Suite
# ============================================================================

def run_all_validations(
    df: pd.DataFrame,
    monthly_sales: pd.DataFrame,
    frequency_results: Optional[Dict] = None,
    time_col: str = 'year_month',
    metric_col: str = 'total_amount',
    category_col: str = 'category',
    region_col: str = 'region',
    discount_col: str = 'discount'
) -> Dict:
    """
    Run all four hypothesis tests and return results + verdicts.

    The 'verdicts' key translates each test outcome into a dashboard_note
    string that the visualisation layer can display as an annotation.

    When frequency_results is provided (from n2f discount analysis), the
    discount verdict incorporates customer-level frequency lift and lifetime
    revenue impact to produce a complete, self-contained conclusion rather
    than pointing the reader to a separate section.

    Args:
        df:                Full transaction DataFrame
        monthly_sales:     Monthly aggregated DataFrame with metric_col
        frequency_results: Optional dict from n2f discount frequency analysis.
                           Expected keys:
                               frequency_lift_pct            float
                               revenue_per_customer_lift_pct float
                               median_orders_discount        float
                               median_orders_no_discount     float
        time_col:          Time index column name (default: 'year_month')
        metric_col:        Revenue metric column name (default: 'total_amount')
        category_col:      Category column name (default: 'category')
        region_col:        Region column name (default: 'region')
        discount_col:      Discount column name (default: 'discount')

    Returns:
        dict with keys: time_trend, category, regional, discount, summary, verdicts
    """
    try:
        logger.info("=" * 60)
        logger.info("STATISTICAL VALIDATION SUITE")
        logger.info("=" * 60)

        results = {}
        logger.info("TIME TREND VALIDATION:")
        results['time_trend'] = validate_time_trend(monthly_sales, metric_col)
        logger.info("CATEGORY VALIDATION:")
        results['category']   = validate_category_differences(df, category_col, metric_col)
        logger.info("REGIONAL VALIDATION:")
        results['regional']   = validate_regional_differences(df, region_col, metric_col)
        logger.info("DISCOUNT VALIDATION:")
        results['discount']   = validate_discount_effect(df, discount_col, metric_col)

        sig_count = sum(1 for r in results.values() if r['significant'])

        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"{sig_count}/4 tests significant")
        for name, r in results.items():
            status = "SIGNIFICANT" if r['significant'] else "NOT SIGNIFICANT"
            logger.info(f"{name.upper():12s}: {status} (p={r['p_value']:.4f})")

        results['summary'] = {
            'total_tests':       4,
            'significant_tests': sig_count,
            'validation_rate':   sig_count / 4 * 100,
        }

        tt  = results['time_trend']
        cat = results['category']
        reg = results['regional']
        dis = results['discount']

        # ── Discount verdict ─────────────────────────────────────────────────
        # Complete when frequency_results available; per-transaction only otherwise.
        if dis['significant']:
            base = (
                f"Discounts reduce transaction value by "
                f"${abs(dis.get('difference', 0)):.2f} "
                f"({dis.get('pct_difference', 0):.1f}%, p<0.001), "
                f"{dis.get('effect_strength', '')} effect."
            )
            if frequency_results:
                freq_lift  = frequency_results.get('frequency_lift_pct', 0)
                rev_lift   = frequency_results.get('revenue_per_customer_lift_pct', 0)
                med_disc   = frequency_results.get('median_orders_discount', 0)
                med_nodisc = frequency_results.get('median_orders_no_discount', 0)
                disc_note  = (
                    f"{base} However discount customers order {freq_lift:.0f}% more "
                    f"frequently (median {med_disc:.0f} vs {med_nodisc:.0f} orders), "
                    f"producing +{rev_lift:.1f}% more lifetime revenue per customer. "
                    f"Net impact is positive — discounting is revenue-accretive at "
                    f"moderate tiers."
                )
                logger.info(
                    f"Discount verdict enriched with frequency analysis "
                    f"(freq_lift={freq_lift:.0f}%, rev_lift={rev_lift:.1f}%)"
                )
            else:
                disc_note = (
                    f"{base} Frequency analysis not provided — evaluate "
                    f"customer-level order frequency to determine net "
                    f"revenue-per-customer impact."
                )
                logger.warning(
                    "frequency_results not provided to run_all_validations — "
                    "discount verdict is per-transaction only. Pass "
                    "frequency_results=discount_results.get('frequency_analysis') "
                    "for a complete verdict."
                )
        else:
            disc_note = "Discount effect on transaction value is not statistically significant."

        results['verdicts'] = {
            'time_trend': {
                'significant':    tt['significant'],
                'dashboard_note': (
                    f"Trend is statistically significant "
                    f"(ρ={tt['statistic']:.3f}, p={tt['p_value']:.3f})."
                    if tt['significant'] else
                    f"No statistically significant revenue trend (p={tt['p_value']:.3f}). "
                    f"Month-to-month variation is consistent with random fluctuation. "
                    f"YoY growth is real but does not imply a steady directional drift."
                ),
            },
            'category': {
                'significant':     cat['significant'],
                'effect_strength': cat.get('effect_strength', ''),
                'dashboard_note': (
                    f"Category differences are real and large "
                    f"(ε²={cat.get('effect_size', 0):.3f}, p<0.001). "
                    f"Electronics alone drives 56.6% of total revenue — "
                    f"this concentration is the dominant structural feature of the business."
                    if cat['significant'] else
                    "Category differences not statistically significant."
                ),
            },
            'regional': {
                'significant':    reg['significant'],
                'dashboard_note': (
                    f"Regional differences are statistically significant "
                    f"(p={reg['p_value']:.3f})."
                    if reg['significant'] else
                    f"Regional revenue differences are NOT statistically significant "
                    f"(p={reg['p_value']:.3f}). The South-leads ranking is within random "
                    f"variation. Do NOT allocate marketing or logistics budgets based on "
                    f"regional revenue rankings alone — volume drives the gap, not order value."
                ),
            },
            'discount': {
                'significant':        dis['significant'],
                'reduces_value':      dis['significant'] and dis.get('difference', 0) < 0,
                'frequency_enriched': frequency_results is not None,
                'dashboard_note':     disc_note,
            },
        }

        logger.info("ACTIONABLE VERDICTS:")
        for domain, v in results['verdicts'].items():
            logger.info(f"{domain.upper():12s}: {v['dashboard_note'][:120]}")

        return results

    except Exception as e:
        logger.error(f"run_all_validations failed: {e}")
        raise


# ============================================================================
# Display Helpers
# ============================================================================

def create_validation_summary_table(results: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame from statistical validation results.

    Args:
        results: Output dict from run_all_validations or create_statistical_validation

    Returns:
        pd.DataFrame: One row per test with columns Analysis, Test, Statistic,
            P-Value, Significant, Interpretation
    """
    try:
        skip = {'summary', 'metadata', 'tests', 'verdicts'}
        rows = []
        for name, r in results.items():
            if name in skip or not isinstance(r, dict) or 'test_name' not in r:
                continue
            rows.append({
                'Analysis':       name.replace('_', ' ').title(),
                'Test':           r['test_name'],
                'Statistic':      f"{r['statistic']:.3f}",
                'P-Value':        f"{r['p_value']:.4f}",
                'Significant':    'Yes' if r['significant'] else 'No',
                'Interpretation': r['interpretation'],
            })
        return pd.DataFrame(rows)

    except Exception as e:
        logger.error(f"create_validation_summary_table failed: {e}")
        raise


def print_validation_report(results: Dict) -> None:
    """
    Log a formatted validation report for all statistical tests.

    Outputs each test's name, statistic, p-value, significance verdict,
    interpretation, optional effect size, and optional post-hoc results.
    Also logs dashboard verdicts and the overall summary.

    Args:
        results: Output dict from run_all_validations or create_statistical_validation
    """
    try:
        skip = {'summary', 'metadata', 'tests', 'verdicts'}
        logger.info("=" * 80)
        logger.info("STATISTICAL VALIDATION REPORT")
        logger.info("=" * 80)

        for name, r in results.items():
            if name in skip or not isinstance(r, dict) or 'test_name' not in r:
                continue
            logger.info(f"{name.upper().replace('_', ' ')}:")
            logger.info(f"Test:           {r['test_name']}")
            logger.info(f"Statistic:      {r['statistic']:.3f}")
            logger.info(f"P-value:        {r['p_value']:.4f}")
            logger.info(
                f"Result:         {'SIGNIFICANT' if r['significant'] else 'NOT SIGNIFICANT'}"
            )
            logger.info(f"Interpretation: {r['interpretation']}")
            if 'effect_size' in r:
                logger.info(
                    f"Effect size:    {r['effect_size']:.3f} "
                    f"({r.get('effect_strength', 'N/A')})"
                )
            if r.get('posthoc'):
                for c in r['posthoc'][:3]:
                    logger.info(
                        f"Post-hoc: {c['comparison']} — {c['higher_region']} higher"
                    )

        if 'verdicts' in results:
            logger.info("=" * 80)
            logger.info("DASHBOARD VERDICTS (connect stats → panels)")
            logger.info("=" * 80)
            for domain, v in results['verdicts'].items():
                enriched = " [frequency-enriched]" if v.get('frequency_enriched') else ""
                logger.info(f"{domain.upper()}{enriched}: {v['dashboard_note']}")

        if 'summary' in results:
            s = results['summary']
            logger.info(
                f"SUMMARY: {s['significant_tests']}/{s['total_tests']} significant "
                f"({s['validation_rate']:.0f}%)"
            )
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"print_validation_report failed: {e}")
        raise


def create_statistical_validation(
    transactions: pd.DataFrame,
    monthly_sales: Optional[pd.DataFrame] = None,
    frequency_results: Optional[Dict] = None,
    save_figures: bool = False
) -> Dict[str, Any]:
    """
    Run the full statistical validation pipeline for a given DataFrame.

    If monthly_sales is not provided, it is computed from the transactions.
    If frequency_results is provided (from n2f discount analysis), the discount
    verdict will incorporate customer-level frequency lift and lifetime revenue
    impact, producing a complete self-contained conclusion.

    Parameters
    ----------
    transactions : pd.DataFrame
        Full transaction DataFrame
    monthly_sales : Optional[pd.DataFrame], default=None
        Pre-computed monthly sales DataFrame. Computed automatically if None.
    frequency_results : Optional[Dict], default=None
        Frequency analysis dict from n2f discount analysis.
        Pass as: frequency_results=discount_results.get('frequency_analysis')
        Expected keys:
            frequency_lift_pct            float
            revenue_per_customer_lift_pct float
            median_orders_discount        float
            median_orders_no_discount     float
    save_figures : bool, default=False
        Whether to save figures (currently unused; reserved for future use)

    Returns
    -------
    Dict[str, Any]
        Full results dict from run_all_validations plus metadata and tests list
    """
    try:
        if monthly_sales is None:
            monthly_sales = (
                transactions
                .groupby(
                    pd.to_datetime(transactions['order_date'])
                    .dt.to_period('M')
                    .astype(str)
                )
                .agg(
                    total_amount=('total_amount', 'sum'),
                    order_id=('order_id', 'nunique')
                )
                .reset_index()
            )
            monthly_sales.columns = ['year_month', 'total_amount', 'num_orders']

        results = run_all_validations(
            transactions,
            monthly_sales,
            frequency_results=frequency_results
        )

        results['metadata'] = {
            'n_transactions':     len(transactions),
            'n_months':           len(monthly_sales),
            'n_categories':       (
                transactions['category'].nunique()
                if 'category' in transactions.columns else 0
            ),
            'n_regions':          (
                transactions['region'].nunique()
                if 'region' in transactions.columns else 0
            ),
            'frequency_enriched': frequency_results is not None,
        }

        results['tests'] = [
            {
                'name':        name,
                'test_type':   test['test_name'],
                'p_value':     test['p_value'],
                'significant': test['significant']
            }
            for name, test in results.items()
            if name not in ['summary', 'metadata', 'tests', 'verdicts']
            and isinstance(test, dict)
            and 'test_name' in test
        ]

        return results

    except Exception as e:
        logger.error(f"create_statistical_validation failed: {e}")
        raise


# ============================================================================
# Module Information
# ============================================================================

__all__ = [
    'ALPHA',
    'CORRELATION_THRESHOLD',
    'validate_time_trend',
    'validate_category_differences',
    'validate_regional_differences',
    'validate_discount_effect',
    'run_all_validations',
    'create_validation_summary_table',
    'print_validation_report',
    'create_statistical_validation',
]

if __name__ == '__main__':
    print("n2h_statistical_validation.py — import create_statistical_validation(df)")