"""
n6i_insights.py
---------------
Business insight generation and revenue scenario modelling for Notebook 06.

Synthesises outputs from retention, revenue, and LTV modules into
prioritised, metric-anchored recommendations.

Functions
---------
generate_cohort_insights    — Auto-generate insight dicts from computed metrics
plot_insights_summary       — Horizontal bar chart of insight priority scores
compute_retention_scenarios — Revenue uplift table for incremental retention gains
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from n6_utils import get_output_paths, get_run_id, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Insight generation
# ---------------------------------------------------------------------------

def generate_cohort_insights(
    retention_matrix: pd.DataFrame,
    revenue_retention: pd.DataFrame,
    agg_retention: pd.DataFrame,
    ltv_matrix: pd.DataFrame,
    segment_retention: dict[str, dict[int, float]],
    ltv_benchmarks: dict[int, float],
    dropout_df: pd.DataFrame,
    config: dict[str, Any],
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Generate a prioritised list of business insights from cohort metrics.

    Each insight is a dict with keys: ``title``, ``finding``, ``metric``,
    ``action``, ``priority``, ``priority_score``.

    Parameters
    ----------
    retention_matrix:
        Output of ``compute_retention_rates``.
    revenue_retention:
        Output of ``compute_revenue_retention``.
    agg_retention:
        Output of ``compute_aggregate_retention``.
    ltv_matrix:
        Output of ``compute_cumulative_ltv``.
    segment_retention:
        Output of ``compute_segment_retention`` (loyalty tier).
    ltv_benchmarks:
        Output of ``compute_ltv_benchmarks``.
    dropout_df:
        Output of ``compute_dropout_rates``.
    config:
        Loaded project configuration dictionary.
    run_id:
        Pipeline run identifier.

    Returns
    -------
    list[dict]
        Insights sorted by priority score descending.
    """
    run_id   = run_id or get_run_id()
    rules    = config.get("notebook6", {}).get("business_rules", {})
    m1_bench = rules.get("healthy_retention_month1", 0.25)

    insights: list[dict[str, Any]] = []

    # ── Pre-compute peak dropout so Insight 1 can absorb it when both
    #    fire on the same period (M+1), avoiding redundant output.
    dropout_clean = dropout_df.dropna(subset=["dropout_rate"])
    m1_plus       = dropout_clean[dropout_clean["period"] > 0]
    peak_period: int | None = None
    peak_rate:   float | None = None
    if len(m1_plus):
        peak_row    = m1_plus.loc[m1_plus["dropout_rate"].idxmax()]
        peak_period = int(peak_row["period"])
        peak_rate   = float(peak_row["dropout_rate"])

    # ---- Insight 1: M+1 retention vs benchmark ----------------------------
    # When peak dropout is also at M+1, the dropout finding is folded in
    # here rather than appearing as a separate (redundant) insight.
    m1_ret = agg_retention.loc[1, "retention_rate"] if 1 in agg_retention.index else None
    if m1_ret is not None:
        gap = m1_ret - m1_bench

        dropout_context = ""
        if peak_period == 1 and peak_rate is not None:
            dropout_context = (
                f" This corresponds to an M+1 dropout rate of {peak_rate:.1%} —"
                f" the steepest single-period customer loss in the analysis window."
            )

        if gap < -0.05:
            insights.append({
                "title":          "M+1 Retention Below Benchmark — Onboarding Failure",
                "finding":        (
                    f"First-month retention of {m1_ret:.1%} is {abs(gap):.1%} below"
                    f" the {m1_bench:.0%} benchmark.{dropout_context}"
                ),
                "metric":         (
                    f"M+1 retention = {m1_ret:.1%} (benchmark: {m1_bench:.0%})"
                    + (f"; M+1 dropout = {peak_rate:.1%}" if peak_period == 1 and peak_rate else "")
                ),
                "action":         (
                    "Implement a post-purchase onboarding sequence triggered within 7 days"
                    " of first order. Target a second-purchase incentive (e.g. 10% off next"
                    " order, personalised recommendation email). A/B test incentive vs"
                    " non-incentive messaging to isolate the effect."
                ),
                "priority":       "High" if gap < -0.10 else "Medium",
                "priority_score": 90 if gap < -0.10 else 65,
            })
        else:
            insights.append({
                "title":          "M+1 Retention Meeting Benchmark",
                "finding":        (
                    f"First-month retention of {m1_ret:.1%} meets or exceeds"
                    f" the {m1_bench:.0%} benchmark.{dropout_context}"
                ),
                "metric":         f"M+1 retention = {m1_ret:.1%}",
                "action":         (
                    "Maintain current onboarding cadence. Focus optimisation efforts"
                    " on M+3 to M+6 retention window."
                ),
                "priority":       "Low",
                "priority_score": 20,
            })

    # ---- Insight 2: Peak dropout — only fires when peak_period > 1 --------
    # When peak is at M+1, already absorbed into Insight 1 above.
    # When peak is at a later period it signals a distinct secondary cliff.
    if peak_period is not None and peak_period > 1 and peak_rate is not None:
        insights.append({
            "title":          f"Secondary Dropout Spike at M+{peak_period}",
            "finding":        (
                f"{peak_rate:.1%} of active customers drop off between"
                f" M+{peak_period - 1} and M+{peak_period} — a secondary cliff"
                f" after the initial M+1 loss, indicating a structural engagement gap."
            ),
            "metric":         f"M+{peak_period} dropout rate = {peak_rate:.1%}",
            "action":         (
                f"Deploy a re-engagement campaign targeting customers inactive for"
                f" {(peak_period - 1) * 30}–{peak_period * 30} days."
                f" Intervention window closes at M+{max(0, peak_period - 1)}."
            ),
            "priority":       "High" if peak_rate > 0.4 else "Medium",
            "priority_score": 85 if peak_rate > 0.4 else 50,
        })

    # ---- Insight 3: Revenue vs customer retention divergence ---------------
    common = agg_retention.index.intersection(revenue_retention.index)
    if len(common) >= 2:
        spreads = [
            float(revenue_retention.loc[p, "revenue_retention_rate"]) - float(agg_retention.loc[p, "retention_rate"])
            for p in common
        ]
        avg_spread = np.mean(spreads)

        if avg_spread < -0.05:
            insights.append({
                "title":          "High-Value Customers Churning Faster Than Average",
                "finding":        (
                    f"Revenue retention trails customer retention by {abs(avg_spread):.1%}"
                    f" on average — high-value customers are leaving at a disproportionate rate."
                ),
                "metric":         f"Avg revenue–customer retention spread = {avg_spread:+.1%}",
                "action":         (
                    "Identify top-decile customers and trigger a proactive retention outreach"
                    " at 60 days post-purchase. Offer VIP tier benefits, early access,"
                    " or dedicated support."
                ),
                "priority":       "High",
                "priority_score": 88,
            })
        elif avg_spread > 0.05:
            insights.append({
                "title":          "High-Value Customers Retained at Above-Average Rates",
                "finding":        (
                    f"Revenue retention exceeds customer retention by {avg_spread:.1%}"
                    f" — high-value customers are stickier than average."
                ),
                "metric":         f"Avg revenue–customer retention spread = {avg_spread:+.1%}",
                "action":         (
                    "Double down on acquisition channels that attract high-value customers."
                    " Analyse first-purchase category and payment method of retained"
                    " high-value customers."
                ),
                "priority":       "Medium",
                "priority_score": 55,
            })
        else:
            # FIX: was silently omitted — now surfaces as an explicit neutral finding.
            # Silence looks like a missing analysis; a neutral result is still a result.
            insights.append({
                "title":          "Revenue and Customer Churn Are Proportional",
                "finding":        (
                    f"Revenue retention tracks customer retention within {abs(avg_spread):.1%}"
                    f" — high-value and average customers are churning at the same rate."
                    f" The business is losing volume and value in equal measure."
                ),
                "metric":         f"Avg revenue–customer retention spread = {avg_spread:+.1%}",
                "action":         (
                    "No differential churn by value tier detected. Retention investment"
                    " benefits all segments equally — prioritise broad onboarding improvements"
                    " over VIP-specific retention programmes."
                ),
                "priority":       "Low",
                "priority_score": 22,
            })

    # ---- Insight 4: Segment retention gap ----------------------------------
    if segment_retention:
        m1_by_seg = {seg: ret.get(1, np.nan) for seg, ret in segment_retention.items()}
        m1_by_seg = {k: v for k, v in m1_by_seg.items() if not np.isnan(v)}
        if len(m1_by_seg) >= 2:
            best    = max(m1_by_seg.keys(), key=m1_by_seg.get)
            worst   = min(m1_by_seg.keys(), key=lambda k: m1_by_seg[k])
            gap_seg = m1_by_seg[best] - m1_by_seg[worst]
            insights.append({
                "title":          f"Loyalty Tier Retention Gap: {gap_seg:.0%} at M+1",
                "finding":        (
                    f"{best} tier retains at {m1_by_seg[best]:.1%} vs {worst} tier"
                    f" at {m1_by_seg[worst]:.1%} — a {gap_seg:.0%} gap that compounds"
                    f" over the LTV window."
                ),
                "metric":         f"M+1 retention gap ({best} vs {worst}) = {gap_seg:.1%}",
                # FIX: was hardcoded "Bronze" — now uses dynamic `worst` variable
                "action":         (
                    f"Prioritise {worst}-tier upgrade pathways: offer frequency-boosting"
                    f" incentives (bundles, subscription options) within the first 30 days"
                    f" to accelerate loyalty progression from {worst} toward {best}."
                ),
                "priority":       "High" if gap_seg > 0.15 else "Medium",
                "priority_score": 78 if gap_seg > 0.15 else 48,
            })

    # ---- Insight 5: LTV plateau detection ----------------------------------
    ltv_12m = ltv_benchmarks.get(12)
    ltv_3m  = ltv_benchmarks.get(3)
    if ltv_12m and ltv_3m and ltv_3m > 0:
        growth_ratio = ltv_12m / ltv_3m
        if growth_ratio < 1.5:
            insights.append({
                "title":          "LTV Plateaus Early — Single-Purchase Dynamics Dominant",
                "finding":        (
                    f"12-month LTV is only {growth_ratio:.1f}x the 3-month LTV"
                    f" (${ltv_3m:,.2f} → ${ltv_12m:,.2f}), indicating most value"
                    f" is captured in the first purchase cycle."
                ),
                "metric":         (
                    f"LTV 3M = ${ltv_3m:,.2f}, LTV 12M = ${ltv_12m:,.2f},"
                    f" ratio = {growth_ratio:.2f}x"
                ),
                "action":         (
                    "Launch a subscription or replenishment programme to create structural"
                    " repeat-purchase triggers. Evaluate category mix for naturally"
                    " recurring product types."
                ),
                "priority":       "High" if growth_ratio < 1.2 else "Medium",
                "priority_score": 80 if growth_ratio < 1.2 else 52,
            })
        else:
            insights.append({
                "title":          "LTV Continues Accumulating — Repeat Purchase Behaviour Confirmed",
                "finding":        (
                    f"12-month LTV is {growth_ratio:.1f}x the 3-month LTV, confirming"
                    f" genuine repeat-purchase dynamics with strong compounding returns."
                ),
                "metric":         f"LTV growth ratio (12M / 3M) = {growth_ratio:.2f}x",
                "action":         (
                    "Invest in retention infrastructure (loyalty programme, personalised"
                    " email sequences) to extend the compounding window beyond 12 months."
                ),
                "priority":       "Low",
                "priority_score": 25,
            })

    insights.sort(key=lambda x: -x["priority_score"])
    logger.info("[%s] %d insights generated", run_id, len(insights))

    return insights


# ---------------------------------------------------------------------------
# Insights summary chart
# ---------------------------------------------------------------------------

def plot_insights_summary(
    insights: list[dict[str, Any]],
    config: dict[str, Any],
    run_id: str | None = None,
    save: bool = True,
    show: bool = True,
) -> None:
    """Horizontal bar chart of insight priority scores."""
    run_id = run_id or get_run_id()
    colors = config["visualization"]["colors"]

    priority_color = {
        "High":   colors["danger"],
        "Medium": colors["warning"],
        "Low":    colors["success"],
    }

    titles = [ins["title"] for ins in insights]
    scores = [ins["priority_score"] for ins in insights]
    prios  = [ins["priority"] for ins in insights]
    bars_c = [priority_color.get(p, colors["primary"]) for p in prios]

    fig, ax = plt.subplots(figsize=(12, max(4, len(insights) * 0.9)))

    bars = ax.barh(titles[::-1], scores[::-1], color=bars_c[::-1],
                   edgecolor="white", height=0.6, alpha=0.88)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            str(score),
            va="center", ha="left", fontsize=9, color="dimgray",
        )

    ax.set_xlabel("Priority Score", fontsize=11)
    ax.set_title("Cohort Retention Insight Priority", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=priority_color["High"],   label="High"),
        Patch(facecolor=priority_color["Medium"], label="Medium"),
        Patch(facecolor=priority_color["Low"],    label="Low"),
    ]
    ax.legend(handles=legend_elements, title="Priority", fontsize=9, loc="lower right")

    plt.tight_layout()

    if save:
        path = get_output_paths(config)["figures"] / f"insights_summary_{run_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("[%s] Figure saved: %s", run_id, path.name)

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Revenue uplift scenarios
# ---------------------------------------------------------------------------

def compute_retention_scenarios(
    agg_retention: pd.DataFrame,
    ltv_benchmarks: dict[int, float],
    cohort_sizes: pd.Series,
    improvement_increments: list[float] | None = None,
    periods: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Estimate annual revenue uplift for incremental retention improvements."""
    if improvement_increments is None:
        improvement_increments = [0.05, 0.10, 0.15]
    if periods is None:
        periods = [1, 3]

    avg_cohort_size = float(cohort_sizes.mean())
    monthly_cohorts = 12

    scenarios = []
    for inc in improvement_increments:
        row: dict[str, Any] = {"increment": inc}
        for period in periods:
            ltv_at_period   = ltv_benchmarks.get(period, 0)
            extra_customers = avg_cohort_size * inc
            annual_uplift   = extra_customers * ltv_at_period * monthly_cohorts
            row[f"m{period}_uplift"] = round(annual_uplift, 0)
        scenarios.append(row)

    return scenarios