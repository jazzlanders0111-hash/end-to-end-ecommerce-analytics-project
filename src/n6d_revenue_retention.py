"""
n6e_revenue_retention.py
------------------------
Revenue-dimension retention analysis for Notebook 06.

Separates customer-count retention from revenue retention to identify
whether high-value customers churn at different rates from the average,
and tracks AOV evolution across customer tenure.

Functions
---------
build_revenue_matrix              — Cohort x period revenue aggregation
compute_revenue_retention         — Revenue retained relative to M0 baseline
plot_revenue_vs_customer_retention — Side-by-side comparison chart
compute_aov_by_period             — Average order value across period offsets
plot_aov_decay                    — AOV trend line chart
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from n6_utils import get_output_paths, get_run_id, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Revenue matrix
# ---------------------------------------------------------------------------

def build_revenue_matrix(
    df: pd.DataFrame,
    incomplete_cutoff: pd.Timestamp,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Aggregate total_amount per cohort per period_offset.

    Only non-returned orders contribute revenue.  Periods on or after
    ``incomplete_cutoff`` are masked to NaN.

    Parameters
    ----------
    df:
        Output of ``assign_acquisition_cohorts`` — must contain
        ``cohort_month``, ``period_offset``, ``returned``, ``total_amount``.
    incomplete_cutoff:
        First partial-month timestamp; observations from this point forward
        are excluded.
    run_id:
        Pipeline run identifier.

    Returns
    -------
    pd.DataFrame
        Index = cohort_month, columns = period_offset, values = total revenue.
    """
    run_id = run_id or get_run_id()
    logger.info("[%s] Building revenue matrix", run_id)

    cutoff_period = incomplete_cutoff.to_period("M")

    active = df[df["returned"] == 0].copy()
    active = active[active["order_month"] < cutoff_period]

    matrix = (
        active.groupby(["cohort_month", "period_offset"])["total_amount"]
        .sum()
        .unstack(fill_value=0.0)
    )

    # Mask incomplete periods per cohort row
    for cohort in matrix.index:
        for offset in matrix.columns:
            if (cohort + offset) >= cutoff_period:
                matrix.at[cohort, offset] = np.nan

    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    logger.info("[%s] Revenue matrix: %d cohorts x %d periods", run_id, *matrix.shape)

    return matrix


# ---------------------------------------------------------------------------
# Revenue retention
# ---------------------------------------------------------------------------

def compute_revenue_retention(
    revenue_matrix: pd.DataFrame,
    cohort_sizes: pd.Series,
) -> pd.DataFrame:
    """Compute revenue retained relative to M0 cohort revenue.

    Revenue retention at period t = total revenue[t] / total revenue[0].
    This is a cohort-level measure, not per-customer.

    Parameters
    ----------
    revenue_matrix:
        Output of :func:`build_revenue_matrix`.
    cohort_sizes:
        Series mapping cohort_month → n_customers (used for weighting the
        aggregate curve).

    Returns
    -------
    pd.DataFrame
        Index = period_offset; column: ``revenue_retention_rate``.
    """
    logger.info("Computing revenue retention rates")

    # M0 revenue per cohort
    m0_revenue = revenue_matrix[0].replace(0, np.nan)

    # Retention rate per cohort per period
    ret_rates = revenue_matrix.div(m0_revenue, axis=0)

    # Weighted aggregate (weight = M0 revenue, i.e. natural economic weight)
    sizes = cohort_sizes.reindex(revenue_matrix.index).fillna(0)

    records = []
    for period in revenue_matrix.columns:
        col   = ret_rates[period].dropna()
        w     = sizes.reindex(col.index)
        if w.sum() == 0:
            continue
        rate = np.average(col, weights=w)
        records.append({"period": period, "revenue_retention_rate": rate})

    return pd.DataFrame(records).set_index("period")


# ---------------------------------------------------------------------------
# Side-by-side comparison chart
# ---------------------------------------------------------------------------

def plot_revenue_vs_customer_retention(
    customer_retention: pd.DataFrame,
    revenue_retention: pd.DataFrame,
    config: dict[str, Any],
    run_id: str | None = None,
    save: bool = True,
    show: bool = True,
) -> None:
    """Dual-line chart comparing customer-count and revenue retention curves.

    Parameters
    ----------
    customer_retention:
        Output of :func:`compute_aggregate_retention`.
    revenue_retention:
        Output of :func:`compute_revenue_retention`.
    config:
        Loaded project configuration dictionary.
    run_id / save / show:
        Standard plotting controls.
    """
    run_id = run_id or get_run_id()
    colors = config["visualization"]["colors"]

    # Align on common periods
    common = customer_retention.index.intersection(revenue_retention.index)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        common,
        customer_retention.loc[common, "retention_rate"],
        color=colors["primary"],
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="Customer retention",
    )

    ax.plot(
        common,
        revenue_retention.loc[common, "revenue_retention_rate"],
        color=colors["secondary"],
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=5,
        label="Revenue retention",
    )

    # Shade the spread
    c_vals = customer_retention.loc[common, "retention_rate"].values
    r_vals = revenue_retention.loc[common, "revenue_retention_rate"].values
    ax.fill_between(
        common,
        c_vals,
        r_vals,
        alpha=0.12,
        color=colors["warning"],
    )

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Period Offset (months since acquisition)", fontsize=11)
    ax.set_ylabel("Retention Rate", fontsize=11)
    ax.set_title(
        "Customer Retention vs Revenue Retention\n"
        "(gap indicates high-value customer churn differential)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save:
        path = get_output_paths(config)["figures"] / f"revenue_vs_customer_retention_{run_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("[%s] Figure saved: %s", run_id, path.name)

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# AOV by period
# ---------------------------------------------------------------------------

def compute_aov_by_period(
    revenue_matrix: pd.DataFrame,
    activity_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute average order value per period offset across all cohorts.

    AOV(period) = total revenue across all cohorts / active customers across
    all cohorts.  Only periods present in both matrices are included.

    Parameters
    ----------
    revenue_matrix:
        Output of :func:`build_revenue_matrix`.
    activity_matrix:
        Output of ``build_activity_matrix`` — active customer counts.

    Returns
    -------
    pd.DataFrame
        Index = period_offset; column: ``aov``.
    """
    common_periods = revenue_matrix.columns.intersection(activity_matrix.columns)

    records = []
    for period in common_periods:
        total_rev   = revenue_matrix[period].sum(skipna=True)
        total_custs = activity_matrix[period].sum(skipna=True)
        if total_custs > 0:
            records.append({"period": period, "aov": total_rev / total_custs})

    return pd.DataFrame(records).set_index("period")


def plot_aov_decay(
    aov_by_period: pd.DataFrame,
    config: dict[str, Any],
    run_id: str | None = None,
    save: bool = True,
    show: bool = True,
) -> None:
    """Line chart of average order value across customer tenure.

    Parameters
    ----------
    aov_by_period:
        Output of :func:`compute_aov_by_period`.
    config:
        Loaded project configuration dictionary.
    run_id / save / show:
        Standard plotting controls.
    """
    run_id = run_id or get_run_id()
    colors = config["visualization"]["colors"]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        aov_by_period.index,
        aov_by_period["aov"],
        color=colors["primary"],
        linewidth=2.5,
        marker="o",
        markersize=5,
    )

    # Reference line: M0 AOV
    m0_aov = aov_by_period["aov"].iloc[0] if len(aov_by_period) else 0
    ax.axhline(
        m0_aov,
        color=colors["neutral"],
        linestyle=":",
        linewidth=1.2,
        label=f"M+0 baseline (${m0_aov:,.2f})",
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.set_xlabel("Period Offset (months since acquisition)", fontsize=11)
    ax.set_ylabel("Average Order Value (active customers)", fontsize=11)
    ax.set_title(
        "Average Order Value by Customer Tenure",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save:
        path = get_output_paths(config)["figures"] / f"aov_decay_{run_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("[%s] Figure saved: %s", run_id, path.name)

    if show:
        plt.show()

    plt.close(fig)
