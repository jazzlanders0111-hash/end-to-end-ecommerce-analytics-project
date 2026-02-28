"""
n6c_cohort_builder.py
---------------------
Cohort assignment, activity matrix construction, and retention rate
computation for Notebook 06: Cohort Retention & Lifecycle Analysis.

Key functions
-------------
assign_acquisition_cohorts  — Tag each customer with their first-order month
build_activity_matrix       — Customer x period-offset active flag matrix
compute_retention_rates     — Proportion of original cohort active each period
flag_low_n_cohorts          — Identify cohorts below minimum size threshold
plot_cohort_sizes           — Bar chart of cohort acquisition volumes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from n6_utils import get_output_paths, get_run_id, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Cohort assignment
# ---------------------------------------------------------------------------

def assign_acquisition_cohorts(
    df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    date_col: str = "order_date",
    returned_col: str = "returned",
    run_id: str | None = None,
) -> pd.DataFrame:
    """Assign each transaction to an acquisition cohort.

    Acquisition cohort = calendar month of the customer's first
    **non-returned** order.  Customers whose entire order history is
    returns receive ``cohort_month = NaT`` and are dropped.

    Parameters
    ----------
    df:
        Transaction-level DataFrame from NB01.
    customer_id_col:
        Column name for customer identifiers.
    date_col:
        Column name for order date (datetime).
    returned_col:
        Binary column (0/1) indicating a returned order.
    run_id:
        Pipeline run identifier for log tracing.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        ``cohort_month`` (Period[M]), ``order_month`` (Period[M]),
        ``period_offset`` (int — months since acquisition).
    """
    run_id = run_id or get_run_id()
    logger.info("[%s] Assigning acquisition cohorts", run_id)

    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Month period columns
    df["order_month"] = df[date_col].dt.to_period("M")

    # Acquisition month: earliest non-returned order per customer
    successful = df[df[returned_col] == 0].copy()
    first_order = (
        successful.groupby(customer_id_col)[date_col]
        .min()
        .dt.to_period("M")
        .rename("cohort_month")
        .reset_index()
    )

    df = df.merge(first_order, on=customer_id_col, how="left")

    # Customers with no successful orders → drop (return-only customers)
    n_return_only = df[df["cohort_month"].isna()][customer_id_col].nunique()
    if n_return_only:
        logger.warning(
            "[%s] Dropping %d return-only customers (no successful orders)",
            run_id, n_return_only,
        )
    df = df[df["cohort_month"].notna()].copy()

    # Period offset in months
    df["period_offset"] = (
        df["order_month"].astype("int64") - df["cohort_month"].astype("int64")
    )

    # Negative offsets are data anomalies (order before acquisition) — clamp to 0
    neg_offsets = (df["period_offset"] < 0).sum()
    if neg_offsets:
        logger.warning(
            "[%s] %d rows have negative period offset — clamped to 0", run_id, neg_offsets
        )
        df["period_offset"] = df["period_offset"].clip(lower=0)

    n_cohorts = df["cohort_month"].nunique()
    logger.info("[%s] Acquisition cohorts assigned: %d cohorts", run_id, n_cohorts)

    return df


# ---------------------------------------------------------------------------
# Activity matrix
# ---------------------------------------------------------------------------

def build_activity_matrix(
    df: pd.DataFrame,
    incomplete_cutoff: pd.Timestamp,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Build a cohort x period_offset activity matrix.

    A cell is 1 if at least one non-returned order was placed by any customer
    in that cohort in that period offset, 0 otherwise.  Periods that fall
    entirely on or after ``incomplete_cutoff`` are masked to NaN to prevent
    survivorship bias in rate calculations.

    Parameters
    ----------
    df:
        Output of :func:`assign_acquisition_cohorts`.
    incomplete_cutoff:
        Timestamp of the first incomplete (partial) calendar month.
        Period offsets whose calendar month >= cutoff are excluded.
    run_id:
        Pipeline run identifier.

    Returns
    -------
    pd.DataFrame
        Index = cohort_month (Period[M]), columns = period_offset (int).
        Values are active-customer counts.
    """
    run_id = run_id or get_run_id()
    logger.info("[%s] Building activity matrix", run_id)

    cutoff_period = incomplete_cutoff.to_period("M")

    # Only count activity on non-returned orders
    active = df[df["returned"] == 0].copy()

    # Mask periods that fall within or after the incomplete month
    active["calendar_month"] = active["order_month"]
    active = active[active["calendar_month"] < cutoff_period]

    # Count distinct active customers per (cohort, offset)
    matrix = (
        active.groupby(["cohort_month", "period_offset"])["customer_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    # Ensure period 0 exists (M0 = acquisition month)
    if 0 not in matrix.columns:
        matrix.insert(0, 0, 0)

    matrix = matrix.sort_index(axis=0).sort_index(axis=1)

    logger.info(
        "[%s] Activity matrix: %d cohorts x %d periods",
        run_id, *matrix.shape,
    )
    return matrix


# ---------------------------------------------------------------------------
# Retention rate matrix
# ---------------------------------------------------------------------------

def compute_retention_rates(
    activity_matrix: pd.DataFrame,
    cohort_sizes: pd.Series,
    incomplete_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Compute retention rates from the activity matrix.

    Retention rate = active customers in period / cohort size at acquisition.

    Parameters
    ----------
    activity_matrix:
        Output of :func:`build_activity_matrix`.
    cohort_sizes:
        Series mapping cohort_month → number of unique customers acquired.
    incomplete_cutoff:
        Periods at or after this month are set to NaN (incomplete observation).

    Returns
    -------
    pd.DataFrame
        Same shape as ``activity_matrix``; values in [0, 1].
    """
    logger.info("Computing retention rate matrix")

    cutoff_period = incomplete_cutoff.to_period("M")

    # Align cohort sizes to matrix index
    sizes = cohort_sizes.reindex(activity_matrix.index).fillna(1)

    # Divide each row by cohort size
    retention = activity_matrix.div(sizes, axis=0)

    # Mask (cohort_month + offset) cells that fall within incomplete months
    for cohort in retention.index:
        for offset in retention.columns:
            obs_month = cohort + offset
            if obs_month >= cutoff_period:
                retention.at[cohort, offset] = np.nan

    # Period 0 should always be exactly 1.0 by construction; enforce this
    if 0 in retention.columns:
        retention[0] = 1.0

    # Clip to [0, 1] for numerical safety
    retention = retention.clip(lower=0.0, upper=1.0)

    return retention


# ---------------------------------------------------------------------------
# Low-n cohort flagging
# ---------------------------------------------------------------------------

def flag_low_n_cohorts(
    cohort_sizes: pd.Series,
    min_size: int = 30,
) -> pd.Series:
    """Return cohorts with fewer than ``min_size`` customers.

    Parameters
    ----------
    cohort_sizes:
        Series mapping cohort_month → n_customers.
    min_size:
        Minimum cohort size threshold.

    Returns
    -------
    pd.Series
        Subset of ``cohort_sizes`` for cohorts below the threshold.
    """
    return cohort_sizes[cohort_sizes < min_size]


# ---------------------------------------------------------------------------
# Acquisition statistics
# ---------------------------------------------------------------------------

def compute_acquisition_stats(cohort_sizes: pd.Series) -> dict[str, Any]:
    """Compute summary statistics for customer acquisition volume across cohorts.

    Extracted from the notebook setup cell so the same calculation is reusable
    and testable independently of display logic.

    Parameters
    ----------
    cohort_sizes:
        Series mapping cohort_month → n_customers, in any order.

    Returns
    -------
    dict with keys:
        ``monthly_growth_pct`` (float) — average month-over-month growth rate
            as a percentage; negative = decline.
        ``cv``  (float)  — coefficient of variation (std / mean); > 0.3 = volatile.
        ``trend`` (str)  — ``'Growth'`` or ``'Decline'``.
        ``volume`` (str) — ``'Stable'`` or ``'Volatile'``.
    """
    sizes_sorted    = cohort_sizes.sort_index()
    monthly_growth  = float(sizes_sorted.pct_change().mean() * 100)
    cv              = float(cohort_sizes.std() / cohort_sizes.mean())

    return {
        "monthly_growth_pct": round(monthly_growth, 2),
        "cv":                  round(cv, 4),
        "trend":               "Growth" if monthly_growth > 0 else "Decline",
        "volume":              "Stable" if cv < 0.3 else "Volatile",
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_cohort_sizes(
    cohort_sizes: pd.Series,
    low_n_cohorts: pd.Series,
    min_cohort_size: int,
    config: dict[str, Any],
    run_id: str | None = None,
    save: bool = True,
    show: bool = True,
) -> None:
    """Bar chart of customer acquisition volume by cohort month.

    Low-n cohorts are highlighted in a warning colour.

    Parameters
    ----------
    cohort_sizes:
        Series mapping cohort_month → n_customers.
    low_n_cohorts:
        Output of :func:`flag_low_n_cohorts`.
    min_cohort_size:
        Threshold for the reference line.
    config:
        Loaded project configuration dictionary.
    run_id:
        Pipeline run identifier for figure naming.
    save:
        Persist figure to disk when True.
    show:
        Display figure in notebook when True.
    """
    run_id  = run_id or get_run_id()
    colors  = config["visualization"]["colors"]
    palette = config["visualization"]["color_palette"]

    bar_colors = [
        colors["warning"] if c in low_n_cohorts.index else palette[0]
        for c in cohort_sizes.index
    ]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(
        range(len(cohort_sizes)),
        cohort_sizes.values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.4,
        width=0.8,
    )

    ax.axhline(
        min_cohort_size,
        color=colors["danger"],
        linestyle="--",
        linewidth=1.2,
        label=f"Min cohort size ({min_cohort_size})",
    )

    # X-axis labels — show every other label if many cohorts
    tick_step = max(1, len(cohort_sizes) // 20)
    ax.set_xticks(range(0, len(cohort_sizes), tick_step))
    ax.set_xticklabels(
        [str(c) for c in cohort_sizes.index[::tick_step]],
        rotation=45, ha="right", fontsize=8,
    )

    for i, (cohort, n) in enumerate(cohort_sizes.items()):
        if n == cohort_sizes.max() or n == cohort_sizes.min():
            ax.text(i, n + cohort_sizes.max() * 0.01, f"{n:,}", ha="center",
                    va="bottom", fontsize=7, color="dimgray")

    ax.set_xlabel("Acquisition Cohort (Month)", fontsize=11)
    ax.set_ylabel("Customers Acquired", fontsize=11)
    ax.set_title("Customer Acquisition Volume by Cohort", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        paths   = get_output_paths(config)
        fig_dir = paths["figures"]
        path    = fig_dir / f"cohort_sizes_{run_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("[%s] Figure saved: %s", run_id, path.name)

    if show:
        plt.show()

    plt.close(fig)