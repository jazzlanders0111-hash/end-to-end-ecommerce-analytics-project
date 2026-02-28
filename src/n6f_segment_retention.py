"""
n6g_segment_retention.py
------------------------
Segment-stratified retention and LTV analysis for Notebook 06.

Joins cohort transaction data with RFM loyalty tiers and first-purchase
product category to reveal whether certain customer types retain at
materially different rates.

Functions
---------
attach_rfm_segments          — Merge loyalty tier from rfm_df onto cohorted transactions
compute_segment_retention    — Retention rate curves per segment label
plot_retention_by_segment    — Multi-line retention chart by segment
compute_segment_ltv          — Cumulative LTV at standard windows per segment
compute_first_category       — Tag each customer with their first-purchase category
plot_category_retention_comparison — Grouped bar chart across categories and periods
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
# Loyalty tier assignment
# ---------------------------------------------------------------------------

def assign_loyalty_tiers(
    rfm_df: pd.DataFrame,
    loyalty_labels: list[str],
    score_col: str = "loyalty_score",
) -> pd.DataFrame:
    """Assign loyalty tier labels to customers via quantile binning.

    Extracted from the notebook setup cell (previously inline ``pd.qcut``)
    so the binning logic is reusable and testable independently.

    Parameters
    ----------
    rfm_df:
        Customer-level RFM DataFrame containing ``score_col``.
    loyalty_labels:
        Ordered list of tier names from ``config['notebook1']['loyalty_segments']['labels']``,
        e.g. ``['Bronze', 'Silver', 'Gold', 'Platinum']``.
        The number of labels determines the number of equal-frequency bins.
    score_col:
        Column in ``rfm_df`` to bin (default: ``loyalty_score``).

    Returns
    -------
    pd.DataFrame
        Copy of ``rfm_df`` with a ``loyalty_tier`` column added.

    Raises
    ------
    ValueError
        If ``score_col`` is not present in ``rfm_df``.
    """
    if score_col not in rfm_df.columns:
        raise ValueError(
            f"Column '{score_col}' not found in rfm_df. "
            f"Available: {list(rfm_df.columns)}"
        )

    rfm_df = rfm_df.copy()
    rfm_df["loyalty_tier"] = pd.qcut(
        rfm_df[score_col],
        q=len(loyalty_labels),
        labels=loyalty_labels,
        duplicates="drop",
    )
    logger.info(
        "Loyalty tiers assigned: %s",
        dict(rfm_df["loyalty_tier"].value_counts().sort_index()),
    )
    return rfm_df


# ---------------------------------------------------------------------------
# Attach RFM segments
# ---------------------------------------------------------------------------

def attach_rfm_segments(
    df_cohorted: pd.DataFrame,
    rfm_df: pd.DataFrame,
    segment_col: str = "loyalty_tier",
    run_id: str | None = None,
) -> pd.DataFrame:
    """Merge a segment label from rfm_df onto the cohorted transaction DataFrame.

    Parameters
    ----------
    df_cohorted:
        Output of ``assign_acquisition_cohorts``.
    rfm_df:
        Customer-level RFM DataFrame containing ``customer_id`` and the
        column specified by ``segment_col``.
    segment_col:
        Column in ``rfm_df`` to merge (default: ``loyalty_tier``).
    run_id:
        Pipeline run identifier.

    Returns
    -------
    pd.DataFrame
        ``df_cohorted`` with the segment column appended.  Customers not
        present in rfm_df receive NaN and are excluded from segment analyses.
    """
    run_id = run_id or get_run_id()

    if segment_col not in rfm_df.columns:
        raise ValueError(
            f"Column '{segment_col}' not found in rfm_df. "
            f"Available: {list(rfm_df.columns)}"
        )

    mapping = rfm_df[["customer_id", segment_col]].drop_duplicates("customer_id")
    merged  = df_cohorted.merge(mapping, on="customer_id", how="left")

    n_unmatched = merged[segment_col].isna().sum()
    if n_unmatched:
        logger.warning(
            "[%s] %d transactions have no matching segment — excluded from segment analysis",
            run_id, n_unmatched,
        )

    return merged


# ---------------------------------------------------------------------------
# Segment-level retention curves
# ---------------------------------------------------------------------------

def compute_segment_retention(
    df: pd.DataFrame,
    segment_col: str,
    incomplete_cutoff: pd.Timestamp,
) -> dict[str, dict[int, float]]:
    """Compute M+t retention rates for each segment label.

    Retention = active customers in segment at period t / total customers
    in segment at acquisition (period 0).

    Parameters
    ----------
    df:
        Cohorted transaction DataFrame with segment column attached.
    segment_col:
        Column name containing the segment label.
    incomplete_cutoff:
        Periods at or after this month are excluded.

    Returns
    -------
    dict
        Maps segment_label → {period_offset: retention_rate}.
    """
    cutoff_period = incomplete_cutoff.to_period("M")

    # Only non-returned orders count as activity
    active = df[(df["returned"] == 0) & (df["order_month"] < cutoff_period)].copy()
    active = active[active[segment_col].notna()]

    results: dict[str, dict[int, float]] = {}

    for seg, seg_df in active.groupby(segment_col):
        # Cohort sizes within this segment
        seg_sizes = (
            seg_df[seg_df["period_offset"] == 0]
            .groupby("cohort_month")["customer_id"]
            .nunique()
        )
        total_acquired = seg_sizes.sum()
        if total_acquired == 0:
            continue

        # Active customers by period offset
        period_active = (
            seg_df.groupby("period_offset")["customer_id"]
            .nunique()
        )

        retention = {
            int(p): round(float(n) / total_acquired, 6)
            for p, n in period_active.items()
        }
        results[str(seg)] = retention

    return results


# ---------------------------------------------------------------------------
# Segment retention chart
# ---------------------------------------------------------------------------

def plot_retention_by_segment(
    segment_retention: dict[str, dict[int, float]],
    config: dict[str, Any],
    run_id: str | None = None,
    save: bool = True,
    show: bool = True,
    segment_type: str | None = None,
    title: str | None = None,
) -> None:
    """Multi-line chart overlaying retention curves by segment.

    Colour resolution order (first match wins):
    1. ``config['notebook6']['segment_colors'][segment_type]``  — explicit per-segment
       hex map keyed on segment label (e.g. ``{'Bronze': '#CD7F32', ...}``).
    2. ``config['visualization']['color_palette']``             — cycled by index.

    Parameters
    ----------
    segment_retention:
        Output of :func:`compute_segment_retention`.
    config:
        Loaded project configuration dictionary.
    run_id:
        Pipeline run identifier for figure naming.
    save / show:
        Persist / display controls.
    segment_type:
        Key into ``config['notebook6']['segment_colors']`` that selects the
        colour map for this call.  Recognised values:
        ``'loyalty_tiers'``, ``'nb03_segments'``, ``'first_category'``.
        Pass ``None`` to always fall back to palette cycling.
    title:
        Chart title.  Defaults to ``'Retention Rate by Segment'`` when not
        provided (replaces the previous hardcoded 'Retention Rate by Loyalty Tier').
    """
    run_id  = run_id or get_run_id()
    palette = config["visualization"]["color_palette"]

    # Resolve per-segment color map from config when segment_type is given
    seg_color_map: dict[str, str] | None = None
    if segment_type:
        nb6_colors = config.get("notebook6", {}).get("segment_colors", {})
        candidate  = nb6_colors.get(segment_type)
        # null in YAML → None in Python; skip if not a dict
        if isinstance(candidate, dict):
            seg_color_map = candidate

    chart_title  = title or "Retention Rate by Segment"
    legend_title = segment_type.replace("_", " ").title() if segment_type else "Segment"

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (seg, ret_map) in enumerate(segment_retention.items()):
        periods = sorted(ret_map.keys())
        rates   = [ret_map[p] for p in periods]

        if seg_color_map and seg in seg_color_map:
            color = seg_color_map[seg]
        else:
            color = palette[i % len(palette)]

        ax.plot(periods, rates, color=color, linewidth=2.2,
                marker="o", markersize=5, label=seg)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Period Offset (months since acquisition)", fontsize=11)
    ax.set_ylabel("Retention Rate", fontsize=11)
    ax.set_title(chart_title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, title=legend_title)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save:
        path = get_output_paths(config)["figures"] / f"retention_by_segment_{run_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("[%s] Figure saved: %s", run_id, path.name)

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Segment LTV
# ---------------------------------------------------------------------------

def compute_segment_ltv(
    df: pd.DataFrame,
    segment_col: str,
    incomplete_cutoff: pd.Timestamp,
    windows: list[int] | None = None,
) -> dict[str, dict[int, float]]:
    """Compute cumulative LTV at standard windows per segment.

    LTV(segment, t) = cumulative revenue up to period t / customers acquired
    in segment at period 0.

    Parameters
    ----------
    df:
        Cohorted transaction DataFrame with segment and total_amount columns.
    segment_col:
        Segment label column.
    incomplete_cutoff:
        Periods at or after this cutoff are excluded.
    windows:
        Period offsets at which to report LTV (default: [3, 6, 12]).

    Returns
    -------
    dict
        Maps segment_label → {window: cumulative_ltv}.
    """
    if windows is None:
        windows = [3, 6, 12]

    cutoff_period = incomplete_cutoff.to_period("M")
    active = df[
        (df["returned"] == 0)
        & (df["order_month"] < cutoff_period)
        & (df[segment_col].notna())
    ].copy()

    results: dict[str, dict[int, float]] = {}

    for seg, seg_df in active.groupby(segment_col):
        total_acquired = seg_df[seg_df["period_offset"] == 0]["customer_id"].nunique()
        if total_acquired == 0:
            continue

        cum_rev_by_period = (
            seg_df.groupby("period_offset")["total_amount"]
            .sum()
            .sort_index()
            .cumsum()
        )

        ltv_at_window: dict[int, float] = {}
        for w in windows:
            available = [p for p in cum_rev_by_period.index if p <= w]
            if available:
                ltv_at_window[w] = round(
                    float(cum_rev_by_period.loc[max(available)]) / total_acquired, 4
                )

        results[str(seg)] = ltv_at_window

    return results


# ---------------------------------------------------------------------------
# First-purchase category
# ---------------------------------------------------------------------------

def compute_first_category(
    df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    date_col: str = "order_date",
    category_col: str = "category",
    returned_col: str = "returned",
) -> pd.DataFrame:
    """Tag each transaction with the customer's first non-returned purchase category.

    Parameters
    ----------
    df:
        Cohorted transaction DataFrame.
    customer_id_col / date_col / category_col / returned_col:
        Column name overrides.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``first_category`` column added.
    """
    logger.info("Computing first-purchase category per customer")

    successful = df[df[returned_col] == 0].sort_values(date_col)

    first_cat = (
        successful.groupby(customer_id_col)
        .first()[[category_col]]
        .rename(columns={category_col: "first_category"})
        .reset_index()
    )

    return df.merge(first_cat, on=customer_id_col, how="left")


# ---------------------------------------------------------------------------
# Category retention comparison chart
# ---------------------------------------------------------------------------

def plot_category_retention_comparison(
    cat_retention: dict[str, dict[int, float]],
    periods: list[int],
    config: dict[str, Any],
    run_id: str | None = None,
    save: bool = True,
    show: bool = True,
) -> None:
    """Grouped bar chart comparing category retention at selected periods.

    Parameters
    ----------
    cat_retention:
        Output of :func:`compute_segment_retention` with ``first_category``
        as the segment column.
    periods:
        Period offsets to display (e.g. [1, 3, 6]).
    config:
        Loaded project configuration dictionary.
    run_id / save / show:
        Standard plotting controls.
    """
    run_id  = run_id or get_run_id()
    palette = config["visualization"]["color_palette"]

    categories = sorted(cat_retention.keys())
    n_cats     = len(categories)
    n_periods  = len(periods)

    x       = np.arange(n_cats)
    width   = 0.8 / n_periods
    offsets = np.linspace(-(0.4 - width / 2), 0.4 - width / 2, n_periods)

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.5), 6))

    for j, period in enumerate(periods):
        heights = [
            cat_retention.get(cat, {}).get(period, np.nan)
            for cat in categories
        ]
        valid_h = [h if not np.isnan(h) else 0 for h in heights]
        bars    = ax.bar(
            x + offsets[j],
            valid_h,
            width=width,
            color=palette[j % len(palette)],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            label=f"M+{period}",
        )

        for bar, h in zip(bars, heights):
            if not np.isnan(h) and h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{h:.0%}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("First-Purchase Category", fontsize=11)
    ax.set_ylabel("Retention Rate", fontsize=11)
    ax.set_title(
        "Retention Rate by First-Purchase Category",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, title="Period")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save:
        path = get_output_paths(config)["figures"] / f"category_retention_{run_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("[%s] Figure saved: %s", run_id, path.name)

    if show:
        plt.show()

    plt.close(fig)