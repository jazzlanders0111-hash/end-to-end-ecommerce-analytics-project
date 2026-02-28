"""
n6f_ltv.py
----------
Lifetime value accumulation curves for Notebook 06.

LTV is defined as cumulative revenue per **original cohort member** — not per
retained customer — so cohorts are directly comparable regardless of their
retention profiles.

Functions
---------
compute_cumulative_ltv   — Cumulative LTV matrix (cohort x period offset)
plot_ltv_curves          — LTV accumulation line chart per cohort
compute_ltv_benchmarks   — Weighted-average LTV at standard time windows
project_ltv              — Linear extrapolation for immature cohorts
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
# Cumulative LTV matrix
# ---------------------------------------------------------------------------

def compute_cumulative_ltv(
    revenue_matrix: pd.DataFrame,
    cohort_sizes: pd.Series,
) -> pd.DataFrame:
    """Compute cumulative LTV per original cohort member.

    LTV(cohort, t) = SUM(revenue[cohort, 0..t]) / cohort_size[cohort]

    NaN cells in ``revenue_matrix`` (incomplete periods) propagate as NaN in
    the cumulative sum so immature cohorts show NaN beyond their last
    observed period.

    Parameters
    ----------
    revenue_matrix:
        Output of ``build_revenue_matrix`` — total revenue per cohort per period.
    cohort_sizes:
        Series mapping cohort_month → n_customers (original acquisition count).

    Returns
    -------
    pd.DataFrame
        Same index/columns as ``revenue_matrix``; values = cumulative LTV.
    """
    logger.info("Computing cumulative LTV matrix")

    sizes = cohort_sizes.reindex(revenue_matrix.index).fillna(1)

    # Cumulative sum row-wise; NaN stops propagation at incomplete periods
    cum_rev = revenue_matrix.cumsum(axis=1)

    # Divide by original cohort size
    ltv = cum_rev.div(sizes, axis=0)

    return ltv


# ---------------------------------------------------------------------------
# LTV curve visualization
# ---------------------------------------------------------------------------

def plot_ltv_curves(
    ltv_matrix: pd.DataFrame,
    window_months: int,
    low_n_cohorts: pd.Series,
    config: dict[str, Any],
    run_id: str | None = None,
    save: bool = True,
    show: bool = True,
) -> None:
    """Line chart of cumulative LTV accumulation per cohort.

    Parameters
    ----------
    ltv_matrix:
        Output of :func:`compute_cumulative_ltv`.
    window_months:
        Reference window for vertical annotation line (e.g. 12 months).
    low_n_cohorts:
        Low-n cohorts — rendered dashed to indicate lower confidence.
    config:
        Loaded project configuration dictionary.
    run_id / save / show:
        Standard plotting controls.
    """
    run_id = run_id or get_run_id()
    colors = config["visualization"]["colors"]

    fig, ax = plt.subplots(figsize=(14, 7))

    palette = plt.cm.Blues(np.linspace(0.3, 0.85, len(ltv_matrix)))

    for i, (cohort, row) in enumerate(ltv_matrix.iterrows()):
        valid = row.dropna()
        if len(valid) == 0:
            continue
        ls    = "--" if cohort in low_n_cohorts.index else "-"
        alpha = 0.35 if cohort in low_n_cohorts.index else 0.65
        ax.plot(
            valid.index,
            valid.values,
            color=palette[i],
            linewidth=1.1,
            linestyle=ls,
            alpha=alpha,
        )

    # Weighted average LTV curve
    agg = _weighted_avg_ltv(ltv_matrix)
    if len(agg):
        ax.plot(
            agg.index,
            agg.values,
            color=colors["secondary"],
            linewidth=2.8,
            label="Weighted average LTV",
            zorder=10,
        )

    # Reference window vertical line
    if window_months in ltv_matrix.columns:
        ax.axvline(
            window_months,
            color=colors["danger"],
            linestyle=":",
            linewidth=1.5,
            label=f"{window_months}M reference",
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Period Offset (months since acquisition)", fontsize=11)
    ax.set_ylabel("Cumulative LTV per Acquired Customer", fontsize=11)
    ax.set_title(
        "Customer Lifetime Value Accumulation by Cohort\n(dashed = low-n cohort)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save:
        path = get_output_paths(config)["figures"] / f"ltv_curves_{run_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("[%s] Figure saved: %s", run_id, path.name)

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# LTV benchmarks
# ---------------------------------------------------------------------------

def compute_ltv_benchmarks(
    ltv_matrix: pd.DataFrame,
    windows: list[int] | None = None,
) -> dict[int, float]:
    """Compute weighted-average LTV at standard time windows.

    Parameters
    ----------
    ltv_matrix:
        Output of :func:`compute_cumulative_ltv`.
    windows:
        List of period offsets to evaluate (default: [3, 6, 12]).

    Returns
    -------
    dict[int, float]
        Mapping from window → weighted-average LTV.
    """
    if windows is None:
        windows = [3, 6, 12]

    agg  = _weighted_avg_ltv(ltv_matrix)
    result = {}
    for w in windows:
        if w in agg.index:
            result[w] = float(agg.loc[w])
        else:
            # Use the last available period as a proxy
            available = [p for p in agg.index if p <= w]
            if available:
                result[w] = float(agg.loc[max(available)])

    return result


# ---------------------------------------------------------------------------
# LTV projection (linear extrapolation for immature cohorts)
# ---------------------------------------------------------------------------

def project_ltv(
    ltv_matrix: pd.DataFrame,
    target_window: int,
    incomplete_cutoff: pd.Timestamp,
) -> dict[Any, float] | None:
    """Linearly extrapolate LTV to ``target_window`` for immature cohorts.

    Uses the slope between the last two observed data points to project
    forward.  Only cohorts with at least 3 observed periods are projected.

    Parameters
    ----------
    ltv_matrix:
        Output of :func:`compute_cumulative_ltv`.
    target_window:
        Target period offset (e.g. 12 months).
    incomplete_cutoff:
        Used to identify which cohorts are still maturing.

    Returns
    -------
    dict mapping cohort → projected LTV, or None if no immature cohorts exist.
    """
    cutoff_period = incomplete_cutoff.to_period("M")
    projections   = {}

    for cohort, row in ltv_matrix.iterrows():
        valid = row.dropna()
        last_period = max(valid.index) if len(valid) else -1

        # Only project if cohort hasn't yet reached target_window
        if last_period >= target_window:
            continue
        if len(valid) < 3:
            continue

        # Linear regression on the last 3 observed points
        x = np.array(valid.index[-3:], dtype=float)
        y = valid.values[-3:]
        slope, intercept = np.polyfit(x, y, 1)

        projected = intercept + slope * target_window
        projections[str(cohort)] = round(max(projected, 0), 2)

    return projections if projections else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_avg_ltv(ltv_matrix: pd.DataFrame) -> pd.Series:
    """Return the simple column-wise mean of LTV across cohorts (skipna).

    A simple mean is used here because cumulative LTV already incorporates
    cohort-size effects (denominator = original cohort size).
    """
    return ltv_matrix.mean(axis=0, skipna=True)
