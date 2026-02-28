"""
n6d_retention_viz.py
--------------------
Retention visualization and aggregate metric computation for Notebook 06.

Functions
---------
plot_retention_heatmap       — Annotated cohort x period heatmap
compute_aggregate_retention  — Weighted-mean retention curve across all cohorts
plot_retention_curves        — Overlaid per-cohort retention curves
identify_retention_outliers  — Flag cohorts with z > threshold from mean
compute_dropout_rates        — Period-over-period dropout computation
plot_dropout_rates           — Bar chart of dropout rates by period
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from n6_utils import get_output_paths, get_run_id, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Retention heatmap
# ---------------------------------------------------------------------------

def plot_retention_heatmap(
    retention_matrix: pd.DataFrame,
    cohort_sizes: pd.Series,
    low_n_cohorts: pd.Series,
    config: dict[str, Any],
    run_id: str | None = None,
    save: bool = True,
    show: bool = True,
) -> None:
    """Render an annotated cohort retention heatmap.

    Parameters
    ----------
    retention_matrix:
        Output of ``compute_retention_rates`` — index cohort_month,
        columns period_offset, values in [0, 1].
    cohort_sizes:
        Series mapping cohort_month → n_customers.
    low_n_cohorts:
        Low-n cohorts flagged by ``flag_low_n_cohorts``.
    config:
        Loaded project configuration dictionary.
    run_id:
        Pipeline run identifier for figure naming.
    save / show:
        Persist / display controls.
    """
    run_id = run_id or get_run_id()
    logger.info("[%s] Rendering retention heatmap", run_id)

    # Build annotation matrix: "xx%" — blank for NaN
    annot = retention_matrix.map(
        lambda v: f"{v:.0%}" if pd.notna(v) else ""
    )

    n_rows = len(retention_matrix)
    fig_h  = max(6, n_rows * 0.35)
    fig, ax = plt.subplots(figsize=(min(20, len(retention_matrix.columns) * 0.9 + 3), fig_h))

    cmap = sns.light_palette(config["visualization"]["colors"]["primary"], as_cmap=True)

    sns.heatmap(
        retention_matrix,
        annot=annot,
        fmt="",
        cmap=cmap,
        linewidths=0.3,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Retention Rate", "shrink": 0.6},
        vmin=0.0,
        vmax=1.0,
    )

    # Y-axis labels: cohort_month + cohort size
    ylabels = []
    for cohort in retention_matrix.index:
        n      = cohort_sizes.get(cohort, 0)
        suffix = "*" if cohort in low_n_cohorts.index else ""
        ylabels.append(f"{cohort}{suffix}  (n={n:,})")

    ax.set_yticklabels(ylabels, rotation=0, fontsize=8)
    ax.set_xticklabels(
        [f"M+{c}" for c in retention_matrix.columns],
        rotation=45, ha="right", fontsize=9,
    )

    ax.set_xlabel("Period Offset (months since acquisition)", fontsize=11)
    ax.set_ylabel("Acquisition Cohort", fontsize=11)
    ax.set_title(
        "Cohort Retention Heatmap\n* = low-n cohort (<30 customers)",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()

    if save:
        path = get_output_paths(config)["figures"] / f"retention_heatmap_{run_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("[%s] Figure saved: %s", run_id, path.name)

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate retention curve
# ---------------------------------------------------------------------------

def compute_aggregate_retention(
    retention_matrix: pd.DataFrame,
    cohort_sizes: pd.Series,
) -> pd.DataFrame:
    """Compute weighted-mean retention across all cohorts per period offset.

    Cohort sizes are used as weights so that larger cohorts have proportionally
    more influence on the aggregate curve.  NaN cells (incomplete periods) are
    excluded from both numerator and denominator.

    Parameters
    ----------
    retention_matrix:
        Output of ``compute_retention_rates``.
    cohort_sizes:
        Series mapping cohort_month → n_customers.

    Returns
    -------
    pd.DataFrame
        Index = period_offset; columns: ``retention_rate``, ``delta``,
        ``n_cohorts_observed``.
    """
    sizes = cohort_sizes.reindex(retention_matrix.index).fillna(0)

    records = []
    for period in retention_matrix.columns:
        col    = retention_matrix[period]
        valid  = col.dropna()
        w      = sizes.reindex(valid.index)
        if w.sum() == 0:
            continue
        rate = np.average(valid, weights=w)
        records.append({"period": period, "retention_rate": rate, "n_cohorts": len(valid)})

    agg = pd.DataFrame(records).set_index("period")
    agg["delta"] = agg["retention_rate"].diff()

    return agg


# ---------------------------------------------------------------------------
# Per-cohort retention curve overlay
# ---------------------------------------------------------------------------

def plot_retention_curves(
    retention_matrix: pd.DataFrame,
    agg_retention: pd.DataFrame,
    low_n_cohorts: pd.Series,
    config: dict[str, Any],
    run_id: str | None = None,
    save: bool = True,
    show: bool = True,
) -> None:
    """Overlay individual cohort retention curves with the aggregate benchmark.

    Parameters
    ----------
    retention_matrix:
        Output of ``compute_retention_rates``.
    agg_retention:
        Output of :func:`compute_aggregate_retention`.
    low_n_cohorts:
        Low-n cohorts — rendered as dashed lines to indicate lower confidence.
    config:
        Loaded project configuration dictionary.
    run_id / save / show:
        Standard plotting controls.
    """
    run_id = run_id or get_run_id()
    colors = config["visualization"]["colors"]

    fig, ax = plt.subplots(figsize=(14, 7))

    palette = plt.cm.Blues(np.linspace(0.3, 0.85, len(retention_matrix)))

    for i, (cohort, row) in enumerate(retention_matrix.iterrows()):
        valid   = row.dropna()
        ls      = "--" if cohort in low_n_cohorts.index else "-"
        alpha   = 0.35 if cohort in low_n_cohorts.index else 0.55
        ax.plot(
            valid.index,
            valid.values,
            color=palette[i],
            linewidth=1.0,
            linestyle=ls,
            alpha=alpha,
        )

    # Aggregate curve — prominent
    if len(agg_retention):
        ax.plot(
            agg_retention.index,
            agg_retention["retention_rate"],
            color=colors["secondary"],
            linewidth=2.8,
            linestyle="-",
            label="Aggregate (weighted avg)",
            zorder=10,
        )

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Period Offset (months since acquisition)", fontsize=11)
    ax.set_ylabel("Retention Rate", fontsize=11)
    ax.set_title(
        "Cohort Retention Curves\n(dashed = low-n cohort)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save:
        path = get_output_paths(config)["figures"] / f"retention_curves_{run_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("[%s] Figure saved: %s", run_id, path.name)

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Outlier cohort identification
# ---------------------------------------------------------------------------

def identify_retention_outliers(
    retention_matrix: pd.DataFrame,
    agg_retention: pd.DataFrame,
    z_threshold: float = 1.5,
) -> list[tuple[Any, str, float]]:
    """Identify cohorts whose mean retention deviates by |z| > threshold.

    Parameters
    ----------
    retention_matrix:
        Output of ``compute_retention_rates``.
    agg_retention:
        Output of :func:`compute_aggregate_retention`.
    z_threshold:
        Minimum absolute z-score to qualify as an outlier.

    Returns
    -------
    list of (cohort, direction, z_score)
        Sorted by absolute z-score descending.
    """
    # Mean retention per cohort across all observed periods
    cohort_means = retention_matrix.mean(axis=1, skipna=True).dropna()

    if len(cohort_means) < 3:
        return []

    pop_mean = cohort_means.mean()
    pop_std  = cohort_means.std()

    if pop_std == 0:
        return []

    outliers = []
    for cohort, mean_ret in cohort_means.items():
        z = (mean_ret - pop_mean) / pop_std
        if abs(z) >= z_threshold:
            direction = "Outperformer" if z > 0 else "Underperformer"
            outliers.append((cohort, direction, round(z, 3)))

    return sorted(outliers, key=lambda x: -abs(x[2]))


# ---------------------------------------------------------------------------
# Dropout rates
# ---------------------------------------------------------------------------

def compute_dropout_rates(agg_retention: pd.DataFrame) -> pd.DataFrame:
    """Compute period-over-period dropout rates from the aggregate curve.

    Dropout rate at period t = (retention[t-1] - retention[t]) / retention[t-1].

    Parameters
    ----------
    agg_retention:
        Output of :func:`compute_aggregate_retention`.

    Returns
    -------
    pd.DataFrame
        Columns: ``period``, ``retention_rate``, ``dropout_rate``.
    """
    df = agg_retention[["retention_rate"]].reset_index()
    df = df.rename(columns={"period": "period"})
    df["dropout_rate"] = (
        df["retention_rate"].shift(1) - df["retention_rate"]
    ) / df["retention_rate"].shift(1)
    df["dropout_rate"] = df["dropout_rate"].clip(lower=0)
    return df


def plot_dropout_rates(
    dropout_df: pd.DataFrame,
    critical_threshold: float,
    config: dict[str, Any],
    run_id: str | None = None,
    save: bool = True,
    show: bool = True,
) -> None:
    """Bar chart of period-over-period customer dropout rates.

    Parameters
    ----------
    dropout_df:
        Output of :func:`compute_dropout_rates`.
    critical_threshold:
        Dropout rate above which a bar is coloured in the danger colour.
    config:
        Loaded project configuration dictionary.
    run_id / save / show:
        Standard plotting controls.
    """
    run_id = run_id or get_run_id()
    colors = config["visualization"]["colors"]
    df     = dropout_df.dropna(subset=["dropout_rate"])

    bar_colors = [
        colors["danger"] if r >= critical_threshold else colors["primary"]
        for r in df["dropout_rate"]
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        df["period"],
        df["dropout_rate"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.4,
        width=0.7,
    )

    ax.axhline(
        critical_threshold,
        color=colors["danger"],
        linestyle="--",
        linewidth=1.2,
        label=f"Critical threshold ({critical_threshold:.0%})",
    )

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Period Offset (months)", fontsize=11)
    ax.set_ylabel("Dropout Rate (% of prior active)", fontsize=11)
    ax.set_title("Period-over-Period Customer Dropout Rate", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save:
        path = get_output_paths(config)["figures"] / f"dropout_rates_{run_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("[%s] Figure saved: %s", run_id, path.name)

    if show:
        plt.show()

    plt.close(fig)