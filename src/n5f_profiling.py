"""
n5f_profiling.py
================
Customer risk profiling and visualisation for Notebook 05.

Combines rule-based and unsupervised anomaly scores into a composite fraud
risk score, assigns risk tiers, and produces all visualisations needed for
the notebook sections.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def build_composite_risk_score(
    df: pd.DataFrame,
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge rule-based and anomaly detection scores into a composite risk score.

    Composite = w_rule * rule_fraud_score_norm
               + w_iso  * iso_anomaly_score
               + w_lof  * lof_anomaly_score

    Parameters
    ----------
    df:
        DataFrame containing rule_fraud_score_norm, iso_anomaly_score,
        lof_anomaly_score columns.
    config:
        Project configuration.
    run_id:
        Pipeline run identifier.
    verbose:
        Print summary if True.

    Returns
    -------
    pd.DataFrame — with 'composite_fraud_score' and 'risk_tier' columns added.
    """
    nb5 = config.get("notebook5", {})
    blend = nb5.get("fraud", {}).get("scoring", {}).get("blend_weights", {})

    w_rule = blend.get("rule_score", 0.40)
    w_iso  = blend.get("iso_score", 0.30)
    w_lof  = blend.get("lof_score", 0.30)

    result = df.copy()

    result["composite_fraud_score"] = (
        w_rule * result.get("rule_fraud_score_norm", 0)
        + w_iso  * result.get("iso_anomaly_score", 0)
        + w_lof  * result.get("lof_anomaly_score", 0)
    )

    # Normalise composite to [0, 1]
    c_max = result["composite_fraud_score"].max()
    if c_max > 0:
        result["composite_fraud_score"] = result["composite_fraud_score"] / c_max

    # Assign risk tiers
    result = assign_risk_tiers(result, config)

    if verbose:
        _print_composite_summary(result, w_rule, w_iso, w_lof)

    logger.info(
        "[%s] Composite scoring complete. Risk distribution: %s",
        run_id,
        result["risk_tier"].value_counts().to_dict(),
    )

    return result


def assign_risk_tiers(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Assign Low / Medium / High / Critical risk tiers based on composite score.

    Thresholds sourced from config['notebook5']['risk_tiers'].

    Parameters
    ----------
    df:
        DataFrame with 'composite_fraud_score' column.
    config:
        Project configuration.

    Returns
    -------
    pd.DataFrame — with 'risk_tier' column added.
    """
    nb5 = config.get("notebook5", {})
    tiers = nb5.get("risk_tiers", {})

    t_critical = tiers.get("critical_threshold", 0.80)
    t_high     = tiers.get("high_threshold", 0.60)
    t_medium   = tiers.get("medium_threshold", 0.35)

    def _assign(score: float) -> str:
        if score >= t_critical:
            return "Critical"
        if score >= t_high:
            return "High"
        if score >= t_medium:
            return "Medium"
        return "Low"

    df = df.copy()
    df["risk_tier"] = df["composite_fraud_score"].apply(_assign)

    tier_order = ["Critical", "High", "Medium", "Low"]
    df["risk_tier"] = pd.Categorical(df["risk_tier"], categories=tier_order, ordered=True)

    return df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_score_distributions(
    df: pd.DataFrame,
    config: dict[str, Any],
    save: bool = True,
    show: bool = True,
    subfolder: str = "notebook5_fig",
    run_id: str = "",
) -> list[Path]:
    """
    Plot distribution of fraud scores (rule-based, ISO, LOF, composite).

    Returns list of saved figure paths.
    """
    figures_dir = _get_figures_dir(config, subfolder)
    saved: list[Path] = []

    color_primary   = config.get("visualization", {}).get("colors", {}).get("primary",   "#1B1F5E")
    color_secondary = config.get("visualization", {}).get("colors", {}).get("secondary", "#EA731D")
    color_danger    = config.get("visualization", {}).get("colors", {}).get("danger",    "#C62828")
    color_success   = config.get("visualization", {}).get("colors", {}).get("success",   "#2E7D32")

    score_cols = {
        "rule_fraud_score_norm": ("Rule-Based Score", color_primary),
        "iso_anomaly_score":     ("Isolation Forest Score", color_secondary),
        "lof_anomaly_score":     ("LOF Score", color_danger),
        "composite_fraud_score": ("Composite Fraud Score", color_success),
    }

    available = {k: v for k, v in score_cols.items() if k in df.columns}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (col, (label, color)) in enumerate(available.items()):
        ax = axes[idx]
        s  = df[col].dropna()

        ax.hist(s, bins=50, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(s.mean(),   color="black",  linestyle="--", linewidth=1.2, label=f"Mean: {s.mean():.3f}")
        ax.axvline(s.median(), color="dimgray", linestyle=":",  linewidth=1.2, label=f"Median: {s.median():.3f}")

        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Customers", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide any unused subplots
    for idx in range(len(available), 4):
        axes[idx].set_visible(False)

    fig.suptitle("Fraud Score Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = figures_dir / f"score_distributions_{run_id}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
        logger.info("Saved: %s", path)

    if show:
        plt.show()
    plt.close(fig)

    return saved


def plot_risk_tier_breakdown(
    df: pd.DataFrame,
    config: dict[str, Any],
    save: bool = True,
    show: bool = True,
    subfolder: str = "notebook5_fig",
    run_id: str = "",
) -> list[Path]:
    """
    Plot risk tier distribution and key metrics per tier.

    Returns list of saved figure paths.
    """
    figures_dir = _get_figures_dir(config, subfolder)
    saved: list[Path] = []

    tier_colors = {
        "Low":      "#2E7D32",
        "Medium":   "#F57C00",
        "High":     "#C62828",
        "Critical": "#4A148C",
    }

    tier_order = ["Critical", "High", "Medium", "Low"]
    tier_counts = (
        df["risk_tier"]
        .value_counts()
        .reindex(tier_order)
        .fillna(0)
        .astype(int)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar chart of customer counts per tier
    ax = axes[0]
    bars = ax.barh(
        tier_counts.index,
        tier_counts.values,
        color=[tier_colors[t] for t in tier_counts.index],
        edgecolor="white",
        height=0.6,
    )
    for bar, val in zip(bars, tier_counts.values):
        pct = val / len(df) * 100
        ax.text(
            bar.get_width() + len(df) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=10,
        )
    ax.set_xlabel("Number of Customers", fontsize=10)
    ax.set_title("Customers by Risk Tier", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, tier_counts.max() * 1.25)

    # Right: Box plot of composite score by tier
    ax = axes[1]
    tier_data = [
        df.loc[df["risk_tier"] == t, "composite_fraud_score"].dropna().values
        for t in tier_order if t in df["risk_tier"].values
    ]
    valid_tiers = [t for t in tier_order if t in df["risk_tier"].values]
    bps = ax.boxplot(
        tier_data,
        labels=valid_tiers,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    for patch, tier in zip(bps["boxes"], valid_tiers):
        patch.set_facecolor(tier_colors[tier])
        patch.set_alpha(0.85)

    ax.set_xlabel("Risk Tier", fontsize=10)
    ax.set_ylabel("Composite Fraud Score", fontsize=10)
    ax.set_title("Score Distribution by Risk Tier", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    path = figures_dir / f"risk_tier_breakdown_{run_id}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
        logger.info("Saved: %s", path)

    if show:
        plt.show()
    plt.close(fig)

    return saved


def plot_feature_radar(
    df: pd.DataFrame,
    config: dict[str, Any],
    save: bool = True,
    show: bool = True,
    subfolder: str = "notebook5_fig",
    run_id: str = "",
) -> list[Path]:
    """
    Radar chart comparing mean feature values across risk tiers.

    Shows the behavioural fingerprint of each risk tier.
    """
    figures_dir = _get_figures_dir(config, subfolder)
    saved: list[Path] = []

    radar_features = [
        "return_rate",
        "max_discount_rate",
        "negative_margin_rate",
        "velocity_burst_flag",
        "high_value_rate_p95",
        "category_hhi",
        "discount_usage_rate",
        "order_value_cv",
    ]
    radar_features = [f for f in radar_features if f in df.columns]
    if not radar_features:
        logger.warning("No radar features available — skipping radar plot")
        return saved

    tier_colors = {
        "Low": "#2E7D32", "Medium": "#F57C00",
        "High": "#C62828", "Critical": "#4A148C",
    }
    tier_order = [t for t in ["Critical", "High", "Medium", "Low"] if t in df["risk_tier"].values]

    # Normalise features to [0, 1] for comparability
    means_by_tier: dict[str, list[float]] = {}
    for tier in tier_order:
        sub = df[df["risk_tier"] == tier][radar_features]
        means_by_tier[tier] = sub.mean().tolist()

    # Build radar
    n = len(radar_features)
    angles = [k * 2 * np.pi / n for k in range(n)]
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})

    for tier in tier_order:
        values = means_by_tier[tier]
        # Normalise each feature to [0, 1] relative to max across tiers
        norm_values = []
        for i, feat in enumerate(radar_features):
            all_means = [means_by_tier[t][i] for t in tier_order]
            mx = max(all_means) if max(all_means) > 0 else 1.0
            norm_values.append(values[i] / mx)

        norm_values += norm_values[:1]  # close polygon
        ax.plot(angles, norm_values, color=tier_colors[tier], linewidth=2.0, label=tier)
        ax.fill(angles, norm_values, color=tier_colors[tier], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [f.replace("_", " ").title() for f in radar_features],
        fontsize=9,
    )
    ax.set_yticklabels([])
    ax.set_title("Risk Tier Behavioural Fingerprint\n(Normalised Feature Means)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=10,
        title="Risk Tier",
    )
    ax.spines["polar"].set_visible(False)

    plt.tight_layout()

    path = figures_dir / f"feature_radar_{run_id}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
        logger.info("Saved: %s", path)

    if show:
        plt.show()
    plt.close(fig)

    return saved


def plot_anomaly_scatter(
    df: pd.DataFrame,
    config: dict[str, Any],
    save: bool = True,
    show: bool = True,
    subfolder: str = "notebook5_fig",
    run_id: str = "",
) -> list[Path]:
    """
    Scatter plot of Isolation Forest vs LOF anomaly scores,
    coloured by risk tier.
    """
    figures_dir = _get_figures_dir(config, subfolder)
    saved: list[Path] = []

    if "iso_anomaly_score" not in df.columns or "lof_anomaly_score" not in df.columns:
        logger.warning("Anomaly score columns not found — skipping scatter plot")
        return saved

    tier_colors = {
        "Low": "#2E7D32", "Medium": "#F57C00",
        "High": "#C62828", "Critical": "#4A148C",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for tier, color in tier_colors.items():
        sub = df[df["risk_tier"] == tier]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["iso_anomaly_score"],
            sub["lof_anomaly_score"],
            c=color, label=tier,
            alpha=0.55, s=18, linewidths=0,
        )

    ax.set_xlabel("Isolation Forest Anomaly Score", fontsize=11)
    ax.set_ylabel("LOF Anomaly Score", fontsize=11)
    ax.set_title(
        "Model Agreement: Isolation Forest vs LOF\n(Coloured by Risk Tier)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Risk Tier", fontsize=10, title_fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    path = figures_dir / f"anomaly_scatter_{run_id}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
        logger.info("Saved: %s", path)

    if show:
        plt.show()
    plt.close(fig)

    return saved


def plot_top_risk_heatmap(
    df: pd.DataFrame,
    config: dict[str, Any],
    top_n: int = 50,
    save: bool = True,
    show: bool = True,
    subfolder: str = "notebook5_fig",
    run_id: str = "",
) -> list[Path]:
    """
    Heatmap of fraud feature values for the top-N highest-risk customers.
    """
    figures_dir = _get_figures_dir(config, subfolder)
    saved: list[Path] = []

    heatmap_features = [
        "return_rate",
        "max_discount_rate",
        "high_discount_return_rate",
        "negative_margin_rate",
        "max_orders_7d",
        "velocity_burst_flag",
        "high_value_rate_p95",
        "category_hhi",
        "order_value_cv",
        "discount_usage_rate",
    ]
    heatmap_features = [f for f in heatmap_features if f in df.columns]

    top_customers = df.nlargest(top_n, "composite_fraud_score")
    heat_data = top_customers[heatmap_features]

    # Normalise each column to [0, 1]
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min()).replace(0, 1)

    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = LinearSegmentedColormap.from_list(
        "fraud_risk", ["#EBF5FB", "#F0B27A", "#C0392B"]
    )
    sns.heatmap(
        heat_norm,
        ax=ax,
        cmap=cmap,
        xticklabels=[f.replace("_", " ").title() for f in heatmap_features],
        yticklabels=False,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Normalised Feature Value", "shrink": 0.6},
    )
    ax.set_title(
        f"Feature Heatmap: Top {top_n} Highest-Risk Customers",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Fraud Feature", fontsize=10)
    ax.set_ylabel("Customer (sorted by risk score)", fontsize=10)
    plt.xticks(rotation=35, ha="right", fontsize=9)

    plt.tight_layout()

    path = figures_dir / f"top_risk_heatmap_{run_id}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
        logger.info("Saved: %s", path)

    if show:
        plt.show()
    plt.close(fig)

    return saved


# ---------------------------------------------------------------------------
# Risk profile summary
# ---------------------------------------------------------------------------

def build_risk_profile_summary(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a summary table of key metrics per risk tier.

    Returns pd.DataFrame indexed by risk tier.
    """
    tier_order = ["Critical", "High", "Medium", "Low"]
    available_tiers = [t for t in tier_order if t in df["risk_tier"].values]

    profile_cols = {
        "composite_fraud_score": "Avg Fraud Score",
        "return_rate":           "Avg Return Rate",
        "max_discount_rate":     "Avg Max Disc. Rate",
        "negative_margin_rate":  "Avg Neg. Margin Rate",
        "max_orders_7d":         "Avg Max Orders/7d",
        "order_count":           "Avg Order Count",
        "avg_order_value":       "Avg Order Value",
    }

    rows: list[dict] = []
    for tier in available_tiers:
        sub = df[df["risk_tier"] == tier]
        row: dict[str, Any] = {
            "Risk Tier":        tier,
            "Customers":        len(sub),
            "% of Total":       f"{len(sub)/len(df)*100:.1f}%",
        }
        for col, label in profile_cols.items():
            if col in sub.columns:
                row[label] = sub[col].mean()
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("Risk Tier")

    if verbose:
        print(f"\n{'='*80}")
        print("RISK TIER PROFILE SUMMARY".center(80))
        print("=" * 80)
        print()
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(summary.to_string())
        print("=" * 80)

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_figures_dir(config: dict[str, Any], subfolder: str) -> Path:
    """Resolve and create the figures output directory."""
    from n5a_utils import get_project_root

    root = get_project_root()
    base = config.get("paths", {}).get("figures_dir", "outputs/figures")
    figures_dir = root / base / subfolder
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _print_composite_summary(
    df: pd.DataFrame,
    w_rule: float,
    w_iso: float,
    w_lof: float,
) -> None:
    """Print composite scoring summary."""
    print(f"\n{'='*80}")
    print("COMPOSITE RISK SCORING".center(80))
    print("=" * 80)
    print(f"\nBlend weights:")
    print(f"  Rule-based:        {w_rule:.0%}")
    print(f"  Isolation Forest:  {w_iso:.0%}")
    print(f"  LOF:               {w_lof:.0%}")
    print()

    tier_order = ["Critical", "High", "Medium", "Low"]
    tier_colors_txt = {
        "Critical": "CRITICAL", "High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"
    }
    print(f"{'Risk Tier':<12} {'Customers':>12} {'% Total':>10} {'Mean Score':>12}")
    print("-" * 50)
    for tier in tier_order:
        sub = df[df["risk_tier"] == tier]
        if len(sub) == 0:
            continue
        print(
            f"{tier:<12} {len(sub):>12,} "
            f"{len(sub)/len(df)*100:>9.1f}% "
            f"{sub['composite_fraud_score'].mean():>12.4f}"
        )
    print("=" * 80)