"""
n5g_insights.py
===============
Business insights and recommendations for Notebook 05.

Translates risk scores and anomaly flags into actionable business outputs:
  - Financial exposure estimates
  - Fraud typology analysis
  - Pattern correlation analysis
  - Intervention recommendations by risk tier
  - Executive summary generation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Financial exposure
# ---------------------------------------------------------------------------

def estimate_financial_exposure(
    df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Estimate financial exposure attributable to high-risk customers.

    Each order is assigned to exactly ONE exposure bucket using priority ordering,
    preventing double-counting of orders that trigger multiple signals:

      Priority 1 — Return Bucket (returned == 1)
          Exposure = total_amount (revenue paid out for returned goods).
          Discount and margin signals are irrelevant once an item is returned.

      Priority 2 — Discount Bucket (not returned AND discount > 0)
          Exposure = price × quantity × discount (revenue foregone vs. baseline).
          Negative margin may be caused by the discount itself; discount is root cause.

      Priority 3 — Negative Margin Bucket (not returned, no discount, profit_margin < 0)
          Exposure = abs(profit_margin) (structural pricing loss).

    Overlap orders (e.g. returned + discounted) are counted once in the
    highest-priority bucket. Raw overlap counts are reported for transparency.

    Parameters
    ----------
    df:
        Customer risk profile DataFrame with risk_tier.
    transactions_df:
        Raw transaction DataFrame.
    config:
        Project configuration.
    run_id:
        Pipeline run identifier.
    verbose:
        Print exposure summary if True.

    Returns
    -------
    dict containing deduplicated exposure estimates by component and tier.
    """
    high_risk_customers = df[df["risk_tier"].isin(["Critical", "High"])]["customer_id"]

    tx_high = transactions_df[transactions_df["customer_id"].isin(high_risk_customers)].copy()
    tx_all  = transactions_df.copy()

    # Ensure returned is numeric
    for frame in [tx_high, tx_all]:
        if frame["returned"].dtype == object:
            frame["returned"] = frame["returned"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    def _deduplicated_exposure(tx: pd.DataFrame) -> dict[str, float]:
        """
        Compute deduplicated exposure components for a transaction DataFrame.

        Returns dict with return_, discount_, neg_margin_, total_ exposure
        and overlap order counts.
        """
        # --- Priority 1: Return bucket ---
        mask_returned = tx["returned"] == 1
        return_exp = tx.loc[mask_returned, "total_amount"].sum()

        # --- Priority 2: Discount bucket (not returned, has discount) ---
        mask_discount = (~mask_returned) & (tx["discount"] > 0)
        discount_exp = (
            tx.loc[mask_discount, "price"]
            * tx.loc[mask_discount, "quantity"]
            * tx.loc[mask_discount, "discount"]
        ).sum()

        # --- Priority 3: Negative margin bucket (not returned, no discount) ---
        mask_neg_margin = (
            (~mask_returned)
            & (tx["discount"] == 0)
            & (tx["profit_margin"] < 0)
        )
        neg_margin_exp = tx.loc[mask_neg_margin, "profit_margin"].abs().sum()

        # --- Overlap counts (for transparency reporting) ---
        # Orders that would have been double-counted without deduplication
        overlap_return_discount  = (mask_returned & (tx["discount"] > 0)).sum()
        overlap_return_negmargin = (mask_returned & (tx["profit_margin"] < 0)).sum()
        overlap_disc_negmargin   = (
            mask_discount & (tx["profit_margin"] < 0)
        ).sum()
        total_overlap = overlap_return_discount + overlap_return_negmargin + overlap_disc_negmargin

        return {
            "return_exp":             return_exp,
            "discount_exp":           discount_exp,
            "neg_margin_exp":         neg_margin_exp,
            "total_exp":              return_exp + discount_exp + neg_margin_exp,
            "overlap_return_disc":    int(overlap_return_discount),
            "overlap_return_negmarg": int(overlap_return_negmargin),
            "overlap_disc_negmarg":   int(overlap_disc_negmargin),
            "total_overlap_orders":   int(total_overlap),
        }

    high_exp = _deduplicated_exposure(tx_high)
    all_exp  = _deduplicated_exposure(tx_all)

    # Total revenue from high-risk customers (not deduplicated — revenue is not exposure)
    revenue_high = tx_high["total_amount"].sum()
    revenue_all  = tx_all["total_amount"].sum()

    exposure = {
        # Customer counts
        "high_risk_customers":         len(high_risk_customers),
        "high_risk_pct":               len(high_risk_customers) / len(df) * 100,

        # Revenue (gross, not deduplicated)
        "revenue_high_risk":           revenue_high,
        "revenue_high_risk_pct":       revenue_high / revenue_all * 100 if revenue_all else 0,

        # Deduplicated exposure — high-risk customers
        "return_exposure_high_risk":   high_exp["return_exp"],
        "discount_exposure_high_risk": high_exp["discount_exp"],
        "neg_margin_exposure_high":    high_exp["neg_margin_exp"],
        "total_exposure_high_risk":    high_exp["total_exp"],

        # Deduplicated exposure — all customers (for comparison)
        "return_exposure_all":         all_exp["return_exp"],
        "discount_exposure_all":       all_exp["discount_exp"],
        "neg_margin_exposure_all":     all_exp["neg_margin_exp"],
        "total_exposure_all":          all_exp["total_exp"],

        # Overlap transparency — high-risk customers
        "overlap_return_discount":     high_exp["overlap_return_disc"],
        "overlap_return_negmargin":    high_exp["overlap_return_negmarg"],
        "overlap_discount_negmargin":  high_exp["overlap_disc_negmarg"],
        "total_overlap_orders":        high_exp["total_overlap_orders"],
    }

    if verbose:
        _print_exposure_summary(exposure)

    logger.info(
        "[%s] Financial exposure (deduplicated): high-risk customers = %d (%.1f%%), "
        "exposure = $%.0f, overlap orders resolved = %d",
        run_id,
        exposure["high_risk_customers"],
        exposure["high_risk_pct"],
        exposure["total_exposure_high_risk"],
        exposure["total_overlap_orders"],
    )

    return exposure


# ---------------------------------------------------------------------------
# Fraud typology
# ---------------------------------------------------------------------------

def classify_fraud_typology(
    df: pd.DataFrame,
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Assign primary fraud typology labels to high-risk customers.

    Typology thresholds are read from config['notebook5']['fraud']['rule_thresholds']
    so they stay consistent with the rule-based scoring thresholds. This prevents
    customers from scoring into High/Critical tiers without qualifying for any
    typology label.

    Typologies (can co-occur):
      - Return Abuser:          return_rate >= rule_thresholds.return_rate_high
      - Discount Exploiter:     max_discount_rate >= rule_thresholds.discount_rate_high
                                AND avg_discount > 0.15
      - Velocity Fraudster:     velocity_burst_flag == 1
      - Margin Exploiter:       negative_margin_rate >= rule_thresholds.negative_margin_rate_high
      - High-Value Anomaly:     high_value_rate_p99 > rule_thresholds.high_value_p99_rate
      - Combined Attack:        3+ typologies simultaneously

    Parameters
    ----------
    df:
        Customer risk DataFrame.
    config:
        Project configuration.
    run_id:
        Pipeline run identifier.
    verbose:
        Print typology summary if True.

    Returns
    -------
    pd.DataFrame — with typology flag columns and 'primary_typology' added.
    """
    result = df.copy()

    # Read scoring thresholds from config so typology boundaries are consistent
    # with the rule-based scoring that determines a customer's risk tier.
    nb5_thresholds = (
        config.get("notebook5", {})
        .get("fraud", {})
        .get("rule_thresholds", {})
    )
    t_return_rate       = nb5_thresholds.get("return_rate_high",          0.20)
    t_discount_rate     = nb5_thresholds.get("discount_rate_high",         0.80)
    t_neg_margin_rate   = nb5_thresholds.get("negative_margin_rate_high",  0.50)
    t_high_value_p99    = nb5_thresholds.get("high_value_p99_rate",        0.20)

    result["type_return_abuser"] = (
        result.get("return_rate", 0) >= t_return_rate
    ).astype(int)

    result["type_discount_exploiter"] = (
        (result.get("max_discount_rate", 0) >= t_discount_rate)
        & (result.get("avg_discount", 0) > 0.15)
    ).astype(int)

    result["type_velocity"] = result.get("velocity_burst_flag", 0).astype(int)

    result["type_margin_exploiter"] = (
        result.get("negative_margin_rate", 0) >= t_neg_margin_rate
    ).astype(int)

    result["type_high_value"] = (
        result.get("high_value_rate_p99", 0) > t_high_value_p99
    ).astype(int)

    typology_cols = [
        "type_return_abuser", "type_discount_exploiter",
        "type_velocity", "type_margin_exploiter", "type_high_value",
    ]

    result["typology_count"] = result[typology_cols].sum(axis=1)
    result["type_combined"]  = (result["typology_count"] >= 3).astype(int)

    # Primary typology label (highest contributing type by score)
    label_map = {
        "type_return_abuser":    "Return Abuse",
        "type_discount_exploiter": "Discount Exploitation",
        "type_velocity":         "Velocity Anomaly",
        "type_margin_exploiter": "Margin Exploitation",
        "type_high_value":       "High-Value Anomaly",
        "type_combined":         "Combined Attack",
    }

    def _primary(row: pd.Series) -> str:
        if row["type_combined"] == 1:
            return "Combined Attack"
        for col, label in label_map.items():
            if col != "type_combined" and row.get(col, 0) == 1:
                return label
        return "Unclassified"

    result["primary_typology"] = result.apply(_primary, axis=1)

    if verbose:
        _print_typology_summary(result, typology_cols, label_map)

    logger.info(
        "[%s] Typology classification complete. Typologies: %s",
        run_id,
        result["primary_typology"].value_counts().to_dict(),
    )

    return result


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def build_intervention_recommendations(
    df: pd.DataFrame,
    exposure: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Generate intervention recommendations based on risk findings.

    Parameters
    ----------
    df:
        Full customer risk DataFrame with 'risk_tier' and 'primary_typology'
        columns. Counts are computed internally from High/Critical customers
        only so that recommendation rationales cite actionable populations,
        not population-wide typology totals.
    exposure:
        Output of estimate_financial_exposure().
    config:
        Project configuration.

    Returns
    -------
    list of recommendation dicts with keys: tier, action, rationale, priority.
    """
    campaigns = config.get("notebook5", {}).get("interventions", {})

    # Compute typology counts from High/Critical customers ONLY.
    # Using the full population would overstate the target audience in each
    # recommendation (e.g. reporting 462 return abusers when only 96 are
    # flagged high-risk and actually actionable).
    high_risk_df = df[df["risk_tier"].isin(["Critical", "High"])]
    typology_summary = high_risk_df["primary_typology"].value_counts()

    recs: list[dict[str, str]] = []

    # Critical tier
    recs.append({
        "tier":     "Critical",
        "action":   "Immediate account review and manual verification",
        "rationale": (
            f"These {exposure.get('high_risk_customers', 0):,} customers account for "
            f"${exposure.get('total_exposure_high_risk', 0):,.0f} in estimated exposure. "
            "Automated flags should trigger human review within 24 hours."
        ),
        "priority": "P0",
    })

    # Return abuse
    if typology_summary.get("Return Abuse", 0) > 0:
        recs.append({
            "tier":     "High",
            "action":   "Implement purchase-to-return verification and return caps",
            "rationale": (
                f"{typology_summary.get('Return Abuse', 0):,} customers identified as "
                "return abusers. Introduce photo verification for high-value returns and "
                "cap returns to 2x per customer per quarter."
            ),
            "priority": "P1",
        })

    # Discount exploitation
    if typology_summary.get("Discount Exploitation", 0) > 0:
        recs.append({
            "tier":     "High",
            "action":   "Implement personalised discount eligibility rules",
            "rationale": (
                f"{typology_summary.get('Discount Exploitation', 0):,} customers "
                "systematically purchase only at maximum discount levels. Shift to "
                "loyalty-based discount allocation rather than blanket promotions."
            ),
            "priority": "P1",
        })

    # Velocity
    if typology_summary.get("Velocity Anomaly", 0) > 0:
        recs.append({
            "tier":     "High",
            "action":   "Add velocity checks to checkout with CAPTCHA escalation",
            "rationale": (
                f"{typology_summary.get('Velocity Anomaly', 0):,} customers show "
                "burst ordering patterns inconsistent with normal retail behaviour. "
                "Restrict to 3 orders per 7-day window pending manual review."
            ),
            "priority": "P1",
        })

    # Combined attack
    if typology_summary.get("Combined Attack", 0) > 0:
        recs.append({
            "tier":     "Critical",
            "action":   "Suspend account and initiate fraud investigation",
            "rationale": (
                f"{typology_summary.get('Combined Attack', 0):,} customers exhibit "
                "3+ concurrent fraud typologies. These represent the highest risk "
                "and should be escalated to the fraud operations team immediately."
            ),
            "priority": "P0",
        })

    # Medium tier
    recs.append({
        "tier":     "Medium",
        "action":   "Enhanced monitoring with friction-based review triggers",
        "rationale": (
            "Medium-risk customers show elevated but not conclusive signals. "
            "Implement silent monitoring: flag the next high-value or high-discount "
            "order for secondary review without blocking the transaction."
        ),
        "priority": "P2",
    })

    return recs


# ---------------------------------------------------------------------------
# Cross-notebook segment analysis
# ---------------------------------------------------------------------------

def build_segment_fraud_crosstab(
    df_fraud: pd.DataFrame,
    segments: pd.DataFrame,
    churn_preds: pd.DataFrame,
    config: dict[str, Any],
    run_id: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build cross-tabulations linking NB03 segments, NB04 churn risk,
    and NB05 fraud risk tiers.

    Joins:
      - df_fraud (NB05) — composite_fraud_score, risk_tier, primary_typology
      - segments (NB03) — segment_name, churn_risk, loyalty_score
      - churn_preds (NB04) — churn_probability, risk_level (churn risk tier)

    Parameters
    ----------
    df_fraud:
        Customer fraud profile with risk_tier and primary_typology.
    segments:
        NB03 customer_segments DataFrame.
    churn_preds:
        NB04 churn predictions DataFrame.
    config:
        Project configuration.
    run_id:
        Pipeline run identifier.
    verbose:
        Print results if True.

    Returns
    -------
    dict with cross-tab DataFrames and summary statistics.
    """
    logger.info("[%s] Building segment x fraud risk cross-tabulations", run_id)

    # Merge fraud with NB03 segments
    merged = df_fraud[["customer_id", "composite_fraud_score", "risk_tier",
                        "primary_typology"]].merge(
        segments[[c for c in ["customer_id", "segment_name", "churn_risk"] if c in segments.columns]],
        on="customer_id", how="left",
    )

    # Merge with NB04 churn predictions (optional — not all customers may be present)
    if len(churn_preds) > 0 and "churn_probability" in churn_preds.columns:
        churn_cols = ["customer_id", "churn_probability"]
        if "risk_level" in churn_preds.columns:
            churn_cols.append("risk_level")
        merged = merged.merge(churn_preds[churn_cols], on="customer_id", how="left")
        merged.rename(columns={"risk_level": "churn_risk_level"}, inplace=True)
    else:
        merged["churn_probability"] = np.nan
        merged["churn_risk_level"]  = np.nan

    n_unmatched = merged["segment_name"].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            "[%s] %d fraud customers not matched to NB03 segments",
            run_id, n_unmatched,
        )
    merged["segment_name"] = merged["segment_name"].fillna("Unknown")

    # --- Cross-tab 1: Segment x Fraud Risk Tier ---
    seg_fraud_ct = pd.crosstab(
        merged["segment_name"],
        merged["risk_tier"],
        margins=True,
    )

    # Reorder tiers
    tier_order = [t for t in ["Critical", "High", "Medium", "Low", "All"]
                  if t in seg_fraud_ct.columns]
    seg_fraud_ct = seg_fraud_ct[tier_order]

    # Add high-risk percentage column
    if "Critical" in seg_fraud_ct.columns and "High" in seg_fraud_ct.columns:
        seg_fraud_ct["High+Critical"] = (
            seg_fraud_ct.get("Critical", 0) + seg_fraud_ct.get("High", 0)
        )
        if "All" in seg_fraud_ct.columns:
            seg_fraud_ct["High+Critical %"] = (
                seg_fraud_ct["High+Critical"] / seg_fraud_ct["All"] * 100
            ).round(1)

    # --- Cross-tab 2: Mean fraud score by segment ---
    seg_score = (
        merged.groupby("segment_name")
        .agg(
            customers=("customer_id", "count"),
            avg_fraud_score=("composite_fraud_score", "mean"),
            high_critical_pct=(
                "risk_tier",
                lambda x: (x.isin(["Critical", "High"])).mean() * 100,
            ),
            avg_churn_prob=("churn_probability", "mean"),
        )
        .round(4)
        .sort_values("avg_fraud_score", ascending=False)
    )

    # --- Cross-tab 3: Tri-variate — NB03 segment x NB04 churn tier x NB05 fraud tier ---
    # Focus on High+Critical fraud and High churn risk to find "double risk" customers
    if "churn_risk_level" in merged.columns and merged["churn_risk_level"].notna().any():
        double_risk = merged[
            merged["risk_tier"].isin(["Critical", "High"])
            & merged["churn_risk_level"].isin(["High"])
        ].groupby("segment_name").size().rename("double_risk_count")

        seg_score = seg_score.join(double_risk, how="left")
        seg_score["double_risk_count"] = seg_score["double_risk_count"].fillna(0).astype(int)
    else:
        seg_score["double_risk_count"] = 0

    results = {
        "merged":       merged,
        "seg_fraud_ct": seg_fraud_ct,
        "seg_score":    seg_score,
        "n_unmatched":  n_unmatched,
    }

    if verbose:
        _print_crosstab_results(seg_fraud_ct, seg_score, merged)

    logger.info(
        "[%s] Cross-tab complete. %d customers merged across NB03/NB04/NB05",
        run_id, len(merged),
    )

    return results


def plot_segment_fraud_heatmap(
    crosstab_results: dict[str, Any],
    config: dict[str, Any],
    save: bool = True,
    show: bool = True,
    subfolder: str = "notebook5_fig",
    run_id: str = "",
) -> list[Any]:
    """
    Plot segment x fraud risk tier heatmap and segment fraud score comparison.

    Returns list of saved figure paths.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    saved: list[Path] = []
    figures_dir = _get_figures_dir(config, subfolder)

    seg_fraud_ct = crosstab_results["seg_fraud_ct"]
    seg_score    = crosstab_results["seg_score"]

    # Build heatmap data — counts without margins, normalised by row
    tier_cols = [t for t in ["Critical", "High", "Medium", "Low"]
                 if t in seg_fraud_ct.columns]
    seg_order = ["High-Value at Risk", "Loyal Customers",
                 "Needs Engagement", "Lost Customers", "Unknown"]
    seg_order = [s for s in seg_order if s in seg_fraud_ct.index]

    heat_raw  = seg_fraud_ct.loc[seg_order, tier_cols] if seg_order else seg_fraud_ct[tier_cols]
    heat_pct  = heat_raw.div(heat_raw.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: heatmap (% within segment)
    ax = axes[0]
    sns.heatmap(
        heat_pct,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "% of Segment Customers", "shrink": 0.8},
        annot_kws={"size": 10},
    )
    ax.set_title(
        "Fraud Risk Tier Distribution\nby NB03 Customer Segment (%)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Fraud Risk Tier (NB05)", fontsize=10)
    ax.set_ylabel("Customer Segment (NB03)", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Right: mean fraud score + double-risk bar by segment
    ax = axes[1]
    plot_seg  = seg_score.drop("All", errors="ignore").sort_values("avg_fraud_score", ascending=True)
    colors_map = {
        "High-Value at Risk": config["visualization"]["colors"]["danger"],
        "Lost Customers":     config["visualization"]["colors"]["neutral"],
        "Loyal Customers":    config["visualization"]["colors"]["success"],
        "Needs Engagement":   config["visualization"]["colors"]["secondary"],
    }
    bar_colors = [
        colors_map.get(seg, config["visualization"]["colors"]["primary"])
        for seg in plot_seg.index
    ]

    bars = ax.barh(
        plot_seg.index,
        plot_seg["avg_fraud_score"],
        color=bar_colors,
        alpha=0.85,
        edgecolor="white",
        height=0.55,
    )
    for bar, (seg, row) in zip(bars, plot_seg.iterrows()):
        ax.text(
            bar.get_width() + plot_seg["avg_fraud_score"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row['avg_fraud_score']:.4f}  ({row['high_critical_pct']:.1f}% H+C)",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Mean Composite Fraud Score", fontsize=10)
    ax.set_title(
        "Mean Fraud Score by Segment\n(H+C = % in High or Critical tier)",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, plot_seg["avg_fraud_score"].max() * 1.35)

    plt.tight_layout()

    path = figures_dir / f"segment_fraud_crosstab_{run_id}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)

    if show:
        plt.show()
    plt.close(fig)

    return saved


def _get_figures_dir(config: dict[str, Any], subfolder: str) -> Any:
    """Resolve and create the figures output directory."""
    from pathlib import Path
    from n5a_utils import get_project_root

    root = get_project_root()
    base = config.get("paths", {}).get("figures_dir", "outputs/figures")
    figures_dir = root / base / subfolder
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _print_crosstab_results(
    seg_fraud_ct: "pd.DataFrame",
    seg_score: "pd.DataFrame",
    merged: "pd.DataFrame",
) -> None:
    """Print cross-tab analysis results."""
    print(f"\n{'='*80}")
    print("SEGMENT x FRAUD RISK CROSS-ANALYSIS".center(80))
    print("=" * 80)

    print("\nCross-Tab: NB03 Segment x NB05 Fraud Risk Tier (customer counts)")
    print("-" * 80)
    with pd.option_context("display.max_columns", None, "display.width", 100):
        print(seg_fraud_ct.to_string())

    print(f"\n\nSegment Fraud Risk Profile:")
    print("-" * 80)
    print(
        f"{'Segment':<25} {'Customers':>10} {'Avg Score':>10} "
        f"{'H+C %':>8} {'Avg Churn P':>12} {'Double Risk':>12}"
    )
    print("-" * 80)
    for seg_name, row in seg_score.drop("All", errors="ignore").iterrows():
        churn_p = f"{row['avg_churn_prob']:.4f}" if pd.notna(row.get("avg_churn_prob")) else "N/A"
        dr = int(row.get("double_risk_count", 0))
        print(
            f"{str(seg_name):<25} {int(row['customers']):>10,} "
            f"{row['avg_fraud_score']:>10.4f} {row['high_critical_pct']:>7.1f}% "
            f"{churn_p:>12} {dr:>12,}"
        )

    # Key findings
    print(f"\nKey Findings:")
    print("-" * 40)

    # Highest fraud risk segment
    top_seg = seg_score.drop("All", errors="ignore")["avg_fraud_score"].idxmax()
    print(
        f"  Highest avg fraud score: {top_seg} "
        f"({seg_score.loc[top_seg, 'avg_fraud_score']:.4f})"
    )

    # Segment with most H+C customers
    if "High+Critical" in seg_fraud_ct.columns:
        top_hc_seg = seg_fraud_ct.drop("All", errors="ignore")["High+Critical"].idxmax()
        hc_count = seg_fraud_ct.loc[top_hc_seg, "High+Critical"]
        print(
            f"  Most High+Critical customers: {top_hc_seg} ({hc_count:,})"
        )

    # Double risk customers total
    dr_total = seg_score.drop("All", errors="ignore").get("double_risk_count", pd.Series([0])).sum()
    if dr_total > 0:
        print(
            f"  Customers in both High churn AND High/Critical fraud: {int(dr_total):,}"
        )
        # Which segment they come from
        dr_by_seg = (
            seg_score.drop("All", errors="ignore")["double_risk_count"]
            .sort_values(ascending=False)
        )
        for seg_n, dr_n in dr_by_seg.items():
            if dr_n > 0:
                print(f"    {seg_n:<25}: {int(dr_n):,}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_exposure_summary(exposure: dict[str, Any]) -> None:
    """Print deduplicated financial exposure report."""
    print(f"\n{'='*80}")
    print("FINANCIAL EXPOSURE ANALYSIS".center(80))
    print("=" * 80)
    print(
        f"\nHigh-Risk Customers:        "
        f"{exposure['high_risk_customers']:,} ({exposure['high_risk_pct']:.1f}%)"
    )
    print(
        f"Revenue (High-Risk):         "
        f"${exposure['revenue_high_risk']:>12,.2f} "
        f"({exposure['revenue_high_risk_pct']:.1f}% of total)"
    )

    print()
    print(
        f"{'Exposure Component':<35} {'High-Risk':>15} {'All Customers':>15}  "
        f"{'Bucket Priority'}"
    )
    print("-" * 80)
    print(
        f"{'Return Exposure':<35} "
        f"${exposure['return_exposure_high_risk']:>14,.2f} "
        f"${exposure['return_exposure_all']:>14,.2f}  "
        f"Priority 1 — returned orders (total_amount)"
    )
    print(
        f"{'Discount Exposure':<35} "
        f"${exposure['discount_exposure_high_risk']:>14,.2f} "
        f"${exposure['discount_exposure_all']:>14,.2f}  "
        f"Priority 2 — kept orders with discount (price×qty×disc)"
    )
    print(
        f"{'Negative Margin Losses':<35} "
        f"${exposure['neg_margin_exposure_high']:>14,.2f} "
        f"${exposure['neg_margin_exposure_all']:>14,.2f}  "
        f"Priority 3 — kept, undiscounted, negative-margin orders"
    )
    print("-" * 80)
    print(
        f"{'Total Exposure (deduplicated)':<35} "
        f"${exposure['total_exposure_high_risk']:>14,.2f}"
    )

    # Overlap transparency block
    total_overlap = exposure.get("total_overlap_orders", 0)
    if total_overlap > 0:
        print()
        print(f"  Deduplication report — orders resolved to one bucket only:")
        print(f"    Returned + discounted orders :  {exposure.get('overlap_return_discount', 0):,}"
              f"  → assigned to Return bucket")
        print(f"    Returned + neg-margin orders  :  {exposure.get('overlap_return_negmargin', 0):,}"
              f"  → assigned to Return bucket")
        print(f"    Discounted + neg-margin orders:  {exposure.get('overlap_discount_negmargin', 0):,}"
              f"  → assigned to Discount bucket")
        print(f"    Total overlap orders resolved :  {total_overlap:,}")
    else:
        print(f"\n  No overlapping orders detected — components are fully disjoint.")

    print("=" * 80)


def _print_typology_summary(
    result: pd.DataFrame,
    typology_cols: list[str],
    label_map: dict[str, str],
) -> None:
    """Print typology classification summary."""
    print(f"\n{'='*80}")
    print("FRAUD TYPOLOGY CLASSIFICATION".center(80))
    print("=" * 80)

    total = len(result)
    high_risk = result[result["risk_tier"].isin(["Critical", "High"])]

    print(f"\nCustomers analysed:        {total:,}")
    print(f"High/Critical risk:        {len(high_risk):,}")
    print()
    print(f"{'Typology':<30} {'All Customers':>15} {'High-Risk Only':>15}")
    print("-" * 63)

    for col, label in label_map.items():
        if col in result.columns:
            n_all    = result[col].sum()
            n_high   = high_risk[col].sum() if col in high_risk.columns else 0
            print(f"{label:<30} {n_all:>14,} {n_high:>14,}")

    print()
    print("Primary Typology Distribution:")
    print("-" * 40)
    for typology, count in result["primary_typology"].value_counts().items():
        pct = count / total * 100
        print(f"  {typology:<28} {count:>6,} ({pct:.1f}%)")

    print("=" * 80)