# src/n1b_missing_data_analysis.py - ERROR-HARDENED VERSION

"""
n1b_missing_data_analysis.py - Comprehensive Missing Data Analysis

Error Handling Strategy:
- analyze_missing_patterns         : validates df type; each of the 6 analysis sections
                                     runs in its own try/except; returns partial results
- _generate_imputation_recommendations : validates inputs; each column recommendation
                                         is computed independently; skips bad columns
- _assess_business_impact          : validates inputs; guards division; returns safe defaults
- _plot_missing_data_analysis      : each subplot block is independently guarded;
                                     closes figure and returns None on fatal error
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Any
import logging
import warnings
warnings.filterwarnings("ignore")
from matplotlib.figure import Figure


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_dataframe(df: Any, caller: str, min_rows: int = 1) -> bool:
    if not isinstance(df, pd.DataFrame):
        logger.error(f"[{caller}] expected pd.DataFrame, got {type(df).__name__}")
        return False
    if len(df) < min_rows:
        logger.warning(f"[{caller}] DataFrame has {len(df)} rows (need >= {min_rows})")
        return False
    return True


def _safe_div(num: float, denom: float, fallback: float = 0.0) -> float:
    try:
        return num / denom if denom != 0 else fallback
    except Exception:
        return fallback


# ===========================================================================
# 1. analyze_missing_patterns
# ===========================================================================

def analyze_missing_patterns(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
    plot: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Comprehensive missing data analysis with mechanism detection.

    Error handling:
    - Validates df is a non-empty DataFrame
    - Validates config is a dict (or None); falls back to {} on bad type
    - Each of the 6 sections (summary, patterns, mechanisms, recommendations,
      visualisation, business impact) runs in its own try/except so one
      failure does not suppress later sections
    - Returns whatever was successfully computed (partial dict on errors)
    - verbose print blocks are independently wrapped
    """
    caller = "analyze_missing_patterns"

    if not _require_dataframe(df, caller):
        return {}

    if config is None:
        config = {}
    elif not isinstance(config, dict):
        logger.warning(
            f"[{caller}] config must be a dict or None, got {type(config).__name__} – using {{}}"
        )
        config = {}

    results: Dict[str, Any] = {
        "summary": {},
        "patterns": {},
        "mechanisms": {},
        "recommendations": [],
    }

    # ---- 1. BASIC SUMMARY --------------------------------------------------
    missing_cols = pd.Series(dtype=int)
    try:
        total_cells   = df.size
        missing_cells = int(df.isna().sum().sum())
        missing_pct   = _safe_div(missing_cells, total_cells, 0.0) * 100

        missing_by_col = df.isna().sum()
        missing_cols   = missing_by_col[missing_by_col > 0].sort_values(ascending=False)

        missing_by_row        = df.isna().sum(axis=1)
        rows_with_missing     = int((missing_by_row > 0).sum())
        rows_with_missing_pct = _safe_div(rows_with_missing, len(df), 0.0) * 100

        results["summary"] = {
            "total_cells":           total_cells,
            "missing_cells":         missing_cells,
            "missing_percentage":    round(missing_pct, 2),
            "columns_with_missing":  len(missing_cols),
            "rows_with_missing":     rows_with_missing,
            "rows_with_missing_pct": round(rows_with_missing_pct, 2),
            "complete_rows":         len(df) - rows_with_missing,
            "complete_rows_pct":     round(100 - rows_with_missing_pct, 2),
        }

        if verbose:
            try:
                print("\n" + "=" * 80)
                print("MISSING DATA ANALYSIS".center(80))
                print("=" * 80)
                print(f"\nOverall Summary:")
                print(f"  Total cells      : {total_cells:,}")
                print(f"  Missing cells    : {missing_cells:,} ({missing_pct:.2f}%)")
                print(f"  Complete rows    : {results['summary']['complete_rows']:,} ({results['summary']['complete_rows_pct']:.1f}%)")
                print(f"  Rows w/ missing  : {rows_with_missing:,} ({rows_with_missing_pct:.1f}%)")
                if len(missing_cols) > 0:
                    print(f"\n  Columns with Missing Data ({len(missing_cols)}):")
                    for col, count in missing_cols.items():
                        pct = _safe_div(count, len(df), 0.0) * 100
                        print(f"    {col}: {count:,} ({pct:.1f}%)")
                else:
                    print("\n  No missing data detected!")
            except Exception as e:
                logger.warning(f"[{caller}] summary verbose output failed: {e}")

    except Exception as e:
        logger.error(f"[{caller}] summary section failed: {e}")

    # ---- 2. PATTERNS -------------------------------------------------------
    missing_matrix = pd.DataFrame()
    try:
        if len(missing_cols) > 0:
            missing_matrix = df[missing_cols.index].isna()
            pattern_counts = missing_matrix.groupby(list(missing_matrix.columns)).size()
            pattern_counts = pattern_counts.sort_values(ascending=False)

            results["patterns"]["pattern_counts"]  = pattern_counts.head(10).to_dict()
            results["patterns"]["unique_patterns"] = len(pattern_counts)

            if verbose:
                try:
                    print(f"\n  Missing Data Patterns: {len(pattern_counts)} unique")
                    if len(pattern_counts) > 1:
                        print("  Top 5 patterns:")
                        for i, (pattern, count) in enumerate(pattern_counts.head(5).items(), 1):
                            pct = _safe_div(count, len(df), 0.0) * 100
                            pattern_tuple = (pattern,) if not isinstance(pattern, tuple) else pattern
                            pattern_str = ", ".join([
                                f"{col}={'X' if val else 'OK'}"
                                for col, val in zip(missing_cols.index, pattern_tuple)
                            ])
                            print(f"    {i}. [{pattern_str}]: {count:,} rows ({pct:.1f}%)")
                except Exception as e:
                    logger.warning(f"[{caller}] pattern verbose output failed: {e}")

    except Exception as e:
        logger.error(f"[{caller}] pattern section failed: {e}")

    # ---- 3. MECHANISM DETECTION --------------------------------------------
    mechanisms: Dict[str, Any] = {}
    try:
        if len(missing_cols) > 0:
            for col in missing_cols.index:
                try:
                    mech = _detect_missingness_mechanism(df, col, verbose=False)
                    mechanisms[col] = mech
                    results["mechanisms"][col] = mech
                except Exception as e:
                    logger.warning(f"[{caller}] mechanism detection failed for '{col}': {e}")
                    fallback_mech = {
                        "mechanism": "UNKNOWN", "confidence": "LOW",
                        "evidence": f"Detection error: {e}",
                    }
                    mechanisms[col] = fallback_mech
                    results["mechanisms"][col] = fallback_mech

            if verbose:
                try:
                    print(f"\n  Missingness Mechanism Analysis:")
                    for col, mech in mechanisms.items():
                        print(f"\n    {col}:")
                        print(f"      Mechanism  : {mech.get('mechanism', 'UNKNOWN')}")
                        print(f"      Confidence : {mech.get('confidence', 'N/A')}")
                        print(f"      Evidence   : {mech.get('evidence', 'N/A')}")
                except Exception as e:
                    logger.warning(f"[{caller}] mechanism verbose output failed: {e}")

    except Exception as e:
        logger.error(f"[{caller}] mechanism section failed: {e}")

    # ---- 4. RECOMMENDATIONS ------------------------------------------------
    try:
        recommendations = _generate_imputation_recommendations(df, missing_cols, mechanisms)
        results["recommendations"] = recommendations

        if verbose and recommendations:
            try:
                print(f"\n  Imputation Recommendations:")
                for rec in recommendations:
                    print(f"\n    {rec.get('column', '?')}:")
                    print(f"      Strategy      : {rec.get('strategy', 'N/A')}")
                    print(f"      Rationale     : {rec.get('rationale', 'N/A')}")
                    if "implementation" in rec:
                        print(f"      Implementation: {rec['implementation']}")
            except Exception as e:
                logger.warning(f"[{caller}] recommendations verbose output failed: {e}")

    except Exception as e:
        logger.error(f"[{caller}] recommendations section failed: {e}")

    # ---- 5. VISUALIZATIONS -------------------------------------------------
    try:
        if plot and len(missing_cols) > 0 and not missing_matrix.empty:
            _plot_missing_data_analysis(df, missing_cols, missing_matrix)
    except Exception as e:
        logger.warning(f"[{caller}] visualization section failed: {e}")

    # ---- 6. BUSINESS IMPACT ------------------------------------------------
    try:
        if len(missing_cols) > 0:
            business_impact = _assess_business_impact(df, missing_cols, config)
            results["business_impact"] = business_impact

            if verbose:
                try:
                    print(f"\n  Business Impact Assessment:")
                    print(f"    Data loss if dropped      : {business_impact.get('potential_data_loss_pct', 'N/A'):.1f}%")
                    print(f"    Critical columns affected : {business_impact.get('critical_columns_affected', 'N/A')}")
                    print(f"    Recommended action        : {business_impact.get('recommended_action', 'N/A')}")
                except Exception as e:
                    logger.warning(f"[{caller}] business impact verbose output failed: {e}")

    except Exception as e:
        logger.error(f"[{caller}] business impact section failed: {e}")

    return results


# ===========================================================================
# _detect_missingness_mechanism  (internal helper – hardened)
# ===========================================================================

def _detect_missingness_mechanism(
    df: pd.DataFrame,
    col: str,
    alpha: float = 0.05,
    verbose: bool = False,
) -> Dict:
    """
    Detect missingness mechanism using statistical tests.

    Error handling:
    - Validates col exists in df and alpha is in (0, 1)
    - Each Mann-Whitney test is individually guarded (replaces bare except)
    - MNAR heuristic is independently guarded
    - Returns a safe UNKNOWN result on any unrecoverable error
    """
    result: Dict[str, Any] = {
        "mechanism": "UNKNOWN",
        "confidence": "LOW",
        "evidence": "",
        "p_values": {},
    }

    if not isinstance(col, str) or col not in df.columns:
        result["evidence"] = f"Column '{col}' not found in DataFrame"
        return result

    if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
        logger.warning(f"_detect_missingness_mechanism: invalid alpha={alpha!r}, defaulting to 0.05")
        alpha = 0.05

    try:
        is_missing = df[col].isna()

        if is_missing.sum() == 0:
            result["mechanism"] = "NO_MISSING"
            return result

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if col in numeric_cols:
            numeric_cols.remove(col)

        significant_relationships = 0
        total_tests = 0

        for other_col in numeric_cols[:10]:
            try:
                missing_group = df.loc[is_missing,  other_col].dropna()
                present_group = df.loc[~is_missing, other_col].dropna()

                if len(missing_group) > 5 and len(present_group) > 5:
                    _, p_value = stats.mannwhitneyu(
                        missing_group, present_group, alternative="two-sided"
                    )
                    result["p_values"][other_col] = float(p_value)
                    total_tests += 1
                    if p_value < alpha:
                        significant_relationships += 1
            except Exception as e:
                logger.warning(
                    f"_detect_missingness_mechanism: test failed for '{col}' vs '{other_col}': {e}"
                )

        if total_tests == 0:
            result.update({"mechanism": "MCAR", "confidence": "LOW",
                           "evidence": "Insufficient data for statistical testing"})
        elif significant_relationships == 0:
            result.update({"mechanism": "MCAR", "confidence": "MEDIUM",
                           "evidence": f"No significant relationships (tested {total_tests} vars)"})
        else:
            result.update({"mechanism": "MAR", "confidence": "MEDIUM",
                           "evidence": f"Missingness correlates with {significant_relationships}/{total_tests} vars"})

        # MNAR heuristic
        try:
            if col in df.select_dtypes(include=[np.number]).columns:
                present_values = df.loc[~is_missing, col].dropna()
                if len(present_values) > 10:
                    q1   = float(present_values.quantile(0.25))
                    q3   = float(present_values.quantile(0.75))
                    pmin = float(present_values.min())
                    pmax = float(present_values.max())
                    if pmin != 0 and pmax != 0:
                        if q1 > pmin * 1.5 or q3 < pmax * 0.5:
                            result["mechanism"]  = "POSSIBLY_MNAR"
                            result["confidence"] = "LOW"
                            result["evidence"]  += " | Values may be systematically missing"
        except Exception as e:
            logger.warning(f"_detect_missingness_mechanism: MNAR heuristic failed for '{col}': {e}")

    except Exception as e:
        logger.error(f"_detect_missingness_mechanism: unexpected error for '{col}': {e}")
        result["evidence"] = f"Error during detection: {e}"

    return result


# ===========================================================================
# 2. _generate_imputation_recommendations
# ===========================================================================

def _generate_imputation_recommendations(
    df: pd.DataFrame,
    missing_cols: pd.Series,
    mechanisms: Dict,
) -> List[Dict]:
    """
    Generate imputation strategy recommendations.

    Error handling:
    - Validates df, missing_cols, and mechanisms types
    - Each column recommendation is computed in its own try/except
    - Skips columns with a warning if any step fails
    """
    caller = "_generate_imputation_recommendations"

    if not isinstance(df, pd.DataFrame):
        logger.error(f"[{caller}] df must be pd.DataFrame, got {type(df).__name__}")
        return []
    if not isinstance(missing_cols, pd.Series):
        logger.error(f"[{caller}] missing_cols must be pd.Series, got {type(missing_cols).__name__}")
        return []
    if not isinstance(mechanisms, dict):
        logger.warning(f"[{caller}] mechanisms must be dict – using {{}}")
        mechanisms = {}

    recommendations: List[Dict] = []

    for col in missing_cols.index:
        try:
            if col not in df.columns:
                logger.warning(f"[{caller}] column '{col}' not in df – skipped")
                continue

            missing_pct = _safe_div(int(missing_cols[col]), len(df), 0.0) * 100
            col_mech    = mechanisms.get(col)
            mechanism   = col_mech.get("mechanism", "UNKNOWN") if isinstance(col_mech, dict) else "UNKNOWN"
            dtype       = df[col].dtype

            rec: Dict[str, Any] = {"column": col, "missing_pct": missing_pct}

            if missing_pct > 50:
                rec["strategy"]  = "DROP_COLUMN"
                rec["rationale"] = f"> 50% missing ({missing_pct:.1f}%) – insufficient data for imputation"

            elif missing_pct < 5:
                if mechanism == "MCAR":
                    if pd.api.types.is_numeric_dtype(dtype):
                        rec.update({
                            "strategy":       "MEDIAN_IMPUTATION",
                            "rationale":      "< 5% missing, MCAR – simple median imputation sufficient",
                            "implementation": f"df['{col}'].fillna(df['{col}'].median(), inplace=True)",
                        })
                    else:
                        rec.update({
                            "strategy":       "MODE_IMPUTATION",
                            "rationale":      "< 5% missing, MCAR – mode imputation for categorical",
                            "implementation": f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)",
                        })
                else:
                    rec.update({
                        "strategy":  "PREDICTIVE_IMPUTATION",
                        "rationale": "< 5% missing but MAR mechanism – consider simple regression/classification",
                    })

            else:  # 5–50%
                if mechanism == "MCAR":
                    rec.update({
                        "strategy":  "MULTIPLE_IMPUTATION",
                        "rationale": f"{missing_pct:.1f}% missing, MCAR – multiple imputation recommended",
                    })
                elif mechanism == "MAR":
                    rec.update({
                        "strategy":       "PREDICTIVE_IMPUTATION",
                        "rationale":      f"{missing_pct:.1f}% missing, MAR – use related variables",
                        "implementation": "Consider KNN, regression, or chained equations",
                    })
                else:
                    rec.update({
                        "strategy":       "INDICATOR_VARIABLE",
                        "rationale":      f"{missing_pct:.1f}% missing, mechanism unclear – create indicator",
                        "implementation": f"df['{col}_missing'] = df['{col}'].isna().astype(int)",
                    })

            recommendations.append(rec)

        except Exception as e:
            logger.warning(f"[{caller}] recommendation failed for column '{col}': {e}")

    return recommendations


# ===========================================================================
# 3. _assess_business_impact
# ===========================================================================

def _assess_business_impact(
    df: pd.DataFrame,
    missing_cols: pd.Series,
    config: Dict,
) -> Dict:
    """
    Assess business impact of missing data.

    Error handling:
    - Validates df, missing_cols, and config types
    - Guards division for potential_data_loss_pct
    - Returns a safe default dict on any unrecoverable error
    """
    caller = "_assess_business_impact"

    _SAFE_DEFAULT: Dict[str, Any] = {
        "critical_columns_affected": 0,
        "critical_columns":          [],
        "potential_data_loss_pct":   0.0,
        "recommended_action":        "UNKNOWN – assessment failed",
    }

    if not isinstance(df, pd.DataFrame):
        logger.error(f"[{caller}] df must be pd.DataFrame, got {type(df).__name__}")
        return _SAFE_DEFAULT
    if not isinstance(missing_cols, pd.Series):
        logger.error(f"[{caller}] missing_cols must be pd.Series, got {type(missing_cols).__name__}")
        return _SAFE_DEFAULT
    if not isinstance(config, dict):
        logger.warning(f"[{caller}] config must be dict – using {{}}")
        config = {}

    try:
        critical_cols = config.get("critical_columns", ["customer_id", "order_id", "total_amount"])
        if not isinstance(critical_cols, list):
            logger.warning(f"[{caller}] config 'critical_columns' must be a list – using defaults")
            critical_cols = ["customer_id", "order_id", "total_amount"]

        critical_affected = [c for c in missing_cols.index if c in critical_cols and c in df.columns]

        try:
            cols_in_df = [c for c in missing_cols.index if c in df.columns]
            rows_with_missing = int(df[cols_in_df].isna().any(axis=1).sum()) if cols_in_df else 0
        except Exception as e:
            logger.warning(f"[{caller}] row count calculation failed: {e}")
            rows_with_missing = 0

        potential_loss_pct = round(_safe_div(rows_with_missing, len(df), 0.0) * 100, 2)

        if len(critical_affected) > 0:
            recommended_action = "CRITICAL – investigate data collection issues"
        elif potential_loss_pct > 20:
            recommended_action = "HIGH PRIORITY – implement imputation strategies"
        elif potential_loss_pct > 5:
            recommended_action = "MEDIUM PRIORITY – consider imputation or analysis with missing data"
        else:
            recommended_action = "LOW PRIORITY – simple handling sufficient"

        return {
            "critical_columns_affected": len(critical_affected),
            "critical_columns":          critical_affected,
            "potential_data_loss_pct":   potential_loss_pct,
            "recommended_action":        recommended_action,
        }

    except Exception as e:
        logger.error(f"[{caller}] unexpected error: {e}")
        return _SAFE_DEFAULT


# ===========================================================================
# 4. _plot_missing_data_analysis
# ===========================================================================

def _plot_missing_data_analysis(
    df: pd.DataFrame,
    missing_cols: pd.Series,
    missing_matrix: pd.DataFrame,
) -> Optional[plt.Figure]:
    """
    Generate comprehensive missing data visualizations.

    Error handling:
    - Validates all three inputs are the expected types
    - Each of the 4 subplot blocks runs in its own try/except
    - A failure in one panel replaces it with an error message text rather
      than leaving a broken figure
    - Figure is always closed and None is returned on fatal setup error
    """
    caller = "_plot_missing_data_analysis"

    if df is None or not isinstance(df, pd.DataFrame):
        logger.error(f"[{caller}] df is null or not pd.DataFrame")
        return None
    if missing_cols is None or not isinstance(missing_cols, pd.Series):
        logger.error(f"[{caller}] missing_cols is null or not pd.Series")
        return None
    if missing_matrix is None or not isinstance(missing_matrix, pd.DataFrame) or missing_matrix.empty:
        logger.warning(f"[{caller}] missing_matrix is empty or wrong type – nothing to plot")
        return None

    fig = None
    try:
        fig = plt.figure(figsize=(16, 10))
        gs  = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # ---- Panel 1: bar chart -------------------------------------------
        try:
            ax1 = fig.add_subplot(gs[0, :])
            missing_pct = (missing_cols / len(df) * 100).sort_values(ascending=False)
            colors = [
                "#e74c3c" if p > 20 else "#f39c12" if p > 5 else "#3498db"
                for p in missing_pct
            ]
            missing_pct.plot(kind="barh", ax=ax1, color=colors)
            ax1.set_xlabel("Missing Data (%)", fontsize=11)
            ax1.set_title("Missing Data by Column", fontsize=13, fontweight="bold")
            ax1.axvline(5,  color="orange", linestyle="--", alpha=0.5, label="5% threshold")
            ax1.axvline(20, color="red",    linestyle="--", alpha=0.5, label="20% threshold")
            ax1.legend()
            ax1.grid(axis="x", alpha=0.3)
        except Exception as e:
            logger.warning(f"[{caller}] bar chart failed: {e}")
            try:
                ax1 = fig.add_subplot(gs[0, :])
                ax1.text(0.5, 0.5, f"Bar chart unavailable:\n{e}", ha="center", va="center")
                ax1.axis("off")
            except Exception:
                pass

        # ---- Panel 2: missingness heatmap ---------------------------------
        try:
            if len(missing_cols) <= 20:
                ax2 = fig.add_subplot(gs[1, :])
                sample_size = min(1000, len(missing_matrix))
                sample_idx  = np.random.choice(len(missing_matrix), size=sample_size, replace=False)
                sample_mat  = missing_matrix.iloc[sample_idx]
                sns.heatmap(
                    sample_mat.T, cmap="RdYlGn_r",
                    cbar_kws={"label": "Missing"},
                    yticklabels=True, xticklabels=False, ax=ax2,
                )
                ax2.set_title(
                    f"Missing Data Pattern (Sample: {sample_size} rows)",
                    fontsize=13, fontweight="bold",
                )
                ax2.set_xlabel("Row Index")
        except Exception as e:
            logger.warning(f"[{caller}] heatmap failed: {e}")
            try:
                ax2 = fig.add_subplot(gs[1, :])
                ax2.text(0.5, 0.5, f"Heatmap unavailable:\n{e}", ha="center", va="center")
                ax2.axis("off")
            except Exception:
                pass

        # ---- Panel 3: correlation heatmap ---------------------------------
        try:
            ax3 = fig.add_subplot(gs[2, 0])
            if len(missing_cols) > 1:
                missing_corr = missing_matrix.corr()
                sns.heatmap(
                    missing_corr, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax3, square=True,
                    cbar_kws={"label": "Correlation"},
                )
                ax3.set_title("Missing Data Correlations", fontsize=12, fontweight="bold")
            else:
                ax3.text(0.5, 0.5, "Not enough variables\nwith missing data",
                         ha="center", va="center", fontsize=12)
                ax3.axis("off")
        except Exception as e:
            logger.warning(f"[{caller}] correlation heatmap failed: {e}")
            try:
                ax3 = fig.add_subplot(gs[2, 0])
                ax3.text(0.5, 0.5, f"Correlation unavailable:\n{e}", ha="center", va="center")
                ax3.axis("off")
            except Exception:
                pass

        # ---- Panel 4: completeness pie ------------------------------------
        try:
            ax4 = fig.add_subplot(gs[2, 1])
            cols_in_df    = [c for c in missing_cols.index if c in df.columns]
            incomplete    = int(df[cols_in_df].isna().any(axis=1).sum()) if cols_in_df else 0
            complete_rows = len(df) - incomplete

            ax4.pie(
                [complete_rows, incomplete],
                labels=[f"Complete\n({complete_rows:,})", f"Incomplete\n({incomplete:,})"],
                colors=["#2ecc71", "#e74c3c"],
                autopct="%1.1f%%", explode=(0.05, 0),
                startangle=90, textprops={"fontsize": 10},
            )
            ax4.set_title("Data Completeness by Row", fontsize=12, fontweight="bold")
        except Exception as e:
            logger.warning(f"[{caller}] pie chart failed: {e}")
            try:
                ax4 = fig.add_subplot(gs[2, 1])
                ax4.text(0.5, 0.5, f"Pie chart unavailable:\n{e}", ha="center", va="center")
                ax4.axis("off")
            except Exception:
                pass

        plt.suptitle("Comprehensive Missing Data Analysis",
                     fontsize=15, fontweight="bold", y=0.98)
        return fig

    except Exception as e:
        logger.error(f"[{caller}] figure setup failed: {e}")
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass
        return None


if __name__ == "__main__":
    print("Missing Data Analysis Module")
    print("Usage: analyze_missing_patterns(df, config, plot=True)")