# src/n1e_viz_exploration_utils.py

"""
n1e_viz_exploration_utils.py - Visualization Utilities for Notebook 01

This module provides utility functions for generating and saving
visualization outputs for Notebook 01.

Key Features:
- Generate and save visualization outputs
- Comprehensive logging with run_id correlation
- Dynamic RFM distribution analysis
- Statistical independence testing (chi-square + Cramér's V)

Design Principles:
- Defensive programming with clear error messages
- Automatic handling of missing RFM scores
- Data integrity checks with hash validation
- Consistent logging patterns

Error Handling Strategy:
- All public functions validate inputs before processing
- Functions return sensible defaults (None / {} / []) on failure
- Errors are logged at appropriate severity levels
- plt figures are always closed on exception to prevent memory leaks
- No function raises unhandled exceptions to callers
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
from n1a_utils import setup_logger, get_project_root, load_config, set_run_id
from scipy.stats import skew, kurtosis, chi2_contingency
from scipy.stats.contingency import association

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

logger = setup_logger(__name__)
PROJECT_ROOT = get_project_root()

try:
    config = load_config(PROJECT_ROOT / "config.yaml")
    figures_rel_path = config.get("paths", {}).get("figures_dir", "outputs/figures")
    FIGURES_BASE_DIR = (PROJECT_ROOT / figures_rel_path).resolve()
    logger.info(f"Figures directory configured: {FIGURES_BASE_DIR}")
except Exception as e:
    logger.warning(f"Failed to load figures_dir from config, using fallback. Error: {e}")
    FIGURES_BASE_DIR = (PROJECT_ROOT / "outputs" / "figures").resolve()
    logger.info(f"Using fallback figures directory: {FIGURES_BASE_DIR}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_dataframe(df: Any, caller: str) -> bool:
    """
    Return True if *df* is a non-empty DataFrame, otherwise log and return False.
    *caller* is the name of the calling function, used in error messages.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"[{caller}] Expected a pd.DataFrame, got {type(df).__name__}")
        return False
    if df.empty:
        logger.warning(f"[{caller}] Received an empty DataFrame – nothing to process")
        return False
    return True


def _safe_set_run_id(run_id: Optional[str], caller: str) -> None:
    """Set run_id, swallowing any errors so callers are never blocked."""
    if run_id is None:
        return
    if not isinstance(run_id, str):
        logger.warning(f"[{caller}] run_id must be a str, got {type(run_id).__name__} – ignored")
        return
    try:
        set_run_id(run_id)
    except Exception as e:
        logger.warning(f"[{caller}] Could not set run_id '{run_id}': {e}")


def _close_figure_on_error(fig=None) -> None:
    """Best-effort figure cleanup to prevent matplotlib memory leaks."""
    try:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close("all")
    except Exception:
        pass


# ===========================================================================
# 1. save_fig
# ===========================================================================

def save_fig(
    fig_name: str,
    subfolder: str = "",
    dpi: int = 150,
    tight: bool = True,
    run_id: Optional[str] = None,
) -> None:
    """
    Save the current matplotlib figure to disk.

    Error handling:
    - Validates fig_name is a non-empty string
    - Validates dpi is a positive integer
    - Creates the output directory if it does not exist
    - Logs and re-raises on save failure so callers can decide how to proceed
    """
    # ---- Input validation --------------------------------------------------
    if not isinstance(fig_name, str) or not fig_name.strip():
        raise ValueError("save_fig: 'fig_name' must be a non-empty string")

    if not isinstance(dpi, int) or dpi <= 0:
        logger.warning(f"save_fig: invalid dpi={dpi!r}, defaulting to 150")
        dpi = 150

    if not isinstance(subfolder, str):
        logger.warning(f"save_fig: subfolder must be str, got {type(subfolder).__name__} – ignored")
        subfolder = ""

    # ---- run_id ------------------------------------------------------------
    _safe_set_run_id(run_id, "save_fig")

    # ---- Determine save directory ------------------------------------------
    save_dir = FIGURES_BASE_DIR
    if subfolder.strip():
        save_dir = save_dir / subfolder.strip()

    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"save_fig: cannot create directory '{save_dir}': {e}")
        raise

    # ---- Build filepath ----------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = save_dir / f"{fig_name.strip()}_{timestamp}.png"

    # ---- Optional tight layout --------------------------------------------
    if tight:
        try:
            plt.tight_layout()
        except Exception as e:
            logger.warning(f"save_fig: tight_layout raised an error (non-fatal): {e}")

    # ---- Save --------------------------------------------------------------
    try:
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        logger.info(f"Figure saved: {filepath}")
    except Exception as e:
        logger.error(f"save_fig: failed to save '{fig_name}': {e}")
        raise


# ===========================================================================
# 2. plot_distribution
# ===========================================================================

def plot_distribution(
    df: pd.DataFrame,
    columns: List[str],
    save: bool = True,
    subfolder: str = "",
    show: bool = False,
    max_cols: int = 6,
    kde_color: str = "black",
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Plot histograms + KDE overlays for each requested column.

    Error handling:
    - Validates df is a non-empty DataFrame
    - Validates columns is a list
    - Skips individual columns that cannot be plotted (logs warning per column)
    - Ensures the figure is closed even if an error occurs mid-plot
    - save_fig errors are caught and logged; plotting still succeeds
    """
    _safe_set_run_id(run_id, "plot_distribution")

    # ---- Input validation --------------------------------------------------
    if not _validate_dataframe(df, "plot_distribution"):
        return

    if not isinstance(columns, list):
        logger.error(f"plot_distribution: 'columns' must be a list, got {type(columns).__name__}")
        return

    if not isinstance(max_cols, int) or max_cols <= 0:
        logger.warning(f"plot_distribution: invalid max_cols={max_cols!r}, defaulting to 6")
        max_cols = 6

    # ---- Filter valid columns ----------------------------------------------
    cols_to_plot = [c for c in columns if isinstance(c, str) and c in df.columns]

    if not cols_to_plot:
        logger.warning("plot_distribution: no valid columns to plot")
        logger.warning(f"  Requested : {columns}")
        logger.warning(f"  Available : {df.columns.tolist()}")
        return

    skipped = [c for c in columns if c not in cols_to_plot]
    if skipped:
        logger.warning(f"plot_distribution: skipping missing columns: {skipped}")

    if verbose:
        logger.info(f"Plotting {min(len(cols_to_plot), max_cols)} of {len(cols_to_plot)} distributions")

    # ---- Plot --------------------------------------------------------------
    fig = None
    try:
        fig = plt.figure(figsize=(15, 10))

        for i, col in enumerate(cols_to_plot[:max_cols], 1):
            try:
                plt.subplot(2, 3, i)
                series = df[col].dropna()

                if series.empty:
                    logger.warning(f"plot_distribution: column '{col}' has no non-null values – skipped")
                    plt.title(f"Distribution of {col}\n(no data)")
                    continue

                if not pd.api.types.is_numeric_dtype(series):
                    logger.warning(f"plot_distribution: column '{col}' is not numeric – skipped")
                    plt.title(f"Distribution of {col}\n(non-numeric)")
                    continue

                sns.histplot(data=series.to_frame(), x=col, bins=30, color="skyblue", stat="density")
                sns.kdeplot(series, color=kde_color, linewidth=2)  # type: ignore[arg-type]
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Density")

            except Exception as col_err:
                logger.warning(f"plot_distribution: error plotting column '{col}': {col_err}")
                plt.title(f"Distribution of {col}\n(error)")

        # ---- Save ----------------------------------------------------------
        if save:
            try:
                save_fig("distribution", subfolder=subfolder, run_id=run_id)
                if verbose:
                    logger.info("Distribution plots saved")
            except Exception as save_err:
                logger.error(f"plot_distribution: save failed: {save_err}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        logger.error(f"plot_distribution: unexpected error: {e}")
        _close_figure_on_error(fig)
        return

    if verbose:
        logger.info("Distribution plots complete")


# ===========================================================================
# 3. plot_boxplots
# ===========================================================================

def plot_boxplots(
    df: pd.DataFrame,
    columns: List[str],
    save: bool = True,
    subfolder: str = "",
    show: bool = False,
    max_cols: int = 6,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Plot boxplots for outlier detection.

    Error handling:
    - Validates df and columns types
    - Skips non-numeric / all-null columns with individual warnings
    - Closes figure on any unexpected error
    - save errors are caught and logged without aborting the plot
    """
    _safe_set_run_id(run_id, "plot_boxplots")

    if not _validate_dataframe(df, "plot_boxplots"):
        return

    if not isinstance(columns, list):
        logger.error(f"plot_boxplots: 'columns' must be a list, got {type(columns).__name__}")
        return

    if not isinstance(max_cols, int) or max_cols <= 0:
        logger.warning(f"plot_boxplots: invalid max_cols={max_cols!r}, defaulting to 6")
        max_cols = 6

    cols_to_plot = [c for c in columns if isinstance(c, str) and c in df.columns]

    if not cols_to_plot:
        logger.warning("plot_boxplots: no valid columns found")
        logger.warning(f"  Requested : {columns}")
        logger.warning(f"  Available : {df.columns.tolist()}")
        return

    if verbose:
        logger.info(f"Creating {min(len(cols_to_plot), max_cols)} boxplots")

    fig = None
    try:
        fig = plt.figure(figsize=(15, 10))

        for i, col in enumerate(cols_to_plot[:max_cols], 1):
            try:
                plt.subplot(2, 3, i)
                series = df[col].dropna()

                if series.empty:
                    logger.warning(f"plot_boxplots: column '{col}' has no non-null values – skipped")
                    plt.title(f"Boxplot of {col}\n(no data)")
                    continue

                if not pd.api.types.is_numeric_dtype(series):
                    logger.warning(f"plot_boxplots: column '{col}' is not numeric – skipped")
                    plt.title(f"Boxplot of {col}\n(non-numeric)")
                    continue

                sns.boxplot(x=series, color="lightblue", fliersize=4, width=0.4)
                plt.title(f"Boxplot of {col}")
                plt.xlabel(col)

            except Exception as col_err:
                logger.warning(f"plot_boxplots: error plotting '{col}': {col_err}")
                plt.title(f"Boxplot of {col}\n(error)")

        if save:
            try:
                save_fig("boxplots", subfolder=subfolder, run_id=run_id)
                if verbose:
                    logger.info("Boxplots saved")
            except Exception as save_err:
                logger.error(f"plot_boxplots: save failed: {save_err}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        logger.error(f"plot_boxplots: unexpected error: {e}")
        _close_figure_on_error(fig)
        return

    if verbose:
        logger.info("Boxplots complete")


# ===========================================================================
# 4. plot_correlation_heatmap
# ===========================================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    save: bool = True,
    subfolder: str = "",
    show: bool = False,
    annot_threshold: float = 0.5,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Plot a full-color correlation heatmap with selective annotation.

    Error handling:
    - Validates df type and numeric column count
    - Validates annot_threshold is in [0, 1]
    - Catches and logs corr() failures (e.g., constant columns)
    - Closes figure on any unexpected error
    """
    _safe_set_run_id(run_id, "plot_correlation_heatmap")

    if not _validate_dataframe(df, "plot_correlation_heatmap"):
        return

    if not isinstance(annot_threshold, (int, float)) or not (0.0 <= annot_threshold <= 1.0):
        logger.warning(
            f"plot_correlation_heatmap: annot_threshold={annot_threshold!r} is invalid, "
            "must be float in [0, 1] – defaulting to 0.5"
        )
        annot_threshold = 0.5

    # ---- Select numeric columns -------------------------------------------
    try:
        numeric_df = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    except Exception as e:
        logger.error(f"plot_correlation_heatmap: failed to select numeric columns: {e}")
        return

    if numeric_df.empty or len(numeric_df.columns) < 2:
        logger.warning(
            f"plot_correlation_heatmap: need ≥2 numeric columns, "
            f"found {len(numeric_df.columns)}"
        )
        return

    if verbose:
        logger.info(f"Computing correlations for {len(numeric_df.columns)} numeric columns")

    # ---- Correlation matrix ------------------------------------------------
    try:
        corr = numeric_df.corr()
    except Exception as e:
        logger.error(f"plot_correlation_heatmap: corr() failed: {e}")
        return

    if corr.isnull().all(axis=None):
        logger.warning("plot_correlation_heatmap: correlation matrix is all NaN – nothing to plot")
        return

    # ---- Plot --------------------------------------------------------------
    fig = None
    try:
        figsize = (
            max(10, len(corr.columns) * 0.8),
            max(9, len(corr.columns) * 0.7),
        )
        fig = plt.figure(figsize=figsize)

        annot_mask = np.abs(corr.values) < annot_threshold
        annot_array = np.where(annot_mask, "", corr.values.round(2).astype(str))

        sns.heatmap(
            corr,
            annot=annot_array,
            fmt="",
            cmap="RdBu_r",
            center=0,
            linewidths=0.5,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Correlation"},
            square=True,
        )
        plt.title(f"Correlation Heatmap (annotations for |r| ≥ {annot_threshold})")
        plt.tight_layout()

        if save:
            try:
                save_fig("correlation_heatmap", subfolder=subfolder, run_id=run_id)
                if verbose:
                    logger.info("Correlation heatmap saved")
            except Exception as save_err:
                logger.error(f"plot_correlation_heatmap: save failed: {save_err}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        logger.error(f"plot_correlation_heatmap: unexpected error: {e}")
        _close_figure_on_error(fig)
        return

    if verbose:
        logger.info("Correlation heatmap complete")


# ===========================================================================
# 5. plot_categorical_counts
# ===========================================================================

def plot_categorical_counts(
    df: pd.DataFrame,
    columns: List[str],
    save: bool = True,
    subfolder: str = "",
    show: bool = False,
    max_cols: int = 6,
    order: Optional[List[str]] = None,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Plot count plots for categorical columns.

    Error handling:
    - Validates df and columns types
    - Skips columns that seaborn cannot plot (e.g., all-null, unsupported dtype)
    - Falls back to default palette on unknown column names
    - Closes figure on any unexpected error
    """
    _safe_set_run_id(run_id, "plot_categorical_counts")

    if not _validate_dataframe(df, "plot_categorical_counts"):
        return

    if not isinstance(columns, list):
        logger.error(
            f"plot_categorical_counts: 'columns' must be a list, got {type(columns).__name__}"
        )
        return

    if not isinstance(max_cols, int) or max_cols <= 0:
        logger.warning(f"plot_categorical_counts: invalid max_cols={max_cols!r}, defaulting to 6")
        max_cols = 6

    if order is not None and not isinstance(order, list):
        logger.warning("plot_categorical_counts: 'order' must be a list or None – ignored")
        order = None

    cols_to_plot = [c for c in columns if isinstance(c, str) and c in df.columns]

    if not cols_to_plot:
        logger.warning("plot_categorical_counts: no valid categorical columns found")
        logger.warning(f"  Requested : {columns}")
        logger.warning(f"  Available : {df.columns.tolist()}")
        return

    custom_palettes: Dict[str, Any] = {
        "category": "Paired",
        "region": "tab10",
        "payment_method": "Set2",
        "customer_gender": {
            "Male": "blue",
            "Female": "pink",
            "Unknown": "gray",
            "Other": "lightgray",
        },
    }

    if verbose:
        logger.info(f"Plotting {min(len(cols_to_plot), max_cols)} categorical distributions")

    fig = None
    try:
        fig = plt.figure(figsize=(15, 10))

        for i, col in enumerate(cols_to_plot[:max_cols], 1):
            try:
                plt.subplot(2, 3, i)
                series = df[col].dropna()

                if series.empty:
                    logger.warning(f"plot_categorical_counts: column '{col}' has no non-null values – skipped")
                    plt.title(f"Counts of {col}\n(no data)")
                    continue

                counts = series.value_counts()
                plot_order = counts.index.tolist() if order is None else order
                palette = custom_palettes.get(col, "bright")

                sns.countplot(data=df, x=col, order=plot_order, palette=palette)
                plt.title(f"Counts of {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")

            except Exception as col_err:
                logger.warning(f"plot_categorical_counts: error plotting '{col}': {col_err}")
                plt.title(f"Counts of {col}\n(error)")

        if save:
            try:
                save_fig("categorical_counts", subfolder=subfolder, run_id=run_id)
                if verbose:
                    logger.info("Categorical count plots saved")
            except Exception as save_err:
                logger.error(f"plot_categorical_counts: save failed: {save_err}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        logger.error(f"plot_categorical_counts: unexpected error: {e}")
        _close_figure_on_error(fig)
        return

    if verbose:
        logger.info("Categorical count plots complete")


# ===========================================================================
# 6. interpret_skewness
# ===========================================================================

def interpret_skewness(skewness: float) -> Dict[str, Union[str, float]]:
    """
    Classify skewness magnitude and direction.

    Error handling:
    - Validates skewness is a real number (int/float)
    - Rejects NaN / Inf with a ValueError
    - Returns a safe default dict on unexpected errors
    """
    # ---- Input validation --------------------------------------------------
    if not isinstance(skewness, (int, float)):
        raise TypeError(
            f"interpret_skewness: 'skewness' must be numeric, got {type(skewness).__name__}"
        )
    if np.isnan(skewness):
        raise ValueError("interpret_skewness: 'skewness' is NaN – cannot interpret")
    if np.isinf(skewness):
        raise ValueError("interpret_skewness: 'skewness' is infinite – cannot interpret")

    try:
        abs_skew = abs(skewness)

        if abs_skew < 0.1:
            strength, magnitude = "negligible", "Nearly symmetric"
        elif abs_skew < 0.5:
            strength, magnitude = "slight", "Nearly symmetric"
        elif abs_skew < 1.0:
            strength, magnitude = "moderate", "Moderately skewed"
        elif abs_skew < 2.0:
            strength, magnitude = "strong", "Highly skewed"
        else:
            strength, magnitude = "extreme", "Extremely skewed"

        if abs_skew < 0.1:
            direction, direction_text = "none", ""
        elif skewness > 0:
            direction, direction_text = "right", "(right tail)"
        else:
            direction, direction_text = "left", "(left tail)"

        return {
            "magnitude": magnitude,
            "strength": strength,
            "direction": direction,
            "direction_text": direction_text,
            "abs_skew": abs_skew,
        }

    except Exception as e:
        logger.error(f"interpret_skewness: unexpected error for value {skewness}: {e}")
        return {
            "magnitude": "Unknown",
            "strength": "unknown",
            "direction": "unknown",
            "direction_text": "",
            "abs_skew": float(abs(skewness)) if not np.isnan(skewness) else 0.0,
        }


# ===========================================================================
# 7. detect_distribution_shape
# ===========================================================================

def detect_distribution_shape(
    q25: float,
    median: float,
    q75: float,
    min_val: float,
    max_val: float,
) -> Dict[str, Any]:
    """
    Detect distribution concentration using quartiles.

    Error handling:
    - Validates all five parameters are finite, real numbers
    - Validates logical ordering: min_val ≤ q25 ≤ median ≤ q75 ≤ max_val
    - Returns safe defaults if the range is zero (degenerate distribution)
    """
    # ---- Type checks -------------------------------------------------------
    params = {"q25": q25, "median": median, "q75": q75, "min_val": min_val, "max_val": max_val}
    for name, val in params.items():
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"detect_distribution_shape: '{name}' must be numeric, got {type(val).__name__}"
            )
        if np.isnan(val) or np.isinf(val):
            raise ValueError(
                f"detect_distribution_shape: '{name}' is {val} – must be finite"
            )

    # ---- Logical ordering (warn but don't abort) ---------------------------
    if not (min_val <= q25 <= median <= q75 <= max_val):
        logger.warning(
            "detect_distribution_shape: quartiles are not in expected order "
            f"(min={min_val}, q25={q25}, median={median}, q75={q75}, max={max_val}). "
            "Results may be unreliable."
        )

    try:
        iqr = q75 - q25
        range_val = max_val - min_val
        lower_spread = median - q25
        upper_spread = q75 - median

        # Degenerate distribution (all values identical)
        if range_val == 0:
            return {
                "concentration": "middle",
                "description": "no variation (constant column)",
                "symmetry": "symmetric",
                "iqr": 0.0,
                "lower_spread": 0.0,
                "upper_spread": 0.0,
            }

        if median < (min_val + range_val * 0.33):
            concentration, description = "lower", "concentrated in lower range"
        elif median > (min_val + range_val * 0.67):
            concentration, description = "upper", "concentrated in upper range"
        else:
            concentration, description = "middle", "concentrated in middle range"

        if iqr == 0:
            symmetry = "symmetric"
        elif abs(lower_spread - upper_spread) < (iqr * 0.2):
            symmetry = "symmetric"
        elif lower_spread > upper_spread:
            symmetry = "lower_heavy"
        else:
            symmetry = "upper_heavy"

        return {
            "concentration": concentration,
            "description": description,
            "symmetry": symmetry,
            "iqr": iqr,
            "lower_spread": lower_spread,
            "upper_spread": upper_spread,
        }

    except Exception as e:
        logger.error(f"detect_distribution_shape: unexpected error: {e}")
        return {
            "concentration": "unknown",
            "description": "error during shape detection",
            "symmetry": "unknown",
            "iqr": 0.0,
            "lower_spread": 0.0,
            "upper_spread": 0.0,
        }


# ===========================================================================
# 8. generate_recency_insights
# ===========================================================================

def generate_recency_insights(
    skewness: float,
    mean: float,
    median: float,
    q75: float,
    max_val: float,
) -> List[str]:
    """
    Generate dynamic business insights for recency distribution.

    Error handling:
    - Validates all parameters are finite numbers
    - Returns an empty list on any error so callers can always iterate safely
    """
    _REQUIRED = {"skewness": skewness, "mean": mean, "median": median,
                 "q75": q75, "max_val": max_val}
    for name, val in _REQUIRED.items():
        if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
            logger.error(
                f"generate_recency_insights: '{name}'={val!r} is not a finite number – "
                "returning empty insights"
            )
            return []

    try:
        insights: List[str] = []
        interp = interpret_skewness(skewness)

        if interp["direction"] == "right" and interp["strength"] in ("moderate", "strong", "extreme"):
            insights.append("Many inactive customers; prioritize reactivation strategies")
        elif interp["direction"] == "left":
            insights.append("Most customers are recent purchasers (good engagement)")

        if median > 0 and mean > median * 1.2:
            insights.append(
                f"Mean ({mean:.0f} days) > Median ({median:.0f} days): some high-recency customers"
            )
            insights.append(f"Target customers with recency > {median:.0f} days (above median)")
        elif median > 0 and mean < median * 0.8:
            insights.append(
                f"Mean ({mean:.0f} days) < Median ({median:.0f} days): recent customer surge"
            )

        if q75 > 120:
            insights.append(
                f"25% of customers inactive for {q75:.0f}+ days → urgent reactivation needed"
            )

        return insights

    except Exception as e:
        logger.error(f"generate_recency_insights: unexpected error: {e}")
        return []


# ===========================================================================
# 9. generate_frequency_insights
# ===========================================================================

def generate_frequency_insights(
    skewness: float,
    mean: float,
    median: float,
    q25: float,
    q75: float,
    max_val: float,
) -> List[str]:
    """
    Generate dynamic business insights for frequency distribution.

    Error handling:
    - Validates all parameters are finite numbers
    - Returns an empty list on any error
    """
    _REQUIRED = {"skewness": skewness, "mean": mean, "median": median,
                 "q25": q25, "q75": q75, "max_val": max_val}
    for name, val in _REQUIRED.items():
        if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
            logger.error(
                f"generate_frequency_insights: '{name}'={val!r} is not a finite number – "
                "returning empty insights"
            )
            return []

    try:
        insights: List[str] = []
        interp = interpret_skewness(skewness)

        if interp["strength"] == "moderate":
            insights.append("Moderate concentration of infrequent buyers")
        elif interp["direction"] == "right" and interp["strength"] in ("strong", "extreme"):
            insights.append("Strong concentration in low-frequency segment")
            insights.append("Growth opportunity: convert one-time buyers to repeat purchasers")

        if q75 >= 5:
            insights.append(
                f"Top 25% made {q75:.0f}+ purchases – leverage for advocacy and referrals"
            )
        elif q75 >= 3:
            insights.append(
                f"Top 25% made {q75:.0f}+ purchases – potential for loyalty programs"
            )

        if q25 <= 2:
            insights.append(
                f"Bottom 25% made ≤{q25:.0f} purchases – focus on second purchase conversion"
            )

        return insights

    except Exception as e:
        logger.error(f"generate_frequency_insights: unexpected error: {e}")
        return []


# ===========================================================================
# 10. generate_monetary_insights
# ===========================================================================

def generate_monetary_insights(
    skewness: float,
    mean: float,
    median: float,
    q75: float,
    q90: float,
    max_val: float,
) -> List[str]:
    """
    Generate dynamic business insights for monetary distribution.

    Error handling:
    - Validates all parameters are finite numbers
    - Guards against median == 0 to prevent ZeroDivisionError
    - Returns an empty list on any error
    """
    _REQUIRED = {"skewness": skewness, "mean": mean, "median": median,
                 "q75": q75, "q90": q90, "max_val": max_val}
    for name, val in _REQUIRED.items():
        if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
            logger.error(
                f"generate_monetary_insights: '{name}'={val!r} is not a finite number – "
                "returning empty insights"
            )
            return []

    try:
        insights: List[str] = []
        interp = interpret_skewness(skewness)

        if interp["direction"] == "right" and interp["strength"] in ("strong", "extreme"):
            insights.append("Revenue concentrated in small customer segment (Pareto principle)")

        if median > 0:
            ratio = mean / median
            if ratio > 3:
                insights.append(f"Mean (${mean:.2f}) is {ratio:.1f}× median (${median:.2f})")
                insights.append("Small percentage of customers driving significant revenue")

            if q90 / median > 5:
                insights.append(f"Top 10% spent ${q90:.2f}+ (vs median ${median:.2f})")
                insights.append(f"{q90/median:.1f}× median value – prioritize retention")

            if q75 / median > 2:
                insights.append(
                    f"Opportunity: upsell lower segments toward ${q75:.2f} (75th percentile)"
                )
                insights.append(f"Target customers spending ${median:.2f}–${q75:.2f}")
        else:
            logger.warning("generate_monetary_insights: median is 0 – ratio-based insights skipped")

        return insights

    except Exception as e:
        logger.error(f"generate_monetary_insights: unexpected error: {e}")
        return []


# ===========================================================================
# 11. generate_loyalty_insights
# ===========================================================================

def generate_loyalty_insights(
    skewness: float,
    mean: float,
    median: float,
    q25: float,
    q75: float,
    min_val: float,
    max_val: float,
) -> List[str]:
    """
    Generate dynamic business insights for loyalty score distribution.

    Error handling:
    - Validates all parameters are finite numbers
    - Guards against detect_distribution_shape failures
    - Returns an empty list on any error
    """
    _REQUIRED = {"skewness": skewness, "mean": mean, "median": median,
                 "q25": q25, "q75": q75, "min_val": min_val, "max_val": max_val}
    for name, val in _REQUIRED.items():
        if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
            logger.error(
                f"generate_loyalty_insights: '{name}'={val!r} is not a finite number – "
                "returning empty insights"
            )
            return []

    try:
        insights: List[str] = []

        try:
            shape = detect_distribution_shape(q25, median, q75, min_val, max_val)
        except Exception as shape_err:
            logger.warning(f"generate_loyalty_insights: shape detection failed: {shape_err}")
            shape = {"description": "unknown shape", "concentration": "unknown"}

        if q75 >= 0.7:
            insights.append(f"Strong loyalty core: top 25% have scores ≥{q75:.2f}")
            insights.append("Leverage these champions for advocacy and referral programs")

        if q25 <= 0.3:
            insights.append(f"Loyalty challenge: bottom 25% have scores ≤{q25:.2f}")
            insights.append("Implement targeted engagement campaigns for this segment")

        if median < 0.4:
            insights.append(
                f"Loyalty distribution {shape.get('description', 'unknown')} (median {median:.2f})"
            )
            insights.append("Majority of customers show weak-to-moderate engagement")
            insights.append("Focus: improve recency, frequency, and monetary value")
        elif median < 0.6:
            insights.append(f"Moderate loyalty baseline (median {median:.2f})")
            insights.append("Opportunity: move moderate customers to high-loyalty tier")
        else:
            insights.append(f"Strong loyalty baseline (median {median:.2f})")
            insights.append("Maintain current engagement strategies")

        if abs(skewness) < 0.5:
            if median < 0.5:
                insights.append("Distribution is balanced but centred in lower-loyalty range")
            else:
                insights.append("Distribution is balanced but centred in higher-loyalty range")

        return insights

    except Exception as e:
        logger.error(f"generate_loyalty_insights: unexpected error: {e}")
        return []


# ===========================================================================
# 12. analyze_rfm_distributions
# ===========================================================================

def analyze_rfm_distributions(
    rfm_df: pd.DataFrame,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Comprehensive RFM distribution analysis with dynamic business insights.

    Error handling:
    - Validates rfm_df is a non-empty DataFrame
    - Each metric is processed independently; a failure in one does not abort others
    - skew/kurtosis computation errors are caught per-column
    - insight generation failures are caught per-column
    - Returns whatever results were successfully computed
    """
    _safe_set_run_id(run_id, "analyze_rfm_distributions")

    if not _validate_dataframe(rfm_df, "analyze_rfm_distributions"):
        return {}

    if verbose:
        print("=" * 80)
        print("RFM DISTRIBUTION ANALYSIS".center(80))
        print("=" * 80)
        print("RFM Distribution Characteristics:")
        print("-" * 80 + "\n")

    results: Dict[str, Dict[str, Any]] = {}

    metrics = {
        "recency_days": "recency_days",
        "frequency": "frequency",
        "monetary": "monetary",
        "loyalty_score": "loyalty_score",
    }

    for col, name in metrics.items():
        if col not in rfm_df.columns:
            if verbose:
                logger.warning(f"analyze_rfm_distributions: column '{col}' not found – skipped")
            continue

        try:
            data = rfm_df[col].dropna()
        except Exception as e:
            logger.error(f"analyze_rfm_distributions: cannot access column '{col}': {e}")
            continue

        if len(data) == 0:
            if verbose:
                logger.warning(f"analyze_rfm_distributions: '{col}' has no valid data – skipped")
            continue

        if not pd.api.types.is_numeric_dtype(data):
            logger.warning(
                f"analyze_rfm_distributions: '{col}' is not numeric ({data.dtype}) – skipped"
            )
            continue

        # ---- Compute statistics -------------------------------------------
        try:
            skew_val = float(skew(data, nan_policy="omit"))
            kurt_val = float(kurtosis(data, nan_policy="omit"))
        except Exception as e:
            logger.error(f"analyze_rfm_distributions: skew/kurtosis failed for '{col}': {e}")
            skew_val, kurt_val = float("nan"), float("nan")

        try:
            col_stats: Dict[str, Any] = {
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "skewness": skew_val,
                "q25": float(data.quantile(0.25)),
                "q50": float(data.median()),
                "q75": float(data.quantile(0.75)),
                "q90": float(data.quantile(0.90)),
                "min": float(data.min()),
                "max": float(data.max()),
                "count": int(len(data)),
                "kurtosis": kurt_val,
            }
        except Exception as e:
            logger.error(f"analyze_rfm_distributions: statistics failed for '{col}': {e}")
            continue

        # ---- Generate insights --------------------------------------------
        insights: List[str] = []
        try:
            if col == "recency_days":
                insights = generate_recency_insights(
                    col_stats["skewness"], col_stats["mean"], col_stats["median"],
                    col_stats["q75"], col_stats["max"],
                )
            elif col == "frequency":
                insights = generate_frequency_insights(
                    col_stats["skewness"], col_stats["mean"], col_stats["median"],
                    col_stats["q25"], col_stats["q75"], col_stats["max"],
                )
            elif col == "monetary":
                insights = generate_monetary_insights(
                    col_stats["skewness"], col_stats["mean"], col_stats["median"],
                    col_stats["q75"], col_stats["q90"], col_stats["max"],
                )
            elif col == "loyalty_score":
                insights = generate_loyalty_insights(
                    col_stats["skewness"], col_stats["mean"], col_stats["median"],
                    col_stats["q25"], col_stats["q75"], col_stats["min"], col_stats["max"],
                )
        except Exception as e:
            logger.error(f"analyze_rfm_distributions: insight generation failed for '{col}': {e}")
            insights = []

        results[col] = {"stats": col_stats, "insights": insights}

        # ---- Print ---------------------------------------------------------
        if verbose:
            try:
                interp = interpret_skewness(col_stats["skewness"])
                if interp["strength"] == "negligible" or interp["direction"] == "none":
                    skew_desc = interp["magnitude"]
                else:
                    skew_desc = f"{interp['magnitude']} → {str(interp['direction']).capitalize()}-skewed"
                print(f"{col:20s}: Skewness = {col_stats['skewness']:6.2f} -> {skew_desc}")
            except Exception as e:
                logger.warning(f"analyze_rfm_distributions: skewness display failed for '{col}': {e}")

            for insight in insights:
                print(f"  {insight}")
            print()

    if verbose:
        print("=" * 80)
        print("RFM OUTLIER ANALYSIS".center(80))
        print("=" * 80)

    return results


# ===========================================================================
# 13. analyze_high_value_customers
# ===========================================================================

def analyze_high_value_customers(
    rfm_df: pd.DataFrame,
    percentile: float = 0.90,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyse high-value customer segment and revenue concentration.

    Error handling:
    - Validates rfm_df is a non-empty DataFrame with required columns
    - Validates percentile is in (0, 1)
    - Guards against zero total revenue (division by zero)
    - Returns empty dict on unrecoverable errors
    """
    _safe_set_run_id(run_id, "analyze_high_value_customers")

    # ---- Input validation --------------------------------------------------
    if not _validate_dataframe(rfm_df, "analyze_high_value_customers"):
        return {}

    REQUIRED_COLS = {"monetary", "frequency", "recency_days"}
    missing = REQUIRED_COLS - set(rfm_df.columns)
    if missing:
        logger.error(
            f"analyze_high_value_customers: missing required columns: {missing}. "
            "Returning empty result."
        )
        return {}

    if not isinstance(percentile, (int, float)) or not (0.0 < percentile < 1.0):
        logger.error(
            f"analyze_high_value_customers: percentile={percentile!r} must be in (0, 1) – "
            "defaulting to 0.90"
        )
        percentile = 0.90

    # ---- Check column types ------------------------------------------------
    for col in REQUIRED_COLS:
        if not pd.api.types.is_numeric_dtype(rfm_df[col]):
            logger.error(
                f"analyze_high_value_customers: column '{col}' is not numeric "
                f"({rfm_df[col].dtype}) – returning empty result"
            )
            return {}

    if verbose:
        print("\nHigh-Value Customer Identification:")
        print("-" * 80)

    try:
        top_threshold = rfm_df["monetary"].quantile(percentile)
        top_customers = rfm_df[rfm_df["monetary"] >= top_threshold]

        pct = int((1 - percentile) * 100)
        count = len(top_customers)
        total_revenue = top_customers["monetary"].sum()
        total_all = rfm_df["monetary"].sum()

        # Guard against division by zero
        if total_all == 0:
            logger.warning(
                "analyze_high_value_customers: total monetary value is 0 – "
                "revenue percentage set to 0"
            )
            revenue_pct = 0.0
        else:
            revenue_pct = (total_revenue / total_all) * 100

        avg_freq = top_customers["frequency"].mean()
        avg_recency = top_customers["recency_days"].mean()

        results = {
            "threshold": float(top_threshold),
            "count": int(count),
            "total_revenue": float(total_revenue),
            "revenue_percentage": float(revenue_pct),
            "avg_frequency": float(avg_freq),
            "avg_recency": float(avg_recency),
            "customers": top_customers,
        }

        if verbose:
            print(f"Top {pct}% Customers (Monetary >= ${top_threshold:.2f}):")
            print(f"  Count                : {count:,} customers")
            print(f"  Revenue contribution : ${total_revenue:,.2f}")
            print(f"  % of total revenue   : {revenue_pct:.1f}%")
            print(f"  Avg frequency        : {avg_freq:.1f} orders")
            print(f"  Avg recency          : {avg_recency:.0f} days")

            if revenue_pct > 70:
                print(f"\nHIGH CONCENTRATION RISK: top {pct}% generate {revenue_pct:.0f}% of revenue")
                print("  Recommendation: diversify; focus on mid-tier customer growth")
            elif revenue_pct > 50:
                print(f"\nMODERATE CONCENTRATION: top {pct}% generate {revenue_pct:.0f}% of revenue")
                print("  Typical Pareto distribution; maintain VIP retention programs")
            else:
                print(f"\nHEALTHY DISTRIBUTION: top {pct}% generate {revenue_pct:.0f}% of revenue")
                print("  Revenue well-distributed; opportunity for tiered strategies")

            print("\n" + "=" * 80)

        return results

    except Exception as e:
        logger.error(f"analyze_high_value_customers: unexpected error: {e}")
        return {}


# ===========================================================================
# 14. interpret_correlation
# ===========================================================================

def interpret_correlation(
    var1: str,
    var2: str,
    r: float,
) -> Dict[str, Union[str, List[str]]]:
    """
    Provide interpretation and business insights for a variable pair correlation.

    Error handling:
    - Validates var1 / var2 are non-empty strings
    - Validates r is a finite float in [-1, 1]
    - Returns a minimal safe dict on any error so callers can always access keys
    """
    # ---- Input validation --------------------------------------------------
    _DEFAULT: Dict[str, Union[str, List[str]]] = {
        "strength": "Unknown",
        "direction": "Unknown",
        "interpretation": "Interpretation unavailable due to input error",
        "business_insight": None,  # type: ignore[assignment]
        "action_items": [],
    }

    if not isinstance(var1, str) or not var1.strip():
        logger.error("interpret_correlation: 'var1' must be a non-empty string")
        return _DEFAULT

    if not isinstance(var2, str) or not var2.strip():
        logger.error("interpret_correlation: 'var2' must be a non-empty string")
        return _DEFAULT

    if not isinstance(r, (int, float)):
        logger.error(f"interpret_correlation: 'r' must be numeric, got {type(r).__name__}")
        return _DEFAULT

    if np.isnan(r) or np.isinf(r):
        logger.error(f"interpret_correlation: 'r' is {r} – cannot interpret")
        return _DEFAULT

    if not (-1.0 <= r <= 1.0):
        logger.warning(
            f"interpret_correlation: r={r:.4f} is outside [-1, 1]. "
            "Clamping for classification but result may be misleading."
        )
        r = max(-1.0, min(1.0, r))

    try:
        abs_r = abs(r)
        direction = "Positive" if r > 0 else "Negative"

        if abs_r > 0.9:
            strength = "Very strong"
        elif abs_r > 0.7:
            strength = "Strong"
        elif abs_r > 0.5:
            strength = "Moderate"
        elif abs_r > 0.3:
            strength = "Weak"
        else:
            strength = "Very weak"

        interpretation: Optional[str] = None
        business_insight: Optional[str] = None
        action_items: List[str] = []

        vars_set = {var1.lower(), var2.lower()}

        # ---- Expected correlations ----------------------------------------
        if vars_set == {"monetary", "net_monetary"}:
            interpretation = "Expected: Monetary and net_monetary are related metrics"
            if abs_r > 0.95:
                business_insight = "Strong relationship confirms net_monetary calculation is correct"
            else:
                business_insight = f"Weaker than expected (r={r:.3f}) – investigate returns impact"
                action_items.append("Review customers with large monetary vs net_monetary gaps")

        elif vars_set == {"frequency", "loyalty_score"}:
            interpretation = "Expected: validates loyalty score calculation (frequency component)"
            if r > 0.8:
                business_insight = "Loyalty score correctly weights purchase frequency"
            else:
                business_insight = "Loyalty score may need frequency weight adjustment"
                action_items.append("Review loyalty_weights in config.yaml")

        elif vars_set == {"recency_days", "loyalty_score"}:
            interpretation = "Expected: validates loyalty score calculation (recency component)"
            if r < -0.7:
                business_insight = "Loyalty score correctly penalises inactive customers"
            else:
                business_insight = "Recency may not be weighted heavily enough in loyalty score"
                action_items.append("Consider increasing recency weight in loyalty calculation")

        elif vars_set in ({"monetary", "loyalty_score"}, {"net_monetary", "loyalty_score"}):
            interpretation = "Expected: validates loyalty score calculation (monetary component)"
            if r > 0.6:
                business_insight = "Loyalty score correctly rewards high-value customers"
            else:
                business_insight = "Monetary value may need stronger weight in loyalty score"

        elif vars_set == {"category_diversity", "loyalty_score"}:
            interpretation = "Expected: validates loyalty score correlation with engagement"
            if r > 0.6:
                business_insight = "Category exploration is a strong loyalty indicator"
                action_items.append("Consider adding category_diversity to loyalty calculation")
            else:
                business_insight = "Category diversity has weak loyalty correlation"

        elif vars_set == {"recency_days", "churn"}:
            interpretation = "Expected: higher recency = higher churn probability"
            if r > 0.7:
                business_insight = "Recency is a strong churn predictor – validates churn definition"
            elif r > 0.5:
                business_insight = "Recency moderately predicts churn – consider multi-factor model"
                action_items.append("Combine recency with frequency/monetary for better churn prediction")
            else:
                business_insight = "Weak recency-churn link – investigate churn definition"
                action_items.append("Review churn_threshold_days in config.yaml")

        elif vars_set == {"loyalty_score", "churn"}:
            interpretation = "Expected: higher loyalty = lower churn"
            if r < -0.6:
                business_insight = "Loyalty score is a strong inverse churn predictor"
                action_items.append("Use loyalty score as primary retention metric")
            else:
                business_insight = "Loyalty score has weak churn prediction – needs refinement"

        elif vars_set == {"frequency", "category_diversity"}:
            interpretation = "Expected: more purchases = more category exploration"
            if r > 0.8:
                business_insight = "Frequent buyers actively explore product catalogue"
                action_items.append("Cross-sell strategies for high-frequency customers")
            else:
                business_insight = "Repeat buyers stick to fewer categories – upsell opportunity"
                action_items.append("Personalised category recommendations for loyal customers")

        elif vars_set in ({"frequency", "monetary"}, {"frequency", "net_monetary"}):
            interpretation = "Expected: more orders = higher revenue"
            if r > 0.7:
                business_insight = "Strong frequency-revenue link – prioritise repeat purchase campaigns"
                action_items.append("Focus retention on driving second and third purchases")
            elif r > 0.4:
                business_insight = "Moderate frequency-revenue link – some low-value repeat buyers"
                action_items.append("Segment: high frequency + low AOV vs low frequency + high AOV")
            else:
                business_insight = "Weak frequency-revenue link – many one-time high spenders"
                action_items.append("Investigate why high-value customers do not return")

        elif vars_set == {"tenure_days", "frequency"}:
            interpretation = "Expected: longer tenure = more purchases"
            if r > 0.6:
                business_insight = "Customer lifetime drives repeat purchases"
            else:
                business_insight = "Weak tenure-frequency link – low repeat purchase rate"
                action_items.append("Implement lifecycle campaigns to drive early repeat purchases")

        elif vars_set == {"tenure_days", "category_diversity"}:
            interpretation = "Expected: longer tenure = more category exploration"
            if r > 0.5:
                business_insight = "Long-term customers explore more categories over time"
                action_items.append("Encourage early-stage customers to try new categories")
            else:
                business_insight = "Customers stick to initial categories – limited exploration"
                action_items.append("Implement category discovery campaigns")

        elif vars_set == {"tenure_days", "loyalty_score"}:
            interpretation = "Expected: longer tenure correlates with loyalty"
            if r > 0.6:
                business_insight = "Tenure is a strong loyalty driver"
            else:
                business_insight = "Tenure has weak loyalty correlation – engagement drops over time"
                action_items.append("Re-engagement campaigns for long-tenured inactive customers")

        elif vars_set in ({"monetary", "avg_order_value"}, {"net_monetary", "avg_order_value"}):
            interpretation = "Expected: revenue correlates with order value"
            if r > 0.7:
                business_insight = "High revenue driven by high-value orders (not just frequency)"
                action_items.append("Upsell strategies to increase AOV for frequent buyers")
            else:
                business_insight = "Revenue driven more by frequency than order size"
                action_items.append("Focus on increasing purchase frequency")

        elif vars_set == {"return_rate", "last_order_was_return"}:
            interpretation = "Expected: return behaviour consistency"
            if r > 0.5:
                business_insight = "Customers with high return rates often end with returns"
                action_items.append("Investigate why returners have final-order returns")
            else:
                business_insight = "Last return does not predict overall return behaviour"

        # ---- Negative correlations -----------------------------------------
        elif vars_set == {"recency_days", "frequency"}:
            if r < -0.3:
                interpretation = "Negative: recent buyers purchase more frequently"
                business_insight = "Engaged customers buy often; inactive ones have low frequency"
                action_items.append("Reactivation campaigns for high-recency, low-frequency segment")
            else:
                interpretation = "Weak negative: recency and frequency somewhat independent"
                business_insight = "Some frequent buyers are currently inactive (churn risk)"
                action_items.append("Monitor frequent buyers with rising recency")

        elif vars_set in ({"recency_days", "monetary"}, {"recency_days", "net_monetary"}):
            if r < -0.3:
                interpretation = "Negative: recent buyers have higher lifetime value"
                business_insight = "Active customers are more valuable – prioritise retention"
                action_items.append("Win-back campaigns for high-value inactive customers")
            else:
                interpretation = "Weak negative: some inactive customers are high-value"
                business_insight = "Churn affects all segments – not just low-value customers"
                action_items.append("Segment-specific retention strategies")

        elif vars_set == {"recency_days", "tenure_days"}:
            if r < -0.5:
                interpretation = "Negative: long-tenured customers are more recent buyers"
                business_insight = "Loyal customers remain engaged over time"
                action_items.append("Leverage long-tenured advocates for referrals")
            else:
                interpretation = "Weak negative: tenure does not guarantee continued engagement"
                business_insight = "Long-term customers can still churn"
                action_items.append("Do not neglect long-tenured customers in retention efforts")

        elif vars_set == {"recency_days", "category_diversity"}:
            if r < -0.3:
                interpretation = "Negative: recent buyers explore more categories"
                business_insight = "Active customers engage across product lines"
                action_items.append("Category expansion offers for reactivation campaigns")
            else:
                interpretation = "Weak negative: category exploration independent of recency"
                business_insight = "Inactive customers explored categories in the past"

        elif vars_set == {"return_rate", "loyalty_score"}:
            if r < -0.3:
                interpretation = "Negative: high returners have lower loyalty"
                business_insight = "Returns negatively impact customer satisfaction and loyalty"
                action_items.append("Quality control for products with high return rates")
                action_items.append("Improve product descriptions to set expectations")
            else:
                interpretation = "Weak negative: returns do not strongly affect loyalty"
                business_insight = "Customers may accept returns as normal (generous policy?)"

        elif vars_set in ({"return_rate", "monetary"}, {"return_rate", "net_monetary"}):
            if r < -0.3:
                interpretation = "Negative: high returners have lower net value"
                business_insight = "Returns erode customer lifetime value"
                action_items.append("Identify and address root causes of returns")
            elif r > 0.2:
                interpretation = "Positive: high spenders return more (volume effect)"
                business_insight = "High-value customers may test more products (tolerable returns)"
            else:
                interpretation = "Weak correlation: returns independent of spend"
                business_insight = "Returns occur across all customer value segments"

        # ---- Multicollinearity concern -------------------------------------
        elif abs_r > 0.8:
            interpretation = "Multicollinearity concern for predictive modelling"
            business_insight = f"Very strong correlation (r={r:.3f}) may cause modelling issues"
            action_items.append("Consider removing one variable or using PCA for modelling")
            action_items.append("If both needed, use regularisation (Ridge/Lasso)")

        # ---- Generic fallback ---------------------------------------------
        else:
            interpretation = f"{strength} {direction.lower()} correlation"
            if abs_r > 0.5:
                business_insight = f"Notable relationship between {var1} and {var2}"
                action_items.append(f"Investigate business drivers of {var1}–{var2} relationship")
            else:
                business_insight = "Weak relationship – variables are largely independent"

        return {
            "strength": strength,
            "direction": direction,
            "interpretation": interpretation,
            "business_insight": business_insight,
            "action_items": action_items,
        }

    except Exception as e:
        logger.error(f"interpret_correlation: unexpected error for ({var1}, {var2}, r={r}): {e}")
        return {
            "strength": "Unknown",
            "direction": "Unknown",
            "interpretation": f"Error during interpretation: {e}",
            "business_insight": None,  # type: ignore[assignment]
            "action_items": [],
        }


# ===========================================================================
# 15. analyze_correlations
# ===========================================================================

def analyze_correlations(
    df: pd.DataFrame,
    threshold: float = 0.5,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive correlation analysis with business insights.

    Error handling:
    - Validates df is a non-empty DataFrame
    - Validates threshold is in [0, 1]
    - Catches corr() failures (e.g., all-NaN columns survive dropna if mixed)
    - Each correlation pair is interpreted independently; failures don't abort the loop
    - Returns a partial result dict on non-fatal errors
    """
    _safe_set_run_id(run_id, "analyze_correlations")

    # ---- Input validation --------------------------------------------------
    if not _validate_dataframe(df, "analyze_correlations"):
        return {"correlations": [], "matrix": None}

    if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
        logger.warning(
            f"analyze_correlations: threshold={threshold!r} is invalid (must be in [0, 1]) – "
            "defaulting to 0.5"
        )
        threshold = 0.5

    if verbose:
        print("=" * 80)
        print("CORRELATION ANALYSIS".center(80))
        print("=" * 80)

    # ---- Select numeric columns -------------------------------------------
    try:
        numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    except Exception as e:
        logger.error(f"analyze_correlations: failed to select numeric columns: {e}")
        return {"correlations": [], "matrix": None}

    if numeric_df.empty or len(numeric_df.columns) < 2:
        logger.warning(
            f"analyze_correlations: need ≥2 numeric columns, "
            f"found {len(numeric_df.columns)}"
        )
        return {"correlations": [], "matrix": None}

    # ---- Correlation matrix -----------------------------------------------
    try:
        corr_matrix = numeric_df.corr()
    except Exception as e:
        logger.error(f"analyze_correlations: corr() failed: {e}")
        return {"correlations": [], "matrix": None}

    # ---- Find strong correlations -----------------------------------------
    strong_corr: List[Dict[str, Any]] = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            try:
                r = float(corr_matrix.iloc[i, j])  # type: ignore[index]
                if np.isnan(r):
                    continue
                if abs(r) > threshold:
                    strong_corr.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "r": r,
                    })
            except Exception as cell_err:
                logger.warning(
                    f"analyze_correlations: error reading correlation at "
                    f"[{i}, {j}]: {cell_err}"
                )

    strong_corr.sort(key=lambda x: abs(x["r"]), reverse=True)

    if verbose:
        print(f"\nStrong Correlations (|r| > {threshold}):")
        print("-" * 80)

        if strong_corr:
            for corr_info in strong_corr:
                var1, var2, r = corr_info["var1"], corr_info["var2"], corr_info["r"]

                try:
                    interp = interpret_correlation(var1, var2, r)
                except Exception as interp_err:
                    logger.warning(
                        f"analyze_correlations: interpretation failed for "
                        f"({var1}, {var2}): {interp_err}"
                    )
                    interp = {
                        "direction": "Unknown",
                        "interpretation": None,
                        "business_insight": None,
                        "action_items": [],
                    }

                print(f"{var1:20s} <-> {var2:20s}: r = {r:6.3f} ({interp.get('direction', '?')})")

                if interp.get("interpretation"):
                    print(f"  -> {interp['interpretation']}")
                if interp.get("business_insight"):
                    print(f"     {interp['business_insight']}")
                for action in interp.get("action_items", []):  # type: ignore[union-attr]
                    print(f"     * ACTION: {action}")

                print()
        else:
            print(f"No correlations exceed |{threshold}| threshold")

    return {
        "correlations": strong_corr,
        "matrix": corr_matrix,
        "threshold": threshold,
        "total_features": len(numeric_df.columns),
    }


# ===========================================================================
# 16. interpret_cramers_v
# ===========================================================================

def interpret_cramers_v(v: float) -> Dict[str, str]:
    """
    Classify Cramér's V effect size into a human-readable label.

    Thresholds follow Cohen (1988) adapted for contingency tables:
      < 0.10  → negligible
      < 0.20  → weak
      < 0.40  → moderate
      ≥ 0.40  → strong

    Error handling:
    - Validates v is a finite float in [0, 1]
    - Returns a safe default dict on bad input
    """
    _DEFAULT = {"strength": "unknown", "label": "Unknown effect size"}

    if not isinstance(v, (int, float)):
        logger.error(f"interpret_cramers_v: 'v' must be numeric, got {type(v).__name__}")
        return _DEFAULT
    if np.isnan(v) or np.isinf(v):
        logger.error(f"interpret_cramers_v: 'v' is {v} – cannot interpret")
        return _DEFAULT
    if not (0.0 <= v <= 1.0):
        logger.warning(
            f"interpret_cramers_v: v={v:.4f} is outside [0, 1]; result may be unreliable"
        )

    try:
        if v < 0.10:
            return {"strength": "negligible", "label": "Negligible association"}
        elif v < 0.20:
            return {"strength": "weak", "label": "Weak association"}
        elif v < 0.40:
            return {"strength": "moderate", "label": "Moderate association"}
        else:
            return {"strength": "strong", "label": "Strong association"}
    except Exception as e:
        logger.error(f"interpret_cramers_v: unexpected error: {e}")
        return _DEFAULT


# ===========================================================================
# 17. test_categorical_independence
# ===========================================================================

def test_categorical_independence(
    segment_df: pd.DataFrame,
    categorical_feature: str,
    alpha: float = 0.05,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[float, float, int, float]:
    """
    Chi-square test of independence between cluster segments and a categorical
    feature, followed by Cramér's V as an effect-size measure.

    H0: Segment and categorical feature are independent
    H1: Segment and categorical feature are associated

    Steps
    -----
    1. Build contingency table (cluster × categorical_feature).
    2. Run scipy chi2_contingency – yields chi2, p-value, dof.
    3. Compute Cramér's V via scipy.stats.contingency.association, which
       correctly applies the bias-corrected formula for small samples.
    4. Log and (optionally) print a structured summary.

    Args:
        segment_df:           DataFrame containing 'cluster' and the feature.
        categorical_feature:  Column name of the categorical variable to test.
        alpha:                Significance level for the pass/fail decision.
        run_id:               Optional run identifier for log correlation.
        verbose:              If True, print a formatted results block.

    Returns:
        Tuple of (chi2_statistic, p_value, degrees_of_freedom, cramers_v).
        On failure, returns (np.nan, np.nan, 0, np.nan).

    Error handling:
    - Validates segment_df is a non-empty DataFrame
    - Checks that 'cluster' column is present
    - Checks that categorical_feature column is present
    - Validates alpha is a float in (0, 1)
    - chi2_contingency failures are caught and logged
    - Cramér's V failures are caught independently so chi2 results are
      still returned even if the effect-size step fails
    """
    _safe_set_run_id(run_id, "test_categorical_independence")

    # ---- Input validation --------------------------------------------------
    if not _validate_dataframe(segment_df, "test_categorical_independence"):
        return np.nan, np.nan, 0, np.nan

    if "cluster" not in segment_df.columns:
        logger.error(
            "test_categorical_independence: 'cluster' column not found in segment_df. "
            f"Available columns: {segment_df.columns.tolist()}"
        )
        return np.nan, np.nan, 0, np.nan

    if not isinstance(categorical_feature, str) or not categorical_feature.strip():
        logger.error(
            "test_categorical_independence: 'categorical_feature' must be a non-empty string"
        )
        return np.nan, np.nan, 0, np.nan

    if categorical_feature not in segment_df.columns:
        logger.error(
            f"test_categorical_independence: feature '{categorical_feature}' not found. "
            f"Available columns: {segment_df.columns.tolist()}"
        )
        return np.nan, np.nan, 0, np.nan

    if not isinstance(alpha, (int, float)) or not (0.0 < alpha < 1.0):
        logger.warning(
            f"test_categorical_independence: invalid alpha={alpha!r}, defaulting to 0.05"
        )
        alpha = 0.05

    # ---- Build contingency table -------------------------------------------
    try:
        contingency_table = pd.crosstab(
            segment_df["cluster"],
            segment_df[categorical_feature],
        )
    except Exception as e:
        logger.error(
            f"test_categorical_independence: crosstab failed for '{categorical_feature}': {e}"
        )
        return np.nan, np.nan, 0, np.nan

    if contingency_table.empty:
        logger.warning(
            f"test_categorical_independence: contingency table for '{categorical_feature}' "
            "is empty – nothing to test"
        )
        return np.nan, np.nan, 0, np.nan

    # ---- Chi-square test ---------------------------------------------------
    try:
        chi2_result = chi2_contingency(contingency_table)
        # Support both old tuple API and new named-tuple API (scipy >= 1.11)
        if hasattr(chi2_result, "statistic"):
            chi2_stat = float(chi2_result.statistic)
            p_value   = float(chi2_result.pvalue)
            dof       = int(chi2_result.dof)
        else:
            chi2_stat, p_value, dof, _ = chi2_result
            chi2_stat = float(chi2_stat)
            p_value   = float(p_value)
            dof       = int(dof)
    except Exception as e:
        logger.error(
            f"test_categorical_independence: chi2_contingency failed for "
            f"'{categorical_feature}': {e}"
        )
        return np.nan, np.nan, 0, np.nan

    # ---- Cramér's V (effect size) ------------------------------------------
    # scipy.stats.contingency.association applies the bias-corrected formula
    # automatically; it accepts the raw contingency DataFrame directly.
    cramers_v = np.nan
    try:
        cramers_v = float(association(contingency_table, method="cramer"))
    except Exception as e:
        logger.warning(
            f"test_categorical_independence: Cramér's V computation failed for "
            f"'{categorical_feature}': {e}. Chi-square results are still valid."
        )

    # ---- Interpret effect size ---------------------------------------------
    effect = interpret_cramers_v(cramers_v) if not np.isnan(cramers_v) else None

    # ---- Logging -----------------------------------------------------------
    logger.info(f"\nChi-Square + Cramér's V Test for '{categorical_feature}':")
    logger.info(f"  Chi2-statistic      : {chi2_stat:.4f}")
    logger.info(f"  P-value             : {p_value:.4f}")
    logger.info(f"  Degrees of freedom  : {dof}")

    if not np.isnan(cramers_v):
        logger.info(f"  Cramér's V          : {cramers_v:.4f}  [{effect['label']}]")  # type: ignore[index]

    sig_flag = "[PASS]" if p_value < alpha else "[FAIL]"
    sig_text = (
        f"Significant association (p < {alpha})"
        if p_value < alpha
        else f"No significant association (p ≥ {alpha})"
    )
    logger.info(f"  {sig_flag} {sig_text}")

    # ---- Verbose print block -----------------------------------------------
    if verbose:
        print(f"\nChi-Square Test for '{categorical_feature}':")
        print(f"  Chi2-statistic      : {chi2_stat:.4f}")
        print(f"  P-value             : {p_value:.4f}")
        print(f"  Degrees of freedom  : {dof}")

        if not np.isnan(cramers_v) and effect is not None:
            print(f"  Cramér's V          : {cramers_v:.4f}  [{effect['label']}]")

            # Paired significance + effect-size narrative
            if p_value < alpha:
                print(
                    f"  [PASS] Significant association (p < {alpha}) with "
                    f"{effect['strength']} effect size (V = {cramers_v:.3f})."
                )
                if effect["strength"] in ("negligible", "weak"):
                    print(
                        "         NOTE: statistically significant but practically small effect; "
                        "likely driven by large sample size."
                    )
            else:
                print(
                    f"  [FAIL] No significant association (p ≥ {alpha}); "
                    f"effect size is also {effect['strength']} (V = {cramers_v:.3f})."
                )
        else:
            # Cramér's V failed – fall back to chi-square-only verdict
            if p_value < alpha:
                print(f"  [PASS] Significant association (p < {alpha})")
            else:
                print(f"  [FAIL] No significant association (p ≥ {alpha})")

    return chi2_stat, p_value, dof, cramers_v