# src/n2c_time_trends.py
"""
n2c_time_trends.py - Time-Based Sales Trends Analysis

This module performs temporal analysis of sales data including:
- Monthly revenue trends with YoY growth
- Seasonality patterns
- Incomplete month filtering

Configuration loaded from config.yaml notebook2.time_trends section.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, Optional, Any, List
import logging
from n2a_utils import (
    setup_logger,
    COLORS,
    save_plotly_figure,
    print_section_header,
    print_subsection,
    get_config
)

# Setup logger
logger = setup_logger(__name__)

# Load configuration
_CONFIG = get_config()
_TIME_CONFIG: Dict[str, Any] = _CONFIG.get('notebook2', {}).get('time_trends', {})
_TS_CONFIG: Dict[str, Any] = _CONFIG.get('time_series', {})

TIME_COLORS: List[str] = _TIME_CONFIG.get('colors', [
    "#1B1F5E", "#EA731D", "#A647AE", "#F34940", "#50DC50",
    "#2B3351", "#FBC500", "#1B1F5E", "#EA731D", "#A647AE",
    "#F34940", "#50DC50", "#2B3351", "#FBC500"
])

INCOMPLETE_MONTH_THRESHOLD: int = _TIME_CONFIG.get('incomplete_month_threshold_days', 28)
YOY_PERIODS: int = _TIME_CONFIG.get('yoy_periods', 12)
MIN_MONTHS_FOR_YOY: int = _TIME_CONFIG.get('min_months_for_yoy', 13)


# ============================================================================
# Data Preparation Functions
# ============================================================================

def filter_incomplete_months(
    df: pd.DataFrame,
    date_column: str = 'order_date'
) -> Tuple[pd.DataFrame, bool]:
    """
    Remove the most recent month if it appears incomplete.

    Logic (in priority order):
    1. If ``time_series.complete_months_only`` is False, skip filtering.
    2. If ``time_series.incomplete_month_cutoff`` is set, use that hard date.
    3. Otherwise fall back to the day-count heuristic using
       ``notebook2.time_trends.incomplete_month_threshold_days``.

    Args:
        df: Transaction DataFrame with a date column
        date_column: Name of the date column (default: 'order_date')

    Returns:
        Tuple[pd.DataFrame, bool]: Filtered DataFrame and whether a month was removed
    """
    try:
        complete_months_only = _TS_CONFIG.get('complete_months_only', True)
        if not complete_months_only:
            logger.info("complete_months_only=False — skipping incomplete-month filter")
            return df, False

        cutoff_str = _TS_CONFIG.get('incomplete_month_cutoff', None)
        if cutoff_str:
            cutoff = pd.Timestamp(cutoff_str)
            if df[date_column].max() >= cutoff:
                df_filtered = df[df[date_column] < cutoff].copy()
                rows_removed = len(df) - len(df_filtered)
                logger.warning(
                    f"Applying hard cutoff {cutoff.date()} from config "
                    f"(removed {rows_removed:,} rows)"
                )
                return df_filtered, True
            else:
                logger.info(f"Data ends before cutoff {cutoff.date()} — no rows removed")
                return df, False

        max_date = df[date_column].max()
        current_month_start = pd.Timestamp(max_date.year, max_date.month, 1)

        if max_date.day < INCOMPLETE_MONTH_THRESHOLD:
            logger.warning(
                f"Detected incomplete month: {max_date.strftime('%Y-%m')} "
                f"(last date: {max_date.date()})"
            )
            logger.info(f"Filtering out data from {current_month_start.date()} onwards...")
            df_filtered = df[df[date_column] < current_month_start].copy()
            rows_removed = len(df) - len(df_filtered)
            logger.info(f"Removed {rows_removed:,} rows from incomplete month")
            return df_filtered, True
        else:
            logger.info("No incomplete months detected")
            return df, False

    except KeyError as e:
        logger.error(f"Missing date column '{date_column}': {e}")
        raise
    except Exception as e:
        logger.error(f"filter_incomplete_months failed: {e}")
        raise


def prepare_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the dataframe.

    Adds year_month, year, month name, and weekday name columns
    derived from the order_date column.

    Args:
        df: Transaction DataFrame with an order_date column

    Returns:
        pd.DataFrame: DataFrame with additional time feature columns
    """
    try:
        logger.info("Adding time-based features...")

        if not pd.api.types.is_datetime64_any_dtype(df['order_date']):
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

        df['year_month'] = df['order_date'].dt.to_period('M').astype(str)
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month_name()
        df['weekday'] = df['order_date'].dt.day_name()

        logger.info("Time features added successfully")
        logger.info("-" * 60)
        return df

    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        raise
    except Exception as e:
        logger.error(f"prepare_time_features failed: {e}")
        raise


def calculate_yoy_growth(monthly_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Safely calculate year-over-year growth with error handling.

    Uses YOY_PERIODS from config (default: 12 months). NaN values are
    intentionally preserved for the first YOY_PERIODS rows that lack a
    prior-year baseline.

    Args:
        monthly_sales: DataFrame with a total_amount column indexed by month

    Returns:
        pd.DataFrame: Input DataFrame with a yoy_growth column added
    """
    try:
        if len(monthly_sales) < MIN_MONTHS_FOR_YOY:
            logger.warning(
                f"Only {len(monthly_sales)} months of data. "
                "YoY growth may not be meaningful."
            )
            monthly_sales['yoy_growth'] = np.nan
        else:
            monthly_sales['yoy_growth'] = (
                monthly_sales['total_amount'].pct_change(periods=YOY_PERIODS) * 100
            ).round(2)
            logger.info("YoY growth calculated successfully")

        return monthly_sales

    except Exception as e:
        logger.error(f"Error calculating YoY growth: {e}")
        monthly_sales['yoy_growth'] = np.nan
        return monthly_sales


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_monthly_revenue(
    monthly_sales: pd.DataFrame,
    save: bool = True
) -> go.Figure:
    """
    Create monthly revenue trend with YoY growth visualization.

    Args:
        monthly_sales: DataFrame with year_month, total_amount, and yoy_growth columns
        save: Whether to save the figure to disk

    Returns:
        go.Figure: Plotly figure with dual-axis revenue and YoY growth lines
    """
    try:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=monthly_sales['year_month'],
                y=monthly_sales['total_amount'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color=TIME_COLORS[0], width=2),
                marker=dict(size=6)
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_sales['year_month'],
                y=monthly_sales['yoy_growth'],
                mode='lines+markers',
                name='YoY Growth (%)',
                line=dict(color=TIME_COLORS[1], width=2),
                marker=dict(size=6),
                opacity=1
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=dict(
                text='Monthly Revenue Trend with YoY Growth',
                font=dict(size=14, family='Arial Black', color='black')
            ),
            xaxis=dict(
                title='Month',
                tickfont=dict(size=11, family='Arial Black', color='black')
            ),
            yaxis=dict(
                title='Total Revenue',
                tickfont=dict(size=11, family='Arial Black', color='black')
            ),
            yaxis2=dict(
                title='YoY Growth (%)',
                tickfont=dict(size=11, family='Arial Black', color='black')
            ),
            hovermode='x unified',
            template='plotly_white',
            height=500
        )

        if save:
            save_plotly_figure(fig, "monthly_revenue_yoy", formats=['html'], logger=logger)

        return fig

    except Exception as e:
        logger.error(f"plot_monthly_revenue failed: {e}")
        raise


def plot_seasonality(
    df: pd.DataFrame,
    save: bool = True
) -> go.Figure:
    """
    Create seasonality analysis visualization.

    Args:
        df: Transaction DataFrame with month and total_amount columns
        save: Whether to save the figure to disk

    Returns:
        go.Figure: Bar chart of average revenue by calendar month
    """
    try:
        seasonal_sales = df.groupby('month')['total_amount'].mean().reset_index()
        seasonal_sales = seasonal_sales.sort_values('total_amount', ascending=False)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=seasonal_sales['month'],
                y=seasonal_sales['total_amount'],
                marker=dict(color=TIME_COLORS[:len(seasonal_sales)])
            )
        )

        fig.update_layout(
            title=dict(
                text='Average Revenue by Month (Seasonality)',
                font=dict(size=14, family='Arial Black', color='black')
            ),
            xaxis=dict(
                title='Month',
                tickfont=dict(size=11, family='Arial Black', color='black')
            ),
            yaxis=dict(
                title='Avg Revenue',
                tickfont=dict(size=11, family='Arial Black', color='black')
            ),
            template='plotly_white',
            showlegend=False,
            height=500
        )

        if save:
            save_plotly_figure(fig, "seasonality", formats=['html'], logger=logger)

        return fig

    except Exception as e:
        logger.error(f"plot_seasonality failed: {e}")
        raise


# ============================================================================
# Main Analysis Function
# ============================================================================

def create_time_trends_analysis(
    df: pd.DataFrame,
    save_figures: bool = True
) -> Dict[str, Any]:
    """
    Perform complete time-based sales trends analysis.

    Runs incomplete month filtering, time feature engineering, monthly
    aggregation, YoY growth calculation, and visualization creation.
    Configuration loaded from config.yaml.

    Args:
        df: Transaction DataFrame
        save_figures: Whether to save generated figures to disk

    Returns:
        dict with keys: df, monthly_sales, figures, metrics
    """
    logger.info("=" * 60)
    logger.info("TIME-BASED SALES TRENDS ANALYSIS")
    logger.info("=" * 60)

    try:
        logger.info("STEP 1: FILTERING INCOMPLETE MONTHS:")
        df_clean, had_incomplete = filter_incomplete_months(df)
        if had_incomplete:
            logger.info("Note: Incomplete month data removed from analysis")
        logger.info("-" * 60)

        logger.info("STEP 2: PREPARING TIME FEATURES:")
        df_clean = prepare_time_features(df_clean)

        logger.info("STEP 3: AGGREGATING MONTHLY SALES:")
        monthly_sales = df_clean.groupby('year_month').agg({
            'total_amount': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        monthly_sales.columns = ['year_month', 'total_amount', 'num_orders']
        logger.info(f"Aggregated {len(monthly_sales)} months of data")
        logger.info("-" * 60)

        logger.info("STEP 4: CALCULATING YOY GROWTH:")
        monthly_sales = calculate_yoy_growth(monthly_sales)
        logger.info("-" * 60)

        logger.info("STEP 5: CREATING VISUALIZATIONS:")
        figures = {}
        figures['monthly_revenue'] = plot_monthly_revenue(monthly_sales, save=save_figures)
        figures['seasonality'] = plot_seasonality(df_clean, save=save_figures)

        metrics = {
            'total_months': len(monthly_sales),
            'avg_monthly_revenue': monthly_sales['total_amount'].mean(),
            'peak_month': monthly_sales.loc[
                monthly_sales['total_amount'].idxmax(), 'year_month'
            ],
            'peak_revenue': monthly_sales['total_amount'].max(),
            'avg_yoy_growth': monthly_sales['yoy_growth'].mean(),
            'latest_yoy_growth': (
                monthly_sales['yoy_growth'].dropna().iloc[-1]
                if monthly_sales['yoy_growth'].notna().any()
                else np.nan
            ),
        }

        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total months analyzed: {metrics['total_months']}")
        logger.info(f"Average monthly revenue: ${metrics['avg_monthly_revenue']:,.2f}")
        logger.info(f"Peak month: {metrics['peak_month']} (${metrics['peak_revenue']:,.2f})")
        logger.info(f"Average YoY growth: {metrics['avg_yoy_growth']:.2f}%")
        logger.info(f"Latest YoY growth: {metrics['latest_yoy_growth']:.2f}%")
        logger.info("=" * 60)

        return {
            'df': df_clean,
            'monthly_sales': monthly_sales,
            'figures': figures,
            'metrics': metrics
        }

    except Exception as e:
        logger.error(f"create_time_trends_analysis failed: {e}")
        raise


# ============================================================================
# Module Information
# ============================================================================

__all__ = [
    'TIME_COLORS',
    'INCOMPLETE_MONTH_THRESHOLD',
    'YOY_PERIODS',
    'MIN_MONTHS_FOR_YOY',
    'filter_incomplete_months',
    'prepare_time_features',
    'calculate_yoy_growth',
    'plot_monthly_revenue',
    'plot_seasonality',
    'create_time_trends_analysis',
]

if __name__ == '__main__':
    print("n2c_time_trends.py - Time Trends Analysis Module")
    print("Configuration loaded from config.yaml")
