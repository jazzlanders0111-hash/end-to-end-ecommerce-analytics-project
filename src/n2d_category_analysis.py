# src/n2d_category_analysis.py
"""
n2d_category_analysis.py - Category & Product Performance Analysis

Analyzes sales performance by category and product including:
- Category revenue and return rates
- Pareto analysis (80/20 rule)
- Category profitability

Functions:
    create_category_analysis() - Main category analysis
    plot_category_revenue_returns() - Category revenue vs returns
    plot_category_profitability() - Category sales and profitability
    plot_category_pareto_chart() - Pareto revenue chart

Usage:
    from n2d_category_analysis import create_category_analysis
    results = create_category_analysis(df)

Unused:
    anything related to product analysis
    top_n_products() - Top product analysis because the products are not the focus
    of this notebook and it adds complexity.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import logging
from n2a_utils import (
    setup_logger,
    save_plotly_figure,
    print_section_header,
    get_config
)

# Setup logger
logger = setup_logger(__name__)

# Load configuration
_CONFIG = get_config()
_CATEGORY_CONFIG: Dict[str, Any] = _CONFIG.get('notebook2', {}).get('category_analysis', {})

CATEGORY_COLORS: List[str] = _CATEGORY_CONFIG.get('colors', [
    "#AECCDB", "#3274A1", "#B3D495", "#33a02c",
    "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"
])

TOP_N_PRODUCTS_DEFAULT: int = _CATEGORY_CONFIG.get('top_n_products', 20)


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate category performance metrics.

    Aggregates revenue, order count, quantity, return rate, and average
    profit margin per category, sorted by revenue descending.

    Args:
        df: Transaction DataFrame with category, total_amount, order_id,
            quantity, returned, and profit_margin columns

    Returns:
        pd.DataFrame: Category performance metrics indexed by category name
    """
    try:
        logger.info("Calculating category performance metrics...")

        cat_performance = df.groupby('category').agg({
            'total_amount': 'sum',
            'order_id': 'nunique',
            'quantity': 'sum',
            'returned': 'mean',
            'profit_margin': 'mean'
        }).sort_values('total_amount', ascending=False).round(2)

        cat_performance.columns = [
            'revenue', 'orders', 'quantity',
            'return_rate', 'avg_profit_margin'
        ]

        logger.info(f"Analyzed {len(cat_performance)} categories")
        return cat_performance

    except KeyError as e:
        logger.error(f"Missing required column in analyze_category_performance: {e}")
        raise
    except Exception as e:
        logger.error(f"analyze_category_performance failed: {e}")
        raise


def calculate_category_pareto(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Perform Pareto (80/20) analysis on categories.

    Args:
        df: Transaction DataFrame with category and total_amount columns

    Returns:
        Tuple[pd.DataFrame, int | None]: DataFrame with Pareto metrics and the
        number of categories that account for 80% of total revenue
    """
    try:
        logger.info("Performing Pareto analysis by category...")

        category_sales = df.groupby('category')['total_amount'].sum().sort_values(
            ascending=False
        )
        cumulative_pct = category_sales.cumsum() / category_sales.sum() * 100

        pareto_df = pd.DataFrame({
            'category': category_sales.index,
            'revenue': category_sales.values,
            'cumulative_pct': cumulative_pct.values
        })

        threshold_idx = next(
            (i for i, val in enumerate(pareto_df['cumulative_pct']) if val >= 80), None
        )
        categories_for_80pct = threshold_idx + 1 if threshold_idx is not None else None

        logger.info(
            f"{categories_for_80pct}/{len(pareto_df)} categories account for 80% of revenue"
        )
        return pareto_df, categories_for_80pct

    except KeyError as e:
        logger.error(f"Missing required column in calculate_category_pareto: {e}")
        raise
    except Exception as e:
        logger.error(f"calculate_category_pareto failed: {e}")
        raise


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_category_revenue_returns(
    cat_performance: pd.DataFrame,
    save: bool = True
) -> go.Figure:
    """
    Plot category revenue with return rates.

    Args:
        cat_performance: DataFrame indexed by category with revenue and
            return_rate columns (output of analyze_category_performance)
        save: Whether to save the figure to disk

    Returns:
        go.Figure: Dual-axis bar and line chart
    """
    try:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=cat_performance.index,
                y=cat_performance['revenue'],
                name='Total Revenue',
                marker=dict(color=CATEGORY_COLORS[:len(cat_performance)])
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=cat_performance.index,
                y=cat_performance['return_rate'] * 100,
                name='Return Rate (%)',
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=dict(
                text='Total Revenue & Return Rate by Category',
                font=dict(size=14, color='black', family='Arial Black'),
                x=0.5
            ),
            xaxis=dict(
                title='Category',
                tickfont=dict(size=11, color='black', family='Arial Black')
            ),
            yaxis=dict(
                title='Total Revenue',
                tickfont=dict(size=11, color='black', family='Arial Black')
            ),
            yaxis2=dict(
                title='Return Rate (%)',
                tickfont=dict(size=11, color='black', family='Arial Black')
            ),
            template='plotly_white',
            hovermode='x unified',
            height=500
        )

        if save:
            save_plotly_figure(
                fig, "category_revenue_returns", formats=['html'], logger=logger
            )

        return fig

    except Exception as e:
        logger.error(f"plot_category_revenue_returns failed: {e}")
        raise


def plot_category_profitability(
    cat_performance: pd.DataFrame,
    save: bool = True
) -> go.Figure:
    """
    Plot category sales and profitability.

    Args:
        cat_performance: DataFrame indexed by category with revenue and
            avg_profit_margin columns (output of analyze_category_performance)
        save: Whether to save the figure to disk

    Returns:
        go.Figure: Dual-axis bar and line chart
    """
    try:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=cat_performance.index,
                y=cat_performance['revenue'],
                name='Total Revenue',
                marker=dict(color=CATEGORY_COLORS[:len(cat_performance)])
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=cat_performance.index,
                y=cat_performance['avg_profit_margin'],
                name='Avg Profit Margin (%)',
                mode='lines+markers',
                line=dict(color='red', width=2.5),
                marker=dict(size=8)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=dict(
                text='Category Sales & Profitability',
                font=dict(size=14, family='Arial Black', color='black'),
                x=0.5
            ),
            xaxis=dict(
                title='Category',
                tickfont=dict(size=11, family='Arial Black', color='black')
            ),
            yaxis=dict(
                title='Total Revenue',
                tickfont=dict(size=11, family='Arial Black', color='black')
            ),
            yaxis2=dict(
                title='Avg Profit Margin (%)',
                tickfont=dict(size=11, family='Arial Black', color='black')
            ),
            template='plotly_white',
            hovermode='x unified',
            height=500
        )

        if save:
            save_plotly_figure(
                fig, "category_profitability", formats=['html'], logger=logger
            )

        return fig

    except Exception as e:
        logger.error(f"plot_category_profitability failed: {e}")
        raise


def plot_category_pareto_chart(
    pareto_df: pd.DataFrame,
    categories_for_80pct: Optional[int],
    save: bool = True
) -> go.Figure:
    """
    Plot category Pareto chart with cumulative revenue percentage.

    Args:
        pareto_df: DataFrame with category, revenue, and cumulative_pct columns
            (output of calculate_category_pareto)
        categories_for_80pct: Number of categories reaching 80% revenue threshold
        save: Whether to save the figure to disk

    Returns:
        go.Figure: Dual-axis Pareto bar and line chart
    """
    try:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=pareto_df['category'],
                y=pareto_df['revenue'],
                name='Revenue',
                marker=dict(color=CATEGORY_COLORS[:len(pareto_df)])
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=pareto_df['category'],
                y=pareto_df['cumulative_pct'],
                name='Cumulative %',
                mode='lines+markers',
                line=dict(color='red', width=2, dash='solid'),
                marker=dict(size=6, color='red')
            ),
            secondary_y=True
        )

        fig.add_hline(y=80, line_dash="dash", line_color="gray", secondary_y=True)

        fig.update_layout(
            title=dict(
                text=f"Pareto Analysis - Revenue by Category "
                     f"({categories_for_80pct} categories ≥ 80%)",
                font=dict(size=14, family='Arial Black'),
                x=0.5
            ),
            xaxis=dict(
                title='Category',
                tickfont=dict(size=11, family='Arial Black', color='black')
            ),
            yaxis=dict(
                title='Revenue',
                tickfont=dict(size=11, family='Arial Black', color='black')
            ),
            yaxis2=dict(
                title='Cumulative %',
                tickfont=dict(size=11, family='Arial Black', color='black'),
                range=[0, 105]
            ),
            template='plotly_white',
            hovermode='x unified',
            height=700
        )

        if save:
            save_plotly_figure(
                fig, "category_pareto_analysis", formats=['html'], logger=logger
            )

        return fig

    except Exception as e:
        logger.error(f"plot_category_pareto_chart failed: {e}")
        raise


# ============================================================================
# Main Analysis Function
# ============================================================================

def create_category_analysis(
    df: pd.DataFrame,
    top_n_products: Optional[int] = None,
    save_figures: bool = True
) -> Dict:
    """
    Perform complete category and product performance analysis.

    Runs category performance aggregation, Pareto analysis, and creates
    all visualizations. top_n_products is loaded from config if not specified.

    Args:
        df: Transaction DataFrame
        top_n_products: Number of top products to analyze (config default if None)
        save_figures: Whether to save generated figures to disk

    Returns:
        dict with keys: category_performance, pareto_analysis, figures, metrics
    """
    if top_n_products is None:
        top_n_products = TOP_N_PRODUCTS_DEFAULT

    logger.info("=" * 60)
    logger.info("CATEGORY PERFORMANCE ANALYSIS")
    logger.info("=" * 60)

    try:
        logger.info("STEP 1: ANALYZING CATEGORY PERFORMANCE:")
        cat_performance = analyze_category_performance(df)
        logger.info("-" * 60)

        logger.info("STEP 2: PARETO ANALYSIS BY CATEGORY:")
        pareto_df, categories_for_80pct = calculate_category_pareto(df)
        logger.info("-" * 60)

        logger.info("STEP 3: CREATING VISUALIZATIONS:")
        figures = {}

        figures['revenue_returns'] = plot_category_revenue_returns(
            cat_performance, save=save_figures
        )
        figures['profitability'] = plot_category_profitability(
            cat_performance, save=save_figures
        )
        figures['pareto'] = plot_category_pareto_chart(
            pareto_df, categories_for_80pct, save=save_figures
        )

        metrics = {
            'total_categories': len(cat_performance),
            'top_category': cat_performance.index[0],
            'top_category_revenue': cat_performance['revenue'].iloc[0],
            'avg_return_rate': cat_performance['return_rate'].mean(),
            'categories_for_80pct': categories_for_80pct,
        }

        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total categories: {metrics['total_categories']}")
        logger.info(
            f"Top category: {metrics['top_category']} "
            f"(${metrics['top_category_revenue']:,.2f})"
        )
        logger.info(f"Average return rate: {metrics['avg_return_rate']:.2%}")
        logger.info(
            f"Pareto: {metrics['categories_for_80pct']}/{metrics['total_categories']} "
            "categories = 80% revenue"
        )
        logger.info("=" * 60)

        return {
            'category_performance': cat_performance,
            'pareto_analysis': pareto_df,
            'figures': figures,
            'metrics': metrics
        }

    except Exception as e:
        logger.error(f"create_category_analysis failed: {e}")
        raise


# ============================================================================
# Module Information
# ============================================================================

__all__ = [
    'CATEGORY_COLORS',
    'TOP_N_PRODUCTS_DEFAULT',
    'analyze_category_performance',
    'calculate_category_pareto',
    'plot_category_revenue_returns',
    'plot_category_profitability',
    'plot_category_pareto_chart',
    'create_category_analysis',
]

if __name__ == '__main__':
    print("n2d_category_analysis.py - Category & Product Analysis Module")
