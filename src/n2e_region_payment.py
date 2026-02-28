# src/n2e_region_payment.py
"""
n2e_region_payment.py - Region & Payment Method Insights

Analyzes sales patterns by:
- Geographic regions
- Payment methods
- Per-customer metrics by region
- Payment method preferences

Functions:
    create_region_payment_analysis() - Main analysis function
    analyze_regional_performance() - Regional metrics
    analyze_payment_methods() - Payment analysis
    plot_region_revenue() - Regional revenue visualization
    plot_region_metrics() - Regional AOV and revenue per customer
    plot_payment_dashboard() - Payment method breakdown

Usage:
    from n2e_region_payment import create_region_payment_analysis
    results = create_region_payment_analysis(df)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List
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
_REGION_PAYMENT_CONFIG: Dict[str, Any] = _CONFIG.get('notebook2', {}).get('region_payment', {})

REGION_COLORS: List[str] = _REGION_PAYMENT_CONFIG.get('region_colors', [
    "#4231B1", "#3B9E3B", "#E4C743", "#d29b41",
    "#d83634", "#9f3dc9", "#40b9d4", "#bf7c39"
])
PAYMENT_COLORS: List[str] = _REGION_PAYMENT_CONFIG.get('payment_colors', [
    "#1D2ADC", "#EA731D", "#33a02c", "#e31a1c",
    "#9f3dc9", "#40b9d4", "#bf7c39"
])


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_regional_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate regional performance metrics with safe AOV calculation.

    Aggregates revenue, return rate, unique customers, order count, AOV,
    and revenue per customer by region. Division-by-zero is handled via
    numpy.where guards.

    Args:
        df: Transaction DataFrame with region, total_amount, returned,
            customer_id, and order_id columns

    Returns:
        pd.DataFrame: Regional metrics indexed by region, sorted by revenue
    """
    try:
        logger.info("Analyzing regional performance...")

        region_perf = df.groupby('region').agg({
            'total_amount': 'sum',
            'returned': 'mean',
            'customer_id': 'nunique',
            'order_id': 'nunique'
        }).sort_values('total_amount', ascending=False)

        region_perf['aov'] = np.where(
            region_perf['order_id'] > 0,
            region_perf['total_amount'] / region_perf['order_id'],
            0
        )

        region_perf['rev_per_customer'] = np.where(
            region_perf['customer_id'] > 0,
            region_perf['total_amount'] / region_perf['customer_id'],
            0
        )

        region_perf = region_perf.round(2)

        region_perf.columns = [
            'revenue', 'return_rate', 'customers',
            'orders', 'aov', 'revenue_per_customer'
        ]

        logger.info(f"Analyzed {len(region_perf)} regions")
        logger.info("Safely calculated AOV and revenue per customer")

        return region_perf

    except KeyError as e:
        logger.error(f"Missing required column in analyze_regional_performance: {e}")
        raise
    except Exception as e:
        logger.error(f"analyze_regional_performance failed: {e}")
        raise


def analyze_payment_methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze payment method distribution and performance.

    Args:
        df: Transaction DataFrame with payment_method, total_amount,
            order_id, and returned columns

    Returns:
        pd.DataFrame: Payment method metrics indexed by payment method,
            sorted by revenue descending
    """
    try:
        logger.info("Analyzing payment methods...")

        payment_perf = df.groupby('payment_method').agg({
            'total_amount': 'sum',
            'order_id': 'nunique',
            'returned': 'mean'
        }).sort_values('total_amount', ascending=False)

        payment_perf['pct_of_total'] = (
            payment_perf['total_amount'] / payment_perf['total_amount'].sum() * 100
        )

        payment_perf = payment_perf.round(2)

        payment_perf.columns = [
            'revenue', 'orders', 'return_rate', 'pct_of_total'
        ]

        logger.info(f"Analyzed {len(payment_perf)} payment methods")

        return payment_perf

    except KeyError as e:
        logger.error(f"Missing required column in analyze_payment_methods: {e}")
        raise
    except Exception as e:
        logger.error(f"analyze_payment_methods failed: {e}")
        raise


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_region_revenue(
    region_perf: pd.DataFrame,
    save: bool = True
) -> go.Figure:
    """
    Create regional revenue bar visualization.

    Args:
        region_perf: DataFrame indexed by region with a revenue column
            (output of analyze_regional_performance)
        save: Whether to save the figure to disk

    Returns:
        go.Figure: Plotly bar chart of revenue by region
    """
    try:
        fig = px.bar(
            region_perf.reset_index(),
            x='region',
            y='revenue',
            color='region',
            title='Revenue by Region',
            color_discrete_sequence=REGION_COLORS,
            labels={'revenue': 'Total Revenue', 'region': 'Region'}
        )

        fig.update_layout(
            template='plotly_white',
            showlegend=False,
            xaxis=dict(
                title='Region',
                tickfont=dict(size=11, family='Arial Black')
            ),
            yaxis=dict(
                title='Total Revenue',
                tickfont=dict(size=11, family='Arial Black')
            ),
            height=500
        )

        if save:
            save_plotly_figure(fig, "region_revenue", formats=['html'], logger=logger)

        return fig

    except Exception as e:
        logger.error(f"plot_region_revenue failed: {e}")
        raise


def plot_region_metrics(
    region_perf: pd.DataFrame,
    save: bool = True
) -> go.Figure:
    """
    Plot regional AOV and revenue per customer with category colors.

    Args:
        region_perf: DataFrame indexed by region with aov and
            revenue_per_customer columns (output of analyze_regional_performance)
        save: Whether to save the figure to disk

    Returns:
        go.Figure: Dual-axis bar and line chart
    """
    try:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        region_list = list(region_perf.index)
        colors = REGION_COLORS[:len(region_list)]

        fig.add_trace(
            go.Bar(
                x=region_list,
                y=region_perf['aov'],
                name='AOV',
                marker=dict(color=colors)
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=region_list,
                y=region_perf['revenue_per_customer'],
                name='Revenue per Customer',
                mode='lines+markers',
                line=dict(color="#000000", width=2),
                marker=dict(size=8)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=dict(
                text='Regional Metrics: AOV & Revenue per Customer',
                font=dict(size=14, family='Arial Black')
            ),
            xaxis=dict(title='Region'),
            yaxis=dict(title='Average Order Value'),
            yaxis2=dict(title='Revenue per Customer'),
            template='plotly_white',
            hovermode='x unified',
            height=500
        )

        if save:
            save_plotly_figure(fig, "region_metrics", formats=['html'], logger=logger)

        return fig

    except Exception as e:
        logger.error(f"plot_region_metrics failed: {e}")
        raise


def plot_payment_dashboard(payment_perf: pd.DataFrame, save: bool = True) -> go.Figure:
    """
    Combine payment distribution (pie) and comparison (bar+line) into one figure.

    Args:
        payment_perf: DataFrame indexed by payment_method with revenue,
            orders, and return_rate columns (output of analyze_payment_methods)
        save: Whether to save the figure to disk

    Returns:
        go.Figure: Side-by-side pie and dual-axis bar+line chart
    """
    try:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Revenue Distribution by Payment Method",
                "Revenue & Orders by Payment Method"
            ),
            specs=[[{"type": "domain"}, {"secondary_y": True}]],
            horizontal_spacing=0.15,
            vertical_spacing=0.15
        )

        pie_fig = px.pie(
            payment_perf.reset_index(),
            names='payment_method',
            values='revenue',
            color_discrete_sequence=PAYMENT_COLORS
        )
        pie_fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=12, family='Arial Black')
        )
        for trace in pie_fig.data:
            fig.add_trace(trace, row=1, col=1)

        fig.add_trace(
            go.Bar(
                x=payment_perf.index,
                y=payment_perf['revenue'],
                name='Revenue',
                marker=dict(color=PAYMENT_COLORS[:len(payment_perf)])
            ),
            row=1, col=2, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=payment_perf.index,
                y=payment_perf['orders'],
                name='Number of Orders',
                mode='lines+markers',
                line=dict(color='black', width=2),
                marker=dict(size=8)
            ),
            row=1, col=2, secondary_y=True
        )

        fig.update_layout(
            template='plotly_white',
            height=600,
            width=1400,
            hovermode='x unified',
            showlegend=True,
            title=dict(
                text="Payment Method Analysis Dashboard",
                font=dict(size=16, family='Arial Black'),
                x=0.5
            )
        )

        fig.update_xaxes(title_text="Payment Method", row=1, col=2)
        fig.update_yaxes(title_text="Total Revenue", row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Number of Orders", row=1, col=2, secondary_y=True)

        if save:
            save_plotly_figure(fig, "payment_dashboard", formats=['html'], logger=logger)

        return fig

    except Exception as e:
        logger.error(f"plot_payment_dashboard failed: {e}")
        raise


# ============================================================================
# Main Analysis Function
# ============================================================================

def create_region_payment_analysis(
    df: pd.DataFrame,
    save_figures: bool = True
) -> Dict:
    """
    Perform complete regional and payment method analysis.

    Runs regional performance aggregation, payment method analysis,
    and creates all visualizations.

    Args:
        df: Transaction DataFrame
        save_figures: Whether to save generated figures to disk

    Returns:
        dict with keys: region_performance, payment_performance, figures, metrics
    """
    logger.info("=" * 60)
    logger.info("REGION & PAYMENT METHOD INSIGHTS")
    logger.info("=" * 60)

    try:
        logger.info("STEP 1: REGIONAL PERFORMANCE ANALYSIS:")
        region_perf = analyze_regional_performance(df)
        logger.info("-" * 60)

        logger.info("STEP 2: PAYMENT METHOD ANALYSIS:")
        payment_perf = analyze_payment_methods(df)
        logger.info("-" * 60)

        logger.info("STEP 3: CREATING VISUALIZATIONS:")
        figures = {}

        figures['region_revenue'] = plot_region_revenue(region_perf, save=save_figures)
        figures['region_metrics'] = plot_region_metrics(region_perf, save=save_figures)
        figures['payment_dashboard'] = plot_payment_dashboard(
            payment_perf, save=save_figures
        )

        metrics = {
            'total_regions': len(region_perf),
            'top_region': region_perf.index[0],
            'top_region_revenue': region_perf['revenue'].iloc[0],
            'avg_aov_all_regions': region_perf['aov'].mean(),
            'top_payment_method': payment_perf.index[0],
            'top_payment_pct': payment_perf['pct_of_total'].iloc[0],
        }

        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total regions: {metrics['total_regions']}")
        logger.info(
            f"Top region: {metrics['top_region']} "
            f"(${metrics['top_region_revenue']:,.2f})"
        )
        logger.info(f"Average AOV across regions: ${metrics['avg_aov_all_regions']:.2f}")
        logger.info(
            f"Top payment method: {metrics['top_payment_method']} "
            f"({metrics['top_payment_pct']:.1f}%)"
        )
        logger.info("=" * 60)

        return {
            'region_performance': region_perf,
            'payment_performance': payment_perf,
            'figures': figures,
            'metrics': metrics
        }

    except Exception as e:
        logger.error(f"create_region_payment_analysis failed: {e}")
        raise


# ============================================================================
# Module Information
# ============================================================================

__all__ = [
    'REGION_COLORS',
    'PAYMENT_COLORS',
    'analyze_regional_performance',
    'analyze_payment_methods',
    'plot_region_revenue',
    'plot_region_metrics',
    'plot_payment_dashboard',
    'create_region_payment_analysis',
]

if __name__ == '__main__':
    print("n2e_region_payment.py - Region & Payment Analysis Module")
