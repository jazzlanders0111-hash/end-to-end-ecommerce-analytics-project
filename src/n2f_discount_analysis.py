# src/n2f_discount_analysis.py
"""
n2f_discount_analysis.py - Discount Effectiveness Analysis

FIX — added analyze_discount_frequency():
  The original analysis only tested whether discounts change transaction VALUE
  (Mann-Whitney on total_amount). That is an incomplete picture: discounts might
  lower per-order value but increase how often customers purchase, making the
  net revenue per customer positive. This function tests exactly that by
  comparing orders-per-customer between customers who ever bought at a discount
  vs those who never did, and also compares average lifetime revenue per customer.

Works with fixed discount levels: 0%, 5%, 10%, 15%, 20%, 30%
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as _stats
from typing import Dict, Any, List
import logging
from n2a_utils import setup_logger, save_plotly_figure, print_section_header, get_config

# Setup logger
logger = setup_logger(__name__)

# Load configuration
_CONFIG = get_config()
_DISCOUNT_CONFIG: Dict[str, Any] = _CONFIG.get('notebook2', {}).get('discount_analysis', {})

DISCOUNT_COLORS: List[str] = _DISCOUNT_CONFIG.get('colors', [
    "#2E7D32", "#7CB342", "#FDD835", "#FB8C00", "#F4511E", "#C62828"
])


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_discount_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate revenue, orders, quantity, and profit margin by exact discount level.

    Discount levels are treated as discrete (not continuous ranges) because they
    are fixed at 0/5/10/15/20/30%.

    Args:
        df: Transaction DataFrame with discount, total_amount, order_id,
            quantity, and profit_margin columns

    Returns:
        pd.DataFrame: Metrics per discount tier sorted by configured discount order
    """
    try:
        logger.info("Analysing discount levels...")

        df_d = df.copy()
        df_d['discount_pct'] = (df_d['discount'] * 100).round(0).astype(int)

        discount_labels = _DISCOUNT_CONFIG.get('discount_levels', {
            0: '0%', 5: '5%', 10: '10%', 15: '15%', 20: '20%', 30: '30%'
        })
        df_d['discount_tier'] = df_d['discount_pct'].map(discount_labels)

        perf = df_d.groupby('discount_tier', observed=True).agg(
            total_amount=('total_amount', 'sum'),
            order_id=('order_id', 'nunique'),
            quantity=('quantity', 'sum'),
            profit_margin=('profit_margin', 'mean')
        ).reset_index()

        order = _DISCOUNT_CONFIG.get(
            'discount_order', ['0%', '5%', '10%', '15%', '20%', '30%']
        )
        perf['discount_tier'] = pd.Categorical(
            perf['discount_tier'], categories=order, ordered=True
        )
        perf = perf.sort_values('discount_tier').round(2)
        perf.columns = ['discount_tier', 'revenue', 'orders', 'quantity', 'avg_profit_margin']

        logger.info(f"Analysed {len(perf)} discount levels")
        return perf

    except KeyError as e:
        logger.error(f"Missing required column in analyze_discount_tiers: {e}")
        raise
    except Exception as e:
        logger.error(f"analyze_discount_tiers failed: {e}")
        raise


def analyze_discount_frequency(df: pd.DataFrame) -> Dict:
    """
    Test whether discounts change how often customers purchase.

    Logic:
      1. Classify each customer: did they EVER buy at a discount?
      2. Compare orders-per-customer between the two groups (Mann-Whitney U).
      3. Compare average lifetime revenue per customer.

    This directly answers the question the value-only analysis cannot:
    "Do discounts bring customers back more often, and does that offset
    the lower per-transaction value?"

    Args:
        df: Transaction DataFrame with customer_id, order_id, discount,
            and total_amount columns

    Returns:
        dict with frequency metrics, Mann-Whitney result, and interpretation
    """
    try:
        logger.info("Analysing discount impact on customer order frequency...")

        ever_discounted = (
            df.groupby('customer_id')['discount']
            .apply(lambda x: (x > 0).any())
            .rename('ever_discounted')
        )

        customer_summary = df.groupby('customer_id').agg(
            order_count=('order_id', 'nunique'),
            total_revenue=('total_amount', 'sum')
        ).join(ever_discounted)

        disc = customer_summary[customer_summary['ever_discounted']]
        nodisc = customer_summary[~customer_summary['ever_discounted']]

        u_stat, p_value = _stats.mannwhitneyu(
            disc['order_count'], nodisc['order_count'], alternative='two-sided'
        )
        significant = bool(p_value < 0.05)

        med_disc = float(disc['order_count'].median())
        med_nodisc = float(nodisc['order_count'].median())
        freq_lift = (med_disc / med_nodisc - 1) * 100 if med_nodisc > 0 else 0.0

        avg_rev_disc = float(disc['total_revenue'].mean())
        avg_rev_nodisc = float(nodisc['total_revenue'].mean())
        rev_lift = (avg_rev_disc / avg_rev_nodisc - 1) * 100 if avg_rev_nodisc > 0 else 0.0

        if significant:
            direction = "MORE" if freq_lift > 0 else "FEWER"
            interpretation = (
                f"Discount customers place significantly {direction} orders per customer "
                f"(median {med_disc:.1f} vs {med_nodisc:.1f}, p={p_value:.4f}). "
                f"Avg lifetime revenue: discount=${avg_rev_disc:,.0f} vs "
                f"no-discount=${avg_rev_nodisc:,.0f} ({rev_lift:+.1f}%)."
            )
        else:
            interpretation = (
                f"No significant difference in order frequency between discount and "
                f"no-discount customers (p={p_value:.4f}, median orders: "
                f"discount={med_disc:.1f} vs no-discount={med_nodisc:.1f}). "
                f"Discounts do not drive repeat purchasing behaviour in this dataset."
            )

        logger.info(interpretation)

        return {
            'n_discount_customers':              len(disc),
            'n_no_discount_customers':           len(nodisc),
            'avg_orders_discount':               float(disc['order_count'].mean()),
            'avg_orders_no_discount':            float(nodisc['order_count'].mean()),
            'median_orders_discount':            med_disc,
            'median_orders_no_discount':         med_nodisc,
            'avg_revenue_discount_customer':     avg_rev_disc,
            'avg_revenue_no_discount_customer':  avg_rev_nodisc,
            'u_statistic':                       float(u_stat),
            'p_value':                           float(p_value),
            'significant':                       significant,
            'frequency_lift_pct':                freq_lift,
            'revenue_per_customer_lift_pct':     rev_lift,
            'interpretation':                    interpretation,
        }

    except KeyError as e:
        logger.error(f"Missing required column in analyze_discount_frequency: {e}")
        raise
    except Exception as e:
        logger.error(f"analyze_discount_frequency failed: {e}")
        raise


# ============================================================================
# Visualization
# ============================================================================

def plot_discount_comparison(discount_perf: pd.DataFrame, save: bool = True) -> go.Figure:
    """
    Revenue & orders by discount level (left) + profit margin (right).

    Args:
        discount_perf: DataFrame with discount_tier, revenue, orders, and
            avg_profit_margin columns (output of analyze_discount_tiers)
        save: Whether to save the figure to disk

    Returns:
        go.Figure: Side-by-side dual-axis chart and margin bar chart
    """
    try:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Revenue & Orders by Discount Level',
                'Profit Margin by Discount Level'
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": False}]]
        )

        fig.add_trace(go.Bar(
            x=discount_perf['discount_tier'],
            y=discount_perf['revenue'],
            name='Revenue',
            marker=dict(color=DISCOUNT_COLORS[:len(discount_perf)]),
            showlegend=True
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=discount_perf['discount_tier'],
            y=discount_perf['orders'],
            name='Orders',
            mode='lines+markers',
            line=dict(color='#333333', width=2),
            marker=dict(size=8),
            showlegend=True
        ), row=1, col=1, secondary_y=True)

        fig.add_trace(go.Bar(
            x=discount_perf['discount_tier'],
            y=discount_perf['avg_profit_margin'],
            name='Avg Profit Margin',
            marker=dict(
                color=discount_perf['avg_profit_margin'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Margin %", x=1.15),
                line=dict(color='black', width=1)
            ),
            text=discount_perf['avg_profit_margin'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            showlegend=False
        ), row=1, col=2)

        fig.update_xaxes(
            title_text="Discount Level", row=1, col=1,
            tickfont=dict(size=10, family='Arial Black')
        )
        fig.update_xaxes(
            title_text="Discount Level", row=1, col=2,
            tickfont=dict(size=10, family='Arial Black')
        )
        fig.update_yaxes(title_text="Revenue ($)",       row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Orders",            row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Profit Margin (%)", row=1, col=2)

        fig.update_layout(
            title=dict(
                text='Comprehensive Discount Level Comparison',
                font=dict(size=14, family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )

        if save:
            save_plotly_figure(fig, "discount_comparison", formats=['html'], logger=logger)

        return fig

    except Exception as e:
        logger.error(f"plot_discount_comparison failed: {e}")
        raise


# ============================================================================
# Main
# ============================================================================

def create_discount_analysis(df: pd.DataFrame, save_figures: bool = True) -> Dict:
    """
    Full discount effectiveness analysis.

    Runs discount tier aggregation, customer-level frequency test, and
    creates the comprehensive visualization.

    Args:
        df: Transaction DataFrame
        save_figures: Whether to save generated figures to disk

    Returns:
        dict with keys: discount_performance, frequency_analysis, figures, metrics
    """
    logger.info("=" * 60)
    logger.info("DISCOUNT EFFECTIVENESS ANALYSIS")
    logger.info("=" * 60)

    try:
        logger.info("STEP 1: DISCOUNT LEVEL ANALYSIS:")
        discount_perf = analyze_discount_tiers(df)
        logger.info("-" * 60)

        logger.info("STEP 2: CREATING COMPREHENSIVE VISUALIZATION:")
        figures = {
            'comprehensive': plot_discount_comparison(discount_perf, save=save_figures)
        }
        logger.info("-" * 60)

        logger.info("STEP 3: DISCOUNT FREQUENCY ANALYSIS (customer-level):")
        frequency_result = analyze_discount_frequency(df)
        logger.info("-" * 60)

        no_disc_mask = discount_perf['discount_tier'] == '0%'
        if no_disc_mask.any():
            no_disc_rev = float(discount_perf[no_disc_mask]['revenue'].iloc[0])
            no_disc_margin = float(discount_perf[no_disc_mask]['avg_profit_margin'].iloc[0])
        else:
            no_disc_rev = 0.0
            no_disc_margin = 0.0
            logger.warning("⚠ No '0%' tier found in data")

        total_disc_rev = float(discount_perf[~no_disc_mask]['revenue'].sum())
        total_rev = float(discount_perf['revenue'].sum())

        metrics = {
            'total_levels':                    len(discount_perf),
            'no_discount_revenue':             no_disc_rev,
            'discounted_revenue':              total_disc_rev,
            'discount_pct_of_total':           (
                total_disc_rev / total_rev * 100 if total_rev > 0 else 0.0
            ),
            'avg_margin_no_discount':          no_disc_margin,
            'total_revenue':                   total_rev,
            'discount_levels':                 discount_perf['discount_tier'].tolist(),
            'avg_orders_discount_customer':    frequency_result['avg_orders_discount'],
            'avg_orders_no_discount_customer': frequency_result['avg_orders_no_discount'],
            'frequency_lift_pct':              frequency_result['frequency_lift_pct'],
            'revenue_per_customer_lift_pct':   frequency_result['revenue_per_customer_lift_pct'],
            'frequency_significant':           frequency_result['significant'],
            'frequency_interpretation':        frequency_result['interpretation'],
        }

        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Non-discounted revenue:  ${metrics['no_discount_revenue']:,.2f}")
        logger.info(f"Discounted revenue:      ${metrics['discounted_revenue']:,.2f}")
        logger.info(f"Discount share of total: {metrics['discount_pct_of_total']:.1f}%")
        logger.info(f"Avg margin (0%):         {metrics['avg_margin_no_discount']:.2f}%")
        logger.info(f"Frequency test:          {metrics['frequency_interpretation']}")
        logger.info("=" * 60)

        return {
            'discount_performance': discount_perf,
            'frequency_analysis':   frequency_result,
            'figures':              figures,
            'metrics':              metrics,
        }

    except Exception as e:
        logger.error(f"create_discount_analysis failed: {e}")
        raise


# ============================================================================
# Module Information
# ============================================================================

__all__ = [
    'DISCOUNT_COLORS',
    'analyze_discount_tiers',
    'analyze_discount_frequency',
    'plot_discount_comparison',
    'create_discount_analysis',
]

if __name__ == '__main__':
    print("n2f_discount_analysis.py - import and call create_discount_analysis(df)")
