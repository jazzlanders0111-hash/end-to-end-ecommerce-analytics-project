# src/n2g_summary_dashboard_enhanced.py
"""
n2g_summary_dashboard_enhanced.py - Enhanced Executive Dashboard with Forecasting

This module creates an enhanced executive dashboard for the e-commerce analytics project, integrating key insights from time trends, category performance, regional analysis, and discount impact. The dashboard features 6 panels:
1. Revenue Trend & Forecast: Monthly revenue with YoY growth and forecast overlay.
2. Top Categories by Revenue: Bar chart of top-performing categories.
3. Regional Performance: Bar chart of revenue by region.
4. Discount Impact Analysis: Bar + line chart showing revenue and profit margin by discount tier.
5. Payment Method Distribution: Pie chart of revenue by payment method.
6. Key Performance Indicators: Table summarizing critical metrics.

Functions:
- generate_summary_dashboard(): Main function to create and save the dashboard figure.
"""

import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from n2a_utils import (
    setup_logger,
    save_plotly_figure,
    get_figures_dir,
    print_section_header,
    get_config,
    COLORS
)

logger = setup_logger(__name__)

_CONFIG        = get_config()
_NB2_CONFIG    = _CONFIG.get('notebook2', {})
_FORECAST_CFG  = _NB2_CONFIG.get('forecasting', {})
_TIME_CONFIG   = _NB2_CONFIG.get('time_trends', {})
_CATEGORY_CONFIG = _NB2_CONFIG.get('category_analysis', {})
_REGION_CONFIG = _NB2_CONFIG.get('region_payment', {})
_DISCOUNT_CONFIG = _NB2_CONFIG.get('discount_analysis', {})

TIME_COLORS     = _TIME_CONFIG.get('colors',          ["#1B1F5E", "#EA731D"])
CATEGORY_COLORS = _CATEGORY_CONFIG.get('colors',      ["#AECCDB", "#3274A1", "#B3D495", "#F4D03F", "#E59866", "#85C1E9", "#A9DFBF"])
REGION_COLORS   = _REGION_CONFIG.get('region_colors', ["#4231B1", "#3B9E3B", "#E4C743", "#E07B39", "#C0392B"])
PAYMENT_COLORS  = _REGION_CONFIG.get('payment_colors',["#1D2ADC", "#EA731D", "#33a02c", "#9B59B6", "#E74C3C", "#1ABC9C"])
DISCOUNT_COLORS = _DISCOUNT_CONFIG.get('colors',      ["#2E7D32", "#7CB342", "#FDD835", "#FB8C00", "#F4511E", "#C62828"])


# ============================================================================
# Helper: pull verdicts safely (graceful fallback if stats not run)
# ============================================================================

def _get_verdict(all_results: Dict, domain: str) -> str:
    """Return dashboard_note for a domain from stats_results verdicts, or '' if absent."""
    stats = all_results.get('statistical_validation', {})
    return stats.get('verdicts', {}).get(domain, {}).get('dashboard_note', '')


def _verdict_significant(all_results: Dict, domain: str) -> Optional[bool]:
    stats = all_results.get('statistical_validation', {})
    v = stats.get('verdicts', {}).get(domain, {})
    return v.get('significant', None)


# ============================================================================
# Executive Sales Dashboard (6 panels)
# ============================================================================

def create_executive_dashboard(
    all_results: Dict[str, Any],
    save: bool = True
) -> go.Figure:
    """
    6-panel Executive Sales Dashboard.

    Layout:
    ┌──────────────────────────┬──────────────────────────┐
    │ Revenue Trend & Forecast │ Top Categories by Revenue│
    ├──────────────────────────┼──────────────────────────┤
    │ Regional Performance     │ Discount Impact Analysis │
    ├──────────────────────────┼──────────────────────────┤
    │ Payment Distribution     │ Key Performance Indicators│
    └──────────────────────────┴──────────────────────────┘

    Statistical verdicts from all_results['statistical_validation']['verdicts']
    are injected as panel annotations so each chart is self-qualifying.
    """
    logger.info("Creating Executive Sales Dashboard...")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Revenue Trend & Forecast',
            'Top Categories by Revenue',
            'Regional Performance',
            'Discount Impact Analysis',
            'Payment Method Distribution',
            'Key Performance Indicators'
        ),
        specs=[
            [{"secondary_y": True}, {"type": "bar"}],
            [{"type": "bar"},       {"secondary_y": True}],
            [{"type": "domain"},    {"type": "table"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
        row_heights=[0.35, 0.35, 0.30]
    )

    # ── Panel 1: Revenue Trend & Forecast ────────────────────────────────────
    if 'time_trends' in all_results and 'forecasting' in all_results:
        time_res     = all_results['time_trends']
        fc_res       = all_results['forecasting']
        monthly_sales = time_res.get('monthly_sales')
        best_model   = fc_res.get('best_model', {})
        weekly_sales = fc_res.get('weekly_sales')
        fc_metrics   = fc_res.get('metrics', {})

        if monthly_sales is not None:
            # Historical monthly revenue
            fig.add_trace(go.Scatter(
                x=monthly_sales['year_month'], y=monthly_sales['total_amount'],
                mode='lines+markers', name='Monthly Revenue',
                line=dict(color=TIME_COLORS[0], width=2.5), marker=dict(size=6),
                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>',
            ), row=1, col=1, secondary_y=False)

            # YoY growth (skip NaN — first 12 months have no baseline)
            valid_yoy = monthly_sales[monthly_sales['yoy_growth'].notna()]
            fig.add_trace(go.Scatter(
                x=valid_yoy['year_month'], y=valid_yoy['yoy_growth'],
                mode='lines+markers', name='YoY Growth %',
                line=dict(color=TIME_COLORS[1], width=2, dash='dot'), marker=dict(size=5),
                hovertemplate='<b>%{x}</b><br>YoY Growth: %{y:.1f}%<extra></extra>',
            ), row=1, col=1, secondary_y=True)

            # Forecast overlay — annotate with WEEKLY unit label to prevent scale confusion
            if best_model and 'forecast' in best_model and weekly_sales is not None:
                train_size = fc_metrics.get('train_size', 0)
                test_size  = fc_metrics.get('test_size',  0)
                weekly_avg = fc_metrics.get('weekly_avg_revenue', 0)

                if train_size > 0 and test_size > 0:
                    last_train_date  = weekly_sales['date'].iloc[train_size - 1]
                    last_train_value = weekly_sales['revenue'].iloc[train_size - 1]
                    test_dates       = weekly_sales['date'].iloc[train_size:train_size + test_size]
                    forecast_values  = best_model['forecast']

                    fx = pd.concat([pd.Series([last_train_date]), test_dates])
                    fy = pd.concat([pd.Series([last_train_value]), pd.Series(forecast_values)])

                    fig.add_trace(go.Scatter(
                        x=fx, y=fy, mode='lines', name='Forecast (weekly)',
                        line=dict(color='#F18F01', width=2.5, dash='dash'),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Forecast (weekly): $%{y:,.0f}<extra></extra>',
                    ), row=1, col=1, secondary_y=False)

                    fig.add_vline(
                        x=last_train_date, line_dash="dot", line_color="gray", opacity=0.5,
                        row=1, col=1  # type: ignore
                    )

                    # Scale annotation — prevents misreading weekly values as monthly collapse
                    avg_wk = float(np.mean(forecast_values))
                    monthly_eq = fc_metrics.get('monthly_equivalent', avg_wk * 4.35)
                    fig.add_annotation(
                        x=fx.iloc[-1], y=avg_wk,
                        text=f"~${avg_wk/1000:.0f}K/wk<br>(≈${monthly_eq/1000:.0f}K/mo equiv.)",
                        showarrow=True, arrowhead=2, arrowcolor="#F18F01",
                        font=dict(size=10, color="#F18F01", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.9)", bordercolor="#F18F01", borderwidth=1,
                        ax=-70, ay=-35,
                        row=1, col=1  # type: ignore
                    )

        # Stat verdict annotation for trend panel
        trend_note = _get_verdict(all_results, 'time_trend')
        if trend_note and _verdict_significant(all_results, 'time_trend') is False:
            fig.add_annotation(
                xref='paper', yref='paper', x=0.02, y=0.995,
                text="⚠ Trend not statistically significant — see validation",
                showarrow=False, font=dict(size=9, color='#888888'),
                bgcolor='rgba(255,255,240,0.85)', bordercolor='#cccc00', borderwidth=1,
                xanchor='left', yanchor='top'
            )

    # ── Panel 2: Category Revenue ─────────────────────────────────────────────
    if 'category' in all_results:
        cat_perf = all_results['category'].get('category_performance')
        if cat_perf is not None:
            top_cats = cat_perf.head(7)
            fig.add_trace(go.Bar(
                x=top_cats.index, y=top_cats['revenue'],
                marker=dict(color=CATEGORY_COLORS[:len(top_cats)], line=dict(color='black', width=1)),
                text=[f"${v/1e6:.2f}M" for v in top_cats['revenue']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>',
                showlegend=False
            ), row=1, col=2)

    # ── Panel 3: Regional Performance ────────────────────────────────────────
    if 'region_payment' in all_results:
        region_perf = all_results['region_payment'].get('region_performance')
        if region_perf is not None:
            regions  = list(region_perf.index)
            revenues = region_perf['revenue'].values
            fig.add_trace(go.Bar(
                x=regions, y=revenues,
                marker=dict(color=REGION_COLORS[:len(regions)], line=dict(color='black', width=1)),
                text=[f"${v/1e6:.2f}M" for v in revenues],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>',
                showlegend=False
            ), row=2, col=1)

    # Regional stat verdict annotation (most important fix — panel showed rankings as real)
    region_note = _get_verdict(all_results, 'regional')
    if region_note and _verdict_significant(all_results, 'regional') is False:
        fig.add_annotation(
            xref='paper', yref='paper', x=0.02, y=0.625,
            text="⚠ Regional differences NOT statistically significant — rankings are noise",
            showarrow=False, font=dict(size=9, color='#a00000'),
            bgcolor='rgba(255,240,240,0.90)', bordercolor='#cc0000', borderwidth=1,
            xanchor='left', yanchor='top'
        )

    # ── Panel 4: Discount Impact ──────────────────────────────────────────────
    if 'discount' in all_results:
        disc_perf = all_results['discount'].get('discount_performance')
        if disc_perf is not None:
            fig.add_trace(go.Bar(
                x=disc_perf['discount_tier'], y=disc_perf['revenue'],
                name='Revenue by Discount',
                marker=dict(color=DISCOUNT_COLORS[:len(disc_perf)], line=dict(color='black', width=1)),
                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>',
                showlegend=True
            ), row=2, col=2, secondary_y=False)

            fig.add_trace(go.Scatter(
                x=disc_perf['discount_tier'], y=disc_perf['avg_profit_margin'],
                name='Profit Margin %', mode='lines+markers',
                line=dict(color='#000000', width=2.5), marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Margin: %{y:.1f}%<extra></extra>',
                showlegend=True
            ), row=2, col=2, secondary_y=True)

    # Discount verdict annotation
    disc_note = _get_verdict(all_results, 'discount')
    if disc_note and _verdict_significant(all_results, 'discount') is True:
        # Significant — show a reminder about frequency analysis
        freq_interp = ''
        if 'discount' in all_results:
            freq_interp = all_results['discount'].get('metrics', {}).get('frequency_interpretation', '')
        annotation_text = "✓ Value effect significant — check frequency analysis for net impact"
        if freq_interp:
            sig = all_results['discount'].get('metrics', {}).get('frequency_significant', False)
            if not sig:
                annotation_text = "✓ Value ↓ (stat. sig.) | Frequency: no change — net effect is negative"
            else:
                annotation_text = "✓ Value ↓ (stat. sig.) | Frequency: ↑ (stat. sig.) — net TBD"
        fig.add_annotation(
            xref='paper', yref='paper', x=0.52, y=0.625,
            text=annotation_text,
            showarrow=False, font=dict(size=9, color='#333333'),
            bgcolor='rgba(255,255,240,0.90)', bordercolor='#999900', borderwidth=1,
            xanchor='left', yanchor='top'
        )

    # ── Panel 5: Payment Distribution ────────────────────────────────────────
    if 'region_payment' in all_results:
        payment_perf = all_results['region_payment'].get('payment_performance')
        if payment_perf is not None:
            fig.add_trace(go.Pie(
                labels=payment_perf.index, values=payment_perf['revenue'],
                marker=dict(colors=PAYMENT_COLORS[:len(payment_perf)]),
                textposition='inside', textinfo='percent+label',
                textfont=dict(size=11, family='Arial Black'),
                showlegend=False
            ), row=3, col=1)

    # ── Panel 6: KPI Table ────────────────────────────────────────────────────
    kpi_data = _extract_kpi_data(all_results)
    fig.add_trace(go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='#1B1F5E',
            font=dict(color='white', size=12, family='Arial Black'),
            align='left', height=30
        ),
        cells=dict(
            values=[list(kpi_data.keys()), list(kpi_data.values())],
            fill_color=['white', '#f0f0f0'],
            font=dict(size=11, family='Arial'),
            align='left', height=28
        )
    ), row=3, col=2)

    # ── Axes labels ───────────────────────────────────────────────────────────
    fig.update_xaxes(title_text="Month",          row=1, col=1, tickfont=dict(size=10, family='Arial Black'))
    fig.update_yaxes(title_text="Revenue ($)",    row=1, col=1, secondary_y=False, tickfont=dict(size=10))
    fig.update_yaxes(title_text="YoY Growth (%)", row=1, col=1, secondary_y=True,  tickfont=dict(size=10))
    fig.update_xaxes(title_text="Category",       row=1, col=2, tickfont=dict(size=10, family='Arial Black'))
    fig.update_yaxes(title_text="Revenue ($)",    row=1, col=2, tickfont=dict(size=10))
    fig.update_xaxes(title_text="Region",         row=2, col=1, tickfont=dict(size=10, family='Arial Black'))
    fig.update_yaxes(title_text="Revenue ($)",    row=2, col=1, tickfont=dict(size=10))
    fig.update_xaxes(title_text="Discount Level", row=2, col=2, tickfont=dict(size=10, family='Arial Black'))
    fig.update_yaxes(title_text="Revenue ($)",    row=2, col=2, secondary_y=False, tickfont=dict(size=10))
    fig.update_yaxes(title_text="Margin (%)",     row=2, col=2, secondary_y=True,  tickfont=dict(size=10))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text='Executive Sales Dashboard',
            font=dict(size=20, family='Arial Black', color='#000000'),
            x=0.5, xanchor='center', y=0.97, yanchor='top',
        ),
        template='plotly_white',
        height=1400,
        margin=dict(t=180, b=40, l=60, r=60),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=1.055, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1,
            font=dict(size=11, family='Arial'), itemsizing='constant', tracegroupgap=5
        ),
        hovermode='closest',
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )

    # Subtitle
    _SUBTITLE = 'Comprehensive Business Performance Overview — Annotations show statistical significance'
    fig.add_annotation(
        text=_SUBTITLE, xref='paper', yref='paper',
        x=0.5, y=1.075, xanchor='center', yanchor='bottom',
        showarrow=False, font=dict(size=12, family='Arial', color='#444444')
    )

    # Fix subplot title fonts (skip our subtitle)
    for ann in fig.layout.annotations:
        if (ann.showarrow is None or ann.showarrow is False) and ann.text != _SUBTITLE:
            if not ann.text.startswith('⚠') and not ann.text.startswith('✓'):
                ann.font.size   = 13
                ann.font.family = 'Arial Black'
                ann.font.color  = '#000000'
                ann.yshift      = 10

    if save:
        save_plotly_figure(fig, "executive_dashboard", formats=['html'], logger=logger)

    logger.info("Executive Sales Dashboard created successfully")
    return fig


# ============================================================================
# KPI Cards
# ============================================================================

def create_kpi_cards(all_results: Dict[str, Any], save: bool = True) -> go.Figure:
    """4 KPI indicator cards: Monthly Revenue | Top Category | Forecast MAPE | Discount %"""
    logger.info("Creating KPI Cards...")

    kpis = _extract_kpi_values(all_results)

    _cx = [(0.00, 0.22), (0.26, 0.48), (0.52, 0.74), (0.78, 1.00)]
    _y  = [0.05, 0.80]

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=kpis['monthly_revenue'],
        number=dict(prefix="$", suffix="K", valueformat=".1f",
                    font=dict(size=46, family='Arial Black', color='#1B1F5E')),
        delta=dict(
            reference=kpis['monthly_revenue'] / (1 + kpis['yoy_growth'] / 100),
            relative=True, valueformat=".1%",
            increasing=dict(color="#2E7D32"), decreasing=dict(color="#C62828"),
            font=dict(size=14)
        ),
        title=dict(text=f"<b>Monthly Revenue</b><br>"
                        f"<span style='font-size:13px;color:#666'>YoY: {kpis['yoy_growth']:+.1f}%</span>",
                   font=dict(size=14, color='#000000')),
        domain=dict(x=list(_cx[0]), y=_y)
    ))

    fig.add_trace(go.Indicator(
        mode="number",
        value=kpis['top_category_revenue'] / 1e6,
        number=dict(prefix="$", suffix="M", valueformat=".2f",
                    font=dict(size=46, family='Arial Black', color='#2E7D32')),
        title=dict(text=f"<b>Top Category</b><br>"
                        f"<span style='font-size:13px;color:#333'>{kpis['top_category']}</span>",
                   font=dict(size=14, color='#000000')),
        domain=dict(x=list(_cx[1]), y=_y)
    ))

    quality_color = '#F57C00' if kpis['forecast_quality'] == 'Good' else '#2E7D32'
    fig.add_trace(go.Indicator(
        mode="number",
        value=kpis['forecast_mape'],
        number=dict(suffix="%", valueformat=".2f",
                    font=dict(size=46, family='Arial Black', color=quality_color)),
        title=dict(text=f"<b>Forecast Quality</b><br>"
                        f"<span style='font-size:13px;color:#666'>{kpis['forecast_quality']} · MAPE</span>",
                   font=dict(size=14, color='#000000')),
        domain=dict(x=list(_cx[2]), y=_y)
    ))

    fig.add_trace(go.Indicator(
        mode="number",
        value=kpis['discount_impact'],
        number=dict(suffix="%", valueformat=".1f",
                    font=dict(size=46, family='Arial Black', color='#C62828')),
        title=dict(text="<b>Discount Impact</b><br>"
                        "<span style='font-size:13px;color:#666'>of total revenue</span>",
                   font=dict(size=14, color='#000000')),
        domain=dict(x=list(_cx[3]), y=_y)
    ))

    fig.update_layout(
        title=dict(text='Key Performance Indicators',
                   font=dict(size=18, family='Arial Black', color='#000000'),
                   x=0.5, xanchor='center', y=0.87, yanchor='top'),
        template='plotly_white', height=250,
        margin=dict(t=55, b=10, l=10, r=10), showlegend=False
    )

    if save:
        save_plotly_figure(fig, "kpi_cards", formats=['html'], logger=logger)

    logger.info("KPI Cards created successfully")
    return fig


# ============================================================================
# Performance Matrix
# ============================================================================

def create_performance_matrix(all_results: Dict[str, Any], save: bool = True) -> go.Figure:
    """Bar chart: revenue by category then by region side by side."""
    logger.info("Creating Performance Matrix...")

    labels = []; values = []; colors = []

    if 'category' in all_results:
        cp = all_results['category'].get('category_performance')
        if cp is not None:
            labels.extend(list(cp.index))
            values.extend(cp['revenue'].values.tolist())
            colors.extend(CATEGORY_COLORS[:len(cp)])

    if 'region_payment' in all_results:
        rp = all_results['region_payment'].get('region_performance')
        if rp is not None:
            labels.extend(list(rp.index))
            values.extend(rp['revenue'].values.tolist())
            colors.extend(REGION_COLORS[:len(rp)])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(color='black', width=1)),
        text=[f"${v/1e6:.2f}M" if v >= 1e6 else f"${v/1e3:.0f}K" for v in values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>',
        showlegend=False
    ))

    # Regional significance note on the matrix too
    region_note = _get_verdict(all_results, 'regional')
    if region_note and _verdict_significant(all_results, 'regional') is False:
        fig.add_annotation(
            xref='paper', yref='paper', x=0.5, y=1.07,
            text="⚠ Regional bars: differences NOT statistically significant (p=0.157) — rankings reflect volume, not order value",
            showarrow=False, font=dict(size=10, color='#a00000'),
            bgcolor='rgba(255,240,240,0.90)', bordercolor='#cc0000', borderwidth=1,
            xanchor='center', yanchor='bottom'
        )

    fig.update_layout(
        title=dict(
            text='Performance Matrix<br><sub>Revenue Distribution — Categories (left) & Regions (right)</sub>',
            font=dict(size=18, family='Arial Black', color='#000000'),
            x=0.5, xanchor='center'
        ),
        xaxis=dict(title='Category / Region', tickfont=dict(size=11, family='Arial Black')),
        yaxis=dict(title='Revenue ($)',        tickfont=dict(size=11, family='Arial Black')),
        template='plotly_white', height=600,
        hovermode='x', hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )

    if save:
        save_plotly_figure(fig, "performance_matrix", formats=['html'], logger=logger)

    logger.info("Performance Matrix created successfully")
    return fig


# ============================================================================
# Legacy Compatibility
# ============================================================================

def create_revenue_forecast_combined(
    monthly_sales: pd.DataFrame,
    forecast_results: Dict[str, Any],
    save: bool = True
) -> go.Figure:
    """Legacy function — monthly trend + forecast vs actual side by side."""
    logger.info("Creating combined revenue trend and forecast (legacy)...")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Monthly Revenue Trend with YoY Growth', 'Revenue Forecast vs Actual'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
        horizontal_spacing=0.12
    )

    fig.add_trace(go.Scatter(
        x=monthly_sales['year_month'], y=monthly_sales['total_amount'],
        mode='lines+markers', name='Revenue',
        line=dict(color=TIME_COLORS[0], width=2.5), marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1, secondary_y=False)

    valid_yoy = monthly_sales[monthly_sales['yoy_growth'].notna()]
    fig.add_trace(go.Scatter(
        x=valid_yoy['year_month'], y=valid_yoy['yoy_growth'],
        mode='lines+markers', name='YoY Growth %',
        line=dict(color=TIME_COLORS[1], width=2), marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>YoY Growth: %{y:.1f}%<extra></extra>'
    ), row=1, col=1, secondary_y=True)

    if 'best_model' in forecast_results:
        best    = forecast_results['best_model']
        ws      = forecast_results.get('weekly_sales')
        metrics = forecast_results.get('metrics', {})
        if ws is not None:
            ts = metrics.get('train_size', 0)
            te = metrics.get('test_size',  0)
            test_dates = ws['date'].iloc[ts:ts+te]
            actual     = ws['revenue'].iloc[ts:ts+te]
            fc_vals    = best.get('forecast')
            fig.add_trace(go.Scatter(
                x=test_dates, y=actual, mode='lines+markers', name='Actual',
                line=dict(color='#2E86AB', width=2.5),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual: $%{y:,.0f}<extra></extra>'
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=test_dates, y=fc_vals, mode='lines+markers', name='Forecast',
                line=dict(color='#F18F01', width=2, dash='dash'),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Forecast (weekly): $%{y:,.0f}<extra></extra>'
            ), row=1, col=2)

    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($)",    row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="YoY Growth (%)", row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Date (weekly)",  row=1, col=2)
    fig.update_yaxes(title_text="Weekly Revenue ($)", row=1, col=2)

    fig.update_layout(
        title=dict(text='Revenue Trend & Forecast Analysis',
                   font=dict(size=16, family='Arial Black'), x=0.5),
        template='plotly_white', height=600, showlegend=True,
        hovermode='closest', hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )
    if save:
        save_plotly_figure(fig, "revenue_forecast_combined", formats=['html'], logger=logger)
    return fig


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_kpi_data(all_results: Dict[str, Any]) -> Dict[str, str]:
    """KPI data for table panel (Panel 6)."""
    kpi = {}

    if 'time_trends' in all_results:
        m = all_results['time_trends'].get('metrics', {})
        kpi['Avg Monthly Revenue'] = f"${m.get('avg_monthly_revenue',0)/1000:.0f}K"
        kpi['Latest YoY Growth']   = f"{m.get('latest_yoy_growth',0):+.1f}%"

    if 'category' in all_results:
        m = all_results['category'].get('metrics', {})
        kpi['Top Category'] = m.get('top_category', 'N/A')
        kpi['Avg Return Rate'] = f"{m.get('avg_return_rate',0):.1%}"

    if 'region_payment' in all_results:
        m = all_results['region_payment'].get('metrics', {})
        kpi['Top Region (by volume)'] = m.get('top_region', 'N/A') + ' ⚠ not stat. sig.'
        kpi['Avg AOV']    = f"${m.get('avg_aov_all_regions',0):.0f}"

    if 'forecasting' in all_results:
        m = all_results['forecasting'].get('metrics', {})
        kpi['Forecast Model'] = m.get('best_model_name', 'N/A')
        kpi['Forecast MAPE']  = f"{m.get('best_model_mape',0):.2f}%"

    if 'discount' in all_results:
        m = all_results['discount'].get('metrics', {})
        kpi['Discount % of Revenue'] = f"{m.get('discount_pct_of_total',0):.1f}%"
        freq_sig = m.get('frequency_significant', None)
        if freq_sig is not None:
            kpi['Discount Frequency Effect'] = 'Significant' if freq_sig else 'Not significant'

    return kpi


def _extract_kpi_values(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """KPI values for indicator cards."""
    kpis = {
        'monthly_revenue':       0,
        'yoy_growth':            0,
        'top_category':          'N/A',
        'top_category_revenue':  0,
        'forecast_quality':      'N/A',
        'forecast_mape':         0,
        'discount_impact':       0,
    }

    if 'time_trends' in all_results:
        m = all_results['time_trends'].get('metrics', {})
        kpis['monthly_revenue'] = m.get('avg_monthly_revenue', 0) / 1000
        kpis['yoy_growth']      = m.get('latest_yoy_growth', 0)

    if 'category' in all_results:
        m = all_results['category'].get('metrics', {})
        kpis['top_category']          = m.get('top_category', 'N/A')
        kpis['top_category_revenue']  = m.get('top_category_revenue', 0)

    if 'forecasting' in all_results:
        m    = all_results['forecasting'].get('metrics', {})
        mape = m.get('best_model_mape', 0)
        kpis['forecast_mape'] = mape
        thresholds = _FORECAST_CFG.get('mape_thresholds', {})
        if mape < thresholds.get('excellent', 10):
            kpis['forecast_quality'] = 'Excellent'
        elif mape < thresholds.get('good', 20):
            kpis['forecast_quality'] = 'Good'
        else:
            kpis['forecast_quality'] = 'Moderate'

    if 'discount' in all_results:
        m = all_results['discount'].get('metrics', {})
        kpis['discount_impact'] = m.get('discount_pct_of_total', 0)

    return kpis


# ============================================================================
# Analysis Summary (Legacy)
# ============================================================================

def generate_analysis_summary(all_results: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Generating analysis summary...")
    summary = {'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), 'sections': {}}
    for key in ['time_trends', 'category', 'region_payment', 'discount', 'forecasting']:
        if key in all_results:
            summary['sections'][key] = all_results[key].get('metrics', {})
    return summary


def print_analysis_summary(summary: Dict[str, Any]) -> None:
    print_section_header("EXECUTIVE SUMMARY")
    print(f"\nAnalysis Timestamp: {summary.get('timestamp','N/A')}")
    s = summary.get('sections', {})

    if 'time_trends' in s:
        tt = s['time_trends']
        print(f"\nTIME TRENDS:")
        print(f"  Avg monthly revenue: ${tt.get('avg_monthly_revenue',0):,.2f}")
        print(f"  Peak month:          {tt.get('peak_month','N/A')}")
        print(f"  Latest YoY growth:   {tt.get('latest_yoy_growth',0):.2f}%")

    if 'category' in s:
        cat = s['category']
        print(f"\nCATEGORY:")
        print(f"  Top: {cat.get('top_category','N/A')} (${cat.get('top_category_revenue',0):,.2f})")

    if 'forecasting' in s:
        fc = s['forecasting']
        print(f"\nFORECASTING:")
        print(f"  Best model: {fc.get('best_model_name','N/A')}")
        print(f"  MAPE: {fc.get('best_model_mape',0):.2f}%")
        print(f"  Trend (rolling): {fc.get('revenue_growth',0):+.1f}%")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    print("n2g_summary_dashboard_enhanced.py — import create_executive_dashboard(all_results)")
