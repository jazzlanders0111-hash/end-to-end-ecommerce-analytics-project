# src/n2g_summary_dashboard.py
"""
n2g_summary_dashboard.py - Summary & Combined Dashboards

Creates combined dashboards and generates analysis summary including:
- Combined multi-chart dashboards
- Key insights extraction
- Summary metrics compilation
- Dashboard saving utilities
- Forecasting integration

Functions:
    create_combined_dashboard() - Combine multiple figures
    create_summary_table() - Build summary metrics table
    generate_key_insights() - Extract key insights from all results
    save_all_dashboards() - Save all generated figures
    create_analysis_summary() - Main summary function

Usage:
    from n2g_summary_dashboard import create_analysis_summary
    summary = create_analysis_summary(all_results)
"""

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any, Optional, cast
import logging
from n2a_utils import (
    setup_logger,
    save_plotly_figure,
    get_figures_dir,
    print_section_header,
    get_config
)

# Setup logger
logger = setup_logger(__name__)

# Load configuration
_CONFIG = get_config()
_VIZ_CONFIG = _CONFIG.get('visualization', {})
_FIG_DEFAULTS = _VIZ_CONFIG.get('figure_defaults', {})
_DASHBOARD_CFG = _CONFIG.get('notebook2', {}).get('summary_dashboard', {})

DASHBOARD_HEIGHT = _FIG_DEFAULTS.get('height', 500)
COMBINED_HEIGHT = 900
SUMMARY_TABLE_HEIGHT = 600


# ============================================================================
# Dashboard Creation Functions
# ============================================================================

def create_combined_dashboard(
    figures_dict: Dict[str, go.Figure],
    title: str = "Sales Analysis Dashboard",
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    save: bool = True
) -> go.Figure:
    """
    Combine multiple figures into a single dashboard.

    Args:
        figures_dict: Dictionary of figure names and Plotly figures
        title: Dashboard title
        rows: Number of rows in subplot grid (defaults from config)
        cols: Number of columns in subplot grid (defaults from config)
        save: Whether to save the dashboard

    Returns:
        go.Figure: Combined Plotly figure
    """
    try:
        logger.info(f"Creating combined dashboard with {len(figures_dict)} figures...")

        dashboard_config = _DASHBOARD_CFG.get('combined_dashboard', {})
        if rows is None:
            rows = dashboard_config.get('rows', 2)
        if cols is None:
            cols = dashboard_config.get('cols', 2)

        specs = [[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=list(figures_dict.keys()),
            specs=specs,
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        positions = [
            (r, c) for r in range(1, rows + 1) for c in range(1, cols + 1)
        ]

        for (fig_name, source_fig), (row, col) in zip(figures_dict.items(), positions):
            if source_fig is None:
                continue

            for trace in source_fig.data:
                use_secondary = False
                if hasattr(trace, 'yaxis') and trace.yaxis is not None and 'y2' in str(trace.yaxis):
                    use_secondary = True

                fig.add_trace(trace, row=row, col=col, secondary_y=use_secondary)

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, family='Arial Black', color='black'),
                x=0.5,
                xanchor='center'
            ),
            template='plotly_white',
            height=COMBINED_HEIGHT,
            showlegend=True
        )

        if save:
            save_plotly_figure(
                fig, "combined_dashboard", formats=['html'], logger=logger
            )

        logger.info("Combined dashboard created")
        return fig

    except Exception as e:
        logger.error(f"create_combined_dashboard failed: {e}")
        raise


def create_summary_table(metrics_dict: Dict[str, Dict]) -> go.Figure:
    """
    Create a summary table from all analysis metrics.

    Handles the statistical_validation section separately as it uses a
    different result structure. All numeric values are formatted by semantic
    key name (revenue → $, rate/pct/growth → %).

    Args:
        metrics_dict: Dictionary of analysis section names and their metrics

    Returns:
        go.Figure: Plotly table figure
    """
    try:
        logger.info("Creating summary metrics table...")

        rows = []
        for section, metrics in metrics_dict.items():
            if section == 'statistical_validation':
                if 'summary' in metrics:
                    summary = metrics['summary']
                    rows.append({
                        'Section': 'Statistical Validation',
                        'Metric': 'Summary',
                        'Value': (
                            f"{summary.get('significant_tests', 0)}/"
                            f"{summary.get('total_tests', 0)} tests significant "
                            f"({summary.get('validation_rate', 0):.1f}%)"
                        )
                    })

                for test_name in ['time_trend', 'category', 'regional', 'discount']:
                    if test_name in metrics:
                        test_result = metrics[test_name]
                        status = (
                            'Significant' if test_result.get('significant', False)
                            else 'Not Significant'
                        )
                        p_val = test_result.get('p_value', 0)
                        rows.append({
                            'Section': 'Statistical Validation',
                            'Metric': test_name.replace('_', ' ').title(),
                            'Value': f"{status} (p={p_val:.4f})"
                        })
                continue

            for key, value in metrics.items():
                if isinstance(value, dict):
                    if 'interpretation' in value:
                        formatted_value = value['interpretation']
                    elif 'test_name' in value and 'significant' in value:
                        test_name = value.get('test_name', 'Test')
                        significant = 'Yes' if value.get('significant', False) else 'No'
                        p_value = value.get('p_value', 'N/A')
                        if isinstance(p_value, (int, float)):
                            formatted_value = (
                                f"{test_name}: Significant={significant}, p={p_value:.4f}"
                            )
                        else:
                            formatted_value = f"{test_name}: Significant={significant}"
                    else:
                        formatted_value = ', '.join(
                            f"{k}={v}" for k, v in list(value.items())[:3]
                        )
                elif isinstance(value, float):
                    key_lower = key.lower()
                    if any(k in key_lower for k in ('revenue', 'amount', 'mae', 'rmse', 'aov')):
                        formatted_value = f"${value:,.2f}"
                    elif any(k in key_lower for k in ('pct', 'rate', 'growth', 'mape', 'margin')):
                        formatted_value = f"{value:.2f}%"
                    else:
                        formatted_value = f"{value:,.2f}"
                elif isinstance(value, list):
                    formatted_value = ', '.join(str(v) for v in value[:5])
                    if len(value) > 5:
                        formatted_value += f" ... ({len(value)} total)"
                else:
                    formatted_value = str(value)

                rows.append({
                    'Section': section.replace('_', ' ').title(),
                    'Metric': key.replace('_', ' ').title(),
                    'Value': formatted_value
                })

        df_summary = pd.DataFrame(rows)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df_summary.columns),
                fill_color='#1B1F5E',
                font=dict(color='white', size=12, family='Arial Black'),
                align='left'
            ),
            cells=dict(
                values=[df_summary[col] for col in df_summary.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=11, family='Arial')
            )
        )])

        fig.update_layout(
            title=dict(
                text='Analysis Summary Metrics',
                font=dict(size=14, family='Arial Black')
            ),
            height=SUMMARY_TABLE_HEIGHT
        )

        logger.info(f"Summary table created with {len(rows)} metrics")
        return fig

    except Exception as e:
        logger.error(f"create_summary_table failed: {e}")
        raise


# ============================================================================
# Key Insights Generation
# ============================================================================

def generate_key_insights(all_results: Dict[str, Dict]) -> List[str]:
    """
    Extract key insights from all analysis results including forecasting.

    Args:
        all_results: Dictionary containing results from all analysis modules

    Returns:
        List[str]: List of human-readable insight strings, one per analysis section
    """
    try:
        logger.info("Generating key insights...")

        insights = []

        if 'time_trends' in all_results:
            metrics = all_results['time_trends']['metrics']
            insights.append(
                f"TIME TRENDS: Peak month was {metrics['peak_month']} with "
                f"${metrics['peak_revenue']:,.2f} revenue. "
                f"Latest YoY growth: {metrics['latest_yoy_growth']:.1f}%"
            )

        if 'category' in all_results:
            metrics = all_results['category']['metrics']
            insights.append(
                f"CATEGORIES: {metrics['top_category']} is the top category with "
                f"${metrics['top_category_revenue']:,.2f} revenue. "
                f"Avg return rate: {metrics['avg_return_rate']:.2%}"
            )

        if 'region_payment' in all_results:
            metrics = all_results['region_payment']['metrics']
            insights.append(
                f"REGIONS: {metrics['top_region']} leads with "
                f"${metrics['top_region_revenue']:,.2f}. "
                f"Avg AOV across regions: ${metrics['avg_aov_all_regions']:.2f}"
            )
            insights.append(
                f"PAYMENTS: {metrics['top_payment_method']} accounts for "
                f"{metrics['top_payment_pct']:.1f}% of revenue"
            )

        if 'discount' in all_results:
            metrics = all_results['discount']['metrics']
            insights.append(
                f"DISCOUNTS: Discounted sales represent "
                f"{metrics['discount_pct_of_total']:.1f}% of total revenue. "
                f"No-discount avg margin: {metrics['avg_margin_no_discount']:.2f}%"
            )

        if 'forecasting' in all_results:
            metrics = all_results['forecasting']['metrics']
            accuracy_level = (
                "Excellent" if metrics['best_model_mape'] < 10
                else "Good" if metrics['best_model_mape'] < 20
                else "Moderate"
            )
            insights.append(
                f"FORECASTING: {metrics['best_model_name']} model achieves "
                f"{accuracy_level} accuracy (MAPE: {metrics['best_model_mape']:.2f}%). "
                f"Revenue trend: {metrics['revenue_growth']:+.1f}%"
            )

        if 'statistical_validation' in all_results:
            val_results = all_results['statistical_validation']
            if 'summary' in val_results:
                summary = val_results['summary']
                sig = summary.get('significant_tests', 0)
                total = summary.get('total_tests', 0)
                insights.append(
                    f"STATISTICAL VALIDATION: {sig}/{total} analyses are statistically "
                    f"significant (validation rate: {summary.get('validation_rate', 0):.1f}%)"
                )

        logger.info(f"Generated {len(insights)} key insights")
        return insights

    except Exception as e:
        logger.error(f"generate_key_insights failed: {e}")
        raise


# ============================================================================
# Dashboard Saving Utilities
# ============================================================================

def save_all_dashboards(
    all_figures: Dict[str, Dict[str, go.Figure]],
    formats: List[str] = ['html']
) -> Dict[str, Path]:
    """
    Save all generated figures from all analysis modules.

    Args:
        all_figures: Nested dict {section_name: {fig_name: figure}}
        formats: List of formats to save

    Returns:
        Dict[str, Path]: Dictionary of saved file paths keyed by figure name
    """
    logger.info("=" * 60)
    logger.info("SAVING ALL DASHBOARDS")
    logger.info("=" * 60)

    try:
        saved_files = {}

        for section_name, figures_dict in all_figures.items():
            logger.info(f"Saving {section_name} figures:")

            for fig_name, fig in figures_dict.items():
                if fig is None:
                    continue

                full_name = f"{section_name}_{fig_name}"
                try:
                    paths = save_plotly_figure(
                        fig,
                        full_name,
                        formats=formats,
                        include_timestamp=False,
                        logger=logger
                    )
                    saved_files[full_name] = paths
                except Exception as e:
                    logger.error(f"Failed to save figure '{full_name}': {e}")

        logger.info(f"Saved {len(saved_files)} figure(s)")
        logger.info("=" * 60)

        return saved_files

    except Exception as e:
        logger.error(f"save_all_dashboards failed: {e}")
        raise


# ============================================================================
# Main Summary Function
# ============================================================================

def create_analysis_summary(
    all_results: Dict[str, Dict],
    save_figures: bool = True
) -> Dict:
    """
    Create comprehensive analysis summary with combined dashboards.

    Generates key insights, compiles all metrics into a summary table,
    and builds a combined 2×2 dashboard from the highest-priority figures.

    Args:
        all_results: Dictionary containing results from all analysis modules.
            Expected keys: time_trends, category, region_payment, discount,
            forecasting, statistical_validation

        save_figures: Whether to save generated dashboards

    Returns:
        dict with keys: insights, metrics_table, combined_dashboard, all_metrics

    Example:
        >>> results = {
        ...     'time_trends': time_results,
        ...     'category': category_results,
        ...     'region_payment': region_results,
        ...     'discount': discount_results,
        ...     'forecasting': forecast_results
        ... }
        >>> summary = create_analysis_summary(results)
    """
    logger.info("=" * 60)
    logger.info("CREATING ANALYSIS SUMMARY")
    logger.info("=" * 60)

    try:
        logger.info("STEP 1: GENERATING KEY INSIGHTS")
        insights = generate_key_insights(all_results)
        for i, insight in enumerate(insights, 1):
            logger.info(f"  {i}. {insight}")
        logger.info("-" * 60)

        logger.info("STEP 2: COMPILING ALL METRICS")
        all_metrics = {}
        for section_name, results in all_results.items():
            if section_name == 'statistical_validation':
                all_metrics[section_name] = results
            elif 'metrics' in results:
                all_metrics[section_name] = results['metrics']
        logger.info(f"Compiled metrics from {len(all_metrics)} sections")
        logger.info("-" * 60)

        logger.info("STEP 3: CREATING SUMMARY METRICS TABLE")
        metrics_table = create_summary_table(all_metrics)

        if save_figures:
            save_plotly_figure(
                metrics_table, "summary_metrics_table", formats=['html'], logger=logger
            )
        logger.info("-" * 60)

        logger.info("STEP 4: CREATING COMBINED DASHBOARD")
        combined_figures = {}

        priority_sections = [
            'time_trends', 'category', 'region_payment', 'discount', 'forecasting'
        ]

        for section_name in priority_sections:
            if section_name in all_results and 'figures' in all_results[section_name]:
                section_figs = all_results[section_name]['figures']
                for fig_name, fig in list(section_figs.items())[:1]:
                    combined_figures[section_name] = fig
                    if len(combined_figures) >= 4:
                        break
            if len(combined_figures) >= 4:
                break

        if len(combined_figures) >= 4:
            combined_dashboard = create_combined_dashboard(
                dict(list(combined_figures.items())[:4]),
                title="Sales Analysis Overview Dashboard",
                save=save_figures
            )
        else:
            combined_dashboard = None
            logger.warning("Not enough figures for combined dashboard")

        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY COMPLETE")
        logger.info(
            f"Generated {len(insights)} insights across {len(all_metrics)} sections"
        )
        logger.info("=" * 60)

        return {
            'insights': insights,
            'metrics_table': metrics_table,
            'combined_dashboard': combined_dashboard,
            'all_metrics': all_metrics
        }

    except Exception as e:
        logger.error(f"create_analysis_summary failed: {e}")
        raise


# ============================================================================
# Module Information
# ============================================================================

__all__ = [
    'DASHBOARD_HEIGHT',
    'COMBINED_HEIGHT',
    'SUMMARY_TABLE_HEIGHT',
    'create_combined_dashboard',
    'create_summary_table',
    'generate_key_insights',
    'save_all_dashboards',
    'create_analysis_summary',
]

if __name__ == '__main__':
    print("n2g_summary_dashboard.py - Summary & Dashboard Module")
    print("This module should be used with results from other analysis modules")
