# src/n2i_forecasting.py
"""
n2i_forecasting.py - Time Series Forecasting Module

This module performs time series forecasting using ARIMA models. It includes:
- Data preparation: Aggregating sales data into weekly revenue.
- Train/test split: Chronological split of the time series.
- Model training: Fitting multiple ARIMA configurations and evaluating performance.
- Visualization: Plotting actual vs forecasted values, residuals, and error distribution.
- Metric interpretation: Structured MAPE tier and rolling trend verdicts.
- Business insights: Interpreting model accuracy and recent revenue trends.

Functions:
- prepare_weekly_sales(df): Prepares weekly sales data from raw transactions.
- create_train_test_split(weekly_sales, train_ratio): Creates chronological train/test split.
- train_arima_models(y_train, y_test): Trains ARIMA models and evaluates performance.
- create_forecast_visualization(y_test, best_model, results): Creates forecast analysis visualizations.
- interpret_forecast_metrics(metrics): Returns structured MAPE and trend verdicts from config thresholds.
- print_business_insights(best_model, weekly_sales): Logs business insights based on model
  performance and revenue trends.
- create_forecasting_analysis(df, train_ratio, save_figures): Main function to run the full
  forecasting analysis pipeline.

Usage:
    from n2i_forecasting import create_forecasting_analysis, interpret_forecast_metrics
    results  = create_forecasting_analysis(df)
    verdicts = interpret_forecast_metrics(results['metrics'])
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from typing import Dict, Tuple, Optional, Any
import warnings
import logging
from n2a_utils import setup_logger, save_plotly_figure, print_section_header, get_config

# Setup logger
logger = setup_logger(__name__)

# Load configuration
_CONFIG = get_config()
_FORECAST_CONFIG: Dict[str, Any] = _CONFIG.get('notebook2', {}).get('forecasting', {})

TRAIN_RATIO_DEFAULT: float = _FORECAST_CONFIG.get('train_ratio', 0.8)
ARIMA_CONFIGS = [tuple(c) for c in _FORECAST_CONFIG.get('arima_configs', [
    [1, 1, 1], [2, 1, 1], [1, 1, 2], [2, 1, 2], [0, 1, 1], [1, 1, 0]
])]
REVENUE_GROWTH_WEEKS: int = _FORECAST_CONFIG.get('revenue_growth_weeks', 13)
MAPE_THRESHOLDS: Dict[str, float] = _FORECAST_CONFIG.get(
    'mape_thresholds', {'excellent': 10.0, 'good': 20.0}
)


# ============================================================================
# Helper
# ============================================================================

def _rolling_trend(weekly_sales: pd.DataFrame, weeks: int) -> Tuple[float, str]:
    """
    Return (pct_change, label) comparing last N weeks to the immediately
    preceding N weeks. Falls back to first-vs-last when series is too short.

    Args:
        weekly_sales: Weekly sales DataFrame with a revenue column
        weeks: Number of weeks to use for each comparison window

    Returns:
        Tuple[float, str]: Percentage change and descriptive label string
    """
    try:
        n = len(weekly_sales)
        if n >= weeks * 2:
            recent = weekly_sales['revenue'].iloc[-weeks:].mean()
            prior  = weekly_sales['revenue'].iloc[-weeks * 2:-weeks].mean()
            label  = f"Last {weeks}wk avg vs prior {weeks}wk avg (rolling)"
        else:
            recent = weekly_sales['revenue'].iloc[-weeks:].mean()
            prior  = weekly_sales['revenue'].iloc[:weeks].mean()
            label  = (
                f"Last {weeks}wk avg vs first {weeks}wk avg "
                f"(fallback: series too short)"
            )
            logger.warning(
                f"Only {n} weeks; rolling comparison needs {weeks * 2}. "
                "Using first-vs-last fallback."
            )
        pct = (recent / prior - 1) * 100 if prior > 0 else 0.0
        return pct, label

    except Exception as e:
        logger.error(f"_rolling_trend failed: {e}")
        raise


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_weekly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare weekly sales data from a DataFrame.

    Filters out returned orders (if column present), groups by ISO week,
    and returns a clean weekly time series with revenue, transaction count,
    and unique customer count.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sales data with order_date and total_amount columns

    Returns
    -------
    pd.DataFrame
        Weekly aggregation with columns: date, revenue, transactions, customers
    """
    try:
        logger.info("Preparing weekly sales data...")

        if 'returned' in df.columns:
            df_p = df[df['returned'] == 0].copy()
            logger.info(f"Filtered out {(df['returned'] == 1).sum():,} returns")
        elif 'is_return' in df.columns:
            df_p = df[df['is_return'] == False].copy()
        else:
            df_p = df.copy()

        ws = df_p.groupby(pd.Grouper(key='order_date', freq='W')).agg(
            total_amount=('total_amount', 'sum'),
            order_id=('order_id', 'count'),
            customer_id=('customer_id', 'nunique')
        ).reset_index()
        ws.columns = ['date', 'revenue', 'transactions', 'customers']
        ws = ws[ws['revenue'] > 0].reset_index(drop=True)

        logger.info(
            f"Prepared {len(ws)} weeks | "
            f"{ws['date'].min().date()} → {ws['date'].max().date()}"
        )
        logger.info(f"Total revenue: ${ws['revenue'].sum():,.2f}")
        return ws

    except KeyError as e:
        logger.error(f"Missing required column in prepare_weekly_sales: {e}")
        raise
    except Exception as e:
        logger.error(f"prepare_weekly_sales failed: {e}")
        raise


def create_train_test_split(
    weekly_sales: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO_DEFAULT
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Create a chronological train/test split from weekly sales data.

    Parameters
    ----------
    weekly_sales : pd.DataFrame
        Weekly sales DataFrame with date and revenue columns
    train_ratio : float, optional
        Proportion of data used for training (default: TRAIN_RATIO_DEFAULT)

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]
        (y_train, y_test, train_data, test_data)
    """
    try:
        logger.info("Creating chronological train/test split...")
        idx        = int(len(weekly_sales) * train_ratio)
        train_data = weekly_sales.iloc[:idx].copy()
        test_data  = weekly_sales.iloc[idx:].copy()
        y_train    = train_data.set_index('date')['revenue'].asfreq('W-SUN')
        y_test     = test_data.set_index('date')['revenue'].asfreq('W-SUN')
        logger.info(f"Train: {len(train_data)} weeks | Test: {len(test_data)} weeks")
        return y_train, y_test, train_data, test_data

    except KeyError as e:
        logger.error(f"Missing required column in create_train_test_split: {e}")
        raise
    except Exception as e:
        logger.error(f"create_train_test_split failed: {e}")
        raise


# ============================================================================
# Model Training
# ============================================================================

def train_arima_models(
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[Dict, list]:
    """
    Train ARIMA models for time series forecasting.

    Iterates over all configured ARIMA orders, fits each to y_train, forecasts
    y_test steps, and selects the best model by MAPE.

    Parameters
    ----------
    y_train : pd.Series
        Training time series indexed by date
    y_test : pd.Series
        Test time series indexed by date

    Returns
    -------
    Tuple[Dict, List[Dict]]
        (best_model_dict, all_results_list) sorted by MAPE ascending

    Raises
    ------
    RuntimeError
        If all ARIMA configurations fail to fit
    """
    try:
        logger.info("Training ARIMA models...")
        results = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            for order in ARIMA_CONFIGS:
                try:
                    fitted   = ARIMA(y_train, order=order).fit()
                    forecast = fitted.forecast(steps=len(y_test))
                    results.append({
                        'model':    f"ARIMA{order}",
                        'order':    order,
                        'forecast': forecast,
                        'mae':      mean_absolute_error(y_test, forecast),
                        'rmse':     float(np.sqrt(mean_squared_error(y_test, forecast))),
                        'mape':     mean_absolute_percentage_error(y_test, forecast) * 100,
                        'aic':      fitted.aic,
                    })
                    logger.info(
                        f"ARIMA{order}: MAPE={results[-1]['mape']:.2f}%  "
                        f"AIC={fitted.aic:.1f}"
                    )
                except Exception as e:
                    logger.warning(f"  ARIMA{order}: failed — {str(e)[:60]}")

        if not results:
            raise RuntimeError("All ARIMA configurations failed.")

        results = sorted(results, key=lambda x: x['mape'])
        best    = results[0]
        logger.info(
            f"Best: {best['model']}  MAPE={best['mape']:.2f}%  "
            f"MAE=${best['mae']:,.2f}  RMSE=${best['rmse']:,.2f}"
        )
        return best, results

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"train_arima_models failed: {e}")
        raise


# ============================================================================
# Visualization
# ============================================================================

def create_forecast_visualization(
    y_test: pd.Series,
    best_model: Dict,
    results: list,
    save: bool = True
) -> go.Figure:
    """
    Create a 2×2 visualization of forecast results.

    Panels: Forecast vs Actual, Model MAPE comparison, Residuals, Error distribution.

    Parameters
    ----------
    y_test : pd.Series
        Actual revenue values for the test period
    best_model : Dict
        Best ARIMA model dict with keys: model, forecast, mae, rmse, mape, aic
    results : list
        All model result dicts, sorted by MAPE ascending
    save : bool, optional
        Whether to save the figure as HTML (default: True)

    Returns
    -------
    go.Figure
        Plotly 2×2 subplot figure
    """
    try:
        logger.info("Creating forecast visualization...")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Forecast vs Actual (Test Period)',
                'Model MAPE Comparison (lower = better)',
                'Forecast Residuals',
                'Error Distribution'
            )
        )

        residuals = y_test.values - best_model['forecast'].values

        fig.add_trace(go.Scatter(
            x=y_test.index, y=y_test.values,
            name='Actual',
            line=dict(color='#2E86AB', width=3),
            mode='lines+markers'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=y_test.index, y=best_model['forecast'].values,
            name=best_model['model'],
            line=dict(color='#A23B72', width=2, dash='dash')
        ), row=1, col=1)

        top5 = results[:5]
        fig.add_trace(go.Bar(
            x=[r['model'] for r in top5],
            y=[r['mape'] for r in top5],
            marker_color='#2E86AB',
            showlegend=False
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=y_test.index, y=residuals,
            mode='markers',
            marker=dict(color='#F18F01', size=8),
            showlegend=False
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)  # type: ignore

        fig.add_trace(go.Histogram(
            x=residuals,
            marker_color='#6A994E',
            nbinsx=15,
            showlegend=False
        ), row=2, col=2)

        fig.update_xaxes(title_text="Date",      row=1, col=1)
        fig.update_xaxes(title_text="Model",     row=1, col=2)
        fig.update_xaxes(title_text="Date",      row=2, col=1)
        fig.update_xaxes(title_text="Error ($)", row=2, col=2)
        fig.update_yaxes(title_text="Revenue ($)",  row=1, col=1)
        fig.update_yaxes(title_text="MAPE (%)",     row=1, col=2)
        fig.update_yaxes(title_text="Residual ($)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency",    row=2, col=2)

        fig.update_layout(
            height=800,
            title=dict(
                text="Time Series Forecast Analysis",
                font=dict(size=16, family='Arial Black'),
                x=0.5
            ),
            template='plotly_white',
            showlegend=True
        )

        if save:
            save_plotly_figure(fig, "forecast_analysis", formats=['html'], logger=logger)

        return fig

    except Exception as e:
        logger.error(f"create_forecast_visualization failed: {e}")
        raise


# ============================================================================
# Metric Interpretation
# ============================================================================

def interpret_forecast_metrics(metrics: Dict) -> Dict[str, str]:
    """
    Return structured, human-readable verdicts for MAPE accuracy tier and
    rolling revenue trend direction.

    Thresholds are sourced from module-level MAPE_THRESHOLDS (config-driven),
    so interpretations update automatically when config changes — no hardcoded
    values in calling code.

    Args:
        metrics: Metrics dict returned by create_forecasting_analysis(), expected
                 keys: 'best_model_mape', 'revenue_growth'

    Returns:
        Dict with keys:
            mape_verdict  str  — accuracy tier label + planning guidance
            trend_verdict str  — rolling trend direction summary
            trend_note    str  — YoY disambiguation when trend is negative;
                                 empty string when trend is flat or positive

    Raises:
        KeyError: if required keys are absent from metrics
        Exception: propagated with logger.error for any unexpected failure

    Example:
        >>> verdicts = interpret_forecast_metrics(forecast_results['metrics'])
        >>> print(verdicts['mape_verdict'])
        'GOOD (< 20%) — suitable for strategic / budget planning'
    """
    try:
        mape      = metrics['best_model_mape']
        growth    = metrics['revenue_growth']
        mape_exc  = MAPE_THRESHOLDS['excellent']
        mape_good = MAPE_THRESHOLDS['good']

        # ── MAPE accuracy tier ───────────────────────────────────────────────
        if mape < mape_exc:
            mape_verdict = (
                f"EXCELLENT (< {mape_exc}%) — suitable for operational planning"
            )
        elif mape < mape_good:
            mape_verdict = (
                f"GOOD (< {mape_good}%) — suitable for strategic / budget planning"
            )
        else:
            mape_verdict = (
                f"MODERATE (≥ {mape_good}%) — directional use only"
            )

        # ── Rolling trend direction ──────────────────────────────────────────
        if growth > 10:
            trend_verdict = "Strong recent momentum vs prior window"
            trend_note    = ""
        elif growth > 0:
            trend_verdict = "Moderate recent growth vs prior window"
            trend_note    = ""
        else:
            trend_verdict = "Recent softening vs prior equal window"
            trend_note    = (
                "Does NOT contradict positive YoY — rolling window compares "
                f"last {REVENUE_GROWTH_WEEKS} weeks to the immediately prior "
                f"{REVENUE_GROWTH_WEEKS} weeks, not the same period last year"
            )

        return {
            "mape_verdict":  mape_verdict,
            "trend_verdict": trend_verdict,
            "trend_note":    trend_note,
        }

    except KeyError as e:
        logger.error(f"interpret_forecast_metrics: missing required key {e}")
        raise
    except Exception as e:
        logger.error(f"interpret_forecast_metrics failed: {e}")
        raise


# ============================================================================
# Business Insights
# ============================================================================

def print_business_insights(best_model: Dict, weekly_sales: pd.DataFrame) -> None:
    """
    Log business insights based on the forecasting results.

    Delegates interpretation to interpret_forecast_metrics for consistent
    threshold handling. Logs accuracy tier, rolling trend verdict, and
    YoY disambiguation note when relevant.

    Args:
        best_model:   Best ARIMA model dict with at minimum a 'mape' key
        weekly_sales: Weekly sales DataFrame with a revenue column
    """
    try:
        weeks          = REVENUE_GROWTH_WEEKS
        pct, label     = _rolling_trend(weekly_sales, weeks)
        revenue_growth = pct

        # Build a minimal metrics dict to reuse interpret_forecast_metrics
        metrics = {
            'best_model_mape': best_model['mape'],
            'revenue_growth':  revenue_growth,
        }
        verdicts = interpret_forecast_metrics(metrics)

        logger.info(f"ACCURACY : {verdicts['mape_verdict']}")
        logger.info(f"TREND    : {label}: {revenue_growth:+.1f}%")
        logger.info(f"VERDICT  : {verdicts['trend_verdict']}")
        if verdicts['trend_note']:
            logger.info(f"NOTE     : {verdicts['trend_note']}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"print_business_insights failed: {e}")
        raise


# ============================================================================
# Main
# ============================================================================

def create_forecasting_analysis(
    df: pd.DataFrame,
    train_ratio: Optional[float] = None,
    save_figures: bool = True
) -> Dict:
    """
    Full ARIMA forecasting pipeline.

    Runs data preparation, train/test split, model training, visualization,
    and business insight logging. train_ratio is read from config if not passed.

    Args:
        df:           Transaction DataFrame
        train_ratio:  Fraction of data for training (config default if None)
        save_figures: Whether to save generated figures to disk

    Returns:
        dict with keys:
            weekly_sales  pd.DataFrame
            best_model    dict
            all_models    list[dict]
            figures       {'forecast': Figure}
            metrics       dict with keys:
                            best_model_name, best_model_mape, best_model_mae,
                            best_model_rmse, train_size, test_size, total_weeks,
                            revenue_growth, weekly_avg_revenue, monthly_equivalent
    """
    ratio = train_ratio if train_ratio is not None else TRAIN_RATIO_DEFAULT

    logger.info("=" * 60)
    logger.info("TIME SERIES FORECASTING ANALYSIS")
    logger.info("=" * 60)

    try:
        logger.info("PREPARING WEEKLY SALES DATA:")
        weekly_sales = prepare_weekly_sales(df)
        logger.info("-" * 60)

        logger.info("CREATING TRAIN/TEST SPLIT:")
        y_train, y_test, train_data, test_data = create_train_test_split(
            weekly_sales, ratio
        )
        logger.info("-" * 60)

        logger.info("TRAINING ARIMA MODELS:")
        best_model, all_models = train_arima_models(y_train, y_test)
        logger.info("-" * 60)

        logger.info("CREATING VISUALIZATION:")
        figures = {
            'forecast': create_forecast_visualization(
                y_test, best_model, all_models, save=save_figures
            )
        }
        logger.info("-" * 60)

        logger.info("GENERATING BUSINESS INSIGHTS:")
        print_business_insights(best_model, weekly_sales)

        weeks              = REVENUE_GROWTH_WEEKS
        revenue_growth, _  = _rolling_trend(weekly_sales, weeks)
        weekly_avg         = float(weekly_sales['revenue'].mean())
        monthly_equivalent = weekly_avg * (365.25 / 12 / 7)

        metrics = {
            'best_model_name':    best_model['model'],
            'best_model_mape':    best_model['mape'],
            'best_model_mae':     best_model['mae'],
            'best_model_rmse':    best_model['rmse'],
            'train_size':         len(train_data),
            'test_size':          len(test_data),
            'total_weeks':        len(weekly_sales),
            'revenue_growth':     revenue_growth,
            'weekly_avg_revenue': weekly_avg,
            'monthly_equivalent': monthly_equivalent,
        }

        logger.info("FORECASTING ANALYSIS COMPLETE")
        logger.info("=" * 60)

        return {
            'weekly_sales': weekly_sales,
            'best_model':   best_model,
            'all_models':   all_models,
            'figures':      figures,
            'metrics':      metrics,
        }

    except Exception as e:
        logger.error(f"create_forecasting_analysis failed: {e}")
        raise


# ============================================================================
# Module Information
# ============================================================================

__all__ = [
    'TRAIN_RATIO_DEFAULT',
    'ARIMA_CONFIGS',
    'REVENUE_GROWTH_WEEKS',
    'MAPE_THRESHOLDS',
    'prepare_weekly_sales',
    'create_train_test_split',
    'train_arima_models',
    'create_forecast_visualization',
    'interpret_forecast_metrics',
    'print_business_insights',
    'create_forecasting_analysis',
]

if __name__ == '__main__':
    print("n2i_forecasting.py - import and call create_forecasting_analysis(df)")