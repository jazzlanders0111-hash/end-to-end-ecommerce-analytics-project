"""
n3h_visualizations.py - Segment Visualizations

Creates comprehensive visualizations for customer segments with configuration-driven styling.

Key Features:
- Segment distribution charts
- RFM metrics by segment
- PCA cluster visualization
- Comprehensive segment comparison dashboard
- All colors from config.yaml
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from typing import Dict, Any

from n3a_utils import setup_logger, load_config, get_output_paths, get_colors

logger = setup_logger(__name__)
_config = load_config()


def plot_segment_distribution(
    rfm_df: pd.DataFrame,
    segment_names: Dict[int, str],
    config: Dict[str, Any]
) -> go.Figure:
    """
    Create pie chart of segment distribution.

    Args:
        rfm_df: RFM DataFrame with cluster column
        segment_names: Dictionary mapping cluster_id to names
        config: Configuration dictionary

    Returns:
        Plotly figure
    """
    try:
        segment_counts = rfm_df['cluster'].value_counts().sort_index()
        labels = [segment_names[i] for i in segment_counts.index]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=segment_counts.values,
            hole=0.4,
            textinfo='label+percent',
            textposition='auto',
            marker=dict(line=dict(color='white', width=2))
        )])

        fig.update_layout(
            title='Customer Distribution by Segment',
            height=500,
            showlegend=True
        )

        output_paths = get_output_paths(config)
        output_file = output_paths['figures'] / 'segment_distribution.html'
        fig.write_html(str(output_file))
        logger.info(f"Saved segment distribution: {output_file}")

        return fig
    except Exception as e:
        logger.error(f"plot_segment_distribution failed: {e}")
        raise


def plot_rfm_by_segment(
    rfm_df: pd.DataFrame,
    segment_names: Dict[int, str],
    config: Dict[str, Any]
) -> go.Figure:
    """
    Create bar charts of RFM metrics by segment.

    Args:
        rfm_df: RFM DataFrame with cluster column
        segment_names: Dictionary mapping cluster_id to names
        config: Configuration dictionary

    Returns:
        Plotly figure
    """
    try:
        colors = get_colors(config)
    
        segment_metrics = rfm_df.groupby('cluster').agg({
            'recency_days': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'loyalty_score': 'mean'
        }).reset_index()

        segment_metrics['segment_name'] = segment_metrics['cluster'].map(segment_names)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary Value ($)', 'Avg Loyalty Score']
        )

        # Recency
        fig.add_trace(
            go.Bar(x=segment_metrics['segment_name'], y=segment_metrics['recency_days'],
                   marker_color=colors.get('primary', '#2E86C1'), showlegend=False),
            row=1, col=1
        )

        # Frequency
        fig.add_trace(
            go.Bar(x=segment_metrics['segment_name'], y=segment_metrics['frequency'],
                   marker_color=colors.get('danger', '#E74C3C'), showlegend=False),
            row=1, col=2
        )

        # Monetary
        fig.add_trace(
            go.Bar(x=segment_metrics['segment_name'], y=segment_metrics['monetary'],
                   marker_color=colors.get('success', '#27AE60'), showlegend=False),
            row=2, col=1
        )

        # Loyalty
        fig.add_trace(
            go.Bar(x=segment_metrics['segment_name'], y=segment_metrics['loyalty_score'],
                   marker_color=colors.get('warning', '#9B59B6'), showlegend=False),
            row=2, col=2
        )

        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=700, title_text="RFM Metrics by Customer Segment")

        output_paths = get_output_paths(config)
        output_file = output_paths['figures'] / 'rfm_by_segment.html'
        fig.write_html(str(output_file))
        logger.info(f"Saved RFM metrics: {output_file}")

        return fig
    except Exception as e:
        logger.error(f"plot_rfm_by_segment failed: {e}")
        raise


def plot_pca_clusters(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    segment_names: Dict[int, str],
    config: Dict[str, Any]
) -> go.Figure:
    """
    Create PCA visualization of clusters in 2D.

    Args:
        X_scaled: Scaled feature matrix
        cluster_labels: Cluster assignments
        segment_names: Dictionary mapping cluster_id to names
        config: Configuration dictionary

    Returns:
        Plotly figure
    """
    # Get PCA config
    try:
        pca_cfg = config.get('notebook3', {}).get('visualizations', {}).get('pca', {})
        n_components = pca_cfg.get('n_components', 2)
        random_state = pca_cfg.get('random_state', 42)

        pca = PCA(n_components=n_components, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'cluster': cluster_labels,
            'segment_name': [segment_names[c] for c in cluster_labels]
        })

        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='segment_name',
            title=f'Customer Segments (PCA) - Variance Explained: {pca.explained_variance_ratio_.sum():.1%}',
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
            opacity=0.6
        )

        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=600)

        output_paths = get_output_paths(config)
        output_file = output_paths['figures'] / 'pca_clusters.html'
        fig.write_html(str(output_file))
        logger.info(f"Saved PCA visualization: {output_file}")

        return fig
    except Exception as e:
        logger.error(f"plot_pca_clusters failed: {e}")
        raise


def plot_segment_comparison(
    rfm_df: pd.DataFrame,
    df: pd.DataFrame,
    segment_names: Dict[int, str],
    config: Dict[str, Any]
) -> go.Figure:
    """
    Create comprehensive segment comparison dashboard.

    Args:
        rfm_df: RFM DataFrame with cluster column
        df: Transaction DataFrame
        segment_names: Dictionary mapping cluster_id to names
        config: Configuration dictionary

    Returns:
        Plotly figure
    """
    try:
        colors = get_colors(config)
    
        df_clean = df.copy()
        if 'cluster' in df_clean.columns:
            df_clean = df_clean.drop(columns=['cluster'])

        df_with_segments = df_clean.merge(
            rfm_df[['customer_id', 'cluster']],
            on='customer_id',
            how='left'
        )
        df_with_segments['segment_name'] = df_with_segments['cluster'].map(segment_names)

        segment_stats = df_with_segments.groupby('segment_name').agg({
            'customer_id': 'nunique',
            'total_amount': ['sum', 'mean'],
            'order_id': 'count'
        }).round(2)

        segment_stats.columns = ['Customers', 'Total Revenue', 'Avg Order Value', 'Total Orders']
        segment_stats = segment_stats.reset_index()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Revenue by Segment', 'Customers by Segment',
                           'Avg Order Value by Segment', 'Orders by Segment'],
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )

        # Revenue
        fig.add_trace(
            go.Bar(x=segment_stats['segment_name'], y=segment_stats['Total Revenue'],
                   marker_color=colors.get('success', '#27AE60'), showlegend=False),
            row=1, col=1
        )

        # Customers
        fig.add_trace(
            go.Bar(x=segment_stats['segment_name'], y=segment_stats['Customers'],
                   marker_color=colors.get('primary', '#2E86C1'), showlegend=False),
            row=1, col=2
        )

        # AOV
        fig.add_trace(
            go.Bar(x=segment_stats['segment_name'], y=segment_stats['Avg Order Value'],
                   marker_color=colors.get('danger', '#E74C3C'), showlegend=False),
            row=2, col=1
        )

        # Orders
        fig.add_trace(
            go.Bar(x=segment_stats['segment_name'], y=segment_stats['Total Orders'],
                   marker_color=colors.get('warning', '#9B59B6'), showlegend=False),
            row=2, col=2
        )

        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=800, title_text="Comprehensive Segment Comparison")

        output_paths = get_output_paths(config)
        output_file = output_paths['figures'] / 'segment_comparison.html'
        fig.write_html(str(output_file))
        logger.info(f"Saved segment comparison: {output_file}")

        return fig
    except Exception as e:
        logger.error(f"plot_segment_comparison failed: {e}")
        raise


__all__ = [
    'plot_segment_distribution',
    'plot_rfm_by_segment',
    'plot_pca_clusters',
    'plot_segment_comparison',
]

if __name__ == "__main__":
    print("Testing n3h_visualizations module...")
    print("Module loaded successfully")
