"""
n3e_cluster_optimizer.py - Optimal Cluster Number Determination

Determines optimal number of clusters using multiple evaluation metrics.

Key Features:
- Multi-metric evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz, Elbow)
- Voting system (4 votes: 3 metrics + elbow)
- Business logic for practical segment sizing
- Comprehensive visualization
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Tuple, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from n3a_utils import setup_logger, load_config, get_output_paths, get_colors

logger = setup_logger(__name__)


def compute_clustering_metrics(X: np.ndarray, k: int, random_state: int = 42, n_init: int = 50) -> dict:
    """
    Compute clustering quality metrics for a given k.

    Args:
        X: Scaled feature matrix
        k: Number of clusters
        random_state: Random seed
        n_init: Number of K-means initializations (increased for stability)

    Returns:
        Dictionary of metrics
    """
    try:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=300, algorithm='lloyd')
        labels = kmeans.fit_predict(X)

        metrics = {
            'k': k,
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels)
        }

        return metrics
    except Exception as e:
        logger.error(f"compute_clustering_metrics failed: {e}")
        raise


def find_optimal_clusters(
    X: np.ndarray,
    k_range: range = None,
    random_state: int = 42,
    config: Dict[str, Any] = None
) -> Tuple[int, pd.DataFrame]:
    """
    Find optimal number of clusters using multiple metrics and voting.

    Evaluates silhouette score, Davies-Bouldin index, Calinski-Harabasz score,
    and elbow method. Uses 4-vote system to determine optimal k.

    Args:
        X: Scaled feature matrix
        k_range: Range of k values to test
        random_state: Random seed
        config: Configuration dictionary

    Returns:
        Tuple of (optimal_k, metrics_dataframe)
    """
    try:
        if config is None:
            config = load_config()

        if k_range is None:
            k_values = config.get('notebook3', {}).get('clustering', {}).get('k_range', [2, 3, 4, 5, 6, 7, 8])
            k_range = range(min(k_values), max(k_values) + 1)

        logger.info(f"Testing k values: {list(k_range)}")
        logger.info(f"Samples: {X.shape[0]:,}, Features: {X.shape[1]}")

        n_init = config.get('notebook3', {}).get('clustering', {}).get('kmeans', {}).get('n_init', 50)

        results = []
        for k in k_range:
            logger.info(f"Evaluating k={k}")
            metrics = compute_clustering_metrics(X, k, random_state, n_init=n_init)
            results.append(metrics)

        metrics_df = pd.DataFrame(results)

        optimal_k = determine_optimal_k(metrics_df, config)

        visualize_metrics(metrics_df, optimal_k, config)

        logger.info(f"Optimal k determined: {optimal_k}")

        return optimal_k, metrics_df
    except Exception as e:
        logger.error(f"find_optimal_clusters failed: {e}")
        raise


def determine_optimal_k(metrics_df: pd.DataFrame, config: Dict[str, Any]) -> int:
    """
    Determine optimal k using 4-vote system + business constraints.

    Voters:
    - Silhouette score (max)
    - Davies-Bouldin index (min)
    - Calinski-Harabasz score (max)
    - Elbow method

    Args:
        metrics_df: DataFrame with clustering metrics
        config: Configuration dictionary

    Returns:
        Optimal k value
    """
    try:
        logger.info("Determining optimal k using voting system")

        clustering_cfg = config.get('notebook3', {}).get('clustering', {})
        min_k = clustering_cfg.get('min_k', 3)
        max_k = clustering_cfg.get('max_k', 8)
        fallback_k = clustering_cfg.get('fallback_k', 4)

        # Criterion 1: Highest silhouette score
        best_silhouette_k = int(metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k'])
        best_silhouette_score = metrics_df['silhouette'].max()

        # Criterion 2: Lowest Davies-Bouldin index
        best_db_k = int(metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k'])
        best_db_score = metrics_df['davies_bouldin'].min()

        # Criterion 3: Highest Calinski-Harabasz score
        best_ch_k = int(metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k'])
        best_ch_score = metrics_df['calinski_harabasz'].max()

        # Criterion 4: Elbow method
        elbow_k = find_elbow(metrics_df['k'].values, metrics_df['inertia'].values)

        logger.info(f"Silhouette (max): k={best_silhouette_k} (score={best_silhouette_score:.4f})")
        logger.info(f"Davies-Bouldin (min): k={best_db_k} (score={best_db_score:.4f})")
        logger.info(f"Calinski-Harabasz (max): k={best_ch_k} (score={best_ch_score:.2f})")
        logger.info(f"Elbow method: k={elbow_k}")

        # 4-vote system: all methods vote
        votes = [best_silhouette_k, best_db_k, best_ch_k, elbow_k]
        vote_counts = {}
        for k in votes:
            vote_counts[k] = vote_counts.get(k, 0) + 1

        logger.info(f"Vote counts: {vote_counts}")

        max_votes = max(vote_counts.values())
        winners = [k for k, v in vote_counts.items() if v == max_votes]

        if len(winners) == 1:
            optimal_k = winners[0]
            logger.info(f"Clear winner: k={optimal_k} with {max_votes}/4 votes")
        else:
            optimal_k = max(winners)
            logger.info(f"Tie between {winners}, choosing k={optimal_k} for business granularity")

        # Business override: k=2 creates only binary split (not useful)
        if optimal_k == 2:
            logger.warning("k=2 creates only binary split (churned/active)")
            logger.warning(f"Overriding to k={max(fallback_k, elbow_k)} for actionable segments")
            optimal_k = max(fallback_k, elbow_k)

        # Apply constraints
        if optimal_k < min_k:
            logger.warning(f"k={optimal_k} too small for useful segmentation, using k={min_k}")
            optimal_k = min_k
        elif optimal_k > max_k:
            logger.warning(f"k={optimal_k} indicates over-segmentation, using k={max_k}")
            optimal_k = max_k

        logger.info(f"Final recommendation: k={optimal_k}")
        return optimal_k
    except Exception as e:
        logger.error(f"determine_optimal_k failed: {e}")
        raise


def find_elbow(k_values: np.ndarray, inertias: np.ndarray) -> int:
    """
    Find elbow point in inertia curve using the angle method.

    Args:
        k_values: Array of k values
        inertias: Array of corresponding inertia values

    Returns:
        K value at the elbow point
    """
    try:
        k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
        inertia_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())

        first_point = np.array([k_norm[0], inertia_norm[0]])
        last_point = np.array([k_norm[-1], inertia_norm[-1]])
        line_vec = last_point - first_point
        line_vec_norm = line_vec / np.linalg.norm(line_vec)

        distances = []
        for i in range(len(k_values)):
            point = np.array([k_norm[i], inertia_norm[i]])
            vec_to_point = point - first_point
            proj_length = np.dot(vec_to_point, line_vec_norm)
            proj_point = first_point + proj_length * line_vec_norm
            distance = np.linalg.norm(point - proj_point)
            distances.append(distance)

        elbow_idx = np.argmax(distances)
        return int(k_values[elbow_idx])
    except Exception as e:
        logger.error(f"find_elbow failed: {e}")
        raise


def visualize_metrics(
    metrics_df: pd.DataFrame,
    optimal_k: int,
    config: Dict[str, Any]
) -> None:
    """
    Create visualizations of clustering metrics.

    Args:
        metrics_df: DataFrame with clustering metrics
        optimal_k: Optimal k value to highlight
        config: Configuration dictionary
    """
    try:
        logger.info("Creating metric visualizations")

        output_paths = get_output_paths(config)
        colors = get_colors(config)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Elbow Method (Inertia)',
                'Silhouette Score',
                'Davies-Bouldin Index',
                'Calinski-Harabasz Score'
            ]
        )

        k_values = metrics_df['k'].values

        color_primary = colors.get('primary', '#2E86C1')
        color_success = colors.get('success', '#27AE60')
        color_danger = colors.get('danger', '#E74C3C')
        color_warning = colors.get('warning', '#9B59B6')
        color_highlight = 'red'

        # 1. Elbow Method
        fig.add_trace(
            go.Scatter(x=k_values, y=metrics_df['inertia'],
                      mode='lines+markers', name='Inertia',
                      line=dict(color=color_primary, width=2),
                      marker=dict(size=8)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[optimal_k],
                      y=[metrics_df.loc[metrics_df['k'] == optimal_k, 'inertia'].values[0]],
                      mode='markers', name='Optimal K',
                      marker=dict(size=15, color=color_highlight, symbol='star')),
            row=1, col=1
        )

        # 2. Silhouette Score
        fig.add_trace(
            go.Scatter(x=k_values, y=metrics_df['silhouette'],
                      mode='lines+markers', name='Silhouette',
                      line=dict(color=color_success, width=2),
                      marker=dict(size=8), showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[optimal_k],
                      y=[metrics_df.loc[metrics_df['k'] == optimal_k, 'silhouette'].values[0]],
                      mode='markers',
                      marker=dict(size=15, color=color_highlight, symbol='star'),
                      showlegend=False),
            row=1, col=2
        )

        # 3. Davies-Bouldin Index
        fig.add_trace(
            go.Scatter(x=k_values, y=metrics_df['davies_bouldin'],
                      mode='lines+markers', name='Davies-Bouldin',
                      line=dict(color=color_danger, width=2),
                      marker=dict(size=8), showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=[optimal_k],
                      y=[metrics_df.loc[metrics_df['k'] == optimal_k, 'davies_bouldin'].values[0]],
                      mode='markers',
                      marker=dict(size=15, color=color_highlight, symbol='star'),
                      showlegend=False),
            row=2, col=1
        )

        # 4. Calinski-Harabasz Score
        fig.add_trace(
            go.Scatter(x=k_values, y=metrics_df['calinski_harabasz'],
                      mode='lines+markers', name='Calinski-Harabasz',
                      line=dict(color=color_warning, width=2),
                      marker=dict(size=8), showlegend=False),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=[optimal_k],
                      y=[metrics_df.loc[metrics_df['k'] == optimal_k, 'calinski_harabasz'].values[0]],
                      mode='markers',
                      marker=dict(size=15, color=color_highlight, symbol='star'),
                      showlegend=False),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Number of Clusters (k)")
        fig.update_yaxes(title_text="Inertia", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        fig.update_yaxes(title_text="Davies-Bouldin Index", row=2, col=1)
        fig.update_yaxes(title_text="Calinski-Harabasz Score", row=2, col=2)

        fig.update_layout(
            height=800,
            title_text=f"Clustering Evaluation Metrics (Optimal k={optimal_k})",
            title_x=0.5,
            showlegend=True
        )

        output_file = output_paths['figures'] / 'cluster_optimization_metrics.html'
        fig.write_html(str(output_file))
        logger.info(f"Saved clustering metrics: {output_file}")
    except Exception as e:
        logger.error(f"visualize_metrics failed: {e}")
        raise


__all__ = [
    'compute_clustering_metrics',
    'find_optimal_clusters',
    'determine_optimal_k',
    'find_elbow',
    'visualize_metrics',
]

if __name__ == "__main__":
    print("Testing n3e_cluster_optimizer module...")
    print("Module loaded successfully")