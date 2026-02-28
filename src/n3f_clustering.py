"""
n3f_clustering.py - K-Means Clustering and Stability Validation

Performs K-Means clustering and validates stability through bootstrap resampling.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from typing import Tuple

from n3a_utils import setup_logger, load_config

logger = setup_logger(__name__)
_config = load_config()


def perform_kmeans_clustering(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 50,
    max_iter: int = 300
) -> Tuple[KMeans, np.ndarray]:
    """
    Perform K-Means clustering on scaled features.

    Args:
        X: Scaled feature matrix
        n_clusters: Number of clusters
        random_state: Random seed
        n_init: Number of initializations
        max_iter: Maximum iterations

    Returns:
        Tuple of (kmeans_model, cluster_labels)
    """
    # Initial logs removed - redundant with output
    
    try:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            algorithm='lloyd'
        )

        cluster_labels = kmeans.fit_predict(X)

        logger.info(f"Clustering complete (inertia={kmeans.inertia_:.2f}, iterations={kmeans.n_iter_})")

        # Log cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        logger.info("Cluster size distribution:")
        for cluster_id, count in zip(unique, counts):
            pct = (count / len(cluster_labels)) * 100
            logger.info(f"  Cluster {cluster_id}: {count:6,} ({pct:5.1f}%)")

        # Check balance
        imbalance_ratio = counts.max() / counts.min()
        if imbalance_ratio > 5:
            logger.warning(f"Clusters imbalanced (ratio={imbalance_ratio:.1f})")
        else:
            logger.info(f"Clusters reasonably balanced (ratio={imbalance_ratio:.1f})")

        return kmeans, cluster_labels
    except Exception as e:
        logger.error(f"perform_kmeans_clustering failed: {e}")
        raise


def validate_clustering_stability(
    X: np.ndarray,
    n_clusters: int,
    n_iterations: int = 50,
    random_state: int = 42,
    sample_fraction: float = 0.8
) -> float:
    """
    Validate clustering stability using bootstrap resampling.

    Tests whether clustering is stable across different random samples
    of the data using Adjusted Rand Index.

    Args:
        X: Scaled feature matrix
        n_clusters: Number of clusters
        n_iterations: Number of bootstrap iterations
        random_state: Random seed
        sample_fraction: Fraction of data to sample per iteration

    Returns:
        Mean Adjusted Rand Index
    """
    # Initial logs removed - process is self-evident from output
    
    # Reference clustering
    try:
        reference_kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        reference_labels = reference_kmeans.fit_predict(X)

        # Bootstrap iterations
        ari_scores = []
        for i in range(n_iterations):
            indices = np.random.choice(len(X), size=int(len(X) * sample_fraction), replace=True)
            X_resampled = X[indices]

            bootstrap_kmeans = KMeans(n_clusters=n_clusters, random_state=random_state + i, n_init=10)
            bootstrap_labels = bootstrap_kmeans.fit_predict(X_resampled)

            reference_resampled = reference_labels[indices]
            ari = adjusted_rand_score(reference_resampled, bootstrap_labels)
            ari_scores.append(ari)

            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{n_iterations} iterations")

        # Results
        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)

        logger.info(f"Stability Results:")
        logger.info(f"Mean ARI: {mean_ari:.4f}")
        logger.info(f"Std Dev: {std_ari:.4f}")
        logger.info(f"Min/Max: {np.min(ari_scores):.4f} / {np.max(ari_scores):.4f}")

        # Interpretation
        if mean_ari >= 0.9:
            stability_label = "Excellent"
        elif mean_ari >= 0.8:
            stability_label = "Good"
        elif mean_ari >= 0.7:
            stability_label = "Moderate"
        else:
            stability_label = "Poor"

        logger.info(f"Stability Assessment: {stability_label}")

        return mean_ari
    except Exception as e:
        logger.error(f"validate_clustering_stability failed: {e}")
        raise


def analyze_cluster_separation(X: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """
    Analyze cluster separation quality.

    Args:
        X: Feature matrix
        cluster_labels: Cluster assignments

    Returns:
        Dictionary with separation metrics
    """
    try:
        from sklearn.metrics import silhouette_score, silhouette_samples

        overall_silhouette = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        cluster_silhouettes = {}
        logger.info("Cluster Separation Analysis:")
        logger.info(f"Overall Silhouette Score: {overall_silhouette:.4f}")

        for i in np.unique(cluster_labels):
            cluster_silhouette = np.mean(sample_silhouette_values[cluster_labels == i])
            cluster_silhouettes[f'cluster_{i}'] = cluster_silhouette

            quality = "Excellent" if cluster_silhouette > 0.5 else "Good" if cluster_silhouette > 0.3 else "Fair"
            logger.info(f"Cluster {i}: {cluster_silhouette:.4f} ({quality})")

        return {
            'overall_silhouette': overall_silhouette,
            'cluster_silhouettes': cluster_silhouettes
        }
    except Exception as e:
        logger.error(f"analyze_cluster_separation failed: {e}")
        raise


def get_cluster_centers(kmeans: KMeans, feature_names: list) -> pd.DataFrame:
    """
    Get cluster centers as DataFrame.

    Args:
        kmeans: Fitted K-Means model
        feature_names: List of feature names

    Returns:
        DataFrame with cluster centers
    """
    try:
        centers_df = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=feature_names
        )
        centers_df.index.name = 'cluster'
        return centers_df
    except Exception as e:
        logger.error(f"get_cluster_centers failed: {e}")
        raise


__all__ = [
    'perform_kmeans_clustering',
    'validate_clustering_stability',
    'analyze_cluster_separation',
    'get_cluster_centers',
]

if __name__ == "__main__":
    print("Testing n3f_clustering module...")
    print("[OK] Module loaded successfully")
