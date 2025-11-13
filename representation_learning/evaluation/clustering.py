"""Clustering evaluation utilities for unsupervised clustering quality assessment."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)

logger = logging.getLogger(__name__)


def eval_clustering(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute clustering metrics using K-means clustering.

    This function performs K-means clustering on the embeddings and evaluates
    the clustering quality against ground truth labels using multiple metrics.

    Parameters
    ----------
    embeds : torch.Tensor, shape (N, D)
        Embedding vectors for clustering.
    labels : torch.Tensor, shape (N,)
        Ground truth labels for evaluation.
    n_clusters : Optional[int], optional
        Number of clusters for K-means. If None, uses the number of unique labels.
    random_state : int, optional
        Random state for reproducible results, by default 42.

    Returns
    -------
    Dict[str, float]
        Dictionary containing clustering metrics:
        - clustering_ari: Adjusted Rand Index
        - clustering_nmi: Normalized Mutual Information
        - clustering_v_measure: V-measure

    Raises
    ------
    ValueError
        If embeddings or labels are empty, or if shapes don't match.
    """
    if embeds.numel() == 0 or labels.numel() == 0:
        logger.warning("Empty embeddings or labels provided to clustering evaluation")
        return _get_empty_clustering_metrics()

    if embeds.shape[0] != labels.shape[0]:
        raise ValueError(f"Embeddings and labels must have same length: {embeds.shape[0]} vs {labels.shape[0]}")

    # Convert to numpy for sklearn compatibility
    embeds_np = embeds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Handle multi-label case by converting to single labels if needed
    if labels_np.ndim > 1:
        if labels_np.shape[1] == 1:
            labels_np = labels_np.squeeze()
        else:
            # For multi-label, use argmax to get primary label
            labels_np = labels_np.argmax(axis=1)

    # Determine number of clusters
    if n_clusters is None:
        unique_labels = np.unique(labels_np)
        # Filter out any invalid labels (e.g., -1 for unknown)
        unique_labels = unique_labels[unique_labels >= 0]
        n_clusters = len(unique_labels)

    if n_clusters < 2:
        logger.warning(f"Need at least 2 clusters for meaningful clustering evaluation, got {n_clusters}")
        return _get_empty_clustering_metrics()

    if n_clusters > embeds_np.shape[0]:
        logger.warning(f"Number of clusters ({n_clusters}) cannot exceed number of samples ({embeds_np.shape[0]})")
        n_clusters = min(n_clusters, embeds_np.shape[0])

    try:
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        cluster_labels = kmeans.fit_predict(embeds_np)

        # Compute clustering metrics
        metrics = {}

        # Metrics comparing clustering to ground truth
        metrics["clustering_ari"] = float(adjusted_rand_score(labels_np, cluster_labels))
        metrics["clustering_nmi"] = float(normalized_mutual_info_score(labels_np, cluster_labels))
        metrics["clustering_v_measure"] = float(v_measure_score(labels_np, cluster_labels))

        return metrics

    except Exception as e:
        logger.error(f"Clustering evaluation failed: {e}")
        return _get_empty_clustering_metrics()


def eval_clustering_multiple_k(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    k_range: Optional[tuple[int, int]] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate clustering with multiple values of K and return the best metrics.

    This function tries different numbers of clusters and returns the best
    clustering results based on Adjusted Rand Index (silhouette removed for speed).

    Parameters
    ----------
    embeds : torch.Tensor, shape (N, D)
        Embedding vectors for clustering.
    labels : torch.Tensor, shape (N,)
        Ground truth labels for evaluation.
    k_range : Optional[tuple[int, int]], optional
        Range of K values to try as (min_k, max_k). If None, uses a reasonable
        range around the true number of classes.
    random_state : int, optional
        Random state for reproducible results, by default 42.

    Returns
    -------
    Dict[str, float]
        Dictionary containing best clustering metrics plus the optimal K:
        - clustering_best_k: Optimal number of clusters
        - clustering_ari_best: Best Adjusted Rand Index
        - clustering_nmi_best: Best Normalized Mutual Information
        - clustering_v_measure_best: Best V-measure
    """
    if embeds.numel() == 0 or labels.numel() == 0:
        logger.warning("Empty embeddings or labels provided to clustering evaluation")
        return _get_empty_clustering_best_metrics()

    # Convert to numpy for sklearn compatibility
    embeds_np = embeds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Handle multi-label case
    if labels_np.ndim > 1:
        if labels_np.shape[1] == 1:
            labels_np = labels_np.squeeze()
        else:
            labels_np = labels_np.argmax(axis=1)

    # Determine K range
    if k_range is None:
        unique_labels = np.unique(labels_np)
        unique_labels = unique_labels[unique_labels >= 0]
        true_k = len(unique_labels)
        min_k = max(2, true_k - 2)
        max_k = min(embeds_np.shape[0] // 2, true_k + 3)
        k_range = (min_k, max_k)

    best_metrics = {}
    best_score = -1.0  # Use ARI as the objective now
    best_k = k_range[0]

    for k in range(k_range[0], k_range[1] + 1):
        if k >= embeds_np.shape[0]:
            break

        metrics = eval_clustering(embeds, labels, n_clusters=k, random_state=random_state)

        if metrics["clustering_ari"] > best_score:
            best_score = metrics["clustering_ari"]
            best_k = k
            best_metrics = {
                "clustering_best_k": float(best_k),
                "clustering_ari_best": metrics["clustering_ari"],
                "clustering_nmi_best": metrics["clustering_nmi"],
                "clustering_v_measure_best": metrics["clustering_v_measure"],
            }

    return best_metrics if best_metrics else _get_empty_clustering_best_metrics()


def _get_empty_clustering_metrics() -> Dict[str, float]:
    """Return empty clustering metrics dictionary.

    Returns
    -------
    Dict[str, float]
        Dictionary with clustering metric keys set to 0.0
    """
    return {
        "clustering_ari": 0.0,
        "clustering_nmi": 0.0,
        "clustering_v_measure": 0.0,
    }


def _get_empty_clustering_best_metrics() -> Dict[str, float]:
    """Return empty best clustering metrics dictionary.

    Returns
    -------
    Dict[str, float]
        Dictionary with best clustering metric keys set to 0.0
    """
    return {
        "clustering_best_k": 0.0,
        "clustering_ari_best": 0.0,
        "clustering_nmi_best": 0.0,
        "clustering_v_measure_best": 0.0,
    }
