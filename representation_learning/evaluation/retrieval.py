"""Retrieval / ranking evaluation utilities."""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------- #
#  Retrieval helper
# -------------------------------------------------------------------- #
def eval_retrieval(
    embeds: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """Compute retrieval metrics using modular evaluation utilities.

    Returns
    -------
    Dict[str, float]
        ``{"retrieval_roc_auc": value, "retrieval_precision_at_1": value}``.
    """
    roc_auc = evaluate_auc_roc(embeds.numpy(), labels.numpy())
    precision_at_1 = evaluate_precision(embeds.numpy(), labels.numpy(), k=1)

    return {
        "retrieval_roc_auc": roc_auc,
        "retrieval_precision_at_1": precision_at_1,
    }


def evaluate_auc_roc(
    embeddings: np.ndarray, labels: Sequence[int] | np.ndarray
) -> float:
    """Compute average ROC-AUC for *instance-level* retrieval.

    For every query embedding *q* we rank *all* database embeddings by cosine
    similarity and compute the binary ROC-AUC where positives are items that
    share the same class label as *q*.  The final score is the mean AUC over
    all queries for which at least one positive and one negative sample exist;
    queries without a positive counterpart are skipped.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        Feature matrix.
    labels : Sequence[int] | np.ndarray, shape (N,)
        Integer class labels.

    Returns
    -------
    float
        Mean ROC-AUC over valid queries (0.0 if none valid).

    Raises
    ------
    ValueError
        If *embeddings* is not 2-D or labels length mismatch.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D (N, D)")
    labels = np.asarray(labels)
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels length must match number of embeddings")

    # Cosine similarity matrix
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    sim = np.matmul(normed, normed.T)

    n = embeddings.shape[0]
    aucs: list[float] = []
    for i in range(n):
        y_true = (labels == labels[i]).astype(int)
        if y_true.sum() <= 1:  # Need at least one positive besides query itself
            continue
        y_score = sim[i]
        # Remove self-match
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            auc = roc_auc_score(y_true[mask], y_score[mask])
            aucs.append(float(auc))
        except ValueError:
            # Happens when only one class present after masking (shouldn't) but be safe
            logger.warning("Only one class present after masking for query %d", i)
            continue

    return float(np.mean(aucs)) if aucs else 0.0


# ----------------------------------------------------------------------------- #
# Precision@k
# ----------------------------------------------------------------------------- #


def evaluate_precision(
    embeddings: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    k: int = 1,
) -> float:
    """Compute mean precision@k for *instance-level* retrieval.

    For every query embedding *q* we rank **all** database embeddings by cosine
    similarity (excluding the query itself) and compute the precision@k where
    positives are items that share the same class label as *q*.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        Feature matrix.
    labels : Sequence[int] | np.ndarray, shape (N,)
        Integer class labels.
    k : int, optional
        Number of top elements to consider (default: 1).

    Returns
    -------
    float
        Mean precision@k over all queries.

    Raises
    ------
    ValueError
        If *embeddings* is not 2-D or if label length mismatches the number of
        embeddings.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D (N, D)")
    labels = np.asarray(labels)
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels length must match number of embeddings")

    n = embeddings.shape[0]
    if n <= 1:
        return 0.0

    # Cosine similarity matrix
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    sim = np.matmul(normed, normed.T)

    # Exclude self-matches by setting similarity to -inf on the diagonal
    np.fill_diagonal(sim, -np.inf)

    k = min(k, n - 1)  # Can't retrieve more than n-1 items

    precisions: list[float] = []
    for i in range(n):
        # Indices of top-k most similar items (highest cosine similarity)
        if k == 1:
            # Fast path for k=1 using argmax
            topk_idx = [int(np.argmax(sim[i]))]
        else:
            # Use argpartition for efficiency when k>1
            topk_idx = np.argpartition(-sim[i], k)[:k]

        # Precision@k: fraction of top-k items that share the same label
        precision_i = np.mean(labels[topk_idx] == labels[i])
        precisions.append(float(precision_i))

    return float(np.mean(precisions)) if precisions else 0.0
