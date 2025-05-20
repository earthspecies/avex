"""Retrieval / ranking evaluation utilities."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def evaluate_ranking(
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
