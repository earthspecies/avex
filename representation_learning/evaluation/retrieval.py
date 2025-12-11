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
    batch_size: int = 2048,
) -> Dict[str, float]:
    """Compute retrieval metrics using modular evaluation utilities.

    Parameters
    ----------
    embeds : torch.Tensor
        Embeddings tensor
    labels : torch.Tensor
        Labels tensor
    batch_size : int, optional
        Batch size for memory-efficient computation (default: 2048)

    Returns
    -------
    Dict[str, float]
        ``{"retrieval_roc_auc": value, "retrieval_precision_at_1": value}``.

    Raises
    ------
    ValueError
        If embeddings or labels are None or empty
    """
    # Input validation
    if embeds is None:
        raise ValueError("Embeddings cannot be None. Check embedding extraction.")
    if labels is None:
        raise ValueError("Labels cannot be None. Check label loading.")
    if embeds.numel() == 0:
        raise ValueError("Embeddings tensor is empty.")
    if labels.numel() == 0:
        raise ValueError("Labels tensor is empty.")

    roc_auc = evaluate_auc_roc_batched(
        embeds.numpy(), labels.numpy(), batch_size=batch_size
    )
    precision_at_1 = evaluate_precision_batched(
        embeds.numpy(), labels.numpy(), k=1, batch_size=batch_size
    )

    return {
        "retrieval_roc_auc": roc_auc,
        "retrieval_precision_at_1": precision_at_1,
    }


def eval_retrieval_cross_set(
    query_embeds: torch.Tensor,
    query_labels: torch.Tensor,
    db_embeds: torch.Tensor,
    db_labels: torch.Tensor,
) -> Dict[str, float]:
    """Compute retrieval metrics using separate query and database sets.

    Parameters
    ----------
    query_embeds : torch.Tensor
        Query embeddings (e.g., from train set)
    query_labels : torch.Tensor
        Query labels
    db_embeds : torch.Tensor
        Database embeddings to search (e.g., from test set)
    db_labels : torch.Tensor
        Database labels

    Returns
    -------
    Dict[str, float]
        Retrieval metrics
    """
    roc_auc = evaluate_auc_roc_cross_set(
        query_embeds.numpy(),
        query_labels.numpy(),
        db_embeds.numpy(),
        db_labels.numpy(),
    )
    precision_at_1 = evaluate_precision_cross_set(
        query_embeds.numpy(),
        query_labels.numpy(),
        db_embeds.numpy(),
        db_labels.numpy(),
        k=1,
    )

    return {
        "retrieval_roc_auc": roc_auc,
        "retrieval_precision_at_1": precision_at_1,
    }


# ------------------------------------------------------------------------- #
#  Helpers
# ------------------------------------------------------------------------- #


def _convert_labels_to_int(labels: np.ndarray) -> np.ndarray:
    """Convert labels to integer indices when they are one-hot encoded.

    If *labels* is a 2-D array and **every** row contains exactly one ``1``
    (i.e. traditional one-hot vectors for single-label classification), the
    function reduces it to a 1-D vector of class indices.  Otherwise the input
    is returned unchanged so that down-stream code can treat it as multi-label
    data (multi-hot encoding).

    Returns
    -------
    np.ndarray
        Either the original labels array or a 1-D array of class indices.
    """

    if labels.ndim == 2 and labels.dtype in (
        np.float32,
        np.float64,
        np.int32,
        np.int64,
    ):
        row_sums = labels.sum(axis=1)
        if np.all(row_sums == 1):
            return labels.argmax(axis=1)
    return labels


def _binary_relevance_matrix(labels: np.ndarray, i: int) -> np.ndarray:
    """Return binary relevance vector for query *i* against *labels*.

    The function supports three situations:

    1. ``labels`` is 1-D (integers): positives are items whose label equals
       ``labels[i]``.
    2. ``labels`` is 2-D and one-hot (each sample has exactly one class): this
       is reduced to case (1) via :func:`_convert_labels_to_int`.
    3. ``labels`` is 2-D **multi-hot** (genuine multi-label data): positives are
       items that share at least one active class with the query, i.e.
       ``(labels & labels[i]).any(axis=1)``.

    Returns
    -------
    np.ndarray
        Binary relevance vector where 1 indicates relevant items for query i.
    """

    if labels.ndim == 1:
        return (labels == labels[i]).astype(int)

    # At this point labels is 2-D.  Try collapsing one-hot situation first.
    collapsed = _convert_labels_to_int(labels)
    if collapsed.ndim == 1:
        return (collapsed == collapsed[i]).astype(int)

    # Genuine multi-label: treat items as positive if **any** label overlaps.
    return np.logical_and(labels, labels[i]).any(axis=1).astype(int)


def _binary_relevance_matrix_cross_set(
    query_labels: np.ndarray, db_labels: np.ndarray, i: int
) -> np.ndarray:
    """Return binary relevance vector for query *i* against database labels.

    The function supports the same label formats as _binary_relevance_matrix
    but compares query labels against database labels for cross-set retrieval.

    Parameters
    ----------
    query_labels : np.ndarray
        Labels for the query set
    db_labels : np.ndarray
        Labels for the database set
    i : int
        Index of the query

    Returns
    -------
    np.ndarray
        Binary relevance vector where 1 indicates relevant database items for query i.
    """
    query_label = query_labels[i]

    if query_labels.ndim == 1:
        return (db_labels == query_label).astype(int)

    # Handle multi-dimensional case
    if query_labels.ndim == 2:
        # Try collapsing one-hot situation first
        collapsed_query = _convert_labels_to_int(query_labels)
        collapsed_db = _convert_labels_to_int(db_labels)

        if collapsed_query.ndim == 1 and collapsed_db.ndim == 1:
            return (collapsed_db == collapsed_query[i]).astype(int)

        # Genuine multi-label: treat items as positive if **any** label overlaps
        if db_labels.ndim == 2:
            return np.logical_and(db_labels, query_label).any(axis=1).astype(int)
        else:
            # db_labels is 1D but query_labels is 2D multi-label
            # This case is less common but we handle it
            return np.zeros(len(db_labels), dtype=int)

    return np.zeros(len(db_labels), dtype=int)


# ------------------------------------------------------------------------- #
#  AUC-ROC
# ------------------------------------------------------------------------- #


def evaluate_auc_roc_batched(
    embeddings: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    batch_size: int = 2048,
) -> float:
    """Compute average ROC-AUC for *instance-level* retrieval using batched processing.

    This is a memory-efficient version of evaluate_auc_roc that processes queries
    in batches to avoid creating the full N×N similarity matrix. Results are
    identical to the non-batched version.

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
    batch_size : int, optional
        Number of queries to process in each batch (default: 2048).

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

    # Normalize embeddings once
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    n = embeddings.shape[0]
    aucs: list[float] = []

    # Process queries in batches
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)

        # Compute similarities for this batch: batch_size × N
        sim_batch = np.matmul(normed[batch_start:batch_end], normed.T)

        # Process each query in the batch
        for i in range(batch_start, batch_end):
            local_i = i - batch_start
            y_true = _binary_relevance_matrix(labels, i)

            # Skip queries without at least one positive *other* than itself.
            if y_true.sum() <= 1:
                continue

            y_score = sim_batch[local_i]

            # Remove self-match
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            try:
                auc = roc_auc_score(y_true[mask], y_score[mask])
                aucs.append(float(auc))
            except ValueError as v:  # pragma: no cover – extremely rare but safe-guard
                logger.warning(
                    "ROC-AUC computation failed for query %d (reason: %s)",
                    i,
                    v,
                )
                continue

    return float(np.mean(aucs)) if aucs else 0.0


def evaluate_auc_roc(
    embeddings: np.ndarray, labels: Sequence[int] | np.ndarray
) -> float:
    """Compute average ROC-AUC for *instance-level* retrieval.

    For every query embedding *q* we rank *all* database embeddings by cosine
    similarity and compute the binary ROC-AUC where positives are items that
    share the same class label as *q*.  The final score is the mean AUC over
    all queries for which at least one positive and one negative sample exist;
    queries without a positive counterpart are skipped.

    In the case of multi-label classification, we treat an item as positive
    if it shares at least one class label with the query.

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
        y_true = _binary_relevance_matrix(labels, i)

        # Skip queries without at least one positive *other* than itself.
        if y_true.sum() <= 1:
            continue

        y_score = sim[i]

        # Remove self-match
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        try:
            auc = roc_auc_score(y_true[mask], y_score[mask])
            aucs.append(float(auc))
        except ValueError as v:  # pragma: no cover – extremely rare but safe-guard
            logger.warning("ROC-AUC computation failed for query %d (reason: %s)", i, v)
            continue

    return float(np.mean(aucs)) if aucs else 0.0


def evaluate_auc_roc_cross_set(
    query_embeds: np.ndarray,
    query_labels: np.ndarray,
    db_embeds: np.ndarray,
    db_labels: np.ndarray,
) -> float:
    """Compute ROC-AUC for cross-set retrieval (queries vs database).

    Parameters
    ----------
    query_embeds : np.ndarray, shape (N_q, D)
        Query embedding matrix.
    query_labels : np.ndarray, shape (N_q,)
        Query labels.
    db_embeds : np.ndarray, shape (N_db, D)
        Database embedding matrix.
    db_labels : np.ndarray, shape (N_db,)
        Database labels.

    Returns
    -------
    float
        Mean ROC-AUC over valid queries (0.0 if none valid).

    Raises
    ------
    ValueError
        If embeddings are not 2-D or if label lengths don't match embeddings.
    """
    if query_embeds.ndim != 2 or db_embeds.ndim != 2:
        raise ValueError("embeddings must be 2-D (N, D)")

    query_labels = np.asarray(query_labels)
    db_labels = np.asarray(db_labels)

    if query_labels.shape[0] != query_embeds.shape[0]:
        raise ValueError("query labels length must match number of query embeddings")
    if db_labels.shape[0] != db_embeds.shape[0]:
        raise ValueError(
            "database labels length must match number of database embeddings"
        )

    # Compute cosine similarity between query and database embeddings
    query_normed = query_embeds / np.linalg.norm(
        query_embeds, axis=1, keepdims=True
    ).clip(1e-12)
    db_normed = db_embeds / np.linalg.norm(db_embeds, axis=1, keepdims=True).clip(1e-12)
    sim = np.matmul(query_normed, db_normed.T)  # Shape: (n_queries, n_db)

    aucs: list[float] = []
    for i in range(query_embeds.shape[0]):
        # For each query, determine which database items are relevant
        y_true = _binary_relevance_matrix_cross_set(query_labels, db_labels, i)

        if y_true.sum() == 0:  # No positive samples
            continue

        y_score = sim[i]  # Similarities to all database items

        try:
            auc = roc_auc_score(y_true, y_score)
            aucs.append(float(auc))
        except ValueError as v:
            logger.warning("ROC-AUC computation failed for query %d (reason: %s)", i, v)
            continue

    return float(np.mean(aucs)) if aucs else 0.0


# ----------------------------------------------------------------------------- #
# Precision@k
# ----------------------------------------------------------------------------- #


def evaluate_precision_batched(
    embeddings: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    k: int = 1,
    batch_size: int = 2048,
) -> float:
    """Compute mean precision@k for *instance-level* retrieval using batched processing.

    This is a memory-efficient version of evaluate_precision that processes queries
    in batches to avoid creating the full N×N similarity matrix. Results are
    identical to the non-batched version.

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
    batch_size : int, optional
        Number of queries to process in each batch (default: 2048).

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

    # Normalize embeddings once
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    k = min(k, n - 1)  # Can't retrieve more than n-1 items
    precisions: list[float] = []

    # Process queries in batches
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)

        # Compute similarities for this batch: batch_size × N
        sim_batch = np.matmul(normed[batch_start:batch_end], normed.T)

        # Set self-similarities to -inf
        for i in range(batch_start, batch_end):
            local_i = i - batch_start
            sim_batch[local_i, i] = -np.inf

        # Process each query in the batch
        for i in range(batch_start, batch_end):
            local_i = i - batch_start
            y_true = _binary_relevance_matrix(labels, i)

            # Skip queries without at least one positive *other* than itself
            if y_true.sum() <= 1:
                continue

            # Identify indices of the top-k most similar items
            if k == 1:
                # Fast path for k=1 using argmax
                topk_idx = [int(np.argmax(sim_batch[local_i]))]
            else:
                # Use argpartition for efficiency when k>1
                topk_idx = np.argpartition(-sim_batch[local_i], k)[:k]

            # Precision is the fraction of relevant items among the retrieved top-k.
            precision_i = float(np.mean(y_true[topk_idx]))
            precisions.append(precision_i)

    return float(np.mean(precisions)) if precisions else 0.0


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
        y_true = _binary_relevance_matrix(labels, i)

        # Skip queries without at least one positive *other* than itself
        # includes "None" class.
        if y_true.sum() <= 1:
            continue

        # Identify indices of the top-k most similar items (self-match already
        # excluded via -inf on the diagonal).
        if k == 1:
            # Fast path for k=1 using argmax
            topk_idx = [int(np.argmax(sim[i]))]
        else:
            # Use argpartition for efficiency when k>1
            topk_idx = np.argpartition(-sim[i], k)[:k]

        # Precision is the fraction of relevant items among the retrieved top-k.
        precision_i = float(np.mean(y_true[topk_idx]))
        precisions.append(precision_i)

    return float(np.mean(precisions)) if precisions else 0.0


def evaluate_precision_cross_set(
    query_embeds: np.ndarray,
    query_labels: np.ndarray,
    db_embeds: np.ndarray,
    db_labels: np.ndarray,
    k: int = 1,
) -> float:
    """Compute precision@k for cross-set retrieval (queries vs database).

    Parameters
    ----------
    query_embeds : np.ndarray, shape (N_q, D)
        Query embedding matrix.
    query_labels : np.ndarray, shape (N_q,)
        Query labels.
    db_embeds : np.ndarray, shape (N_db, D)
        Database embedding matrix.
    db_labels : np.ndarray, shape (N_db,)
        Database labels.
    k : int, optional
        Number of top elements to consider (default: 1).

    Returns
    -------
    float
        Mean precision@k over all queries.

    Raises
    ------
    ValueError
        If embeddings are not 2-D or if label lengths don't match embeddings.
    """
    if query_embeds.ndim != 2 or db_embeds.ndim != 2:
        raise ValueError("embeddings must be 2-D (N, D)")

    query_labels = np.asarray(query_labels)
    db_labels = np.asarray(db_labels)

    if query_labels.shape[0] != query_embeds.shape[0]:
        raise ValueError("query labels length must match number of query embeddings")
    if db_labels.shape[0] != db_embeds.shape[0]:
        raise ValueError(
            "database labels length must match number of database embeddings"
        )

    n_db = db_embeds.shape[0]
    if n_db == 0:
        return 0.0

    # Compute cosine similarity between query and database embeddings
    query_normed = query_embeds / np.linalg.norm(
        query_embeds, axis=1, keepdims=True
    ).clip(1e-12)
    db_normed = db_embeds / np.linalg.norm(db_embeds, axis=1, keepdims=True).clip(1e-12)
    sim = np.matmul(query_normed, db_normed.T)  # Shape: (n_queries, n_db)

    k = min(k, n_db)  # Can't retrieve more than available database items

    precisions: list[float] = []
    for i in range(query_embeds.shape[0]):
        # For each query, determine which database items are relevant
        y_true = _binary_relevance_matrix_cross_set(query_labels, db_labels, i)

        if y_true.sum() == 0:  # No positive samples
            continue

        # Identify indices of the top-k most similar database items
        if k == 1:
            # Fast path for k=1 using argmax
            topk_idx = [int(np.argmax(sim[i]))]
        else:
            # Use argpartition for efficiency when k>1
            topk_idx = np.argpartition(-sim[i], k)[:k]

        # Precision is the fraction of relevant items among the retrieved top-k
        precision_i = float(np.mean(y_true[topk_idx]))
        precisions.append(precision_i)

    return float(np.mean(precisions)) if precisions else 0.0
