from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from representation_learning.evaluation.retrieval import (
    evaluate_auc_roc,
    evaluate_precision,
)


def test_retrieval_metrics_simple() -> None:
    """Two perfectly separable classes should give ROC-AUC = 1.0 and P@1 = 1.0."""
    # Build a toy 2-D feature space with orthogonal class clusters
    class0: NDArray[np.float64] = np.tile([1.0, 0.0], (5, 1))
    class1: NDArray[np.float64] = np.tile([0.0, 1.0], (5, 1))
    embeddings: NDArray[np.float64] = np.vstack([class0, class1])

    labels: NDArray[np.int64] = np.array([0] * 5 + [1] * 5)

    auc: float = evaluate_auc_roc(embeddings, labels)
    assert np.isclose(auc, 1.0), f"Unexpected AUC value: {auc}"

    p_at_1: float = evaluate_precision(embeddings, labels, k=1)
    assert np.isclose(p_at_1, 1.0), f"Unexpected precision@1 value: {p_at_1}"


# ------------------------- random baseline ----------------------------- #


def test_random_embeddings_performance() -> None:
    """Random embeddings should be close to chance (AUC ≈ 0.5, P@1 ≈ 0.1)."""
    rng: np.random.Generator = np.random.default_rng(seed=42)

    n_classes: Final[int] = 10
    samples_per_class: Final[int] = 5
    emb_dim: Final[int] = 16

    labels: NDArray[np.int64] = np.repeat(
        np.arange(n_classes, dtype=np.int64), samples_per_class
    )
    embeddings: NDArray[np.float64] = rng.standard_normal(
        (n_classes * samples_per_class, emb_dim)
    )

    auc: float = evaluate_auc_roc(embeddings, labels)
    assert 0.3 <= auc <= 0.7, f"Random AUC {auc} out of expected range"

    p_at_1: float = evaluate_precision(embeddings, labels, k=1)
    expected_p: float = 1.0 / n_classes
    tol: float = 0.05
    assert (expected_p - tol) <= p_at_1 <= (expected_p + tol), (
        f"Random precision@1 {p_at_1} outside expected range around {expected_p}"
    )
