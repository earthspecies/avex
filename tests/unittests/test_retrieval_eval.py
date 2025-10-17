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


# -------------------- single-label one-hot equivalence -------------------- #


def test_one_hot_equivalence() -> None:  # noqa: D401
    """Single-label one-hot encoding should yield identical metrics."""
    # Two-class toy setup -----------------------------------------------------
    emb0 = np.tile([1.0, 0.0], (3, 1))  # class 0
    emb1 = np.tile([0.0, 1.0], (3, 1))  # class 1
    embeddings: NDArray[np.float64] = np.vstack([emb0, emb1])

    labels_int: NDArray[np.int64] = np.array([0] * 3 + [1] * 3)
    labels_one_hot: NDArray[np.float32] = np.eye(2, dtype=np.float32)[labels_int]

    # Metrics for integer labels --------------------------------------------
    auc_int = evaluate_auc_roc(embeddings, labels_int)
    p_int = evaluate_precision(embeddings, labels_int, k=1)

    # Metrics for one-hot labels --------------------------------------------
    auc_hot = evaluate_auc_roc(embeddings, labels_one_hot)
    p_hot = evaluate_precision(embeddings, labels_one_hot, k=1)

    assert np.isclose(auc_int, auc_hot), (
        f"AUC mismatch between int and one-hot labels: {auc_int} vs {auc_hot}"
    )
    assert np.isclose(p_int, p_hot), (
        f"Precision@1 mismatch between int and one-hot labels: {p_int} vs {p_hot}"
    )


# ------------------------- simple multi-label case ------------------------ #


def test_multilabel_retrieval() -> None:  # noqa: D401
    """Verify metrics behave sensibly on a small multi-label example."""
    # 4 samples, 3 possible classes -----------------------------------------
    labels: NDArray[np.int64] = np.array(
        [
            [1, 0, 0],  # sample 0 – class 0
            [1, 1, 0],  # sample 1 – classes 0 & 1
            [0, 1, 1],  # sample 2 – classes 1 & 2
            [0, 0, 1],  # sample 3 – class 2
        ],
        dtype=np.int64,
    )

    # Use the label vectors themselves as embeddings – cosine similarity will
    # reflect label overlap perfectly.
    embeddings: NDArray[np.float64] = labels.astype(np.float64)

    auc = evaluate_auc_roc(embeddings, labels)
    p_at_1 = evaluate_precision(embeddings, labels, k=1)

    # In this construction every query has at least one *other* sample that
    # shares a label and is its most similar neighbour, so Precision@1 should
    # be 1.0 and AUC should also be 1.0.
    assert np.isclose(p_at_1, 1.0), f"Unexpected P@1 for multi-label toy set: {p_at_1}"
    assert np.isclose(auc, 1.0), f"Unexpected AUC for multi-label toy set: {auc}"


# ------------------------- "None" class skipping --------------------------- #


def test_none_class_skipping() -> None:
    """Verify that samples with no positive labels are skipped in evaluation."""
    # 6 samples: 4 with labels, 2 with no labels ("None")
    labels: NDArray[np.int64] = np.array(
        [
            [1, 0],  # sample 0 – class 0
            [1, 0],  # sample 1 – class 0
            [0, 1],  # sample 2 – class 1
            [0, 1],  # sample 3 – class 1
            [0, 0],  # sample 4 – "None" class
            [0, 0],  # sample 5 – "None" class
        ],
        dtype=np.int64,
    )

    # Create embeddings where labeled samples get perfect precision,
    # but None samples would get 0 precision if not skipped
    embeddings: NDArray[np.float64] = np.array(
        [
            [1.0, 0.0],  # class 0 cluster
            [0.9, 0.1],  # class 0 cluster
            [0.0, 1.0],  # class 1 cluster
            [0.1, 0.9],  # class 1 cluster
            [0.5, 0.5],  # None sample - positioned between clusters
            [0.4, 0.6],  # None sample - positioned between clusters
        ],
        dtype=np.float64,
    )

    p_at_1 = evaluate_precision(embeddings, labels, k=1)

    # If None samples are properly skipped: only 4 labeled samples contribute,
    # each gets precision 1.0, so mean = 1.0
    # If None samples are NOT skipped: all 6 samples contribute,
    # 4 labeled get precision 1.0, 2 None get precision 0.0 (since
    # _binary_relevance_matrix returns all zeros for None samples),
    # so mean = (4*1.0 + 2*0.0)/6 = 0.667

    assert np.isclose(p_at_1, 1.0), (
        f"Expected P@1=1.0 when None samples are skipped, got {p_at_1}. "
        f"If this is ~0.667, None samples are not being skipped properly."
    )

    # Double-check the skipping logic by verifying _binary_relevance_matrix behavior
    from representation_learning.evaluation.retrieval import (
        _binary_relevance_matrix,
    )

    # None samples should have no relevant items (including each other)
    relevance_none_1 = _binary_relevance_matrix(labels, 4)
    relevance_none_2 = _binary_relevance_matrix(labels, 5)

    assert relevance_none_1.sum() == 0, "None sample should have zero relevance vector"
    assert relevance_none_2.sum() == 0, "None sample should have zero relevance vector"


def test_none_class_impact_on_average() -> None:
    """Demonstrate the measurable impact of skipping None samples."""
    import numpy as np

    from representation_learning.evaluation.retrieval import (
        _binary_relevance_matrix,
    )

    # Setup: 3 labeled samples + 3 None samples
    labels: NDArray[np.int64] = np.array(
        [
            [1, 0],  # sample 0 – class 0
            [1, 0],  # sample 1 – class 0
            [0, 1],  # sample 2 – class 1
            [0, 0],  # sample 3 – "None"
            [0, 0],  # sample 4 – "None"
            [0, 0],  # sample 5 – "None"
        ],
        dtype=np.int64,
    )

    # Embeddings: class 0 and class 1 samples cluster together
    embeddings: NDArray[np.float64] = np.array(
        [
            [1.0, 0.0],  # class 0
            [0.9, 0.1],  # class 0
            [0.0, 1.0],  # class 1
            [0.5, 0.5],  # None
            [0.4, 0.6],  # None
            [0.3, 0.7],  # None
        ],
        dtype=np.float64,
    )

    # Test the actual skipping behavior
    p_at_1 = evaluate_precision(embeddings, labels, k=1)

    # With proper skipping:
    # - Sample 0 retrieves sample 1 (same class) → precision 1.0
    # - Sample 1 retrieves sample 0 (same class) → precision 1.0
    # - Sample 2 has no other class 1 samples, so gets skipped
    # - Samples 3,4,5 get skipped (None class)
    # Mean = (1.0 + 1.0) / 2 = 1.0

    # If NOT properly skipped, None samples would contribute 0.0 each:
    # Mean = (1.0 + 1.0 + 0.0 + 0.0 + 0.0 + 0.0) / 6 = 0.333...

    assert np.isclose(p_at_1, 1.0), (
        f"Expected P@1=1.0 with proper skipping, got {p_at_1}. "
        f"If ~0.33, None samples aren't being skipped."
    )

    # Verify None samples would indeed contribute 0.0 if not skipped
    for none_idx in [3, 4, 5]:
        relevance = _binary_relevance_matrix(labels, none_idx)
        assert relevance.sum() == 0, (
            f"None sample {none_idx} should have zero relevance"
        )


def test_none_class_skipping_comparison() -> None:
    """Compare results with and without None class skipping to prove it works."""
    from typing import Sequence

    import numpy as np

    from representation_learning.evaluation.retrieval import (
        _binary_relevance_matrix,
    )

    def evaluate_precision_no_skip(
        embeddings: np.ndarray,
        labels: Sequence[int] | np.ndarray,
        k: int = 1,
    ) -> float:
        """Version of evaluate_precision that does NOT skip None samples.

        Returns
        -------
        float
            Mean precision@k over all queries including None samples.

        Raises
        ------
        ValueError
            If embeddings is not 2-D or labels length mismatches embeddings.
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
        normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(
            1e-12
        )
        sim = np.matmul(normed, normed.T)
        np.fill_diagonal(sim, -np.inf)
        k = min(k, n - 1)

        precisions: list[float] = []
        for i in range(n):
            # NO SKIPPING - process all samples including None
            y_true = _binary_relevance_matrix(labels, i)

            if k == 1:
                topk_idx = [int(np.argmax(sim[i]))]
            else:
                topk_idx = np.argpartition(-sim[i], k)[:k]

            precision_i = float(np.mean(y_true[topk_idx]))
            precisions.append(precision_i)

        return float(np.mean(precisions)) if precisions else 0.0

    # Test setup
    labels: NDArray[np.int64] = np.array(
        [
            [1, 0],  # sample 0 – class 0
            [1, 0],  # sample 1 – class 0
            [0, 1],  # sample 2 – class 1 (isolated)
            [0, 0],  # sample 3 – "None"
            [0, 0],  # sample 4 – "None"
        ],
        dtype=np.int64,
    )

    embeddings: NDArray[np.float64] = np.array(
        [
            [1.0, 0.0],  # class 0
            [0.9, 0.1],  # class 0
            [0.0, 1.0],  # class 1
            [0.5, 0.5],  # None
            [0.4, 0.6],  # None
        ],
        dtype=np.float64,
    )

    # Compare results
    precision_with_skip = evaluate_precision(embeddings, labels, k=1)
    precision_no_skip = evaluate_precision_no_skip(embeddings, labels, k=1)

    # With skipping: only samples 0,1 are evaluated (both get precision 1.0)
    # Sample 2 skipped (no other class 1), samples 3,4 skipped (None)
    # Result: (1.0 + 1.0) / 2 = 1.0

    # Without skipping: all 5 samples evaluated
    # Sample 0: precision 1.0, Sample 1: precision 1.0
    # Sample 2: precision 0.0 (retrieves class 0, but needs class 1)
    # Sample 3: precision 0.0 (None relevance vector is all zeros)
    # Sample 4: precision 0.0 (None relevance vector is all zeros)
    # Result: (1.0 + 1.0 + 0.0 + 0.0 + 0.0) / 5 = 0.4

    assert np.isclose(precision_with_skip, 1.0), (
        f"With skipping: expected 1.0, got {precision_with_skip}"
    )
    assert np.isclose(precision_no_skip, 0.4), (
        f"Without skipping: expected 0.4, got {precision_no_skip}"
    )

    print(f"Precision with skipping: {precision_with_skip}")
    print(f"Precision without skipping: {precision_no_skip}")
    print("✓ None class skipping is working correctly!")
