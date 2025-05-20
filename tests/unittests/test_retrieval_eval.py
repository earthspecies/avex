import numpy as np

from representation_learning.evaluation.retrieval import evaluate_ranking


def test_evaluate_ranking_simple() -> None:
    # Two classes, embeddings clearly separable
    embeddings = np.vstack(
        [
            np.tile([1.0, 0.0], (5, 1)),  # class 0
            np.tile([0.0, 1.0], (5, 1)),  # class 1
        ]
    )
    labels = np.array([0] * 5 + [1] * 5)

    auc = evaluate_ranking(embeddings, labels)
    # Perfect separation should yield ROC-AUC of 1.0
    assert np.isclose(auc, 1.0)
