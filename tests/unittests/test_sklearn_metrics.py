import numpy as np
import torch

from representation_learning.metrics.sklearn_metrics import (
    Accuracy,
    AveragePrecision,
    BalancedAccuracy,
    BinaryF1Score,
    MeanAveragePrecision,
    MulticlassBinaryF1Score,
)


def test_accuracy() -> None:
    metric = Accuracy()

    # Test perfect accuracy
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 1])
    metric.update(logits, y)
    assert metric.get_metric()["acc"] == 1.0

    # Test 50% accuracy
    metric = Accuracy()  # Reset metric
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    y = torch.tensor([0, 1])
    metric.update(logits, y)
    assert metric.get_metric()["acc"] == 0.5  # 1 correct out of 2

    # Test empty case
    metric = Accuracy()
    assert metric.get_metric()["acc"] == 0.0


def test_binary_f1_score() -> None:
    # Test perfect F1
    metric = BinaryF1Score()
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 1])
    metric.update(logits, y)
    metrics = metric.get_metric()
    assert metrics["prec"] == 1.0
    assert metrics["rec"] == 1.0
    assert metrics["f1"] == 1.0

    # Test partial F1 with new metric instance
    metric = BinaryF1Score()  # Reset metric
    logits = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y = torch.tensor(
        [1, 0, 0]
    )  # One true positive, one false positive, one true negative
    metric.update(logits, y)
    metrics = metric.get_metric()
    # Expected: 1 TP, 1 FP, 0 TN, 0 FN
    # Precision = TP / (TP + FP) = 1/2 = 0.5
    # Recall = TP / (TP + FN) = 1/1 = 1.0
    # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 0.67
    assert abs(metrics["prec"] - 0.5) < 1e-6
    assert abs(metrics["rec"] - 1.0) < 1e-6
    assert abs(metrics["f1"] - 0.67) < 0.01


def test_multiclass_binary_f1_score() -> None:
    metric = MulticlassBinaryF1Score(num_classes=3)

    # Test perfect F1 for all classes
    logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    metric.update(logits, y)
    metrics = metric.get_metric()
    assert metrics["macro_prec"] == 1.0
    assert metrics["macro_rec"] == 1.0
    assert metrics["macro_f1"] == 1.0

    # Test partial F1 with new metric instance
    metric = MulticlassBinaryF1Score(num_classes=3)  # Reset metric
    logits = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    y = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    metric.update(logits, y)
    metrics = metric.get_metric()
    assert metrics["macro_prec"] < 1.0
    assert metrics["macro_rec"] < 1.0
    assert metrics["macro_f1"] < 1.0


def test_average_precision() -> None:
    # Test perfect AP
    metric = AveragePrecision()
    output = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    target = torch.tensor([[1, 0], [0, 1]])
    metric.update(output, target)
    ap = metric.get_metric()
    assert np.allclose(ap, np.array([1.0, 1.0]))

    # Test partial AP with new metric instance and more challenging case
    metric = AveragePrecision()

    output = torch.tensor(
        [
            [0.1, 0.9],  # FP for class 1 (label is 0)
            [0.2, 0.8],  # FP for class 1 (label is 0)
            [0.3, 0.7],  # TP for class 1
            [0.4, 0.6],  # TP for class 1
            [0.5, 0.5],  # TP for class 1
            [0.9, 0.1],  # Irrelevant (class 0 only)
        ]
    )

    target = torch.tensor(
        [
            [0, 0],  # false positive for class 1
            [0, 0],  # false positive for class 1
            [0, 1],  # true positive
            [0, 1],  # true positive
            [0, 1],  # true positive
            [1, 0],  # irrelevant
        ]
    )

    metric.update(output, target)
    ap = metric.get_metric()

    print(f"AP scores: {ap}")
    assert ap[1] < 1.0, f"Expected AP[1] < 1.0, got {ap[1]}"


def test_mean_average_precision() -> None:
    # Test perfect mAP
    metric = MeanAveragePrecision()
    output = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    target = torch.tensor([[1, 0], [0, 1]])
    metric.update(output, target)
    assert metric.get_metric()["map"] == 1.0

    # Test partial mAP with new metric instance and more challenging case
    metric = MeanAveragePrecision()  # Reset metric
    output = torch.tensor(
        [
            [0.1, 0.9],  # FP for class 1 (label is 0)
            [0.2, 0.8],  # FP for class 1 (label is 0)
            [0.3, 0.7],  # TP for class 1
            [0.4, 0.6],  # TP for class 1
            [0.5, 0.5],  # TP for class 1
            [0.9, 0.1],  # Irrelevant (class 0 only)
        ]
    )

    target = torch.tensor(
        [
            [0, 0],  # false positive for class 1
            [0, 0],  # false positive for class 1
            [0, 1],  # true positive
            [0, 1],  # true positive
            [0, 1],  # true positive
            [1, 0],  # irrelevant
        ]
    )

    metric.update(output, target)
    map_value = metric.get_metric()["map"]
    assert map_value < 1.0, f"Expected mAP < 1.0, got {map_value}"


def test_balanced_accuracy() -> None:
    metric = BalancedAccuracy()

    # Test perfect balanced accuracy
    logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = torch.tensor([0, 1, 2])
    metric.update(logits, y)
    assert metric.get_metric()["balanced_acc"] == 1.0

    # Test imbalanced case with new metric instance
    metric = BalancedAccuracy()  # Reset metric
    logits = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Wrong prediction
            [0.0, 0.0, 1.0],
        ]
    )
    y = torch.tensor([0, 1, 2])
    metric.update(logits, y)
    assert metric.get_metric()["balanced_acc"] < 1.0

    # Test empty case
    metric = BalancedAccuracy()
    assert metric.get_metric()["balanced_acc"] == 0.0


def test_metrics_with_weights() -> None:
    # Test AveragePrecision with weights
    metric = AveragePrecision()
    output = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    target = torch.tensor([[1, 0], [1, 0]])
    weights = torch.tensor([1.0, 2.0])
    metric.update(output, target, weights)
    ap = metric.get_metric()
    assert ap.shape[0] == 2  # Should have AP for both classes

    # Test MeanAveragePrecision with weights
    metric = MeanAveragePrecision()
    metric.update(output, target, weights)
    assert metric.get_metric()["map"] >= 0.0
    assert metric.get_metric()["map"] <= 1.0


def test_edge_cases() -> None:
    # Test empty inputs
    metric = Accuracy()
    logits = torch.empty(0, 2)
    y = torch.empty(0)
    metric.update(logits, y)
    assert metric.get_metric()["acc"] == 0.0

    # Test single sample
    metric = BinaryF1Score()
    logits = torch.tensor([[0.0, 1.0]])
    y = torch.tensor([1])
    metric.update(logits, y)
    metrics = metric.get_metric()
    assert metrics["f1"] >= 0.0
    assert metrics["f1"] <= 1.0

    # Test all zeros
    metric = MulticlassBinaryF1Score(num_classes=2)
    logits = torch.zeros(3, 2)
    y = torch.zeros(3, 2)
    metric.update(logits, y)
    metrics = metric.get_metric()
    assert metrics["macro_f1"] >= 0.0
    assert metrics["macro_f1"] <= 1.0


def test_gpu_tensors() -> None:
    if torch.cuda.is_available():
        # Test Accuracy with GPU tensors
        metric = Accuracy()
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda")
        y = torch.tensor([0, 1], device="cuda")
        metric.update(logits, y)
        assert metric.get_metric()["acc"] == 1.0

        # Test BinaryF1Score with GPU tensors
        metric = BinaryF1Score()
        logits = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device="cuda")
        y = torch.tensor([1, 1], device="cuda")
        metric.update(logits, y)
        metrics = metric.get_metric()
        assert metrics["f1"] == 1.0
