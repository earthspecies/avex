"""Unit tests for linear probe metrics with different label formats."""

from __future__ import annotations

import numpy as np
import torch

from representation_learning.metrics.sklearn_metrics import (
    Accuracy,
    BalancedAccuracy,
    BinaryF1Score,
)


def test_accuracy_integer_labels() -> None:
    """Test Accuracy metric with integer labels."""
    metric = Accuracy()
    
    # Perfect predictions for 3-class problem
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    labels = torch.tensor([0, 1, 2])
    
    metric.update(logits, labels)
    result = metric.get_metric()
    
    assert result["acc"] == 1.0, f"Expected perfect accuracy, got {result['acc']}"


def test_accuracy_one_hot_labels() -> None:
    """Test Accuracy metric with one-hot encoded labels."""
    metric = Accuracy()
    
    # Perfect predictions for 3-class problem with one-hot labels
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    
    metric.update(logits, labels)
    result = metric.get_metric()
    
    assert result["acc"] == 1.0, f"Expected perfect accuracy, got {result['acc']}"


def test_accuracy_equivalence() -> None:
    """Test that integer and one-hot labels produce identical accuracy."""
    # Same logits for both cases
    logits = torch.tensor([
        [1.5, 0.2, 0.1],  # prediction: class 0
        [0.1, 1.8, 0.3],  # prediction: class 1
        [0.2, 0.1, 1.6],  # prediction: class 2
        [1.2, 0.5, 0.8],  # prediction: class 0
    ])
    
    # Integer labels
    labels_int = torch.tensor([0, 1, 2, 1])  # one wrong prediction
    metric_int = Accuracy()
    metric_int.update(logits, labels_int)
    acc_int = metric_int.get_metric()["acc"]
    
    # One-hot labels (same ground truth)
    labels_one_hot = torch.tensor([
        [1.0, 0.0, 0.0],  # class 0
        [0.0, 1.0, 0.0],  # class 1
        [0.0, 0.0, 1.0],  # class 2
        [0.0, 1.0, 0.0],  # class 1
    ])
    metric_one_hot = Accuracy()
    metric_one_hot.update(logits, labels_one_hot)
    acc_one_hot = metric_one_hot.get_metric()["acc"]
    
    assert acc_int == acc_one_hot, f"Accuracy mismatch: int={acc_int}, one-hot={acc_one_hot}"
    assert acc_int == 0.75, f"Expected 75% accuracy (3/4 correct), got {acc_int}"


def test_balanced_accuracy_equivalence() -> None:
    """Test that BalancedAccuracy works with both label formats."""
    # Imbalanced predictions: mostly predict class 0
    logits = torch.tensor([
        [2.0, 0.0],  # pred: 0, correct
        [2.0, 0.0],  # pred: 0, wrong (should be 1)
        [2.0, 0.0],  # pred: 0, correct
        [0.0, 2.0],  # pred: 1, correct
    ])
    
    # Integer labels: 3 class 0, 1 class 1
    labels_int = torch.tensor([0, 1, 0, 1])
    metric_int = BalancedAccuracy()
    metric_int.update(logits, labels_int)
    bal_acc_int = metric_int.get_metric()["balanced_acc"]
    
    # One-hot labels (same ground truth)
    labels_one_hot = torch.tensor([
        [1.0, 0.0],  # class 0
        [0.0, 1.0],  # class 1
        [1.0, 0.0],  # class 0
        [0.0, 1.0],  # class 1
    ])
    metric_one_hot = BalancedAccuracy()
    metric_one_hot.update(logits, labels_one_hot)
    bal_acc_one_hot = metric_one_hot.get_metric()["balanced_acc"]
    
    assert bal_acc_int == bal_acc_one_hot, f"Balanced accuracy mismatch: int={bal_acc_int}, one-hot={bal_acc_one_hot}"
    
    # For this case: class 0 accuracy = 2/2 = 1.0, class 1 accuracy = 1/2 = 0.5
    # Balanced accuracy = (1.0 + 0.5) / 2 = 0.75
    expected = 0.75
    assert abs(bal_acc_int - expected) < 1e-6, f"Expected balanced accuracy {expected}, got {bal_acc_int}"


def test_binary_f1_equivalence() -> None:
    """Test that BinaryF1Score works with both label formats."""
    # Binary classification logits
    logits = torch.tensor([
        [1.5, 0.5],  # pred: 0, correct
        [0.3, 1.8],  # pred: 1, correct
        [1.2, 0.4],  # pred: 0, wrong (should be 1)
        [0.1, 1.9],  # pred: 1, correct
    ])
    
    # Integer labels
    labels_int = torch.tensor([0, 1, 1, 1])
    metric_int = BinaryF1Score()
    metric_int.update(logits, labels_int)
    f1_int = metric_int.get_metric()["f1"]
    
    # One-hot labels
    labels_one_hot = torch.tensor([
        [1.0, 0.0],  # class 0
        [0.0, 1.0],  # class 1
        [0.0, 1.0],  # class 1
        [0.0, 1.0],  # class 1
    ])
    metric_one_hot = BinaryF1Score()
    metric_one_hot.update(logits, labels_one_hot)
    f1_one_hot = metric_one_hot.get_metric()["f1"]
    
    assert f1_int == f1_one_hot, f"F1 score mismatch: int={f1_int}, one-hot={f1_one_hot}"
    
    # Verify we get a reasonable F1 score
    assert 0.0 <= f1_int <= 1.0, f"F1 score should be between 0 and 1, got {f1_int}"


def test_accumulation_across_batches() -> None:
    """Test that metrics accumulate correctly across multiple update() calls."""
    metric = Accuracy()
    
    # Batch 1: perfect predictions
    logits1 = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # one-hot
    metric.update(logits1, labels1)
    
    # Batch 2: one wrong prediction
    logits2 = torch.tensor([[2.0, 0.0], [2.0, 0.0]])  # both predict class 0
    labels2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # second should be class 1
    metric.update(logits2, labels2)
    
    result = metric.get_metric()
    # Overall: 3 correct out of 4 = 75%
    assert result["acc"] == 0.75, f"Expected 75% accuracy across batches, got {result['acc']}" 