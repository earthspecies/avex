"""ROC-AUC input-contract and regression tests for issue #205."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

from avex.metrics.metric_factory import get_metric_class
from avex.metrics.sklearn_metrics import ROCAUC
from avex.training.metrics_tracker import MetricsTracker
from avex.training.training_strategies import SupervisedStrategy


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("probabilities", [False, True])
def test_roc_auc_class_indices(num_classes: int, probabilities: bool) -> None:
    """Both supported score representations agree with sklearn, not just at 1.0."""
    generator = torch.Generator().manual_seed(17)
    labels = torch.arange(30) % num_classes
    logits = torch.randn(30, num_classes, generator=generator, dtype=torch.float64)
    probs = logits.softmax(dim=1)
    scores = probs if probabilities else logits
    metric = get_metric_class("roc_auc")
    metric.update(scores, labels)

    expected = (
        roc_auc_score(labels, probs[:, 1])
        if num_classes == 2
        else roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    )
    assert 0.0 < expected < 1.0
    assert metric.get_metric()["roc_auc"] == pytest.approx(expected)
    assert metric.get_primary_metric() == pytest.approx(expected)


@pytest.mark.parametrize("num_classes", [2, 3])
def test_roc_auc_perfect_logits(num_classes: int) -> None:
    """The issue's binary and multiclass raw-logit cases both return 1.0."""
    labels = torch.arange(12) % num_classes
    logits = torch.nn.functional.one_hot(labels, num_classes).float() * 6.0
    metric = ROCAUC()
    metric.update(logits, labels)
    assert metric.get_primary_metric() == 1.0


def test_roc_auc_binary_uses_logit_margin() -> None:
    """A two-output classifier ranks by its margin, not its second raw logit."""
    logits = torch.tensor([[100.0, 100.0], [0.0, 0.0], [100.0, 101.0], [1.0, 2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    assert roc_auc_score(labels, logits[:, 1]) == 0.75
    metric = ROCAUC()
    metric.update(logits, labels)
    assert metric.get_primary_metric() == 1.0


def test_roc_auc_binary_does_not_saturate_logits() -> None:
    """Large finite logits retain their ranking even when softmax saturates."""
    logits = torch.tensor([[0.0, 1000.0], [0.0, 1001.0], [0.0, 1002.0], [0.0, 1003.0]])
    metric = ROCAUC()
    metric.update(logits, torch.tensor([0, 0, 1, 1]))
    assert metric.get_primary_metric() == 1.0


@pytest.mark.parametrize("offset", [40.0, 1000.0])
def test_roc_auc_multiclass_does_not_saturate_logits(offset: float) -> None:
    """Stable OVR odds retain rankings when softmax would round to one/zero."""
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    scores = torch.zeros(6, 3, dtype=torch.float64)
    scores[:, 0] = torch.arange(5, -1, -1, dtype=torch.float64) + offset
    # Class 0 is perfectly ranked; class 1's two positives fall between its
    # negatives; class 2 is perfectly ranked: mean(1, 0.5, 1) = 5/6.
    metric = ROCAUC()
    metric.update(scores, labels)
    assert metric.get_primary_metric() == pytest.approx(5.0 / 6.0)


def test_roc_auc_preserves_probability_rankings() -> None:
    """Reapplying softmax to multiclass probabilities can change macro AUC."""
    generator = torch.Generator().manual_seed(17)
    labels = torch.arange(30) % 3
    probs = torch.randn(30, 3, generator=generator, dtype=torch.float64).softmax(dim=1)
    expected = roc_auc_score(labels, probs, multi_class="ovr")
    double_softmax = roc_auc_score(labels, probs.softmax(dim=1), multi_class="ovr")
    assert expected != pytest.approx(double_softmax)

    metric = ROCAUC()
    metric.update(probs, labels)
    assert metric.get_primary_metric() == pytest.approx(expected)


def test_roc_auc_explicit_score_representation() -> None:
    """An override disambiguates logits whose values happen to form a simplex."""
    generator = torch.Generator().manual_seed(17)
    labels = torch.arange(30) % 3
    scores = torch.randn(30, 3, generator=generator, dtype=torch.float64).softmax(dim=1)

    for from_logits in (True, False):
        metric = ROCAUC(from_logits=from_logits)
        metric.update(scores, labels)
        expected_scores = scores.softmax(dim=1) if from_logits else scores
        expected = roc_auc_score(labels, expected_scores, multi_class="ovr")
        assert metric.get_primary_metric() == pytest.approx(expected)


@pytest.mark.parametrize("one_hot", [False, True])
def test_roc_auc_indicator_targets_preserve_independent_rankings(one_hot: bool) -> None:
    """Two-dimensional targets retain the existing sklearn multilabel contract."""
    generator = torch.Generator().manual_seed(42)
    scores = torch.randn(30, 3, generator=generator, dtype=torch.float64) * 4.0
    labels = (
        torch.nn.functional.one_hot(torch.arange(30) % 3, 3)
        if one_hot
        else torch.randint(0, 2, (30, 3), generator=generator)
    )
    expected = roc_auc_score(labels, scores, average="macro")
    normalized = roc_auc_score(labels, scores.softmax(dim=1), average="macro")
    assert expected != pytest.approx(normalized)

    metric = ROCAUC()
    metric.update(scores, labels)
    assert metric.get_primary_metric() == pytest.approx(expected)


def test_roc_auc_indicator_probabilities_need_not_sum_to_one() -> None:
    """Independent multilabel probabilities are not categorical probabilities."""
    scores = torch.tensor([[0.8, 0.9], [0.9, 0.2], [0.1, 0.8], [0.2, 0.1]])
    labels = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]])
    metric = ROCAUC()
    metric.update(scores, labels)
    assert metric.get_primary_metric() == roc_auc_score(labels, scores) == 1.0


def test_roc_auc_multilabel_softmax_would_reverse_a_ranking() -> None:
    """A competing label's score must not change a label's independent rank."""
    scores = torch.tensor([[2.0, 0.0], [3.0, 10.0]])
    labels = torch.tensor([[0, 1], [1, 0]])
    assert roc_auc_score(labels[:, 0], scores[:, 0]) == 1.0
    assert roc_auc_score(labels[:, 0], scores.softmax(1)[:, 0]) == 0.0
    metric = ROCAUC()
    metric.update(scores, labels)
    assert metric.get_primary_metric() == roc_auc_score(labels, scores) == 0.5


@pytest.mark.parametrize("multi_label", [False, True])
def test_roc_auc_training_callers_keep_indicator_contract(multi_label: bool) -> None:
    """Actual CE/BCE strategy outputs retain the old independent-column AUC."""
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(30, 3, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 2, (30, 3), generator=generator).double() if multi_label else torch.arange(30) % 3
    criterion = torch.nn.BCEWithLogitsLoss() if multi_label else torch.nn.CrossEntropyLoss()
    strategy = SupervisedStrategy(criterion, torch.device("cpu"))
    result = strategy.forward(Mock(return_value=logits), {"raw_wav": torch.zeros(30, 4), "label": labels})
    predictions, targets = result.metrics_data
    assert targets.shape == predictions.shape
    expected = roc_auc_score(targets, predictions, average="macro")

    tracker = MetricsTracker(["roc_auc"], num_classes=3)
    tracker.update_batch_metrics(result.loss, result.metrics_data, result.batch_size)
    assert tracker.get_epoch_metrics()[1]["roc_auc"] == pytest.approx(expected)


@pytest.mark.parametrize("indicator_targets", [False, True])
def test_roc_auc_aggregates_batches_before_computing(indicator_targets: bool) -> None:
    """A class may be absent from a minibatch but present in the complete epoch."""
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    scores = torch.nn.functional.one_hot(labels, 3).float() * 6.0
    if indicator_targets:
        labels = torch.nn.functional.one_hot(labels, 3)
    metric = ROCAUC()
    for start in range(0, 6, 2):
        metric.update(scores[start : start + 2], labels[start : start + 2])
    assert metric.get_primary_metric() == 1.0


def test_roc_auc_binary_zero_is_a_valid_score() -> None:
    """Genuine reversed rankings still report zero, without an error fallback."""
    metric = ROCAUC()
    metric.update(torch.tensor([[0.1, 0.9], [0.9, 0.1]]), torch.tensor([0, 1]))
    assert metric.get_primary_metric() == 0.0


def test_roc_auc_constant_scores_are_defined() -> None:
    """Missing predictions are not a reason to skip an observed target class."""
    metric = ROCAUC()
    metric.update(torch.zeros(4, 3), torch.tensor([0, 1, 2, 1]))
    assert metric.get_primary_metric() == 0.5


@pytest.mark.parametrize("indicator_targets", [False, True])
def test_roc_auc_skips_undefined_classes(indicator_targets: bool) -> None:
    """Missing classes do not erase AUC for classes with positives and negatives."""
    labels = torch.tensor([0, 2, 0, 2])
    scores = torch.tensor([[4.0, 1.0, 0.0], [0.0, 2.0, 3.0], [1.0, 0.0, 2.0], [1.0, 3.0, 2.0]])
    if indicator_targets:
        labels = torch.nn.functional.one_hot(labels, 3)
        expected = np.mean([roc_auc_score(labels[:, k], scores[:, k]) for k in (0, 2)])
    else:
        probs = scores.double().softmax(dim=1)
        expected = np.mean([roc_auc_score(labels == k, probs[:, k]) for k in (0, 2)])

    metric = ROCAUC()
    metric.update(scores, labels)
    with pytest.warns(UndefinedMetricWarning, match="excluding.*1"):
        result = metric.get_primary_metric()
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("num_classes", [2, 3])
def test_roc_auc_single_observed_class_is_nan(num_classes: int) -> None:
    """A single-class epoch is explicitly undefined rather than a plausible 0.0."""
    metric = ROCAUC()
    metric.update(torch.zeros(3, num_classes), torch.ones(3, dtype=torch.int64))
    with pytest.warns(UndefinedMetricWarning):
        assert np.isnan(metric.get_primary_metric())


def test_roc_auc_all_constant_indicator_columns_are_nan() -> None:
    """Both all-negative and all-positive label columns are undefined."""
    metric = ROCAUC()
    metric.update(torch.randn(4, 2), torch.tensor([[0, 1]] * 4))
    with pytest.warns(UndefinedMetricWarning):
        assert np.isnan(metric.get_primary_metric())


@pytest.mark.parametrize(
    ("scores", "labels", "message"),
    [
        (torch.ones(3), torch.tensor([0, 1, 0]), "shape"),
        (torch.ones(3, 2), torch.tensor([0, 1]), "samples"),
        (torch.ones(3, 2), torch.ones(3, 3), "shape"),
        (torch.ones(3, 2), torch.tensor([0, 1, 2]), "class indices"),
        (torch.ones(3, 2), torch.tensor([0.0, 0.5, 1.0]), "class indices"),
        (torch.ones(3, 2), torch.full((3, 2), 0.5), "binary indicators"),
        (torch.full((3, 2), float("nan")), torch.tensor([0, 1, 0]), "finite"),
        (torch.full((3, 2), float("inf")), torch.tensor([0, 1, 0]), "finite"),
        (torch.ones(3, 2), torch.tensor([0.0, float("nan"), 1.0]), "finite"),
    ],
)
def test_roc_auc_rejects_invalid_inputs(scores: torch.Tensor, labels: torch.Tensor, message: str) -> None:
    """Malformed data must not be caught and converted to zero."""
    metric = ROCAUC()
    with pytest.raises(ValueError, match=message):
        metric.update(scores, labels)


def test_roc_auc_rejects_invalid_explicit_probabilities() -> None:
    """The explicit probability contract does not silently reinterpret logits."""
    metric = ROCAUC(from_logits=False)
    metric.update(torch.tensor([[3.0, 1.0, 2.0]] * 3), torch.arange(3))
    with pytest.raises(ValueError, match="probabilities"):
        metric.get_metric()


@pytest.mark.parametrize("change_targets", [False, True])
def test_roc_auc_rejects_inconsistent_batches(change_targets: bool) -> None:
    """Class count and target representation must remain fixed across updates."""
    metric = ROCAUC()
    metric.update(torch.zeros(2, 3), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="batches"):
        if change_targets:
            metric.update(torch.zeros(2, 3), torch.tensor([[1, 0, 0], [0, 1, 0]]))
        else:
            metric.update(torch.zeros(2, 2), torch.tensor([0, 1]))


def test_roc_auc_detaches_tensors() -> None:
    """Training predictions requiring gradients remain valid metric inputs."""
    metric = ROCAUC()
    metric.update(torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True), torch.tensor([0, 1]))
    assert metric.get_primary_metric() == 1.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_roc_auc_low_precision_logits(dtype: torch.dtype) -> None:
    """Mixed-precision outputs can be collected without numpy dtype errors."""
    metric = ROCAUC()
    metric.update(torch.eye(3, dtype=dtype) * 4.0, torch.arange(3))
    assert metric.get_primary_metric() == 1.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("from_logits", [None, False])
def test_roc_auc_low_precision_probabilities(dtype: torch.dtype, from_logits: bool | None) -> None:
    """Probability rounding must not trigger another cross-class softmax."""
    generator = torch.Generator().manual_seed(17)
    labels = torch.arange(30) % 3
    scores = torch.randn(30, 3, generator=generator, dtype=torch.float64).softmax(1).to(dtype)
    indicator_targets = torch.nn.functional.one_hot(labels, 3)
    expected = roc_auc_score(indicator_targets, scores.double(), average="macro")

    metric = ROCAUC(from_logits=from_logits)
    for start in range(0, 30, 10):
        metric.update(scores[start : start + 10], labels[start : start + 10])
    assert metric.get_primary_metric() == pytest.approx(expected)


def test_roc_auc_empty_retains_existing_sentinel() -> None:
    """Empty metrics retain the existing no-data value shared by other metrics."""
    metric = ROCAUC()
    assert metric.get_metric() == {"roc_auc": 0.0}
    metric.update(torch.empty(0, 2), torch.empty(0, dtype=torch.int64))
    assert metric.get_metric() == {"roc_auc": 0.0}
