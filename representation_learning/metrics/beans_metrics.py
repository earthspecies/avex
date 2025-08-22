"""
Classification and detection metrics from BEANS.
https://github.com/earthspecies/beans/blob/main/beans/metrics.py
"""

import math

import numpy as np
import torch


class Accuracy:
    """Simple running accuracy meter."""

    def __init__(self) -> None:
        self.num_total = 0
        self.num_correct = 0

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        """Accumulate predictions.

        Args
        ----
        logits:
            Raw model outputs of shape ``(N, C)``.
        y:
            Ground-truth labels of shape ``(N,)``.
        """
        self.num_total += logits.shape[0]
        self.num_correct += torch.sum(logits.argmax(axis=1) == y).cpu().item()

    def get_metric(self) -> dict[str, float]:
        """Return current accuracy value.

        Returns
        -------
        dict[str, float]
            Dictionary with key ``"acc"`` holding the running accuracy.
        """
        return {
            "acc": 0.0 if self.num_total == 0 else self.num_correct / self.num_total
        }

    def get_primary_metric(self) -> float:  # noqa: ANN001 (keep interface)
        """Primary scalar value (accuracy).

        Returns
        -------
        float
            The current accuracy.
        """
        return self.get_metric()["acc"]


class BinaryF1Score:
    """Binary classification precision/recall/F1 tracker."""

    def __init__(self) -> None:
        self.num_positives = 0
        self.num_trues = 0
        self.num_tps = 0

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        positives = logits.argmax(axis=1) == 1
        trues = y == 1
        tps = trues & positives
        self.num_positives += torch.sum(positives).cpu().item()
        self.num_trues += torch.sum(trues).cpu().item()
        self.num_tps += torch.sum(tps).cpu().item()

    def get_metric(self) -> dict[str, float]:
        prec = 0.0 if self.num_positives == 0 else self.num_tps / self.num_positives
        rec = 0.0 if self.num_trues == 0 else self.num_tps / self.num_trues
        if prec + rec > 0.0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        return {"prec": prec, "rec": rec, "f1": f1}

    def get_primary_metric(self) -> float:
        return self.get_metric()["f1"]


class MulticlassBinaryF1Score:
    """Macro-averaged binary F1 score for multi-label problems."""

    def __init__(self, num_classes: int) -> None:
        self.metrics = [BinaryF1Score() for _ in range(num_classes)]
        self.num_classes = num_classes

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        """Update metrics with new predictions and ground truth.

        Args:
            logits: Model output logits of shape (N, C)
            y: Ground truth labels of shape (N, C) for one-hot encoded labels
        """
        probs = torch.sigmoid(logits)
        # Convert one-hot labels back to class indices
        y_indices = y.argmax(dim=1)
        for i in range(self.num_classes):
            binary_logits = torch.stack((1 - probs[:, i], probs[:, i]), dim=1)
            # Create binary labels for this class
            # (1 if this class is present, 0 otherwise)
            binary_y = (y_indices == i).long()
            self.metrics[i].update(binary_logits, binary_y)

    def get_metric(self) -> dict[str, float]:
        macro_prec = 0.0
        macro_rec = 0.0
        macro_f1 = 0.0
        for i in range(self.num_classes):
            metrics = self.metrics[i].get_metric()
            macro_prec += metrics["prec"]
            macro_rec += metrics["rec"]
            macro_f1 += metrics["f1"]
        return {
            "macro_prec": macro_prec / self.num_classes,
            "macro_rec": macro_rec / self.num_classes,
            "macro_f1": macro_f1 / self.num_classes,
        }

    def get_primary_metric(self) -> float:
        return self.get_metric()["macro_f1"]


class AveragePrecision:
    """
    Taken from https://github.com/amdegroot/tnt
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Resets the meter with empty member variables"""
        self.scores = torch.tensor(
            torch.FloatStorage(), dtype=torch.float32, requires_grad=False
        )
        self.targets = torch.tensor(
            torch.LongStorage(), dtype=torch.int64, requires_grad=False
        )
        self.weights = torch.tensor(
            torch.FloatStorage(), dtype=torch.float32, requires_grad=False
        )

    def update(
        self,
        output: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
        weight: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """
        Args:
            output (Tensor): NxK tensor of raw logits from the model for each of
                the N examples and K classes. These will be converted to probabilities
                using sigmoid for mAP computation.
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        # Convert logits to probabilities using sigmoid for mAP computation
        output = torch.sigmoid(output)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, (
                "wrong output size (should be 1D or 2D with one column \
                per class)"
            )
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, (
                "wrong target size (should be 1D or 2D with one column \
                per class)"
            )
        if weight is not None:
            assert weight.dim() == 1, "Weight dimension should be 1"
            assert weight.numel() == target.size(0), (
                "Weight dimension 1 should be the same as that of target"
            )
            assert torch.min(weight) >= 0, "Weight should be non-negative only"
        assert torch.equal(target**2, target), "targets should be binary (0 or 1)"
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), (
                "dimensions for output should match previously added examples."
            )

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))
            if weight is not None:
                self.weights.storage().resize_(int(new_weight_size + output.size(0)))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output.detach())
        self.targets.narrow(0, offset, target.size(0)).copy_(target.detach())

        if weight is not None:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

    def get_metric(self) -> torch.Tensor | int:
        """Return per-class average precision scores.

        Returns
        -------
        torch.Tensor | int
            1-D tensor of length *K* containing AP for each class, or ``0`` if
            no data has been accumulated yet.
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0) + 1).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(truth.sum(), 1)
        return ap


class MeanAveragePrecision:
    """Mean Average Precision (mAP) metric for multi-label classification.
    averages Average Precision across classes.
    """

    def __init__(self) -> None:
        self.ap = AveragePrecision()

    def reset(self) -> None:
        self.ap.reset()

    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> None:
        self.ap.update(output, target, weight)

    def get_metric(self) -> dict[str, float]:
        return {"map": self.ap.get_metric().mean().item()}

    def get_primary_metric(self) -> float:
        return self.get_metric()["map"]


class BalancedAccuracy:
    """Balanced accuracy across classes (handles class imbalance)."""

    def __init__(self) -> None:
        self.class_correct = {}
        self.class_total = {}

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        predictions = logits.argmax(axis=1)
        for pred, true in zip(predictions, y, strict=False):
            true_label = true.item()
            if true_label not in self.class_total:
                self.class_total[true_label] = 0
                self.class_correct[true_label] = 0

            self.class_total[true_label] += 1
            if pred.item() == true_label:
                self.class_correct[true_label] += 1

    def get_metric(self) -> dict[str, float]:
        if not self.class_total:  # If no updates have been made
            return {"balanced_acc": 0.0}

        class_recalls = []
        for class_label in self.class_total:
            recall = (
                self.class_correct[class_label] / self.class_total[class_label]
                if self.class_total[class_label] > 0
                else 0.0
            )
            class_recalls.append(recall)

        balanced_acc = sum(class_recalls) / len(class_recalls)
        return {"balanced_acc": balanced_acc}

    def get_primary_metric(self) -> float:
        return self.get_metric()["balanced_acc"]
