"""
Classification and detection metrics using scikit-learn implementations.
"""

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Accuracy:
    def __init__(self) -> None:
        self.y_true = []
        self.y_pred = []

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        """Update the metric with new predictions and ground truth.

        Args:
            logits: Model output logits of shape (N, C)
            y: Ground truth labels of shape (N,) or (N, C) for one-hot encoded labels
        """
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

        # Handle one-hot encoded labels by converting to class indices
        if y_true.ndim == 2:
            y_true = y_true.argmax(axis=1)

        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    def get_metric(self) -> Dict[str, float]:
        """Get the current metric value.

        Returns:
            Dictionary containing the accuracy score
        """
        if not self.y_true:
            return {"acc": 0.0}
        acc = accuracy_score(self.y_true, self.y_pred)
        return {"acc": acc}

    def get_primary_metric(self) -> float:
        """Get the primary metric value.

        Returns:
            Accuracy score
        """
        return self.get_metric()["acc"]


class BinaryF1Score:
    def __init__(self) -> None:
        self.y_true = []
        self.y_pred = []
        self.y_scores = []

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        """Update the metric with new predictions and ground truth.

        Args:
            logits: Model output logits of shape (N, 2)
            y: Ground truth labels of shape (N,)
        """
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = y.cpu().numpy()
        y_scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        # Handle one-hot (or soft) encoded labels by converting to class
        # indices for *metric* computation.  We deliberately keep the raw
        # scores as probabilities to compute AUC/AP later on.
        if y_true.ndim == 2:
            y_true = y_true.argmax(axis=1)

        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
        self.y_scores.extend(y_scores)

    def get_metric(self) -> Dict[str, float]:
        """Get the current metric values.

        Returns:
            Dictionary containing precision, recall, and F1 score
        """
        if not self.y_true:
            return {"prec": 0.0, "rec": 0.0, "f1": 0.0}

        prec = precision_score(self.y_true, self.y_pred, zero_division=0)
        rec = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)

        return {"prec": prec, "rec": rec, "f1": f1}

    def get_primary_metric(self) -> float:
        """Get the primary metric value.

        Returns:
            F1 score
        """
        return self.get_metric()["f1"]


class MulticlassBinaryF1Score:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.y_true = []
        self.y_scores = []

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        """Update the metric with new predictions and ground truth.

        Args:
            logits: Model output logits of shape (N, C)
            y: Ground truth labels of shape (N, C) for one-hot encoded labels
        """
        y_pred = logits.argmax(dim=1).cpu().numpy()
        # Convert one-hot labels back to class indices
        y_true = y.argmax(dim=1).cpu().numpy()
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    def get_metric(self) -> Dict[str, float]:
        """Get the current metric values.

        Returns:
            Dictionary containing macro-averaged precision, recall, and F1 score
        """
        if not self.y_true:
            return {"macro_prec": 0.0, "macro_rec": 0.0, "macro_f1": 0.0}

        y_true = np.array(self.y_true)
        y_scores = np.array(self.y_scores)
        y_pred = (y_scores > 0.5).astype(int)

        macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return {"macro_prec": macro_prec, "macro_rec": macro_rec, "macro_f1": macro_f1}

    def get_primary_metric(self) -> float:
        """Get the primary metric value.

        Returns:
            Macro-averaged F1 score
        """
        return self.get_metric()["macro_f1"]


class AveragePrecision:
    def __init__(self) -> None:
        self.y_true = []
        self.y_scores = []
        self.sample_weights = []

    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Update the metric with new predictions and ground truth.

        Args:
            output: Model output scores of shape (N, K)
            target: Ground truth labels of shape (N, K)
            weight: Optional sample weights of shape (N,)
        """
        y_scores = output.cpu().numpy()
        y_true = target.cpu().numpy()
        self.y_true.extend(y_true)
        self.y_scores.extend(y_scores)

        if weight is not None:
            sample_weights = weight.cpu().numpy()
            self.sample_weights.extend(sample_weights)

    def get_metric(self) -> np.ndarray:
        """Get the current metric values.

        Returns:
            Array of average precision scores for each class
        """
        if not self.y_true:
            return np.array([0.0])

        y_true = np.array(self.y_true)
        y_scores = np.array(self.y_scores)
        if len(self.sample_weights) > 0:
            sample_weights = np.array(self.sample_weights)
        else:
            sample_weights = None

        ap_scores = []
        for k in range(y_scores.shape[1]):
            # For each class, calculate AP using sklearn's average_precision_score
            # This handles the precision-recall curve calculation correctly
            ap = average_precision_score(
                y_true[:, k], y_scores[:, k], sample_weight=sample_weights
            )
            ap_scores.append(ap)

        return np.array(ap_scores)


class MeanAveragePrecision:
    def __init__(self) -> None:
        self.ap = AveragePrecision()

    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Update the metric with new predictions and ground truth.

        Args:
            output: Model output scores of shape (N, K)
            target: Ground truth labels of shape (N, K)
            weight: Optional sample weights of shape (N,)
        """
        self.ap.update(output, target, weight)

    def get_metric(self) -> Dict[str, float]:
        """Get the current metric value.

        Returns:
            Dictionary containing the mean average precision score
        """
        ap_scores = self.ap.get_metric()
        return {"map": float(np.mean(ap_scores))}

    def get_primary_metric(self) -> float:
        """Get the primary metric value.

        Returns:
            Mean average precision score
        """
        return self.get_metric()["map"]


class BalancedAccuracy:
    def __init__(self) -> None:
        self.y_true = []
        self.y_pred = []

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        """Update the metric with new predictions and ground truth.

        Args:
            logits: Model output logits of shape (N, C)
            y: Ground truth labels of shape (N,) or (N, C) for one-hot encoded labels
        """
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

        # Handle one-hot encoded labels by converting to class indices
        if y_true.ndim == 2:
            y_true = y_true.argmax(axis=1)

        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    def get_metric(self) -> Dict[str, float]:
        """Get the current metric value.

        Returns:
            Dictionary containing the balanced accuracy score
        """
        if not self.y_true:
            return {"balanced_acc": 0.0}

        balanced_acc = balanced_accuracy_score(self.y_true, self.y_pred)
        return {"balanced_acc": balanced_acc}

    def get_primary_metric(self) -> float:
        """Get the primary metric value.

        Returns:
            Balanced accuracy score
        """
        return self.get_metric()["balanced_acc"]


class ROCAUC:
    """Compute (macro) ROC-AUC for (multi-)class classification.

    This implementation collects logits and targets and computes a *macro-
    averaged* ROC-AUC using ``sklearn.metrics.roc_auc_score``.  For binary
    problems it yields the standard AUC of the positive class.  If a class is
    missing in the ground-truth or predictions the sample is skipped to avoid
    ``ValueError``.
    """

    def __init__(self) -> None:
        self.y_true: list[np.ndarray] = []
        self.y_scores: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:  # noqa: D401
        """Accumulate a minibatch.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model outputs of shape (N, C).
        y : torch.Tensor
            Integer class labels (N,) or one-hot / multi-hot (N, C).
        """
        self.y_scores.append(logits.detach().cpu().numpy())
        self.y_true.append(y.detach().cpu().numpy())

    def _stack(self) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.concatenate(self.y_true, axis=0)
        y_scores = np.concatenate(self.y_scores, axis=0)
        return y_true, y_scores

    def get_metric(self) -> Dict[str, float]:
        """Return a dict with the macro ROC-AUC.

        Returns
        -------
        Dict[str, float]
            ``{"roc_auc": value}``
        """

        if not self.y_true:
            return {"roc_auc": 0.0}
        y_true, y_scores = self._stack()

        # Handle binary vs multi-class automatically
        try:
            auc = roc_auc_score(
                y_true,
                y_scores,
                multi_class="ovr" if y_scores.shape[1] > 2 else "raise",
                average="macro",
            )
        except ValueError:
            auc = 0.0  # Occurs when only one class present
        return {"roc_auc": float(auc)}

    def get_primary_metric(self) -> float:  # noqa: D401
        return self.get_metric()["roc_auc"]
