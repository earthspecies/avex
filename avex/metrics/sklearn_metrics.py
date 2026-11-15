"""
Classification and detection metrics using scikit-learn implementations.
"""

import warnings
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Suppress sklearn warnings about no positive class when using zero_division=0
warnings.filterwarnings("ignore", message="No positive class found in y_true", category=UserWarning)


class Accuracy:
    """Accuracy metric for classification tasks.

    Computes the fraction of correct predictions over total predictions.
    """

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
    """Binary F1 score metric for binary classification tasks.

    Computes precision, recall, and F1 score for binary classification.
    """

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
    """Multiclass F1 score metric using macro averaging.

    Computes macro-averaged precision, recall, and F1 score for multiclass
    classification tasks.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.y_true = []
        self.y_pred = []

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
        y_pred = np.array(self.y_pred)

        macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return {
            "macro_prec": macro_prec,
            "macro_rec": macro_rec,
            "macro_f1": macro_f1,
        }

    def get_primary_metric(self) -> float:
        """Get the primary metric value.

        Returns:
            Macro-averaged F1 score
        """
        return self.get_metric()["macro_f1"]


class AveragePrecision:
    """Average precision metric for multilabel classification.

    Computes the average precision score for each class in multilabel
    classification tasks.
    """

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
            output: Model output logits of shape (N, K)
            target: Ground truth labels of shape (N, K)
            weight: Optional sample weights of shape (N,)
        """
        # Convert logits to probabilities using sigmoid for mAP computation
        y_scores = torch.sigmoid(output).cpu().numpy()
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
            ap = average_precision_score(y_true[:, k], y_scores[:, k], sample_weight=sample_weights)
            ap_scores.append(ap)

        return np.array(ap_scores)


class MeanAveragePrecision:
    """Mean average precision metric for multilabel classification.

    Computes the mean of average precision scores across all classes.
    """

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
    """Balanced accuracy metric for classification tasks.

    Computes the balanced accuracy which accounts for class imbalance by
    averaging the recall obtained on each class.
    """

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
    """Compute binary or macro one-vs-rest ROC-AUC over accumulated batches.

    Class-index targets ``(N,)`` describe a categorical classifier with score
    columns ordered by class index. Logits are ranked by stable categorical
    log-odds (equivalent to softmax probabilities); probability matrices are
    used unchanged. Binary classifiers use class 1.

    Indicator targets ``(N, C)`` (including one-hot targets) retain the existing
    multilabel contract: each column is ranked independently, without softmax.
    Convert single-label one-hot targets to indices before ``update`` when
    categorical normalization is desired. One-hot indicator matrices are not
    automatically reinterpreted as categorical targets.

    Classes lacking either positives or negatives are excluded from the macro
    average with an ``UndefinedMetricWarning``. If none remain, the result is
    NaN. With no accumulated samples, the existing 0.0 sentinel is retained.

    Args:
        from_logits: For class-index targets, True always interprets inputs as
            logits and False requires categorical probabilities. The default
            None recognizes probabilities by values in [0, 1] and rows summing
            to 1 within input-precision tolerance; otherwise inputs are logits.
            Set this explicitly if logits happen to satisfy that probability
            check. All batches must use the same score representation.
            Indicator scores are always used as-is.
    """

    def __init__(self, *, from_logits: Optional[bool] = None) -> None:
        self.from_logits = from_logits
        self.y_true: list[np.ndarray] = []
        self.y_scores: list[np.ndarray] = []
        self._probabilities = True

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:  # noqa: D401
        """Accumulate a minibatch.

        Parameters
        ----------
        logits : torch.Tensor
            Finite raw model outputs or probabilities of shape (N, C).
        y : torch.Tensor
            Class indices in [0, C) of shape (N,), or binary indicators (N, C).

        Raises
        ------
        ValueError
            If input shapes, labels, finite values or batch formats are invalid.
        """
        if logits.ndim != 2 or logits.shape[1] == 0:
            raise ValueError("ROC-AUC scores must have shape (N, C) with C > 0.")
        if y.ndim not in (1, 2) or (y.ndim == 2 and y.shape != logits.shape):
            raise ValueError("ROC-AUC targets must have shape (N,) or the same shape (N, C) as scores.")
        if y.shape[0] != logits.shape[0]:
            raise ValueError("ROC-AUC scores and targets must contain the same number of samples.")
        if logits.is_complex() or y.is_complex() or not torch.isfinite(logits).all() or not torch.isfinite(y).all():
            raise ValueError("ROC-AUC scores and targets must be finite real values.")
        if y.ndim == 1:
            if logits.shape[1] < 2 or not ((y >= 0) & (y < logits.shape[1]) & (y == y.long())).all():
                raise ValueError("ROC-AUC class indices must be integers in [0, C), with C >= 2.")
        elif not ((y == 0) | (y == 1)).all():
            raise ValueError("ROC-AUC two-dimensional targets must be binary indicators.")
        if self.y_scores and (logits.shape[1] != self.y_scores[0].shape[1] or y.ndim != self.y_true[0].ndim):
            raise ValueError("ROC-AUC batches must have consistent score columns and target dimensions.")

        # float64 also supports bfloat16 model outputs and avoids low-precision
        # normalization/subtraction. Detaching keeps metric state off autograd.
        scores = logits.detach().to(device="cpu", dtype=torch.float64).numpy()
        # Account for rounding of genuine fp16/bfloat16 probabilities before
        # upcasting. Each batch uses its own precision tolerance, but every
        # batch must satisfy the check to use the probability path.
        atol = max(1e-8, torch.finfo(logits.dtype).eps) if logits.is_floating_point() else 1e-8
        self._probabilities = self._probabilities and bool(
            np.all((scores >= 0) & (scores <= 1)) and np.allclose(scores.sum(axis=1), 1.0, atol=atol)
        )
        self.y_scores.append(scores)
        self.y_true.append(y.detach().to(device="cpu", dtype=torch.int64).numpy())

    def _stack(self) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate all batches before deciding which classes are defined.

        Returns:
            Tuple of accumulated targets and scores.
        """
        y_true = np.concatenate(self.y_true, axis=0)
        y_scores = np.concatenate(self.y_scores, axis=0)
        return y_true, y_scores

    def _categorical_scores(self, y_scores: np.ndarray) -> np.ndarray:
        """Return stable categorical log-odds or unchanged probabilities.

        Returns:
            Binary class-1 scores or a multiclass one-vs-rest score matrix.

        Raises:
            ValueError: If explicitly supplied probabilities are invalid.
        """
        if self.from_logits is False and not self._probabilities:
            raise ValueError("ROC-AUC probabilities must be in [0, 1] and sum to 1 over classes.")
        if self.from_logits is True or (self.from_logits is None and not self._probabilities):
            if y_scores.shape[1] == 2:
                # The logit margin has the same ranking as P(class 1), without
                # softmax saturation collapsing distinct large scores to ties.
                return y_scores[:, 1] - y_scores[:, 0]
            # logit_k - logsumexp(other logits) ranks exactly like softmax_k
            # without rounding near-one probabilities or underflowing to zero.
            # Prefix/suffix sums avoid O(N*C*C) leave-one-class-out work.
            shifted = y_scores - y_scores.max(axis=1, keepdims=True)
            prefix = np.logaddexp.accumulate(shifted, axis=1)
            suffix = np.logaddexp.accumulate(shifted[:, ::-1], axis=1)[:, ::-1]
            other_classes = np.empty_like(shifted)
            other_classes[:, 0] = suffix[:, 1]
            other_classes[:, -1] = prefix[:, -2]
            other_classes[:, 1:-1] = np.logaddexp(prefix[:, :-2], suffix[:, 2:])
            return shifted - other_classes
        return y_scores[:, 1] if y_scores.shape[1] == 2 else y_scores

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
        if y_true.shape[0] == 0:
            return {"roc_auc": 0.0}

        if y_true.ndim == 1:
            y_scores = self._categorical_scores(y_scores)
            if y_scores.ndim == 1:
                y_true = y_true[:, None]
                y_scores = y_scores[:, None]
            else:
                y_true = y_true[:, None] == np.arange(y_scores.shape[1])

        defined = (y_true == 0).any(axis=0) & (y_true == 1).any(axis=0)
        if not defined.all():
            warnings.warn(
                "ROC-AUC is undefined for classes without both positive and negative samples; "
                f"excluding columns {np.flatnonzero(~defined).tolist()} from the macro average. "
                "The result is NaN if no defined classes remain.",
                UndefinedMetricWarning,
                stacklevel=2,
            )
        aucs = [roc_auc_score(y_true[:, k], y_scores[:, k]) for k in np.flatnonzero(defined)]
        return {"roc_auc": float(np.mean(aucs)) if aucs else float("nan")}

    def get_primary_metric(self) -> float:  # noqa: D401
        """Return the binary or macro ROC-AUC value.

        Returns:
            ROC-AUC, or NaN if no observed class has positives and negatives.
        """
        return self.get_metric()["roc_auc"]
