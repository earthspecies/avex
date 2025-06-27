from __future__ import annotations

"""Frame-wise strong-detection metrics operating directly on tensors.

This module provides an in-memory alternative to
`representation_learning.metrics.strong_metrics.StrongDetectionF1Metric`,
which expects *paths* to Raven-format CSV files.  The new class
`StrongDetectionF1Tensor` exposes the same public interface as metrics in
`beans_metrics.py` or `sklearn_metrics.py`:

    >>> metric = StrongDetectionF1Tensor(iou=0.5, threshold=0.5)
    >>> metric.update(logits, targets, padding_mask)
    >>> metric.get_metric()  # {"f1": 0.83, ...}

The implementation re-uses the *event-matching* utilities from
`representation_learning.metrics.detection_metric_helpers` to guarantee
numerical equivalence with the legacy file-based metric.
"""

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from representation_learning.metrics.strong_detection.detection_metric_helpers import match_events, _frames_to_events

__all__ = ["StrongDetectionF1Tensor"]

class StrongDetectionF1Tensor:
    """Frame-level F1 metric for *strong* sound-event detection.

    The metric mirrors the logic in ``StrongDetectionF1Metric`` but operates
    on tensors directly, avoiding any disk I/O.  It supports both *binary*
    and *multi-class* (multi-label) scenarios.

    Notes
    -----
    • **Inputs:**
        * ``logits``  – (B, T, C) raw model outputs; `C` may be 1 for binary.
        * ``targets`` – (B, T, C) binary ground-truth labels (multi-hot).
        * ``padding_mask`` (optional) – (B, T) bool tensor indicating padded
          frames that should be ignored.
    • Predictions are thresholded with ``sigmoid(logits) > prob_threshold``.
    • Events are extracted per-class via contiguous positive regions and
      matched to reference events using *intersection-over-union* (IoU).
    """

    def __init__(
        self,
        iou: float = 0.5,
        prob_threshold: float = 0.5,
        fps: float = 50.0,
        label_set: Optional[List[str]] = None,
    ) -> None:
        self.iou = iou
        self.prob_threshold = prob_threshold
        self.fps = fps
        self.label_set = label_set  # Optional explicit class names/order

        # Running counters (event-level)
        self.tp: List[int] = []
        self.fp: List[int] = []
        self.fn: List[int] = []

    # ------------------------------------------------------------ ------
    # Public API – mirrors *Accuracy* / *AveragePrecision* etc.
    # ------------------------------------------------------------------
    def update(
        self,
        logits: Tensor,
        targets_events: list[list[np.ndarray]],
        padding_mask: Optional[Tensor] = None,
    ) -> None:
        """Accumulate a minibatch.

        Parameters
        ----------
        logits : Tensor
            Raw model outputs of shape ``(B, T, C)`` containing **frame-level**
            predictions.
        targets_events : list[list[np.ndarray]]
            Nested list where ``targets_events[b][c]`` is a ``(2, N)`` numpy
            array holding *onset* / *offset* times (in seconds) for **clip
            ``b``**, **class ``c``**.  This matches the format expected by
            ``match_events`` (row-stacked onsets/offsets).
        padding_mask : Optional[Tensor]
            Optional boolean mask of shape ``(B, T)`` indicating padded
            frames that should be ignored when converting predictions to
            events.  ``True`` = padded.
        """

        B, T, C = logits.shape
        if len(targets_events) != B:
            raise ValueError("targets_events must have length B (batch size)")

        # Default: no padding anywhere
        if padding_mask is None:
            padding_mask = torch.zeros(B, T, dtype=torch.bool, device=logits.device)

        # -------------------------------------------------------------
        # 1. Threshold predictions → binary frames
        # -------------------------------------------------------------
        probs = torch.sigmoid(logits) if logits.dtype.is_floating_point else logits
        preds_bin = (probs > self.prob_threshold).cpu().int()  # (B, T, C)

        # -------------------------------------------------------------
        # 2. Loop over batch & classes
        # -------------------------------------------------------------
        for b in range(B):
            if len(targets_events[b]) != C:
                raise ValueError(
                    f"targets_events[{b}] should have length {C} (num classes)"
                )

            valid = ~padding_mask[b].cpu().numpy()  # (T,)

            for c in range(C):
                # 2.a. Predictions → events
                pred_frames = preds_bin[b, :, c].numpy()[valid]
                pred_events = _frames_to_events(pred_frames, self.fps)

                # 2.b. Ground-truth events (already provided)
                true_events = targets_events[b][c]

                if true_events.size == 0 and pred_events.size == 0:
                    continue  # nothing to count/compare

                matching = match_events(true_events, pred_events, self.iou, method="fast")
                tp = len(matching)
                fp = pred_events.shape[1] - tp
                fn = true_events.shape[1] - tp

                self.tp.append(tp)
                self.fp.append(fp)
                self.fn.append(fn)

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.tp.clear()
        self.fp.clear()
        self.fn.clear()

    # ------------------------------------------------------------------
    # Metric retrieval helpers
    # ------------------------------------------------------------------
    def get_metric(self) -> dict[str, float]:
        """Return dictionary with event-level F1-score."""
        tp = int(np.sum(self.tp))
        fp = int(np.sum(self.fp))
        fn = int(np.sum(self.fn))
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else 2 * tp / denom
        return {"f1": f1, "tp": tp, "fp": fp, "fn": fn}

    def get_primary_metric(self) -> float:
        return self.get_metric()["f1"]

