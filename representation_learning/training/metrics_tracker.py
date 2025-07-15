"""
Metrics tracking and computation for training.

This module handles metrics computation, aggregation, and tracking
across training epochs and batches.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch

from representation_learning.metrics.metric_factory import get_metric_class
from representation_learning.training.distributed import (
    gather_metrics_from_all_ranks,
    synchronize_scalar,
)

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks and computes metrics during training."""

    def __init__(
        self,
        metrics: List[str],
        num_classes: int,
        device: torch.device,
        training_mode: str = "supervised",
    ):
        """Initialize metrics tracker.

        Parameters
        ----------
        metrics : List[str]
            List of metric names to track
        num_classes : int
            Number of classes for classification metrics
        device : torch.device
            Device for tensor operations
        training_mode : str, optional
            Training mode (supervised, clip, eat_ssl), by default "supervised"
        """
        self.metrics = metrics
        self.num_classes = num_classes
        self.device = device
        self.training_mode = training_mode
        self.primary_metric_name = metrics[0] if metrics else "accuracy"

        # Initialize metric calculators
        self.metric_calculators = {}
        if training_mode == "supervised":
            self.metric_calculators = {
                name: get_metric_class(name, num_classes) for name in metrics
            }

        # State tracking
        self.reset_epoch_state()

    def reset_epoch_state(self) -> None:
        """Reset epoch-level state."""
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_correct = 0
        self.total_correct_a2t = 0  # For CLIP
        self.total_correct_t2a = 0  # For CLIP
        self.component_totals = {}  # For EAT SSL

        # Reset metric calculators
        if self.training_mode == "supervised":
            self.metric_calculators = {
                name: get_metric_class(name, self.num_classes) for name in self.metrics
            }

    def update_batch_metrics(
        self,
        loss: torch.Tensor,
        metrics_data: Union[int, Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update metrics with batch results.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value for this batch
        metrics_data : Union[int, Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]
            Metrics data from the batch (format depends on training mode)
        batch_size : int
            Size of the batch
        additional_metrics : Optional[Dict[str, float]], optional
            Additional metrics (e.g., component losses), by default None
        """
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size

        if self.training_mode == "clip":
            # CLIP mode: metrics_data is Tuple[int, int]
            correct_a2t, correct_t2a = metrics_data
            self.total_correct_a2t += correct_a2t
            self.total_correct_t2a += correct_t2a
        elif self.training_mode == "eat_ssl":
            # EAT SSL mode: metrics_data is int (always 0)
            self.total_correct += metrics_data
            # Handle component losses
            if additional_metrics:
                for k, v in additional_metrics.items():
                    self.component_totals[k] = (
                        self.component_totals.get(k, 0.0) + v * batch_size
                    )
        else:
            # Supervised mode: metrics_data is Tuple[torch.Tensor, torch.Tensor]
            if isinstance(metrics_data, tuple):
                predictions, targets = metrics_data
                # Update metric calculators
                for metric_calculator in self.metric_calculators.values():
                    metric_calculator.update(predictions, targets)
                self.total_correct += 0  # Will be computed from metrics
            else:
                # Legacy format: int count
                self.total_correct += metrics_data

    def get_batch_metrics(self) -> Tuple[float, float]:
        """Get current batch-level metrics.

        Returns
        -------
        Tuple[float, float]
            Average loss and accuracy so far
        """
        if self.total_samples == 0:
            return 0.0, 0.0

        avg_loss = self.total_loss / self.total_samples

        # ------------------------------------------------------------------ #
        #  Accuracy computation depends on training mode
        # ------------------------------------------------------------------ #
        if self.training_mode == "clip":
            # Retrieval accuracies are tracked separately
            avg_acc = (
                (self.total_correct_a2t + self.total_correct_t2a)
                / 2.0
                / self.total_samples
            )
        elif self.training_mode == "supervised":
            # Derive running accuracy directly from the metric calculator if present
            if "accuracy" in self.metric_calculators:
                avg_acc = self.metric_calculators["accuracy"].get_primary_metric()
            else:
                avg_acc = 0.0
        else:
            # eat_ssl or other modes that don't track accuracy
            avg_acc = self.total_correct / self.total_samples

        return float(avg_loss), float(avg_acc)

    def get_component_metrics(self) -> Dict[str, float]:
        """Get component metrics (for EAT SSL).

        Returns
        -------
        Dict[str, float]
            Component metrics averaged over samples
        """
        if not self.component_totals or self.total_samples == 0:
            return {}

        return {k: v / self.total_samples for k, v in self.component_totals.items()}

    def get_epoch_metrics(self) -> Tuple[float, Dict[str, float]]:
        """Get final epoch metrics.

        Returns
        -------
        Tuple[float, Dict[str, float]]
            Average loss and dictionary of metric values
        """
        # Synchronize final metrics across ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        avg_loss, avg_acc = gather_metrics_from_all_ranks(
            self.total_loss,
            self.total_correct,
            self.total_samples,
            self.device,
            is_clip_mode=(self.training_mode == "clip"),
            total_correct_a2t=self.total_correct_a2t,
            total_correct_t2a=self.total_correct_t2a,
        )

        # Compute final metrics based on training mode
        if self.training_mode == "clip":
            # Calculate individual accuracy components
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                total_samples_sync = synchronize_scalar(self.total_samples, self.device)
                avg_acc_a2t = (
                    synchronize_scalar(self.total_correct_a2t, self.device)
                    / total_samples_sync
                    if total_samples_sync > 0
                    else 0.0
                )
                avg_acc_t2a = (
                    synchronize_scalar(self.total_correct_t2a, self.device)
                    / total_samples_sync
                    if total_samples_sync > 0
                    else 0.0
                )
            else:
                avg_acc_a2t = (
                    self.total_correct_a2t / self.total_samples
                    if self.total_samples
                    else 0.0
                )
                avg_acc_t2a = (
                    self.total_correct_t2a / self.total_samples
                    if self.total_samples
                    else 0.0
                )

            final_metrics = {
                self.primary_metric_name: avg_acc,
                "acc_a2t": avg_acc_a2t,
                "acc_t2a": avg_acc_t2a,
            }
        elif self.training_mode == "eat_ssl":
            # For EAT SSL, return 0 accuracy
            final_metrics = {self.primary_metric_name: 0.0}
        else:
            # For supervised learning, compute metrics from calculators
            final_metrics = {}
            for name, metric_calculator in self.metric_calculators.items():
                final_metrics[name] = metric_calculator.get_primary_metric()

        return avg_loss, final_metrics

    def get_clip_additional_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """Get additional CLIP metrics like logit scale.

        Parameters
        ----------
        model : torch.nn.Module
            Model to extract metrics from

        Returns
        -------
        Dict[str, float]
            Additional CLIP metrics
        """
        if self.training_mode != "clip":
            return {}

        # Get logit scale from model
        current_scale = 1.0
        if hasattr(model, "logit_scale"):
            current_scale = model.logit_scale.exp().item()
        elif hasattr(model, "module") and hasattr(model.module, "logit_scale"):
            current_scale = model.module.logit_scale.exp().item()

        return {
            "logit_scale": current_scale,
        }
