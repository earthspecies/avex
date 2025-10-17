"""
Training strategies for different learning modes.

This module implements the Strategy pattern to handle different training modes
(supervised, CLIP, EAT SSL) in a modular way.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from representation_learning.training.losses import ClipLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result of a single training step."""

    loss: torch.Tensor
    metrics_data: Union[int, Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]
    batch_size: int
    additional_metrics: Dict[str, float] = None


class TrainingStrategy(ABC):
    """Base class for training strategies."""

    def __init__(self, criterion: nn.Module, device: torch.device) -> None:
        self.criterion = criterion
        self.device = device
        self.additional_metrics: Dict[str, float] = {}

    @abstractmethod
    def forward(self, model: nn.Module, batch: Dict[str, Any]) -> TrainingResult:
        """Execute forward pass for this training strategy."""
        pass

    @abstractmethod
    def get_expected_metrics_format(self) -> str:
        """Return the expected format for metrics computation."""
        pass


class SupervisedStrategy(TrainingStrategy):
    """Strategy for supervised learning."""

    def get_expected_metrics_format(self) -> str:
        return "predictions_targets"

    def forward(self, model: nn.Module, batch: Dict[str, Any]) -> TrainingResult:
        """Forward pass for supervised learning.

        Returns
        -------
        TrainingResult
            The result containing loss and metrics for supervised learning.
        """
        # Get the inputs
        audio = batch["raw_wav"]
        target = batch["label"]
        padding_mask = batch.get("padding_mask")

        # Forward pass
        outputs = model(audio, padding_mask=padding_mask)

        # DEBUG: Check for NaN in model outputs
        if torch.isnan(outputs).any():
            logger.warning(f"NaN detected in model outputs! Shape: {outputs.shape}")
            logger.warning(
                f"Output stats: min={outputs.min():.6f}, max={outputs.max():.6f}, "
                f"mean={outputs.mean():.6f}"
            )
            nan_count = torch.isnan(outputs).sum().item()
            logger.warning(f"Number of NaN values in outputs: {nan_count}")

        # DEBUG: Check for extreme values in model outputs
        if torch.isinf(outputs).any():
            logger.warning("Inf detected in model outputs!")
            inf_count = torch.isinf(outputs).sum().item()
            logger.warning(f"Number of Inf values in outputs: {inf_count}")

        # Match target format to criterion expectations
        if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
            # BCE expects float multi-hot targets (shape [B, C])
            if target.dim() == 1:
                target = torch.nn.functional.one_hot(
                    target.long(), num_classes=outputs.size(1)
                ).float()
        elif isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            # Cross-entropy expects class indices (shape [B])
            if target.dim() > 1:
                target = target.argmax(dim=1)

        loss = self.criterion(outputs, target)

        # DEBUG: Check for NaN loss
        if torch.isnan(loss).any():
            logger.warning(
                f"NaN loss detected! outputs stats: min={outputs.min():.6f}, "
                f"max={outputs.max():.6f}, mean={outputs.mean():.6f}"
            )
            logger.warning(
                f"Target stats: min={target.min():.6f}, max={target.max():.6f}, "
                f"mean={target.mean():.6f}"
            )
            logger.warning(f"Loss value: {loss.item()}")

        # Return predictions and targets for metric computation
        with torch.no_grad():
            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                # Multi-label: return logits and targets as they are
                predictions = outputs.detach()
                targets = target.detach()
            else:  # Cross-entropy (single-label)
                # Single-label: return logits and one-hot targets for consistency
                predictions = outputs.detach()
                if target.dim() == 1:
                    # Convert indices to one-hot for metric computation
                    targets = (
                        torch.nn.functional.one_hot(
                            target.long(), num_classes=outputs.size(1)
                        )
                        .float()
                        .detach()
                    )
                else:
                    targets = target.detach()

        return TrainingResult(
            loss=loss,
            metrics_data=(predictions, targets),
            batch_size=target.size(0),
        )


class CLIPStrategy(TrainingStrategy):
    """Strategy for CLIP training."""

    def get_expected_metrics_format(self) -> str:
        return "clip_accuracy"

    def forward(self, model: nn.Module, batch: Dict[str, Any]) -> TrainingResult:
        """Forward pass for CLIP training.

        Returns
        -------
        TrainingResult
            The result containing loss and metrics for CLIP training.
        """
        audio = batch["raw_wav"]
        text = batch["text_label"]
        padding_mask = batch.get("padding_mask")

        # Forward pass through CLIPModel
        audio_emb, text_emb, logit_scale = model(
            audio, text=text, padding_mask=padding_mask
        )

        # Debug: Log before loss computation
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            if not hasattr(self, "_debug_first_loss"):
                logger.info("[DEBUG] Starting first CLIP loss computation")
                self._debug_first_loss = True

        # Get loss and logits from criterion
        if isinstance(self.criterion, ClipLoss):
            loss, logits = self.criterion(
                audio_emb, text_emb, logit_scale, output_logits=True
            )
        else:
            loss = self.criterion(audio_emb, text_emb, logit_scale)
            with torch.no_grad():
                logits = audio_emb @ text_emb.T * logit_scale

        # Debug: Log after loss computation
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            if hasattr(self, "_debug_first_loss") and not hasattr(
                self, "_debug_first_loss_done"
            ):
                logger.info("[DEBUG] First CLIP loss computation completed")
                self._debug_first_loss_done = True

        # Compute accuracy metrics
        with torch.no_grad():
            local_bs = audio.size(0)

            rank = 0
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()

            start = rank * local_bs
            end = start + local_bs

            local_logits = logits[start:end, start:end]
            ground_truth = torch.arange(local_bs, device=self.device, dtype=torch.long)

            # Audio to text accuracy
            pred_a2t = torch.argmax(local_logits, dim=1)
            correct_a2t = (pred_a2t == ground_truth).sum().item()

            # Text to audio accuracy
            pred_t2a = torch.argmax(local_logits, dim=0)
            correct_t2a = (pred_t2a == ground_truth).sum().item()

            # Store additional metrics for epoch-level logging
            self.additional_metrics = {
                "acc_a2t": correct_a2t / local_bs,
                "acc_t2a": correct_t2a / local_bs,
            }

            # Add logit scale monitoring
            if hasattr(model, "logit_scale"):
                self.additional_metrics["logit_scale"] = model.logit_scale.exp().item()
            elif hasattr(model, "module") and hasattr(model.module, "logit_scale"):
                self.additional_metrics["logit_scale"] = (
                    model.module.logit_scale.exp().item()
                )

        return TrainingResult(
            loss=loss,
            metrics_data=(correct_a2t, correct_t2a),
            batch_size=local_bs,
            additional_metrics=self.additional_metrics,
        )


class EATSSLStrategy(TrainingStrategy):
    """Strategy for EAT self-supervised learning."""

    def get_expected_metrics_format(self) -> str:
        return "ssl_dummy"

    def forward(self, model: nn.Module, batch: Dict[str, Any]) -> TrainingResult:
        """Forward pass for EAT SSL training.

        Returns
        -------
        TrainingResult
            The result containing loss and metrics for EAT SSL training.

        Raises
        ------
        RuntimeError
            If EAT model doesn't support SSL method.
        """
        audio = batch["raw_wav"]
        padding_mask = batch.get("padding_mask")

        out = model(audio, padding_mask=padding_mask)

        if not isinstance(out, dict) or "losses" not in out:
            raise RuntimeError(
                "EAT model did not return expected loss dict in SSL mode"
            )

        # Per-component averages for logging
        sample_size = out["sample_size"].clamp(min=1).item()
        component_metrics = {
            k: v.sum().item() / sample_size for k, v in out["losses"].items()
        }

        # Add additional EAT metrics
        if "masked_pct" in out:
            component_metrics["masked_pct"] = out["masked_pct"].item()
        if "target_var" in out:
            component_metrics["target_var"] = out["target_var"].item()

        # Add prediction variance metrics
        for key, value in out.items():
            if key.startswith("pred_var_"):
                component_metrics[key] = value.item()

        self.additional_metrics = component_metrics

        # Compute total loss
        total_loss = sum(v.sum() for v in out["losses"].values())
        sample_size_tensor = out.get("sample_size", audio.size(0)).clamp(min=1).float()
        loss = total_loss / sample_size_tensor

        return TrainingResult(
            loss=loss,
            metrics_data=0,  # No accuracy for SSL
            batch_size=audio.size(0),
            additional_metrics=component_metrics,
        )

    def update_teacher(self, model: nn.Module, global_updates: int) -> None:
        """Update EAT SSL teacher model with global update count."""
        # Get unwrapped model if it's wrapped with DDP
        unwrapped_model = model.module if hasattr(model, "module") else model
        if hasattr(unwrapped_model, "backbone"):
            unwrapped_model.backbone.set_num_updates(global_updates)


class StrategyFactory:
    """Factory for creating training strategies."""

    @staticmethod
    def create_strategy(
        mode: str, criterion: nn.Module, device: torch.device
    ) -> TrainingStrategy:
        """Create appropriate training strategy based on mode.

        Returns
        -------
        TrainingStrategy
            The appropriate training strategy instance.

        Raises
        ------
        ValueError
            If the specified training mode is not supported.
        """
        if mode == "supervised":
            return SupervisedStrategy(criterion, device)
        elif mode == "clip":
            return CLIPStrategy(criterion, device)
        elif mode == "eat_ssl":
            return EATSSLStrategy(criterion, device)
        else:
            raise ValueError(f"Unknown training mode: {mode}")
