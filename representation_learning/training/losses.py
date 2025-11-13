"""Loss functions for representation learning training.

This module provides various loss functions used in representation learning
tasks, including contrastive losses, focal loss, and distributed training utilities.
"""

import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from representation_learning.training.distributed import (
    get_rank,
    get_world_size,
)

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    local_loss: bool = False,
    gather_with_grad: bool = False,
    rank: int = 0,
    world_size: int = 1,
    use_horovod: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather image and text features from all ranks for contrastive learning.

    Parameters
    ----------
    image_features : torch.Tensor
        Local image/audio features
    text_features : torch.Tensor
        Local text features
    local_loss : bool, default=False
        Whether to use local loss computation
    gather_with_grad : bool, default=False
        Whether to gather features with gradients
    rank : int, default=0
        Current process rank
    world_size : int, default=1
        Total number of processes
    use_horovod : bool, default=False
        Whether to use Horovod for distributed training

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of (all_image_features, all_text_features) gathered from all ranks
    """
    assert has_distributed, "torch.distributed did not import correctly, please use a PyTorch version with support."
    # Optional debug print – enable by setting DEBUG_CLIP_GATHER=1
    if torch.distributed.is_initialized() and os.environ.get("DEBUG_CLIP_GATHER", "0") == "1" and rank == 0:
        print(f"[DEBUG] rank {rank} | batch={image_features.size(0)}")
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    """CLIP contrastive loss implementation with distributed training support.

    Parameters
    ----------
    local_loss : bool, default=False
        Whether to compute loss locally vs. globally across all ranks
    gather_with_grad : bool, default=False
        Whether to gather features with gradients preserved
    cache_labels : bool, default=False
        Whether to cache ground truth labels for efficiency
    rank : int, default=0
        Current process rank (will be overridden dynamically)
    world_size : int, default=1
        Total number of processes (will be overridden dynamically)
    use_horovod : bool, default=False
        Whether to use Horovod for distributed training
    """

    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ) -> None:
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels: Dict[torch.device, torch.Tensor] = {}

    def get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        """Get ground truth labels for contrastive loss.

        Parameters
        ----------
        device : torch.device
            Device to create labels on
        num_logits : int
            Number of logits (batch size)

        Returns
        -------
        torch.Tensor
            Labels tensor
        """
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: float,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits for contrastive loss.

        Parameters
        ----------
        image_features : torch.Tensor
            Image/audio embeddings
        text_features : torch.Tensor
            Text embeddings
        logit_scale : float
            Scale factor for logits
        logit_bias : Optional[torch.Tensor], optional
            Optional bias to add to logits

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of (logits_per_image, logits_per_text)
        """
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: float,
        logit_bias: Optional[torch.Tensor] = None,
        output_dict: bool = False,
        output_logits: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """Forward pass for contrastive loss calculation.

        Parameters
        ----------
        image_features : torch.Tensor
            Image/audio embeddings
        text_features : torch.Tensor
            Text embeddings
        logit_scale : float
            Scale factor for logits
        logit_bias : Optional[torch.Tensor], optional
            Optional bias to add to logits
        output_dict : bool, optional
            Whether to return loss as a dict
        output_logits : bool, optional
            Whether to return logits along with loss

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]
            Loss tensor, (loss, logits) tuple, or dict with loss
        """
        device = image_features.device

        # Dynamically get rank and world_size
        world_size = get_world_size()
        rank = get_rank()
        self.world_size = world_size
        self.rank = rank

        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
            logit_bias=logit_bias,
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        if output_dict:
            return {"contrastive_loss": total_loss}
        if output_logits:
            return total_loss, logits_per_image
        return total_loss


# --------------------------------------------------------------------------- #
#  Focal loss for multi-label classification (sigmoid variant)
# --------------------------------------------------------------------------- #


class FocalLoss(nn.Module):
    """Sigmoid-based focal loss (BCE variant) for multi-label tasks.

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter that down-weights well-classified examples.
    alpha : float | None, default=None
        Balancing factor between positive / negative examples.  Common values
        are ``0.25`` (as in the RetinaNet paper) or ``None`` to disable.
    reduction : str, {"mean", "sum", "none"}
        Reduction method applied to the per-sample loss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if targets.dtype != torch.float32:
            targets = targets.float()

        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Convert BCE loss back to probability space for focal scaling
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)

        focal_factor = (1 - p_t) ** self.gamma
        loss = focal_factor * bce

        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_factor * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def build_criterion(loss_name: str) -> nn.Module:
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name in {
        "bce",
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
    }:
        return nn.BCEWithLogitsLoss()
    elif loss_name in {"clip", "contrastive"}:
        return ClipLoss()
    elif loss_name in {"focal", "focal_loss"}:
        return FocalLoss()
    else:
        raise ValueError(f"Unknown loss_function: {loss_name}")
