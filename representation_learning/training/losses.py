from typing import Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from representation_learning.training.distributed import (
    get_rank,
    get_world_size,
)


def all_gather_features(features: torch.Tensor) -> torch.Tensor:
    """Gather *features* from all ranks **while preserving gradients**.

    Standard ``torch.distributed.all_gather`` fills the *output* tensors with
    detached data, so gradients cannot flow back to the original *features*.
    We fix this by overwriting the entry that corresponds to the *current*
    rank with the original tensor after the collective.

    This keeps the autograd graph intact for the local portion, which is all
    that is required for correct gradient computation.

    Returns
    -------
    torch.Tensor
        Concatenation of features across all ranks (first dimension enlarged
        by ``world_size``). Gradients propagate to the local slice.
    """

    world_size = get_world_size()
    if world_size == 1:
        return features

    gathered: list[torch.Tensor] = [
        torch.zeros_like(features, requires_grad=False) for _ in range(world_size)
    ]

    # Collective: each rank populates *gathered* (detached tensors)
    dist.all_gather(gathered, features.contiguous())

    # Replace this rank's slot with the original tensor (preserves grad)
    rank = get_rank()
    gathered[rank] = features

    return torch.cat(gathered, dim=0)


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        use_horovod: bool = False,
    ) -> None:
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.use_horovod = use_horovod
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
            all_image_features = all_gather_features(image_features)
            all_text_features = all_gather_features(text_features)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
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
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]
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
        world_size = get_world_size()
        rank = get_rank()
        self.world_size = world_size
        self.rank = rank

        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        if output_dict:
            return {"contrastive_loss": total_loss}
        if output_logits:
            return total_loss, logits_per_image
        return total_loss


def _build_criterion(loss_name: str) -> nn.Module:
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
    else:
        raise ValueError(f"Unknown loss_function: {loss_name}")
