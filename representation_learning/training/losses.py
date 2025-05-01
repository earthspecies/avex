import torch
import torch.nn as nn
import torch.nn.functional as F
from representation_learning.training.distributed import get_world_size, get_rank
import torch.distributed as dist


def all_gather_features(features: torch.Tensor) -> torch.Tensor:
    """Gather features from all processes for distributed training."""
    if get_world_size() == 1:
        return features
    gathered = [torch.zeros_like(features) for _ in range(get_world_size())]
    dist.all_gather(gathered, features)
    return torch.cat(gathered, dim=0)


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
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

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features = all_gather_features(image_features)
            all_text_features = all_gather_features(text_features)
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
        image_features,
        text_features,
        logit_scale,
        logit_bias=None,
        output_dict=False,
    ):
        device = image_features.device
        world_size = get_world_size()
        rank = get_rank()
        self.world_size = world_size
        self.rank = rank
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss


def _build_criterion(loss_name: str) -> nn.Module:
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name in {"bce", "binary_cross_entropy_with_logits"}:
        return nn.BCEWithLogitsLoss()
    elif loss_name == "clip":
        return ClipLoss()
    else:
        raise ValueError(f"Unknown loss_function: {loss_name}")