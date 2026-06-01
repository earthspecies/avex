"""Prototypical probe for backbone fine-tuning with prototype-based classification.

Restricted to EfficientNet, ConvNeXt, and AudioProtoPNet backbones.

Reference: Ghaffari et al., "AudioProtoPNet: An Interpretable Deep Learning
Model for Bird Sound Classification", 2024.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from avex.models.base_model import ModelBase
from avex.models.probes.base_probes import BaseProbe2D

_SUPPORTED_BACKBONE_MODULES = frozenset({"convnext", "audioprotopnet", "efficientnet"})


def _check_backbone_supported(model: nn.Module) -> None:
    module = type(model).__module__.lower()
    if not any(name in module for name in _SUPPORTED_BACKBONE_MODULES):
        raise ValueError(
            f"PrototypicalProbe only supports EfficientNet, ConvNeXt, or AudioProtoPNet backbones. "
            f"Got model from module '{type(model).__module__}'. "
            f"Supported: {sorted(_SUPPORTED_BACKBONE_MODULES)}"
        )


class PrototypicalHead(nn.Module):
    """Prototype classification head operating on 1-D embeddings.

    embeddings → cosine similarities with prototype vectors → [0, 1] shift →
    non-negative linear layer → class logits.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the input embedding vector.
    num_classes : int
        Number of target classes.
    num_prototypes_per_class : int
        Number of prototype vectors per class.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        num_prototypes_per_class: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = num_classes * num_prototypes_per_class

        self.prototype_vectors = nn.Parameter(torch.randn(self.num_prototypes, embedding_dim))
        nn.init.xavier_uniform_(self.prototype_vectors)

        self.register_buffer(
            "prototype_class_identity",
            torch.arange(self.num_prototypes) // num_prototypes_per_class,
        )

        self.last_layer = nn.Linear(self.num_prototypes, num_classes, bias=False)
        self._init_last_layer()

    def _init_last_layer(self) -> None:
        with torch.no_grad():
            w = torch.full((self.num_classes, self.num_prototypes), -0.5)
            for p in range(self.num_prototypes):
                c = int(self.prototype_class_identity[p].item())
                w[c, p] = 1.0
            self.last_layer.weight.copy_(w)

    def clamp_last_layer(self) -> None:
        """Enforce non-negativity of correct-class connections. Call after each optimizer step."""
        with torch.no_grad():
            class_ids = self.prototype_class_identity
            proto_ids = torch.arange(self.num_prototypes, device=class_ids.device)
            self.last_layer.weight[class_ids, proto_ids].clamp_(min=0)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        e = F.normalize(embeddings, dim=-1)
        p = F.normalize(self.prototype_vectors, dim=-1)
        similarities = (e @ p.T + 1) / 2
        return self.last_layer(similarities)


class PrototypicalProbe(BaseProbe2D):
    """Prototype-based probe: cosine-similarity prototype layer + non-negative linear head.

    Only compatible with EfficientNet, ConvNeXt, and AudioProtoPNet backbones.

    Parameters
    ----------
    base_model : Optional[ModelBase]
        Backbone model; required when not in feature_mode.
    layers : List[str]
        Layer names to extract embeddings from.
    num_classes : int
        Number of output classes.
    device : str
        Device string.
    feature_mode : bool
        If True, accepts pre-computed embeddings instead of raw audio.
    input_dim : Optional[int]
        Embedding dim when feature_mode=True and base_model is None.
    aggregation : str
        Embedding aggregation strategy.
    target_length : Optional[int]
        Audio length in samples (inferred from backbone if None).
    freeze_backbone : bool
        Whether to freeze the backbone during training.
    num_prototypes_per_class : int
        Number of prototype vectors per class.
    """

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
        aggregation: str = "mean",
        input_processing: str = "pooled",
        target_length: Optional[int] = None,
        freeze_backbone: bool = True,
        num_prototypes_per_class: int = 10,
    ) -> None:
        if base_model is not None and not feature_mode:
            _check_backbone_supported(base_model)
        self.num_prototypes_per_class = num_prototypes_per_class
        super().__init__(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=aggregation,
            input_processing=input_processing,
            target_length=target_length,
            freeze_backbone=freeze_backbone,
        )

    def build_head(self, inferred_dim: int) -> None:
        self.classifier = PrototypicalHead(
            embedding_dim=inferred_dim,
            num_classes=self.num_classes,
            num_prototypes_per_class=self.num_prototypes_per_class,
        )

    def clamp_last_layer(self) -> None:
        """Enforce non-negativity of correct-class connections. Call after each optimizer step."""
        self.classifier.clamp_last_layer()

    def __del__(self) -> None:
        try:
            if hasattr(self, "base_model") and self.base_model is not None:
                if hasattr(self.base_model, "deregister_all_hooks"):
                    self.base_model.deregister_all_hooks()
        except Exception:
            pass

    def forward(self, x: torch.Tensor | dict, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = self._get_embeddings(x, padding_mask)
        embeddings = self._combine_or_reshape_embeddings(embeddings)
        return self.classifier(embeddings)

    def debug_info(self) -> dict:
        return {
            "probe_type": "prototypical",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "num_prototypes_per_class": self.num_prototypes_per_class,
            "has_layer_weights": hasattr(self, "layer_weights"),
        }
