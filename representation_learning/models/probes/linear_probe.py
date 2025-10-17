"""Linear probe implemented using BaseProbe2D."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.base_probes import (
    BaseProbe2D,
)


class LinearProbe(BaseProbe2D):
    """Linear probe built on the shared 2D probe base."""

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
        aggregation: str = "mean",
        target_length: Optional[int] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=aggregation,
            target_length=target_length,
            freeze_backbone=freeze_backbone,
        )

    def build_head(self, inferred_dim: int) -> None:  # noqa: D401
        """Build a single linear classifier head."""
        self.classifier = nn.Linear(inferred_dim, self.num_classes)

    def __del__(self) -> None:
        try:
            if hasattr(self, "base_model") and self.base_model is not None:
                if hasattr(self.base_model, "deregister_all_hooks"):
                    self.base_model.deregister_all_hooks()
        except Exception:
            pass

    def forward(
        self, x: torch.Tensor | dict, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass identical to original LinearProbe.

        Returns:
            torch.Tensor: The output tensor from the linear probe.
        """
        embeddings = self._get_embeddings(x, padding_mask)

        # List case handled via projectors and  sum in base helper
        embeddings = self._combine_or_reshape_embeddings(embeddings)

        # Classify
        logits = self.classifier(embeddings)
        return logits

    def debug_info(self) -> dict:
        info = {
            "probe_type": "linear",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "target_length": self.target_length,
            "has_layer_weights": hasattr(self, "layer_weights"),
            "has_embedding_projectors": hasattr(self, "embedding_projectors")
            and getattr(self, "embedding_projectors", None) is not None,
        }
        if hasattr(self, "layer_weights") and self.layer_weights is not None:
            info["layer_weights"] = self.layer_weights.detach().cpu().numpy().tolist()
        return info
