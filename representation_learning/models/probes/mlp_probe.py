"""MLP probe implemented using BaseProbe2D."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.base_probes import (
    BaseProbe2D,
)


class MLPProbe(BaseProbe2D):
    """MLP probe built on the shared probe base."""

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
        aggregation: str = "mean",
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        target_length: Optional[int] = None,
        freeze_backbone: bool = True,
    ) -> None:
        self.hidden_dims = hidden_dims or [512, 256]
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.num_classes = num_classes

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
        """Build a simple MLP followed by a classifier.

        Raises:
            ValueError: If the inferred dimension is invalid.
        """
        layers_list: List[nn.Module] = []
        current_dim = inferred_dim
        for hidden_dim in self.hidden_dims:
            layers_list.append(nn.Linear(current_dim, hidden_dim))
            if self.activation == "relu":
                layers_list.append(nn.ReLU())
            elif self.activation == "gelu":
                layers_list.append(nn.GELU())
            elif self.activation == "tanh":
                layers_list.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")
            if self.dropout_rate > 0:
                layers_list.append(nn.Dropout(self.dropout_rate))
            current_dim = hidden_dim
        layers_list.append(nn.Linear(current_dim, self.num_classes))
        self.mlp = nn.Sequential(*layers_list)

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
        """Forward pass matching original MLP probe behavior.

        Returns:
            torch.Tensor: The output tensor from the MLP probe.
        """
        embeddings = self._get_embeddings(x, padding_mask)
        embeddings = self._combine_or_reshape_embeddings(embeddings)
        return self.mlp(embeddings)

    def debug_info(self) -> dict:
        info = {
            "probe_type": "mlp",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "target_length": self.target_length,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "has_layer_weights": hasattr(self, "layer_weights"),
            "has_embedding_projectors": hasattr(self, "embedding_projectors")
            and getattr(self, "embedding_projectors", None) is not None,
        }
        if hasattr(self, "layer_weights") and self.layer_weights is not None:
            info["layer_weights"] = self.layer_weights.detach().cpu().numpy().tolist()
        return info
