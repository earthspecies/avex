"""Transformer probe implemented using BaseProbe3D."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from avex.models.base_model import ModelBase
from avex.models.probes.base_probes import (
    BaseProbe3D,
)


class TransformerProbe(BaseProbe3D):
    """Transformer probe built on the shared 3D probe base."""

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
        aggregation: str = "mean",
        num_heads: int = 12,
        attention_dim: int = 768,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
        max_sequence_length: Optional[int] = None,
        use_positional_encoding: bool = False,
        target_length: Optional[int] = None,
        freeze_backbone: bool = True,
    ) -> None:
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.use_positional_encoding = use_positional_encoding

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
        """Build transformer encoder and classifier."""
        # Ensure embed_dim is divisible by num_heads
        if inferred_dim % self.num_heads != 0:
            adjusted = min(self.num_heads, inferred_dim)
            while inferred_dim % adjusted != 0 and adjusted > 1:
                adjusted -= 1
            self.num_heads = adjusted

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=inferred_dim,
            nhead=self.num_heads,
            dim_feedforward=self.attention_dim,
            dropout=self.dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        if self.use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, self.max_sequence_length or 1000, inferred_dim))
        else:
            self.pos_encoding = None  # type: ignore[assignment]

        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.classifier = nn.Linear(inferred_dim, self.num_classes)

    def __del__(self) -> None:
        try:
            if hasattr(self, "base_model") and self.base_model is not None:
                if hasattr(self.base_model, "deregister_all_hooks"):
                    self.base_model.deregister_all_hooks()
        except Exception:
            pass

    def forward(self, x: torch.Tensor | dict, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass matching original transformer probe behavior.

        Returns:
            torch.Tensor: The output tensor from the transformer probe.
        """
        embeddings = self._get_embeddings(x, padding_mask)
        embeddings = self._combine_or_reshape_embeddings(embeddings)

        if getattr(self, "pos_encoding", None) is not None:
            embeddings = embeddings + self.pos_encoding[:, : embeddings.size(1), :]

        if padding_mask is not None and padding_mask.shape[1] != embeddings.shape[1]:
            padding_mask = None

        transformer_out = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        pooled = transformer_out.mean(dim=1)
        if self.dropout is not None:
            pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def debug_info(self) -> dict:
        info = {
            "probe_type": "transformer",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "target_length": self.target_length,
            "num_heads": self.num_heads,
            "attention_dim": self.attention_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "max_sequence_length": self.max_sequence_length,
            "use_positional_encoding": self.use_positional_encoding,
            "has_layer_weights": hasattr(self, "layer_weights"),
            "has_embedding_projectors": hasattr(self, "embedding_projectors")
            and getattr(self, "embedding_projectors", None) is not None,
        }
        if hasattr(self, "layer_weights") and self.layer_weights is not None:
            info["layer_weights"] = self.layer_weights.detach().cpu().numpy().tolist()
        return info
