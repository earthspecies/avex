"""Attention probe implemented using BaseProbe3D."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.base_probes import (
    BaseProbe3D,
)


class AttentionProbe(BaseProbe3D):
    """Attention probe built on the shared 3D probe base."""

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
        aggregation: str = "mean",
        num_heads: int = 8,
        attention_dim: int = 512,
        num_layers: int = 2,
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
        """Build attention stack and classifier."""
        self.attention_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=inferred_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout_rate,
                    batch_first=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(inferred_dim))

        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        if self.use_positional_encoding:
            # Simple sinusoidal encoding
            pe = torch.zeros(self.max_sequence_length or 1000, inferred_dim)
            position = torch.arange(0, pe.shape[0], dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, inferred_dim, 2).float()
                * (-torch.log(torch.tensor(10000.0)) / inferred_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_encoding", pe.unsqueeze(0))
        else:
            self.pos_encoding = None  # type: ignore[assignment]

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
        """Forward pass matching original attention probe behavior.

        Returns:
            torch.Tensor: The output tensor from the attention probe.
        """
        embeddings = self._get_embeddings(x, padding_mask)
        embeddings = self._combine_or_reshape_embeddings(embeddings)

        if getattr(self, "pos_encoding", None) is not None:
            embeddings = embeddings + self.pos_encoding[:, : embeddings.size(1)]

        if padding_mask is not None and padding_mask.shape[1] != embeddings.shape[1]:
            padding_mask = None

        for i, attn in enumerate(self.attention_layers):
            attn_out, _ = attn(
                embeddings, embeddings, embeddings, key_padding_mask=padding_mask
            )
            embeddings = self.layer_norms[i](embeddings + attn_out)
            if self.dropout is not None:
                embeddings = self.dropout(embeddings)

        embeddings = embeddings.mean(dim=1)
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        return self.classifier(embeddings)

    def debug_info(self) -> dict:
        info = {
            "probe_type": "attention",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "num_heads": self.num_heads,
            "attention_dim": self.attention_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "max_sequence_length": self.max_sequence_length,
            "use_positional_encoding": self.use_positional_encoding,
            "target_length": self.target_length,
            "has_layer_weights": hasattr(self, "layer_weights"),
            "has_embedding_projectors": hasattr(self, "embedding_projectors")
            and getattr(self, "embedding_projectors", None) is not None,
        }
        if hasattr(self, "layer_weights") and self.layer_weights is not None:
            info["layer_weights"] = self.layer_weights.detach().cpu().numpy().tolist()
        return info
