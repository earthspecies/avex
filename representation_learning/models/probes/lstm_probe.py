"""LSTM probe implemented using BaseProbe3D."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.base_probes import (
    BaseProbe3D,
)


class LSTMProbe(BaseProbe3D):
    """LSTM probe built on the shared 3D probe base."""

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
        aggregation: str = "mean",
        lstm_hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout_rate: float = 0.1,
        max_sequence_length: Optional[int] = None,
        use_positional_encoding: bool = False,
        target_length: Optional[int] = None,
        freeze_backbone: bool = True,
    ) -> None:
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
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
        """Build LSTM and classifier."""
        # Heuristic to select stable hidden size with short sequences
        lstm_true_hidden_size = int(
            np.maximum(int((self.max_sequence_length or 4) / 4), self.lstm_hidden_size)
        )
        self.lstm = nn.LSTM(
            input_size=inferred_dim,
            hidden_size=lstm_true_hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            batch_first=True,
        )
        classifier_input_dim = lstm_true_hidden_size * (2 if self.bidirectional else 1)
        self.classifier = nn.Linear(classifier_input_dim, self.num_classes)

        if self.use_positional_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, self.max_sequence_length or 1000, inferred_dim)
            )
        else:
            self.pos_encoding = None  # type: ignore[assignment]

        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

    def __del__(self) -> None:
        try:
            if hasattr(self, "base_model") and self.base_model is not None:
                if hasattr(self, "deregister_all_hooks"):
                    self.base_model.deregister_all_hooks()
        except Exception:
            pass

    def forward(
        self, x: torch.Tensor | dict, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass matching original LSTM probe behavior.

        Returns:
            torch.Tensor: The output tensor from the LSTM probe.
        """
        embeddings = self._get_embeddings(x, padding_mask)
        embeddings = self._combine_or_reshape_embeddings(embeddings)

        if getattr(self, "pos_encoding", None) is not None:
            embeddings = embeddings + self.pos_encoding[:, : embeddings.size(1), :]

        embeddings = embeddings.contiguous()
        lstm_out, _ = self.lstm(embeddings)
        pooled = lstm_out.mean(dim=1)
        if self.dropout is not None:
            pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def debug_info(self) -> dict:
        info = {
            "probe_type": "lstm",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "target_length": self.target_length,
            "lstm_hidden_size": self.lstm_hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
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
