"""Fusion modules for combining multiband embeddings.

This module provides various fusion strategies for combining per-band
embeddings into a single representation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseFusion(nn.Module):
    """Base class for fusion modules."""

    def forward(
        self, embeddings: torch.Tensor, scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse band embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            Per-band embeddings of shape (N, B, D)
        scores : torch.Tensor, optional
            Handcrafted scores of shape (N, B) or (N, B, num_scores)

        Returns
        -------
        torch.Tensor
            Fused embedding of shape (N, D) or (N, D')
        """
        raise NotImplementedError

    def get_band_weights(self) -> Optional[torch.Tensor]:
        """Return last computed band weights for interpretability."""
        return getattr(self, "_last_weights", None)


class ConcatFusion(BaseFusion):
    """Concatenates band embeddings and projects to output dimension.

    This is the most information-preserving fusion method. All band
    embeddings are concatenated and a linear projection reduces
    dimensionality.
    """

    def __init__(self, embed_dim: int, num_bands: int, output_dim: Optional[int] = None):
        super().__init__()
        self.proj = nn.Linear(embed_dim * num_bands, output_dim or embed_dim)
        self.num_bands = num_bands

    def forward(
        self, embeddings: torch.Tensor, scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        N, B, D = embeddings.shape
        # Store uniform weights for interpretability
        self._last_weights = torch.ones(B, device=embeddings.device) / B
        return self.proj(embeddings.reshape(N, B * D))


class AttentionFusion(BaseFusion):
    """Self-attention over bands followed by mean pooling.

    Allows bands to attend to each other, learning cross-band
    dependencies. Useful when sounds span multiple frequency bands.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, embeddings: torch.Tensor, scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_out, attn_weights = self.attn(embeddings, embeddings, embeddings)
        x = self.norm(embeddings + attn_out)

        # Store attention weights (averaged over heads) for interpretability
        self._last_weights = attn_weights.mean(dim=(0, 2))  # (B,)

        return x.mean(dim=1)  # Mean pool over bands


class GatedFusion(BaseFusion):
    """Learned gating weights per band.

    Each band embedding is scored by a learned gate, and the final
    embedding is a weighted sum. Simple but effective.
    """

    def __init__(self, embed_dim: int, temperature: float = 1.0):
        super().__init__()
        self.gate = nn.Linear(embed_dim, 1)
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        gate_scores = self.gate(embeddings).squeeze(-1)  # (N, B)
        weights = F.softmax(gate_scores / self.temperature, dim=-1)

        # Store weights for interpretability
        self._last_weights = weights.detach().mean(dim=0)  # (B,)

        return (weights.unsqueeze(-1) * embeddings).sum(dim=1)


class HybridGatedFusion(BaseFusion):
    """Combines learned gates with handcrafted scores.

    Uses entropy/flux scores as additional input to the gating network,
    incorporating domain knowledge into the fusion.
    """

    def __init__(
        self,
        embed_dim: int,
        num_score_types: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        # Gate takes both embedding and scores
        self.gate = nn.Linear(embed_dim + num_score_types, 1)
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if scores is None:
            raise ValueError("HybridGatedFusion requires handcrafted scores")

        # scores: (N, B, num_scores) or (N, B)
        if scores.ndim == 2:
            scores = scores.unsqueeze(-1)

        # Concatenate embeddings with scores for gating
        gate_input = torch.cat([embeddings, scores], dim=-1)
        gate_scores = self.gate(gate_input).squeeze(-1)  # (N, B)
        weights = F.softmax(gate_scores / self.temperature, dim=-1)

        self._last_weights = weights.detach().mean(dim=0)

        return (weights.unsqueeze(-1) * embeddings).sum(dim=1)


class LogitFusion(BaseFusion):
    """Per-band classifiers with confidence-weighted voting.

    Each band has its own classifier head. Final prediction is a
    confidence-weighted combination of per-band predictions.
    Most interpretable: you can see which bands drove each decision.
    """

    def __init__(self, embed_dim: int, num_bands: int, num_classes: int):
        super().__init__()
        self.classifiers = nn.ModuleList(
            [nn.Linear(embed_dim, num_classes) for _ in range(num_bands)]
        )
        self.confidence = nn.ModuleList(
            [nn.Linear(embed_dim, 1) for _ in range(num_bands)]
        )
        self.num_bands = num_bands

    def forward(
        self, embeddings: torch.Tensor, scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns fused logits, not embeddings."""
        N, B, D = embeddings.shape

        all_logits = []
        all_conf = []
        for i in range(B):
            band_embed = embeddings[:, i, :]  # (N, D)
            logits = self.classifiers[i](band_embed)  # (N, C)
            conf = torch.sigmoid(self.confidence[i](band_embed))  # (N, 1)
            all_logits.append(logits)
            all_conf.append(conf)

        logits = torch.stack(all_logits, dim=1)  # (N, B, C)
        conf = torch.stack(all_conf, dim=1)  # (N, B, 1)

        # Normalize confidence to sum to 1
        conf_norm = conf / (conf.sum(dim=1, keepdim=True) + 1e-8)

        self._last_weights = conf_norm.detach().mean(dim=0).squeeze(-1)

        return (conf_norm * logits).sum(dim=1)  # (N, C)


def build_fusion(
    fusion_type: str,
    embed_dim: int,
    num_bands: int,
    num_classes: Optional[int] = None,
    **kwargs,
) -> BaseFusion:
    """Factory function to build fusion modules.

    Parameters
    ----------
    fusion_type : str
        One of "concat", "attention", "gated", "hybrid", "logit"
    embed_dim : int
        Embedding dimension from backbone
    num_bands : int
        Number of frequency bands
    num_classes : int, optional
        Number of classes (required for logit fusion)
    **kwargs
        Additional arguments passed to fusion module

    Returns
    -------
    BaseFusion
        Fusion module instance
    """
    if fusion_type == "concat":
        return ConcatFusion(embed_dim, num_bands, **kwargs)
    elif fusion_type == "attention":
        return AttentionFusion(embed_dim, **kwargs)
    elif fusion_type == "gated":
        return GatedFusion(embed_dim, **kwargs)
    elif fusion_type == "hybrid":
        return HybridGatedFusion(embed_dim, **kwargs)
    elif fusion_type == "logit":
        if num_classes is None:
            raise ValueError("LogitFusion requires num_classes")
        return LogitFusion(embed_dim, num_bands, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
