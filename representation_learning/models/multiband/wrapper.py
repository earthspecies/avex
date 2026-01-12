"""Multiband wrapper for any audio backbone.

This module provides a wrapper that adds multiband processing capability
to ANY existing audio model. The wrapper handles band splitting, runs
each band through the user-provided backbone, and fuses the results.
"""

import logging
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase
from representation_learning.models.multiband.fusion import BaseFusion, build_fusion
from representation_learning.models.multiband.transforms import MultibandTransform

logger = logging.getLogger(__name__)


class MultibandWrapper(nn.Module):
    """Wraps any audio backbone to add multiband processing.

    This wrapper:
    1. Splits input audio into frequency bands via heterodyning
    2. Runs each band through the provided backbone
    3. Fuses the per-band embeddings

    Works with any backbone that accepts (batch, time) waveforms and
    returns (batch, embed_dim) embeddings.

    Parameters
    ----------
    backbone : nn.Module
        Any audio model. Must accept waveform input and return embeddings.
        Can be a ModelBase subclass (BEATs, EAT, etc.) or any nn.Module.
    sample_rate : int
        Input audio sample rate (e.g., 44100, 96000)
    baseband_sr : int
        Sample rate expected by the backbone (typically 16000)
    band_width_hz : int
        Width of each frequency band
    fusion_type : str
        How to fuse band embeddings: "concat", "attention", "gated", "mean"
    embed_dim : int, optional
        Embedding dimension from backbone. If None, will try to detect.
    num_classes : int, optional
        If provided, adds a classification head after fusion.

    Example
    -------
    >>> from representation_learning import load_model
    >>> from representation_learning.models.multiband import MultibandWrapper
    >>>
    >>> # Load any backbone
    >>> backbone = load_model("beats_naturelm", device="cuda")
    >>>
    >>> # Wrap it for multiband processing
    >>> model = MultibandWrapper(
    ...     backbone=backbone,
    ...     sample_rate=44100,
    ...     baseband_sr=16000,
    ...     fusion_type="attention",
    ... )
    >>>
    >>> # Now handles high sample rate input
    >>> audio = torch.randn(4, 44100 * 5)  # 5 sec at 44.1kHz
    >>> embeddings = model(audio)  # (4, embed_dim)
    """

    def __init__(
        self,
        backbone: nn.Module,
        sample_rate: int = 44100,
        baseband_sr: int = 16000,
        band_width_hz: int = 8000,
        fusion_type: str = "attention",
        embed_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.sample_rate = sample_rate
        self.baseband_sr = baseband_sr

        # Multiband transform
        self.multiband = MultibandTransform(
            sample_rate=sample_rate,
            baseband_sr=baseband_sr,
            band_width_hz=band_width_hz,
        )

        # Detect embed_dim if not provided
        if embed_dim is None:
            embed_dim = self._detect_embed_dim()
        self.embed_dim = embed_dim

        logger.info(
            f"MultibandWrapper: {self.multiband.num_bands} bands, "
            f"{sample_rate}Hz -> {baseband_sr}Hz, embed_dim={embed_dim}"
        )

        # Fusion module
        self.fusion: BaseFusion = build_fusion(
            fusion_type=fusion_type,
            embed_dim=embed_dim,
            num_bands=self.multiband.num_bands,
            num_classes=num_classes if fusion_type == "logit" else None,
        )
        self.fusion_type = fusion_type

        # Optional classification head
        if num_classes is not None and fusion_type != "logit":
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = None

    def _detect_embed_dim(self) -> int:
        """Try to detect embedding dimension from backbone."""
        # Check common attributes
        if hasattr(self.backbone, "embed_dim"):
            return self.backbone.embed_dim
        if hasattr(self.backbone, "get_embedding_dim"):
            return self.backbone.get_embedding_dim()
        if hasattr(self.backbone, "hidden_size"):
            return self.backbone.hidden_size

        # Try a forward pass with dummy input
        logger.info("Detecting embed_dim via forward pass...")
        device = next(self.backbone.parameters()).device
        dummy = torch.randn(1, self.baseband_sr, device=device)
        with torch.no_grad():
            try:
                out = self._backbone_forward(dummy)
                if out.ndim == 2:
                    return out.shape[-1]
                elif out.ndim == 3:
                    return out.shape[-1]
            except Exception as e:
                logger.warning(f"Could not detect embed_dim: {e}")

        # Default fallback
        logger.warning("Using default embed_dim=768")
        return 768

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone and extract embeddings.

        Handles different backbone output formats:
        - (batch, embed_dim) -> use directly
        - (batch, time, embed_dim) -> mean pool over time
        - (batch, channels, height, width) -> global avg pool
        """
        # Check if backbone expects padding_mask
        if isinstance(self.backbone, ModelBase):
            out = self.backbone(x, padding_mask=None)
        else:
            out = self.backbone(x)

        # Handle different output shapes
        if out.ndim == 2:
            # Already (batch, embed_dim)
            return out
        elif out.ndim == 3:
            # (batch, time, embed_dim) -> mean pool
            return out.mean(dim=1)
        elif out.ndim == 4:
            # (batch, channels, height, width) -> global avg pool
            return out.mean(dim=(2, 3))
        else:
            raise ValueError(f"Unexpected backbone output shape: {out.shape}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with multiband processing.

        Parameters
        ----------
        x : torch.Tensor
            Input waveform of shape (N, T) at self.sample_rate
        padding_mask : torch.Tensor, optional
            Ignored (for API compatibility)

        Returns
        -------
        torch.Tensor
            Fused embeddings (N, embed_dim) or logits (N, num_classes)
        """
        # Split into bands: (N, T) -> (N, B, T')
        bands = self.multiband(x)
        N, B, T = bands.shape

        # Process all bands through backbone
        # Reshape to batch all bands together: (N*B, T')
        bands_flat = bands.reshape(N * B, T)

        # Forward through backbone
        embeddings_flat = self._backbone_forward(bands_flat)  # (N*B, D)

        # Reshape back: (N, B, D)
        D = embeddings_flat.shape[-1]
        embeddings = embeddings_flat.reshape(N, B, D)

        # Fuse band embeddings
        fused = self.fusion(embeddings)  # (N, D) or (N, num_classes) for logit

        # Apply classifier if present
        if self.classifier is not None:
            return self.classifier(fused)

        return fused

    def get_band_weights(self) -> Optional[torch.Tensor]:
        """Return fusion weights for interpretability."""
        return self.fusion.get_band_weights()

    def get_band_info(self) -> List[tuple]:
        """Return frequency ranges for each band."""
        return self.multiband.get_band_info()

    @property
    def num_bands(self) -> int:
        """Number of frequency bands."""
        return self.multiband.num_bands
