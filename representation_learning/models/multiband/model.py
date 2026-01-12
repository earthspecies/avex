"""Multiband audio model with heterodyne preprocessing and fusion.

This module provides the main MultibandModel class that integrates
multiband transforms, a shared backbone, and fusion into a single
model compatible with the representation-learning framework.
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torchaudio

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase
from representation_learning.models.multiband.fusion import BaseFusion, build_fusion
from representation_learning.models.multiband.transforms import MultibandTransform

logger = logging.getLogger(__name__)


class MultibandModel(ModelBase):
    """Multiband audio model using EfficientNet backbone.

    This model:
    1. Splits input audio into frequency bands via heterodyning
    2. Converts each band to a mel spectrogram
    3. Processes each band through a shared EfficientNet backbone
    4. Fuses band embeddings using a configurable fusion module
    5. Optionally applies a classification head

    The model inherits from ModelBase and is compatible with the
    representation-learning framework including probes and checkpoints.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes. If None, returns embeddings only.
    device : str
        Device to run model on
    audio_config : AudioConfig, optional
        Audio configuration (sample_rate used as input rate)
    sample_rate : int
        Input audio sample rate (overrides audio_config if provided)
    baseband_sr : int
        Target sample rate for each band after heterodyning
    band_width_hz : int
        Width of each frequency band
    fusion_type : str
        Fusion method: "concat", "attention", "gated", "hybrid", "logit"
    pretrained : bool
        Whether to use pretrained EfficientNet weights
    efficientnet_variant : str
        EfficientNet variant: "b0" or "b1"
    return_features_only : bool
        If True, return fused embeddings instead of class logits
    """

    name = "multiband_model"

    def __init__(
        self,
        num_classes: Optional[int] = None,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        sample_rate: int = 44100,
        baseband_sr: int = 16000,
        band_width_hz: int = 8000,
        fusion_type: str = "attention",
        pretrained: bool = True,
        efficientnet_variant: str = "b0",
        return_features_only: bool = False,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # Use sample_rate from audio_config if provided
        if audio_config is not None and hasattr(audio_config, "sample_rate"):
            sample_rate = audio_config.sample_rate

        self.sample_rate = sample_rate
        self.baseband_sr = baseband_sr
        self.return_features_only = return_features_only or (num_classes is None)

        # Multiband transform
        self.multiband = MultibandTransform(
            sample_rate=sample_rate,
            baseband_sr=baseband_sr,
            band_width_hz=band_width_hz,
        )
        logger.info(
            f"MultibandModel: {self.multiband.num_bands} bands, "
            f"{band_width_hz}Hz each, {sample_rate}Hz -> {baseband_sr}Hz"
        )

        # Backbone (EfficientNet)
        self._init_backbone(pretrained, efficientnet_variant)
        self.embed_dim = 1280  # EfficientNet-B0/B1 output dim

        # Mel spectrogram for converting bands to backbone input
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=baseband_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        # Fusion
        self.fusion: BaseFusion = build_fusion(
            fusion_type=fusion_type,
            embed_dim=self.embed_dim,
            num_bands=self.multiband.num_bands,
            num_classes=num_classes,
        )
        self.fusion_type = fusion_type

        # Classification head (skip for logit fusion which has its own)
        if num_classes is not None and fusion_type != "logit":
            self.classifier = nn.Linear(self.embed_dim, num_classes)
        else:
            self.classifier = None

        self.to(device)

    def _init_backbone(self, pretrained: bool, variant: str) -> None:
        """Initialize EfficientNet backbone."""
        from torchvision.models import (
            EfficientNet_B0_Weights,
            EfficientNet_B1_Weights,
            efficientnet_b0,
            efficientnet_b1,
        )

        if variant == "b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
        elif variant == "b1":
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b1(weights=weights)
        else:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")

        # Remove classifier - we use our own
        self.backbone.classifier = nn.Identity()

    def _band_to_spectrogram(self, band: torch.Tensor) -> torch.Tensor:
        """Convert band waveform to mel spectrogram for EfficientNet.

        Parameters
        ----------
        band : torch.Tensor
            Waveform of shape (N, T)

        Returns
        -------
        torch.Tensor
            Spectrogram of shape (N, 3, H, W) for EfficientNet
        """
        spec = self.mel_spec(band)  # (N, n_mels, time)
        spec = torch.log(spec + 1e-8)

        # Normalize
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True) + 1e-5
        spec = (spec - mean) / std

        # EfficientNet expects 3 channels
        spec = spec.unsqueeze(1).repeat(1, 3, 1, 1)
        return spec

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through multiband model.

        Parameters
        ----------
        x : torch.Tensor
            Input waveform of shape (N, T)
        padding_mask : torch.Tensor, optional
            Padding mask (currently ignored)

        Returns
        -------
        torch.Tensor
            Classification logits (N, num_classes) or embeddings (N, D)
        """
        # Split into bands
        bands = self.multiband(x)  # (N, B, T')
        N, B, T = bands.shape

        # Process all bands through backbone efficiently
        bands_flat = bands.reshape(N * B, T)
        specs = self._band_to_spectrogram(bands_flat)  # (N*B, 3, H, W)

        features = self.backbone.features(specs)
        pooled = self.backbone.avgpool(features)
        embeddings_flat = torch.flatten(pooled, 1)  # (N*B, D)

        # Reshape to (N, B, D)
        embeddings = embeddings_flat.reshape(N, B, -1)

        # Fuse bands
        fused = self.fusion(embeddings)  # (N, D) or (N, C) for logit fusion

        # Return based on mode
        if self.return_features_only:
            return fused

        if self.classifier is not None:
            return self.classifier(fused)

        # LogitFusion already returns logits
        return fused

    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embed_dim

    def get_band_weights(self) -> Optional[torch.Tensor]:
        """Return fusion weights for interpretability."""
        return self.fusion.get_band_weights()

    def get_band_info(self) -> List[tuple]:
        """Return frequency ranges for each band."""
        return self.multiband.get_band_info()

    def _discover_linear_layers(self) -> None:
        """Discover linear layers for hook registration."""
        if len(self._layer_names) == 0:
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    self._layer_names.append(name)
            logger.info(f"Discovered {len(self._layer_names)} linear layers")
