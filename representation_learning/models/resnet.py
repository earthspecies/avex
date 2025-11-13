"""ResNet model implementation for audio classification.

This module provides ResNet model implementation for audio classification tasks,
using 2D ResNet backbones on mel-spectrogram inputs.
"""

# ResNet audio classifier integrated with the common **ModelBase** interface
# so it can be used transparently by the training / evaluation utilities.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase


class Model(ModelBase):
    """Audio front-end that feeds a Mel-spectrogram (B × F × T) through a
    2-D ResNet backbone.

    The input spectrogram is converted to 3-channel format so we can reuse the
    ImageNet-pre-trained models from **torchvision**.
    """

    _ALLOWED_VARIANTS = {
        "resnet18": torchvision.models.resnet18,
        "resnet50": torchvision.models.resnet50,
        "resnet152": torchvision.models.resnet152,
    }

    _DEFAULT_WEIGHTS = {
        "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
        "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
        "resnet152": torchvision.models.ResNet152_Weights.DEFAULT,
    }

    def __init__(
        self,
        variant: str = "resnet18",
        *,
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # Validate num_classes: required when return_features_only=False
        if not return_features_only and num_classes is None:
            # Use default from torchvision ResNet (1000 classes)
            num_classes = 1000

        variant = variant.lower()
        if variant not in self._ALLOWED_VARIANTS:
            allowed = list(self._ALLOWED_VARIANTS)
            raise ValueError(f"Unsupported ResNet variant '{variant}'. Supported variants: {allowed}")

        weights = self._DEFAULT_WEIGHTS[variant] if pretrained else None
        # Instantiate the torchvision model
        self.backbone = self._ALLOWED_VARIANTS[variant](weights=weights)

        # Replace the final FC with identity so we get feature vector
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Store configuration
        self.return_features_only = return_features_only

        # Linear classifier for our target dataset (only if not return_features_only)
        if not return_features_only:
            self.classifier = nn.Linear(in_features, num_classes)
            self.classifier.to(self.device)
        else:
            self.classifier = None

        # Send to device
        self.backbone.to(self.device)

    # ------------------------------------------------------------------ #
    #  Audio processing helpers
    # ------------------------------------------------------------------ #
    def process_audio(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Convert a spectrogram into a 3-channel image-like tensor.

        Parameters
        ----------
        x : torch.Tensor
            Either a raw waveform ``(B, T)`` or a spectrogram ``(B, F, T)``
            depending on whether :pyattr:`audio_config` was supplied.

        Returns
        -------
        torch.Tensor
            A 4-D tensor of shape ``(B, 3, F, T)`` normalised to the range
            ``[0, 1]`` so it can be fed into a standard 2-D ResNet.
        """
        x = super().process_audio(x)

        if x.dim() == 3:  # (B, F, T)
            x = x.unsqueeze(1)  # (B, 1, F, T)
            x = x.repeat(1, 3, 1, 1)  # (B, 3, F, T)

        # Normalise to [0, 1] per-sample to mimic image pixel range
        x = x / (x.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return x

    # ------------------------------------------------------------------ #
    #  Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401, N802  (keep same signature as other models)
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw waveform or spectrogram input.
        padding_mask : torch.Tensor | None, optional
            Ignored for ResNet models; kept for API parity.

        Returns
        -------
        torch.Tensor
            • When *return_features_only* is **False**: class-logit tensor of shape
              ``(B, num_classes)``
            • Otherwise: feature tensor of shape ``(B, in_features)``
        """
        x = self.process_audio(x)
        features = self.backbone(x)  # (B, in_features)
        if self.return_features_only:
            return features
        return self.classifier(features)
