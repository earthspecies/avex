from typing import List, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torchvision.models import efficientnet_b0, efficientnet_b1

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase


# EfficientNet (B0/B1). Each class should be called "Model."
class Model(ModelBase):
    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        efficientnet_variant: str = "b0",
    ) -> None:
        # Call parent initializer with audio config
        super().__init__(device=device, audio_config=audio_config)

        # Store the flag
        self.return_features_only = return_features_only
        self.gradient_checkpointing = False

        # Load the appropriate EfficientNet variant based on configuration
        if efficientnet_variant == "b0":
            self.model = efficientnet_b0(pretrained=pretrained)
        elif efficientnet_variant == "b1":
            self.model = efficientnet_b1(pretrained=pretrained)
        else:
            raise ValueError(
                f"Unsupported EfficientNet variant: {efficientnet_variant}"
            )

        # Modify the classifier only if not returning features and num_classes differs.
        if not self.return_features_only and num_classes != 1000:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        # No need to modify classifier if return_features_only is True

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio input and adapt it for EfficientNet's 3-channel input.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor

        Returns
        -------
        torch.Tensor
            Processed audio tensor with 3 channels for EfficientNet (B0/B1)
        """
        # First use parent's audio processing
        x = super().process_audio(x)

        # EfficientNet expects 3 channels, so we need to repeat the spectrogram
        if x.dim() == 3:  # (batch, freq, time)
            x = x.unsqueeze(1)  # Add channel dimension
            x = x.repeat(1, 3, 1, 1)  # Repeat across channels

        return x

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for EfficientNet features."""
        self.gradient_checkpointing = True

    def _checkpointed_features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply features with gradient checkpointing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Features extracted with gradient checkpointing
        """
        return torch.utils.checkpoint.checkpoint(
            self.model.features, x, use_reentrant=False
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor
        padding_mask : torch.Tensor
            Padding mask for the input

        Returns
        -------
        torch.Tensor
            Model output (logits or features based on init flag)
        """
        # Process audio
        x = self.process_audio(x)

        # Extract features with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            features = self._checkpointed_features(x)
        else:
            features = self.model.features(x)

        pooled_features = self.model.avgpool(features)
        flattened_features = torch.flatten(pooled_features, 1)

        # Return features or logits based on the flag
        if self.return_features_only:
            return flattened_features
        else:
            logits = self.model.classifier(flattened_features)
            return logits

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        layers: List[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract embeddings from the model with automatic batch splitting.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav'
        layers : List[str]
            List of layer names (ignored for EfficientNet)
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (ignored for EfficientNet)

        Returns
        -------
        torch.Tensor
            Model embeddings
        """
        # Extract features
        if isinstance(x, dict):
            x = x["raw_wav"]
        x = self.process_audio(x)

        # Extract features with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            features = self._checkpointed_features(x)
        else:
            features = self.model.features(x)

        pooled_features = self.model.avgpool(features)
        flattened_features = torch.flatten(pooled_features, 1)
        return flattened_features
