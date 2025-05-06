from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase


# EfficientNetB0. Each class should be called "Model."
class Model(ModelBase):
    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
    ) -> None:
        # Call parent initializer with audio config
        super().__init__(device=device, audio_config=audio_config)

        # Store the flag
        self.return_features_only = return_features_only

        # Load a pre-trained EfficientNet B0 from torchvision.
        self.model = efficientnet_b0(pretrained=pretrained)

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
            Processed audio tensor with 3 channels for EfficientNet
        """
        # First use parent's audio processing
        x = super().process_audio(x)

        # EfficientNet expects 3 channels, so we need to repeat the spectrogram
        if x.dim() == 3:  # (batch, freq, time)
            x = x.unsqueeze(1)  # Add channel dimension
            x = x.repeat(1, 3, 1, 1)  # Repeat across channels

        return x

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

        # Extract features
        features = self.model.features(x)
        pooled_features = self.model.avgpool(features)
        flattened_features = torch.flatten(pooled_features, 1)

        # Return features or logits based on the flag
        if self.return_features_only:
            return flattened_features
        else:
            logits = self.model.classifier(flattened_features)
            return logits
