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
    ) -> None:
        # Call parent initializer with audio config
        super().__init__(device=device, audio_config=audio_config)

        # Load a pre-trained EfficientNet B0 from torchvision.
        self.model = efficientnet_b0(pretrained=pretrained)

        # If you need a different number of output classes than the default 1000,
        # modify the classifier. Here, classifier[1] is the final fully connected layer.
        if num_classes != 1000:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

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
            Model output
        """
        # Forward pass: process and call the underlying EfficientNet.
        x = self.process_audio(x)
        return self.model(x)
