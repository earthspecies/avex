from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase


class Model(ModelBase):
    """
    A dummy model that produces random 768-dimensional embeddings.
    This serves as a random baseline for evaluation purposes.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = False,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        embedding_dim: int = 768,
        return_features_only: bool = False,
    ) -> None:
        """
        Initialize the dummy model.

        Args:
            num_classes: Number of output classes (ignored for embeddings)
            pretrained: Whether to use pretrained weights (ignored for dummy)
            device: Device to run on
            audio_config: Audio processing configuration (ignored for dummy)
            embedding_dim: Dimension of random embeddings (default: 768)
            return_features_only: Whether to return features or logits
        """
        super().__init__(device=device, audio_config=audio_config)

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.return_features_only = return_features_only

        # Create a simple linear layer for classification if needed
        if not return_features_only:
            self.classifier = nn.Linear(embedding_dim, num_classes)

        # Move to device
        self.to(device)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass that returns random embeddings or logits.

        Args:
            x: Input audio tensor (shape: [batch_size, ...])
            padding_mask: Optional padding mask (ignored)

        Returns:
            Random embeddings or logits
        """
        batch_size = x.shape[0]
        device = x.device

        # Generate random embeddings
        embeddings = torch.randn(batch_size, self.embedding_dim, device=device)

        if self.return_features_only:
            return embeddings
        else:
            # Return logits through classifier
            return self.classifier(embeddings)

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        layers: List[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        average_over_time: bool = True,
        framewise_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Extract random embeddings (ignores layer specification).

        Args:
            x: Input tensor or dictionary
            layers: List of layer names (ignored for dummy model)
            padding_mask: Optional padding mask (ignored)
            average_over_time: Whether to average over time
            framewise_embeddings: Whether to return frame-level embeddings

        Returns:
            Random embeddings tensor
        """
        if isinstance(x, dict):
            x = x["raw_wav"]

        batch_size = x.shape[0]
        device = x.device

        if framewise_embeddings and not average_over_time:
            # Return frame-level embeddings - assume 50 FPS for 5 second clips
            target_frames = int(self.audio_processor.target_length_seconds * 50)
            embeddings = torch.randn(
                batch_size, target_frames, self.embedding_dim, device=device
            )
            return [embeddings]  # Return as list like the base implementation
        else:
            # Return clip-level embeddings
            return torch.randn(batch_size, self.embedding_dim, device=device)

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing (no-op for dummy model).
        """
        pass  # No gradients to checkpoint for random embeddings
