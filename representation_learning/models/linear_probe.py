from typing import List, Optional

import torch

from representation_learning.models.base_model import ModelBase


class LinearProbe(torch.nn.Module):
    """
    Lightweight head for *linear probing* a frozen representation model.

    The probe extracts embeddings from specified layers of a **base_model** and
    feeds their concatenation into a single fully-connected classifier layer.

    Args:
        base_model: Frozen backbone network to pull embeddings from.
        layers: List of layer names to extract embeddings from.
        num_classes: Number of output classes.
        device: Device on which to place both backbone and probe.
        feature_mode: Whether to use the input directly as embeddings.
    """

    def __init__(
        self,
        base_model: ModelBase,
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_model = base_model
        self.layers = layers
        self.feature_mode = feature_mode

        # Calculate input dimension based on concatenated embeddings
        # We'll get this from a forward pass with a dummy input
        with torch.no_grad():
            # Get the target length from the audio config
            target_length = (
                base_model.audio_processor.target_length_seconds
                * base_model.audio_processor.sr
            )
            dummy_input = torch.randn(1, target_length).to(
                device
            )  # Use correct input length
            embeddings = self.base_model.extract_embeddings(dummy_input, self.layers)
            print(f"Embeddings shape: {embeddings.shape}")
            input_dim = embeddings.shape[1]

        self.classifier = torch.nn.Linear(input_dim, num_classes).to(device)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the linear probe.

        Args:
            x: Input tensor of shape (batch_size, time_steps)
            padding_mask: Optional padding mask tensor of shape (batch_size, time_steps)
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        if self.feature_mode:
            embeddings = x  # type: ignore[arg-type]
        else:
            embeddings = self.base_model.extract_embeddings(x, self.layers)

        return self.classifier(embeddings)
