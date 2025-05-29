from typing import List, Optional

import torch

from representation_learning.models.base_model import ModelBase


class LinearProbe(torch.nn.Module):
    """
    Lightweight head for *linear probing* a frozen representation model.

    The probe extracts embeddings from specified layers of a **base_model** and
    feeds their concatenation into a single fully-connected classifier layer.

    Args:
        base_model: Frozen backbone network to pull embeddings from. Can be None if
            feature_mode=True.
        layers: List of layer names to extract embeddings from.
        num_classes: Number of output classes.
        device: Device on which to place both backbone and probe.
        feature_mode: Whether to use the input directly as embeddings.
        input_dim: Input dimension when in feature mode. Required if base_model is None.
    """

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_model = base_model
        self.layers = layers
        self.feature_mode = feature_mode

        if feature_mode:
            if base_model is None and input_dim is None:
                raise ValueError(
                    "input_dim must be provided when base_model is None in feature mode"
                )
            if input_dim is not None:
                self.classifier = torch.nn.Linear(input_dim, num_classes).to(device)
            else:
                # Calculate input dimension based on concatenated embeddings
                with torch.no_grad():
                    # Get the target length from the audio config
                    target_length = (
                        base_model.audio_processor.target_length_seconds
                        * base_model.audio_processor.sr
                    )
                    dummy_input = torch.randn(1, target_length).to(device)
                    embeddings = self.base_model.extract_embeddings(
                        dummy_input, self.layers
                    )
                    input_dim = embeddings.shape[1]
                self.classifier = torch.nn.Linear(input_dim, num_classes).to(device)
        else:
            if base_model is None:
                raise ValueError("base_model must be provided when not in feature mode")
            # Calculate input dimension based on concatenated embeddings
            with torch.no_grad():
                # Get the target length from the audio config
                target_length = (
                    base_model.audio_processor.target_length_seconds
                    * base_model.audio_processor.sr
                )
                dummy_input = torch.randn(1, target_length).to(device)
                embeddings = self.base_model.extract_embeddings(
                    dummy_input, self.layers
                )
                input_dim = embeddings.shape[1]
            self.classifier = torch.nn.Linear(input_dim, num_classes).to(device)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the linear probe.

        Args:
            x: Input tensor of shape (batch_size, time_steps)
                or (batch_size, embedding_dim) in feature mode
            padding_mask: Optional padding mask tensor of shape (batch_size, time_steps)
        Returns:
            Classification logits of shape (batch_size, num_classes)
        Raises:
            ValueError: If base_model is None when not in feature mode
        """
        if self.feature_mode:
            embeddings = x  # type: ignore[arg-type]
        else:
            if self.base_model is None:
                raise ValueError("base_model must be provided when not in feature mode")
            embeddings = self.base_model.extract_embeddings(x, self.layers)

        return self.classifier(embeddings)
