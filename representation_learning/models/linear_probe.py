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

        # Determine classifier input dimension
        if self.feature_mode:
            # Embeddings will be fed directly – base_model may be None.
            if input_dim is not None:
                inferred_dim = input_dim
            else:
                if base_model is None:
                    raise ValueError(
                        "input_dim must be provided when feature_mode=True "
                        "and base_model is None"
                    )
                with torch.no_grad():
                    # Derive dim via one dummy forward
                    target_length = (
                        base_model.audio_processor.target_length_seconds
                        * base_model.audio_processor.sr
                    )
                    dummy = torch.randn(1, target_length, device=device)
                    inferred_dim = base_model.extract_embeddings(dummy, layers).shape[1]
        else:
            # We will compute embeddings inside forward – need base_model.
            if base_model is None:
                raise ValueError("base_model must be provided when feature_mode=False")
            with torch.no_grad():
                target_length = (
                    base_model.audio_processor.target_length_seconds
                    * base_model.audio_processor.sr
                )
                dummy = torch.randn(1, target_length, device=device)
                inferred_dim = base_model.extract_embeddings(dummy, layers).shape[1]

        print(f"inferred_dim: {inferred_dim}")
        print(f"num_classes: {num_classes}")

        self.classifier = torch.nn.Linear(inferred_dim, num_classes).to(device)

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
