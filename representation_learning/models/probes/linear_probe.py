"""Linear probe for representation learning evaluation."""

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
        device: Device to run on.
        feature_mode: Whether to use the input directly as embeddings.
        input_dim: Input dimension when in feature mode. Required if base_model is None.
        aggregation: How to aggregate multiple layer embeddings ('mean', 'max',
                    'concat').
    """

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        self.device = device
        self.base_model = base_model
        self.layers = layers
        self.feature_mode = feature_mode
        self.aggregation = aggregation

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
                    # Handle different audio processor types
                    if hasattr(base_model.audio_processor, "target_length_seconds"):
                        target_length = (
                            base_model.audio_processor.target_length_seconds
                            * base_model.audio_processor.sr
                        )
                    elif hasattr(base_model.audio_processor, "target_length"):
                        # For processors like EAT that use target_length in frames
                        target_length = base_model.audio_processor.target_length
                    else:
                        # Fallback: use a reasonable default
                        target_length = 16000  # 1 second at 16kHz

                    dummy = torch.randn(1, target_length, device=device)
                    inferred_dim = base_model.extract_embeddings(
                        dummy, layers=layers
                    ).shape[1]
        else:
            # We will compute embeddings inside forward – need base_model.
            if base_model is None:
                raise ValueError("base_model must be provided when feature_mode=False")
            with torch.no_grad():
                # Handle different audio processor types
                if hasattr(base_model.audio_processor, "target_length_seconds"):
                    target_length = (
                        base_model.audio_processor.target_length_seconds
                        * base_model.audio_processor.sr
                    )
                elif hasattr(base_model.audio_processor, "target_length"):
                    # For processors like EAT that use target_length in frames
                    target_length = base_model.audio_processor.target_length
                else:
                    # Fallback: use a reasonable default
                    target_length = 16000  # 1 second at 16kHz
                dummy = torch.randn(1, target_length, device=device)
                inferred_dim = base_model.extract_embeddings(
                    dummy, layers=layers
                ).shape[1]

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
            ValueError: If base_model is None when feature_mode=False
        """
        if self.feature_mode:
            embeddings = x  # type: ignore[arg-type]
        else:
            if self.base_model is None:
                raise ValueError("base_model must be provided when feature_mode=False")
            embeddings = self.base_model.extract_embeddings(
                x, self.layers, padding_mask=padding_mask
            )

        # Apply aggregation if multiple layers
        if len(self.layers) > 1:
            embeddings = self._aggregate_embeddings(embeddings)

        return self.classifier(embeddings)

    def _aggregate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate embeddings from multiple layers according to the aggregation
        method.

        Args:
            embeddings: Tensor of shape (batch_size, num_layers, embedding_dim)

        Returns:
            Aggregated embeddings of shape (batch_size, embedding_dim)

        Raises:
            ValueError: If aggregation method is not supported
        """
        if self.aggregation == "mean":
            return embeddings.mean(dim=1)
        elif self.aggregation == "max":
            return embeddings.max(dim=1)[0]
        elif self.aggregation == "concat":
            # Flatten all layer embeddings
            batch_size = embeddings.shape[0]
            return embeddings.view(batch_size, -1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
