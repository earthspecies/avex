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

    def _get_dummy_input_length(self, base_model: ModelBase) -> int:
        """Get appropriate dummy input length for different audio processor types.

        Args:
            base_model: The base model with an audio processor

        Returns:
            int: Length in samples for dummy audio input
        """
        audio_processor = base_model.audio_processor

        # Standard AudioProcessor interface
        if hasattr(audio_processor, "target_length_seconds") and hasattr(
            audio_processor, "sr"
        ):
            return int(audio_processor.target_length_seconds * audio_processor.sr)

        # EAT AudioProcessor interface
        if hasattr(audio_processor, "target_length") and hasattr(
            audio_processor, "sample_rate"
        ):
            # For EAT models, estimate raw audio length from target frames
            # EAT uses 10ms frame shift, so roughly target_length * hop_length
            hop_length = getattr(audio_processor, "hop_length", 160)  # 10ms at 16kHz
            return audio_processor.target_length * hop_length

        # Alternative AudioProcessor interfaces
        if hasattr(audio_processor, "target_length_seconds"):
            # Has target_length_seconds but different sample rate attribute name
            sample_rate = getattr(
                audio_processor, "sample_rate", getattr(audio_processor, "sr", 16000)
            )
            return int(audio_processor.target_length_seconds * sample_rate)

        # Fallback: Use a reasonable default (5 seconds at 16kHz)
        return 80000  # 5 seconds at 16kHz

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
                    target_length = self._get_dummy_input_length(base_model)
                    dummy = torch.randn(1, target_length, device=device)
                    inferred_dim = base_model.extract_embeddings(
                        dummy, layers=layers
                    ).shape[1]
        else:
            # We will compute embeddings inside forward – need base_model.
            if base_model is None:
                raise ValueError("base_model must be provided when feature_mode=False")
            with torch.no_grad():
                target_length = self._get_dummy_input_length(base_model)
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
            ValueError: If base_model is None when not in feature mode
        """
        if self.feature_mode:
            embeddings = x  # type: ignore[arg-type]
        else:
            if self.base_model is None:
                raise ValueError("base_model must be provided when not in feature mode")
            embeddings = self.base_model.extract_embeddings(
                x, self.layers, padding_mask=padding_mask
            )

        return self.classifier(embeddings)
