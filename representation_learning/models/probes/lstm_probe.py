"""LSTM probe for representation learning evaluation."""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


class LSTMProbe(torch.nn.Module):
    """
    LSTM probe for sequence-based representation learning evaluation.

    The probe extracts embeddings from specified layers of a **base_model** and
    processes them through an LSTM for sequence modeling.

    Args:
        base_model: Backbone network to pull embeddings from. Can be None if
            feature_mode=True.
        layers: List of layer names to extract embeddings from.
        num_classes: Number of output classes.
        device: Device to run on.
        feature_mode: Whether to use the input directly as embeddings.
        input_dim: Input dimension when in feature mode. Required if base_model is None.
        aggregation: How to aggregate multiple layer embeddings ('mean', 'max',
                    'cls_token', 'none'). If 'none', each layer gets its own
                    projection head.
        lstm_hidden_size: Hidden size of the LSTM.
        num_layers: Number of LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
        dropout_rate: Dropout rate for regularization.
        max_sequence_length: Maximum sequence length for processing.
        use_positional_encoding: Whether to add positional encoding.
        target_length: Target length in samples for audio processing. If None, will be
            computed from base_model.audio_processor. Required if
            base_model.audio_processor does not have target_length or
            target_length_seconds attributes.

        freeze_backbone: Whether the backbone model is frozen. If True, embeddings
            will be detached to prevent gradient flow back to the backbone.
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
        lstm_hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout_rate: float = 0.1,
        max_sequence_length: Optional[int] = None,
        use_positional_encoding: bool = False,
        target_length: Optional[int] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_model = base_model
        self.layers = layers
        self.feature_mode = feature_mode
        self.aggregation = aggregation
        self.freeze_backbone = freeze_backbone
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.use_positional_encoding = use_positional_encoding
        self.target_length = target_length

        # Register hooks for the specified layers if base_model is provided
        if self.base_model is not None and not self.feature_mode:
            self.base_model.register_hooks_for_layers(self.layers)

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
                    # Use provided target_length or compute from base_model
                    if self.target_length is not None:
                        computed_target_length = self.target_length
                    elif hasattr(base_model.audio_processor, "target_length_seconds"):
                        computed_target_length = (
                            base_model.audio_processor.target_length_seconds
                            * base_model.audio_processor.sr
                        )
                    elif hasattr(base_model.audio_processor, "target_length"):
                        # For processors like EAT that use target_length in frames
                        computed_target_length = (
                            base_model.audio_processor.target_length
                        )
                    else:
                        raise ValueError(
                            "target_length must be provided when "
                            "base_model.audio_processor does not have target_length or "
                            "target_length_seconds attributes"
                        )

                    dummy = torch.randn(1, computed_target_length, device=device)
                    dummy_embeddings = base_model.extract_embeddings(
                        dummy, aggregation=self.aggregation
                    ).detach()

                    # feature_mode=True assumes that the embeddings are not lists
                    assert isinstance(dummy_embeddings, torch.Tensor), (
                        "dummy_embeddings should be a tensor"
                    )
                    if dummy_embeddings.dim() == 3:
                        # For sequence probes, we expect 3D embeddings
                        # (batch_size, sequence_length, embedding_dim)
                        inferred_dim = dummy_embeddings.shape[-1]
                    else:
                        raise ValueError(
                            f"LSTM probe expects 3D embeddings (batch_size, "
                            f"sequence_length, embedding_dim), got shape "
                            f"{dummy_embeddings.shape}"
                        )
        else:
            # Extract embeddings from base_model to determine dimensions
            with torch.no_grad():
                # Use provided target_length or compute from base_model
                if self.target_length is not None:
                    computed_target_length = self.target_length
                elif hasattr(base_model.audio_processor, "target_length_seconds"):
                    computed_target_length = (
                        base_model.audio_processor.target_length_seconds
                        * base_model.audio_processor.sr
                    )
                elif hasattr(base_model.audio_processor, "target_length"):
                    # For processors like EAT that use target_length in frames
                    computed_target_length = base_model.audio_processor.target_length
                else:
                    raise ValueError(
                        "target_length must be provided when "
                        "base_model.audio_processor does not have target_length or "
                        "target_length_seconds attributes"
                    )

                dummy = torch.randn(1, computed_target_length, device=device)
                dummy_embeddings = base_model.extract_embeddings(
                    dummy, aggregation=self.aggregation
                )

                # Handle the case where dummy_embeddings is a list (aggregation="none")
                if isinstance(dummy_embeddings, list):
                    # Detach each embedding in the list
                    dummy_embeddings = [emb.detach() for emb in dummy_embeddings]

                    # For sequence probes, we expect 3D embeddings
                    # Check that all embeddings are 3D
                    for i, emb in enumerate(dummy_embeddings):
                        if emb.dim() != 3:
                            raise ValueError(
                                f"LSTM probe expects 3D embeddings (batch_size, "
                                f"sequence_length, embedding_dim), got shape "
                                f"{emb.shape} for layer {i}"
                            )

                    # Create LSTM projection heads for each layer
                    self.layer_projections = nn.ModuleList(
                        [
                            nn.LSTM(
                                input_size=emb.shape[-1],
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional,
                                dropout=self.dropout_rate if self.num_layers > 1 else 0,
                                batch_first=True,
                            )
                            for emb in dummy_embeddings
                        ]
                    )

                    # Input dimension is lstm_hidden_size * number of layers
                    # (accounting for bidirectional LSTM)
                    lstm_output_size = self.lstm_hidden_size * (
                        2 if self.bidirectional else 1
                    )
                    inferred_dim = lstm_output_size * len(dummy_embeddings)

                    # Log the setup
                    logger.info(
                        f"LSTMProbe init (feature_mode=False, aggregation='none'): "
                        f"dummy_embeddings: list of {len(dummy_embeddings)} tensors, "
                        f"layers: {layers}, aggregation: {self.aggregation}, "
                        f"inferred_dim: {inferred_dim}"
                    )
                else:
                    # Single tensor case
                    logger.debug(
                        f"LSTM probe: Single tensor case - dummy_embeddings type: "
                        f"{type(dummy_embeddings)}, shape: {dummy_embeddings.shape}, "
                        f"aggregation: {self.aggregation}"
                    )

                    if self.freeze_backbone:
                        dummy_embeddings = dummy_embeddings.detach()

                    if dummy_embeddings.dim() == 3:
                        # For sequence probes, we expect 3D embeddings
                        # (batch_size, sequence_length, embedding_dim)
                        inferred_dim = dummy_embeddings.shape[-1]
                        logger.debug(
                            f"LSTM probe: Using 3D tensor with "
                            f"inferred_dim: {inferred_dim}"
                        )
                    else:
                        logger.error(
                            f"LSTM probe: Expected 3D embeddings but got shape "
                            f"{dummy_embeddings.shape}. This suggests the "
                            f"base_model.extract_embeddings did not respect "
                            f"aggregation='{self.aggregation}'"
                        )
                        raise ValueError(
                            f"LSTM probe expects 3D embeddings (batch_size, "
                            f"sequence_length, embedding_dim), got shape "
                            f"{dummy_embeddings.shape}"
                        )

        # Create the main LSTM layer (only used for single layer or when not
        # using projection heads)
        self.lstm = nn.LSTM(
            input_size=inferred_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
        )

        # Calculate output size based on bidirectional setting
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)

        # Create the classifier head
        self.classifier = nn.Linear(lstm_output_size, num_classes)

        # Optional positional encoding
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, max_sequence_length or 1000, inferred_dim)
            )
        else:
            self.pos_encoding = None

        # Optional dropout after LSTM
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        self.to(device)

    def __del__(self) -> None:
        """Cleanup hooks when the probe is destroyed."""
        try:
            if hasattr(self, "base_model") and self.base_model is not None:
                if hasattr(self.base_model, "deregister_all_hooks"):
                    self.base_model.deregister_all_hooks()
        except Exception:
            pass  # Ignore errors during cleanup

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the LSTM probe.

        Args:
            x: Input tensor. If feature_mode=True, this should be embeddings.
                If feature_mode=False, this should be raw audio.
            padding_mask: Optional padding mask for the input. Currently not used
                by LSTM probes but included for compatibility with training loops.

        Returns:
            Classification logits

        Raises:
            ValueError: If embeddings are not 3D (batch_size, sequence_length,
                embedding_dim)
        """
        if isinstance(x, dict):
            padding_mask = x.get("padding_mask")
            x = x["raw_wav"]

        if self.feature_mode:
            # x is already embeddings
            assert isinstance(x, torch.Tensor), "x should be a tensor in feature mode"
            embeddings = x

            # In feature mode, we need to process the embeddings through the main LSTM
            # Ensure embeddings are 3D
            if embeddings.dim() != 3:
                raise ValueError(
                    f"LSTM probe expects 3D embeddings (batch_size, "
                    f"sequence_length, embedding_dim), got shape "
                    f"{embeddings.shape}"
                )

            # Add positional encoding if enabled
            if self.pos_encoding is not None:
                embeddings = embeddings + self.pos_encoding[:, : embeddings.size(1), :]

            # Pass through main LSTM
            lstm_out, _ = self.lstm(embeddings)
        else:
            # Extract embeddings from base_model
            logger.debug(
                f"LSTM probe forward: Calling base_model.extract_embeddings "
                f"with aggregation='{self.aggregation}'"
            )
            embeddings = self.base_model.extract_embeddings(
                x, padding_mask=padding_mask, aggregation=self.aggregation
            )
            logger.debug(
                f"LSTM probe forward: Received embeddings type: {type(embeddings)}, "
                f"shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'list'}"
            )

            if self.freeze_backbone:
                if isinstance(embeddings, list):
                    embeddings = [emb.detach() for emb in embeddings]
                else:
                    embeddings = embeddings.detach()

            # Handle the case where embeddings is a list (aggregation="none")
            if isinstance(embeddings, list):
                # Apply LSTM projection heads to each layer's embeddings
                projected_embeddings = []
                logger.debug(
                    f"Processing {len(embeddings)} embeddings through "
                    f"LSTM projection heads"
                )

                for i, (emb, lstm_proj) in enumerate(
                    zip(embeddings, self.layer_projections, strict=False)
                ):
                    # Ensure embeddings are 3D
                    if emb.dim() == 3:
                        logger.debug(
                            f"Processing embedding {i}: shape={emb.shape}, "
                            f"device={emb.device}"
                        )
                        # Ensure the embedding is contiguous for cuDNN compatibility
                        emb = emb.contiguous()
                        logger.debug(
                            f"After contiguous: shape={emb.shape}, device={emb.device}"
                        )

                        # Pass through LSTM projection head
                        # Note: Using regular LSTM instead of packed sequences
                        # to avoid cuDNN issues
                        lstm_out, _ = lstm_proj(emb)
                        # Take the mean for classification
                        projected_emb = lstm_out.mean(dim=1)
                    else:
                        raise ValueError(
                            f"LSTM probe expects 3D embeddings (batch_size, "
                            f"sequence_length, embedding_dim), got shape "
                            f"{emb.shape} for layer {i}"
                        )
                    projected_embeddings.append(projected_emb)

                # Concatenate along the feature dimension
                embeddings = torch.cat(projected_embeddings, dim=-1)

                # When using projection heads, embeddings are already processed
                # No need to pass through main LSTM - go directly to classifier
                if self.pos_encoding is not None:
                    # Note: positional encoding not applicable for concatenated features
                    pass

                # Skip main LSTM since we already processed each layer with
                # individual LSTMs
                lstm_out = embeddings.unsqueeze(1)  # Add sequence dimension for
                # consistency
            else:
                # Single tensor case - ensure it's 3D
                if embeddings.dim() != 3:
                    raise ValueError(
                        f"LSTM probe expects 3D embeddings (batch_size, "
                        f"sequence_length, embedding_dim), got shape "
                        f"{embeddings.shape}"
                    )

                # Add positional encoding if enabled
                if self.pos_encoding is not None:
                    embeddings = (
                        embeddings + self.pos_encoding[:, : embeddings.size(1), :]
                    )

                # Ensure embeddings are contiguous for cuDNN compatibility
                embeddings = embeddings.contiguous()

                # Pass through main LSTM
                # Note: Using regular LSTM instead of packed sequences "
                "to avoid cuDNN issues"
                lstm_out, _ = self.lstm(embeddings)

        # Apply dropout if enabled
        if self.dropout is not None:
            lstm_out = self.dropout(lstm_out)

        # Take the last output for classification
        # For bidirectional LSTM, this captures information from both directions
        last_output = lstm_out[:, -1, :]

        # Classify
        logits = self.classifier(last_output)

        return logits

    def debug_info(self) -> dict:
        """Get debug information about the probe.

        Returns:
            Dictionary containing debug information
        """
        return {
            "probe_type": "lstm",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "lstm_hidden_size": self.lstm_hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dropout_rate": self.dropout_rate,
            "max_sequence_length": self.max_sequence_length,
            "use_positional_encoding": self.use_positional_encoding,
            "target_length": self.target_length,
            "has_layer_projections": hasattr(self, "layer_projections"),
        }
