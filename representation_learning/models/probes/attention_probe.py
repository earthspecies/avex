"""Attention probe for representation learning evaluation."""

from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase


class AttentionProbe(torch.nn.Module):
    """
    Attention probe for sequence-based representation learning evaluation.

    The probe extracts embeddings from specified layers of a **base_model** and
    processes them through attention mechanisms for sequence modeling.

    Args:
        base_model: Frozen backbone network to pull embeddings from. Can be None if
            feature_mode=True.
        layers: List of layer names to extract embeddings from.
        num_classes: Number of output classes.
        device: Device to run on.
        feature_mode: Whether to use the input directly as embeddings.
        input_dim: Input dimension when in feature mode. Required if base_model is None.
        num_heads: Number of attention heads.
        attention_dim: Dimension of the attention mechanism.
        num_layers: Number of attention layers.
        dropout_rate: Dropout rate for regularization.
        max_sequence_length: Maximum sequence length for processing.
        use_positional_encoding: Whether to add positional encoding.
    """

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
        num_heads: int = 8,
        attention_dim: int = 512,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        max_sequence_length: Optional[int] = None,
        use_positional_encoding: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_model = base_model
        self.layers = layers
        self.feature_mode = feature_mode
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.use_positional_encoding = use_positional_encoding

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
                    inferred_dim = base_model.extract_embeddings(
                        dummy, layers=layers
                    ).shape[1]
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
                inferred_dim = base_model.extract_embeddings(
                    dummy, layers=layers
                ).shape[1]

        # Project embeddings to attention dimension if needed
        if inferred_dim != attention_dim:
            self.input_projection = nn.Linear(inferred_dim, attention_dim)
        else:
            self.input_projection = None

        # Build attention layers
        self.attention_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=attention_dim,
                    nhead=num_heads,
                    dim_feedforward=attention_dim * 4,
                    dropout=dropout_rate,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        ).to(device)

        # Output layer
        self.classifier = nn.Linear(attention_dim, num_classes).to(device)

        # Positional encoding if requested
        if use_positional_encoding and max_sequence_length is not None:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, max_sequence_length, attention_dim)
            )
        else:
            self.pos_encoding = None

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the attention probe.

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

        # Project to attention dimension if needed
        if self.input_projection is not None:
            embeddings = self.input_projection(embeddings)

        # Add positional encoding if enabled
        if self.pos_encoding is not None:
            seq_len = embeddings.size(1)
            if seq_len <= self.pos_encoding.size(1):
                embeddings = embeddings + self.pos_encoding[:, :seq_len, :]
            else:
                # Truncate if sequence is longer than expected
                embeddings = embeddings[:, : self.pos_encoding.size(1), :]
                embeddings = embeddings + self.pos_encoding

        # Process through attention layers
        attention_output = embeddings
        for attention_layer in self.attention_layers:
            attention_output = attention_layer(attention_output)

        # Global average pooling over sequence dimension
        pooled_output = attention_output.mean(dim=1)

        return self.classifier(pooled_output)
