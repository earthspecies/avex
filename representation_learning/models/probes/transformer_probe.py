"""Transformer probe for representation learning evaluation."""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


class TransformerProbe(torch.nn.Module):
    """
    Transformer probe for complex sequence-based representation learning evaluation.

    The probe extracts embeddings from specified layers of a **base_model** and
    processes them through a full transformer architecture for sequence modeling.

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
        num_heads: Number of attention heads.
        attention_dim: Dimension of the attention mechanism.
        num_layers: Number of transformer layers.
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
        num_heads: int = 12,
        attention_dim: int = 768,
        num_layers: int = 4,
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
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.num_layers = num_layers
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
                            f"Transformer probe expects 3D embeddings (batch_size, "
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
                                f"Transformer probe expects 3D embeddings (batch_size, "
                                f"sequence_length, embedding_dim), got shape "
                                f"{emb.shape} for layer {i}"
                            )

                    # Create transformer projection heads for each layer
                    self.layer_projections = nn.ModuleList()

                    for i, emb in enumerate(dummy_embeddings):
                        # Find the optimal number of heads that the embedding "
                        "dimension can be divisible by"
                        d_model = emb.shape[-1]

                        # Find all divisors of d_model that are reasonable for "
                        "attention heads. We want heads that result in head_dim >= 64 "
                        "(PyTorch recommendation)"
                        valid_heads = []
                        for n in range(1, min(self.num_heads + 1, d_model + 1)):
                            if d_model % n == 0 and d_model // n >= 64:
                                valid_heads.append(n)

                        if not valid_heads:
                            # If no valid heads found, use 1 head and adjust "
                            "d_model if needed"
                            nhead = 1
                            if d_model < 64:
                                d_model = 64
                        else:
                            # Use the largest valid number of heads "
                            "(closest to requested)"
                            nhead = max(valid_heads)

                        # Ensure dim_feedforward is reasonable
                        dim_feedforward = max(self.attention_dim * 2, d_model * 2)

                        transformer_proj = nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=nhead,
                            dim_feedforward=dim_feedforward,
                            dropout=self.dropout_rate,
                            batch_first=True,
                            norm_first=True,
                        )
                        self.layer_projections.append(transformer_proj)

                        logger.debug(
                            f"Created transformer projection head {i}: "
                            f"d_model={d_model}, nhead={nhead}, "
                            f"dim_feedforward={dim_feedforward}, "
                            f"for embedding shape {emb.shape}"
                        )

                    # Input dimension is attention_dim * number of layers
                    inferred_dim = self.attention_dim * len(dummy_embeddings)

                    # Log the setup
                    logger.info(
                        f"TransformerProbe init (feature_mode=False, "
                        f"aggregation='none'): "
                        f"dummy_embeddings: list of {len(dummy_embeddings)} tensors, "
                        f"layers: {layers}, aggregation: {self.aggregation}, "
                        f"inferred_dim: {inferred_dim}"
                    )
                else:
                    # Single tensor case
                    if self.freeze_backbone:
                        dummy_embeddings = dummy_embeddings.detach()

                    if dummy_embeddings.dim() == 3:
                        # For sequence probes, we expect 3D embeddings
                        # (batch_size, sequence_length, embedding_dim)
                        inferred_dim = dummy_embeddings.shape[-1]
                    else:
                        raise ValueError(
                            f"Transformer probe expects 3D embeddings (batch_size, "
                            f"sequence_length, embedding_dim), got shape "
                            f"{dummy_embeddings.shape}"
                        )

                    # Create transformer encoder layers
                    # Find the optimal number of heads that the embedding "
                    "dimension can be divisible by"
                    d_model = inferred_dim

                    # Find all divisors of d_model that are reasonable for "
                    "attention heads. We want heads that result in head_dim >= 64 "
                    "(PyTorch recommendation)"
                    valid_heads = []
                    for n in range(1, min(num_heads + 1, d_model + 1)):
                        if d_model % n == 0 and d_model // n >= 64:
                            valid_heads.append(n)

                    if not valid_heads:
                        # If no valid heads found, use 1 head and adjust "
                        "d_model if needed"
                        nhead = 1
                        if d_model < 64:
                            d_model = 64
                    else:
                        # Use the largest valid number of heads "
                        "(closest to requested)"
                        nhead = max(valid_heads)

                    # Ensure dim_feedforward is reasonable
                    dim_feedforward = max(attention_dim * 2, d_model * 2)

                    logger.debug(
                        f"Creating main transformer: "
                        f"d_model={d_model}, nhead={nhead}, "
                        f"dim_feedforward={dim_feedforward}, "
                        f"original inferred_dim={inferred_dim}"
                    )

                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout_rate,
                        batch_first=True,
                        norm_first=True,  # Pre-norm for better training stability
                    )

                    self.transformer = nn.TransformerEncoder(
                        encoder_layer, num_layers=num_layers
                    )

        # Ensure d_model is always defined for consistent dimensions
        if "d_model" not in locals():
            d_model = inferred_dim

        # Classifier - use the adjusted d_model
        self.classifier = nn.Linear(d_model, num_classes)

        # Positional encoding - use the adjusted d_model
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, max_sequence_length or 1000, d_model)
            )
        else:
            self.pos_encoding = None

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Ensure transformer attribute exists for all cases
        if not hasattr(self, "transformer"):
            self.transformer = None

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
        """Forward pass through the transformer probe.

        Args:
            x: Input tensor. If feature_mode=True, this should be embeddings.
                If feature_mode=False, this should be raw audio.

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
        else:
            # Extract embeddings from base_model
            embeddings = self.base_model.extract_embeddings(
                x, padding_mask=padding_mask, aggregation=self.aggregation
            )

            if self.freeze_backbone:
                if isinstance(embeddings, list):
                    embeddings = [emb.detach() for emb in embeddings]
                else:
                    embeddings = embeddings.detach()

        # Handle the case where embeddings is a list (aggregation="none")
        if isinstance(embeddings, list):
            # Apply transformer projection heads to each layer's embeddings
            projected_embeddings = []
            for i, (emb, transformer_proj) in enumerate(
                zip(embeddings, self.layer_projections, strict=False)
            ):
                # Ensure embeddings are 3D
                if emb.dim() == 3:
                    # Apply transformer encoder layer to each layer's embeddings
                    # Note: TransformerEncoderLayer supports src_key_padding_mask
                    if padding_mask is not None:
                        # Ensure padding mask has correct shape for transformer
                        batch_size, seq_len, _ = emb.shape
                        if padding_mask.shape[1] != seq_len:
                            # Reshape padding mask to match the embedding "
                            "sequence length"
                            if padding_mask.shape[1] > seq_len:
                                # Truncate to match sequence length
                                mask = padding_mask[:, :seq_len]
                            else:
                                # Pad with zeros to match sequence length
                                mask = torch.zeros(
                                    batch_size,
                                    seq_len,
                                    device=padding_mask.device,
                                    dtype=padding_mask.dtype,
                                )
                                mask[:, : padding_mask.shape[1]] = padding_mask
                        else:
                            mask = padding_mask

                        projected_emb = transformer_proj(emb, src_key_padding_mask=mask)
                    else:
                        projected_emb = transformer_proj(emb)
                    # Global average pooling over sequence dimension
                    projected_emb = projected_emb.mean(dim=1)
                else:
                    raise ValueError(
                        f"Transformer probe expects 3D embeddings (batch_size, "
                        f"sequence_length, embedding_dim), got shape "
                        f"{emb.shape} for layer {i}"
                    )
                projected_embeddings.append(projected_emb)

            # Concatenate along the feature dimension
            embeddings = torch.cat(projected_embeddings, dim=-1)

            # When using projection heads, embeddings are already processed
            # No need to pass through main transformer - go directly to classifier
            x = embeddings
        else:
            # Single tensor case - ensure it's 3D
            if embeddings.dim() != 3:
                raise ValueError(
                    f"Transformer probe expects 3D embeddings (batch_size, "
                    f"sequence_length, embedding_dim), got shape "
                    f"{embeddings.shape}"
                )

            # Add positional encoding if enabled
            if self.pos_encoding is not None:
                embeddings = embeddings + self.pos_encoding[:, : embeddings.size(1), :]

            # Apply main transformer
            if padding_mask is not None:
                # Ensure padding mask has correct shape for transformer
                batch_size, seq_len, _ = embeddings.shape
                if padding_mask.shape[1] != seq_len:
                    # Reshape padding mask to match the embedding "
                    "sequence length"
                    if padding_mask.shape[1] > seq_len:
                        # Truncate to match sequence length
                        mask = padding_mask[:, :seq_len]
                    else:
                        # Pad with zeros to match sequence length
                        mask = torch.zeros(
                            batch_size,
                            seq_len,
                            device=padding_mask.device,
                            dtype=padding_mask.dtype,
                        )
                        mask[:, : padding_mask.shape[1]] = padding_mask
                else:
                    mask = padding_mask

                x = self.transformer(embeddings, src_key_padding_mask=mask)
            else:
                x = self.transformer(embeddings)

        # Global average pooling over sequence dimension
        if x.dim() == 3:
            x = x.mean(dim=1)

        # Apply dropout
        x = self.dropout(x)

        # Classify
        logits = self.classifier(x)

        return logits

    def debug_info(self) -> dict:
        """Get debug information about the probe.

        Returns:
            Dictionary containing debug information
        """
        return {
            "probe_type": "transformer",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "num_heads": self.num_heads,
            "attention_dim": self.attention_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "max_sequence_length": self.max_sequence_length,
            "use_positional_encoding": self.use_positional_encoding,
            "target_length": self.target_length,
            "has_layer_projections": hasattr(self, "layer_projections"),
        }
