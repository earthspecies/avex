"""Weighted Transformer probe for representation learning evaluation."""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.embedding_projectors import (
    Conv4DProjector,
    EmbeddingProjector,
    Sequence3DProjector,
)

logger = logging.getLogger(__name__)


class WeightedTransformerProbe(torch.nn.Module):
    """
    Weighted Transformer probe for complex sequence-based representation
    learning evaluation.

    The probe extracts embeddings from specified layers of a **base_model** and
    processes them through a single transformer architecture. For multiple layer
    embeddings, it uses learned weights to combine them.

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

        # Hooks are now registered in get_probe() after model mode is set

        # Initialize variables
        inferred_dim = None
        num_embeddings = 1  # Default for single embedding case

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

                    dummy = torch.randn(1, int(computed_target_length), device=device)
                    dummy_embeddings = base_model.extract_embeddings(
                        dummy, aggregation=self.aggregation, freeze_backbone=True
                    ).detach()

                    # feature_mode=True assumes that the embeddings are not lists
                    assert isinstance(dummy_embeddings, torch.Tensor), (
                        "dummy_embeddings should be a tensor"
                    )
                    logger.info(
                        f"Input to Transformer probe shape: {dummy_embeddings.shape}"
                    )
                    if dummy_embeddings.dim() == 3:
                        # For transformer probes, we expect 3D embeddings
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

                dummy = torch.randn(1, int(computed_target_length), device=device)
                dummy_embeddings = base_model.extract_embeddings(
                    dummy, aggregation=self.aggregation, freeze_backbone=True
                )

                # Handle the case where dummy_embeddings is a list (aggregation="none")
                if isinstance(dummy_embeddings, list):
                    # Detach each embedding in the list
                    dummy_embeddings = [emb.detach() for emb in dummy_embeddings]
                    num_embeddings = len(dummy_embeddings)

                    # Analyze embeddings to determine which need projection and
                    # target dimensions
                    embedding_info = []

                    for i, emb in enumerate(dummy_embeddings):
                        logger.info(f"Original embedding {i} shape: {emb.shape}")

                        if emb.dim() == 4:
                            # 4D: (batch, channels, height, width) -> width is sequence,
                            # channels*height is features
                            seq_len = emb.shape[3]  # width
                            feature_dim = (
                                emb.shape[1] * emb.shape[2]
                            )  # channels * height
                        elif emb.dim() == 3:
                            # 3D: (batch, seq_len, features) or
                            # (batch, features, seq_len)
                            if emb.shape[2] > emb.shape[1] * 2:
                                # (batch, seq_len, features)
                                seq_len = emb.shape[1]
                                feature_dim = emb.shape[2]
                            else:
                                # (batch, features, seq_len) - needs transposition
                                seq_len = emb.shape[2]
                                feature_dim = emb.shape[1]
                        elif emb.dim() == 2:
                            # 2D: (batch, features) -> seq_len=1, features=features
                            seq_len = 1
                            feature_dim = emb.shape[1]
                        else:
                            raise ValueError(
                                f"Unsupported embedding dimension: {emb.dim()}. "
                                f"Expected 2D, 3D, or 4D for embedding {i}"
                            )

                        embedding_info.append(
                            {
                                "index": i,
                                "embedding": emb,
                                "seq_len": seq_len,
                                "feature_dim": feature_dim,
                                "dim": emb.dim(),
                            }
                        )

                    # Find the most common dimensions
                    from collections import Counter

                    dimension_counts = Counter(
                        (info["seq_len"], info["feature_dim"])
                        for info in embedding_info
                    )
                    most_common_dims = dimension_counts.most_common(1)[0]
                    most_common_seq_len, most_common_feature_dim = most_common_dims[0]
                    most_common_count = most_common_dims[1]

                    # Check if more than 50% of embeddings have the same dimensions
                    if most_common_count > len(embedding_info) / 2:
                        # Use the most common dimensions as target
                        max_seq_len = most_common_seq_len
                        max_feature_dim = most_common_feature_dim
                        logger.info(
                            f"Using most common dimensions as target: "
                            f"seq_len={max_seq_len}, feature_dim={max_feature_dim} "
                            f"(appears in {most_common_count}/"
                            f"{len(embedding_info)} embeddings)"
                        )
                    else:
                        # Use maximum dimensions as target
                        max_seq_len = max(info["seq_len"] for info in embedding_info)
                        max_feature_dim = max(
                            info["feature_dim"] for info in embedding_info
                        )
                        logger.info(
                            f"Using maximum dimensions as target: "
                            f"seq_len={max_seq_len}, feature_dim={max_feature_dim}"
                        )

                    # Determine which embeddings need projection
                    needs_projection = False
                    for info in embedding_info:
                        if (
                            info["seq_len"] != max_seq_len
                            or info["feature_dim"] != max_feature_dim
                        ):
                            needs_projection = True
                            break

                    logger.info(
                        f"Target dimensions: seq_len={max_seq_len}, "
                        f"feature_dim={max_feature_dim}"
                    )
                    logger.info(f"Needs projection: {needs_projection}")

                    if needs_projection:
                        # Create individual projectors for each embedding
                        # Each projector is tailored to its specific embedding shape
                        self.embedding_projectors = nn.ModuleList()

                        for info in embedding_info:
                            i = info["index"]
                            emb = info["embedding"]
                            seq_len = info["seq_len"]
                            feature_dim = info["feature_dim"]

                            # Always create projector with target dimensions
                            # The projector will detect if no changes are needed and
                            # skip projection
                            if emb.dim() == 4:
                                # 4D: Use Conv4DProjector
                                projector = Conv4DProjector(
                                    target_feature_dim=max_feature_dim,
                                    target_sequence_length=max_seq_len,
                                )
                            elif emb.dim() == 3:
                                # 3D: Use Sequence3DProjector
                                projector = Sequence3DProjector(
                                    target_feature_dim=max_feature_dim,
                                    target_sequence_length=max_seq_len,
                                )
                            elif emb.dim() == 2:
                                # 2D: Use EmbeddingProjector with force_sequence_format
                                projector = EmbeddingProjector(
                                    target_feature_dim=max_feature_dim,
                                    target_sequence_length=max_seq_len,
                                    force_sequence_format=True,
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported embedding dimension: {emb.dim()}"
                                )

                            self.embedding_projectors.append(projector)

                            # Log whether this embedding needs dimension changes
                            if (
                                seq_len == max_seq_len
                                and feature_dim == max_feature_dim
                            ):
                                logger.info(
                                    f"Created projector for embedding {i} "
                                    f"(shape: {emb.shape}) - already matches target "
                                    f"dimensions, will only do format conversion"
                                )
                            else:
                                logger.info(
                                    f"Created projector for embedding {i} "
                                    f"(shape: {emb.shape}) -> target: "
                                    f"({max_seq_len}, {max_feature_dim})"
                                )

                        # Apply projectors to determine final dimensions
                        projected_embeddings = []
                        for i, (emb, projector) in enumerate(
                            zip(
                                dummy_embeddings,
                                self.embedding_projectors,
                                strict=False,
                            )
                        ):
                            # Apply projection (all projectors are now not None)
                            projected_emb = projector(emb)
                            logger.info(
                                f"Projected embedding {i}: {emb.shape} -> "
                                f"{projected_emb.shape}"
                            )
                            projected_embeddings.append(projected_emb)

                        # All projected embeddings should have the same
                        # feature dimension
                        embedding_dims = [emb.shape[-1] for emb in projected_embeddings]
                        if len(set(embedding_dims)) > 1:
                            raise ValueError(
                                f"All projected embeddings must have the same "
                                f"dimension for weighted sum. "
                                f"Got dimensions: {embedding_dims}"
                            )

                        inferred_dim = embedding_dims[0]
                        logger.info(
                            f"Final projected feature dimension: {inferred_dim}"
                        )
                    else:
                        # All embeddings are already 3D with same dimensions
                        embedding_dims = [emb.shape[-1] for emb in dummy_embeddings]
                        if len(set(embedding_dims)) > 1:
                            raise ValueError(
                                f"All embeddings must have the same dimension "
                                f"for weighted sum. "
                                f"Got dimensions: {embedding_dims}"
                            )
                        inferred_dim = embedding_dims[0]

                    # Create learned weights for weighted sum
                    self.layer_weights = nn.Parameter(torch.zeros(num_embeddings))

                    # Log the setup
                    logger.info(
                        f"WeightedTransformerProbe init "
                        f"(feature_mode=False, aggregation='none'): "
                        f"dummy_embeddings: list of {len(dummy_embeddings)} tensors, "
                        f"layers: {layers}, aggregation: {self.aggregation}, "
                        f"inferred_dim: {inferred_dim}, "
                        f"num_embeddings: {num_embeddings}"
                    )
                elif self.aggregation == "none":
                    # Handle case where aggregation="none" but only one layer
                    # was extracted
                    # This happens when the base model returns a single tensor
                    # instead of a list
                    logger.info(
                        f"WeightedTransformerProbe init "
                        f"(feature_mode=False, aggregation='none'): "
                        f"Only one layer extracted, treating as single embedding. "
                        f"dummy_embeddings type: {type(dummy_embeddings)}, "
                        f"shape: {dummy_embeddings.shape}"
                    )

                    if self.freeze_backbone:
                        dummy_embeddings = dummy_embeddings.detach()

                    # Handle single 4D tensor by reshaping to 3D (same as
                    # TransformerProbe)
                    if dummy_embeddings.dim() == 4:
                        # Handle 4D embeddings (B, C, H, W)
                        batch_size, channels, height, width = dummy_embeddings.shape
                        logger.debug(
                            f"Processing 4D embedding: {dummy_embeddings.shape}"
                        )
                        # Reshape to (B, W, C*H) - treat width as sequence length,
                        # C*H as features. Ensure proper memory layout for cuDNN
                        dummy_embeddings = (
                            dummy_embeddings.permute(0, 3, 1, 2)
                            .contiguous()
                            .view(batch_size, width, channels * height)
                        )
                        logger.info(
                            f"Reshaped 4D embedding to: {dummy_embeddings.shape}"
                        )

                    if dummy_embeddings.dim() == 2:
                        # Handle 2D embeddings by adding sequence dimension
                        dummy_embeddings = dummy_embeddings.unsqueeze(2)
                        logger.info(
                            f"Reshaped 2D embedding to: {dummy_embeddings.shape}"
                        )

                    if dummy_embeddings.dim() == 3:
                        inferred_dim = dummy_embeddings.shape[-1]
                        # No layer_weights needed for single embedding
                        self.layer_weights = None
                        logger.info(
                            f"Single embedding case: inferred_dim={inferred_dim}, "
                            f"no layer_weights needed"
                        )
                    else:
                        raise ValueError(
                            f"Transformer probe expects 2D, 3D or 4D embeddings, "
                            f"got shape {dummy_embeddings.shape}"
                        )
                else:
                    # Single tensor case
                    logger.debug(
                        f"Transformer probe: Single tensor case - "
                        f"dummy_embeddings type: "
                        f"{type(dummy_embeddings)}, shape: {dummy_embeddings.shape}, "
                        f"aggregation: {self.aggregation}"
                    )

                    if self.freeze_backbone:
                        dummy_embeddings = dummy_embeddings.detach()
                    logger.info(
                        f"Input to Transformer probe shape: {dummy_embeddings.shape}"
                    )

                    if dummy_embeddings.dim() == 3:
                        # For transformer probes, we expect 3D embeddings
                        # (batch_size, sequence_length, embedding_dim)
                        inferred_dim = dummy_embeddings.shape[-1]

                        logger.debug(
                            f"Transformer probe: Using 3D tensor with "
                            f"inferred_dim: {inferred_dim}"
                        )

                    else:
                        logger.error(
                            f"Transformer probe: Expected 3D embeddings but got shape "
                            f"{dummy_embeddings.shape}. This suggests the "
                            f"base_model.extract_embeddings did not respect "
                            f"aggregation='{self.aggregation}'"
                        )
                        raise ValueError(
                            f"Transformer probe expects 3D embeddings (batch_size, "
                            f"sequence_length, embedding_dim), got shape "
                            f"{dummy_embeddings.shape}"
                        )

        # Create single transformer for all cases
        # Ensure embed_dim is divisible by num_heads
        if inferred_dim % num_heads != 0:
            # Adjust num_heads to be compatible with inferred_dim
            adjusted_num_heads = min(num_heads, inferred_dim)
            while inferred_dim % adjusted_num_heads != 0 and adjusted_num_heads > 1:
                adjusted_num_heads -= 1
            logger.warning(
                f"Adjusted num_heads from {num_heads} to {adjusted_num_heads} "
                f"to be compatible with embed_dim {inferred_dim}"
            )
            num_heads = adjusted_num_heads

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=inferred_dim,
                nhead=num_heads,
                dim_feedforward=attention_dim,
                dropout=dropout_rate,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Optional positional encoding
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, max_sequence_length or 1000, inferred_dim)
            )
        else:
            self.pos_encoding = None

        # Optional dropout after transformer
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        # Final classification layer
        self.classifier = nn.Linear(inferred_dim, num_classes)

        self.to(device)

        # Log final probe parameters
        logger.info(
            f"WeightedTransformerProbe initialized with final parameters: "
            f"layers={self.layers}, feature_mode={self.feature_mode}, "
            f"aggregation={self.aggregation}, freeze_backbone={self.freeze_backbone}, "
            f"num_heads={self.num_heads}, attention_dim={self.attention_dim}, "
            f"num_layers={self.num_layers}, dropout_rate={self.dropout_rate}, "
            f"max_sequence_length={self.max_sequence_length}, "
            f"use_positional_encoding={self.use_positional_encoding}, "
            f"target_length={self.target_length}, "
            f"inferred_dim={inferred_dim}, num_classes={num_classes}, "
            f"num_embeddings={num_embeddings}, "
            f"has_layer_weights={hasattr(self, 'layer_weights')}"
        )

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
        """Forward pass through the weighted transformer probe.

        Args:
            x: Input tensor. If feature_mode=True, this should be embeddings.
                If feature_mode=False, this should be raw audio.
            padding_mask: Optional padding mask for the input.

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
            logger.debug(
                f"Transformer probe forward: Calling base_model.extract_embeddings "
                f"with aggregation='{self.aggregation}'"
            )
            embeddings = self.base_model.extract_embeddings(
                x,
                padding_mask=padding_mask,
                aggregation=self.aggregation,
                freeze_backbone=self.freeze_backbone,
            )
            logger.debug(
                f"Transformer probe forward: Received embeddings type: "
                f"{type(embeddings)}, "
                f"shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'list'}"
            )

            if self.freeze_backbone:
                if isinstance(embeddings, list):
                    embeddings = [emb.detach() for emb in embeddings]
                else:
                    embeddings = embeddings.detach()

        # Handle single 4D tensor by reshaping to 3D (same as TransformerProbe)
        if not isinstance(embeddings, list) and embeddings.dim() == 4:
            # Handle 4D embeddings (B, C, H, W)
            batch_size, channels, height, width = embeddings.shape
            logger.debug(f"Processing 4D embedding: {embeddings.shape}")
            # Reshape to (B, W, C*H) - treat width as sequence length,
            # C*H as features. Ensure proper memory layout for cuDNN
            embeddings = (
                embeddings.permute(0, 3, 1, 2)
                .contiguous()
                .view(batch_size, width, channels * height)
            )
            logger.debug(f"Reshaped 4D embedding to: {embeddings.shape}")
        elif not isinstance(embeddings, list) and embeddings.dim() == 2:
            # Handle 2D embeddings by adding sequence dimension
            embeddings = embeddings.unsqueeze(2)
            logger.debug(f"Reshaped 2D embedding to: {embeddings.shape}")

        # Handle the case where embeddings is a list (aggregation="none")
        if isinstance(embeddings, list):
            # Apply individual embedding projectors to each embedding if enabled
            if (
                hasattr(self, "embedding_projectors")
                and getattr(self, "embedding_projectors", None) is not None
            ):
                projected_embeddings = []
                for i, (emb, projector) in enumerate(
                    zip(embeddings, self.embedding_projectors, strict=False)
                ):
                    # Apply projection (all projectors are now not None)
                    logger.debug(f"Projecting embedding {i}: {emb.shape}")
                    projected_emb = projector(emb)
                    logger.debug(
                        f"Projected embedding {i}: {emb.shape} -> {projected_emb.shape}"
                    )
                    projected_embeddings.append(projected_emb)
                embeddings = projected_embeddings

            # Apply weighted sum to combine embeddings
            # Normalize weights to sum to 1
            weights = torch.softmax(self.layer_weights, dim=0)

            # All embeddings should now have the same shape after projection
            batch_size = embeddings[0].shape[0]
            seq_len = embeddings[0].shape[1]
            embedding_dim = embeddings[0].shape[-1]

            # Verify all embeddings have the same shape
            for i, emb in enumerate(embeddings):
                if emb.shape != (batch_size, seq_len, embedding_dim):
                    raise ValueError(
                        f"Embedding {i} has shape {emb.shape}, expected "
                        f"{(batch_size, seq_len, embedding_dim)}. "
                        f"Projectors should ensure consistent shapes."
                    )

            # Apply weighted sum
            weighted_embeddings = torch.zeros(
                batch_size,
                seq_len,
                embedding_dim,
                device=embeddings[0].device,
                dtype=embeddings[0].dtype,
            )
            for _i, (emb, weight) in enumerate(zip(embeddings, weights, strict=False)):
                weighted_embeddings += weight * emb

            embeddings = weighted_embeddings
            logger.debug(
                f"Applied weighted sum to {len(embeddings)} embeddings "
                f"with weights: {weights.detach().cpu().numpy()}"
            )

        # Single tensor case - ensure it's 3D
        if embeddings.dim() != 3:
            raise ValueError(
                f"Transformer probe expects 3D embeddings (batch_size, "
                f"sequence_length, embedding_dim), got shape {embeddings.shape}"
            )

        # Add positional encoding if enabled
        if self.pos_encoding is not None:
            embeddings = embeddings + self.pos_encoding[:, : embeddings.size(1), :]

        # Adjust padding mask to match embedding sequence length
        if padding_mask is not None:
            # If padding mask is provided, it should match the embedding sequence length
            if padding_mask.shape[1] != embeddings.shape[1]:
                # If the padding mask doesn't match, assume no padding for the
                # embeddings
                # This can happen when the original audio is much longer than the
                # processed embeddings
                padding_mask = None

        # Pass through transformer
        transformer_out = self.transformer(
            embeddings, src_key_padding_mask=padding_mask
        )

        # Global average pooling
        embeddings = transformer_out.mean(dim=1)

        # Apply dropout if enabled
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)

        # Classify
        logits = self.classifier(embeddings)

        return logits

    def print_learned_weights(self) -> None:
        """Print the learned weights for layer embeddings.

        This function prints the raw weights and normalized weights (softmax)
        for each layer when using list embeddings with aggregation='none'.
        """
        if not hasattr(self, "layer_weights") or self.layer_weights is None:
            print(
                "No learned weights found. This probe does not use weighted sum "
                "of embeddings."
            )
            return

        raw_weights = self.layer_weights.detach().cpu().numpy()
        normalized_weights = (
            torch.softmax(self.layer_weights, dim=0).detach().cpu().numpy()
        )

        print("Learned Layer Weights:")
        print("=" * 50)
        print(f"{'Layer':<15} {'Raw Weight':<12} {'Normalized':<12} {'Percentage':<12}")
        print("-" * 50)

        for i, (raw, norm) in enumerate(
            zip(raw_weights, normalized_weights, strict=False)
        ):
            layer_name = self.layers[i] if i < len(self.layers) else f"Layer_{i}"
            percentage = norm * 100
            print(f"{layer_name:<15} {raw:<12.4f} {norm:<12.4f} {percentage:<12.2f}%")

        print("-" * 50)
        print(f"Sum of normalized weights: {normalized_weights.sum():.6f}")
        print(f"Number of layers: {len(raw_weights)}")

    def debug_info(self) -> dict:
        """Get debug information about the probe.

        Returns:
            Dictionary containing debug information
        """
        info = {
            "probe_type": "weighted_transformer",
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
            "has_layer_weights": hasattr(self, "layer_weights"),
            "has_embedding_projectors": hasattr(self, "embedding_projectors")
            and getattr(self, "embedding_projectors", None) is not None,
        }

        if hasattr(self, "layer_weights") and self.layer_weights is not None:
            info["layer_weights"] = self.layer_weights.detach().cpu().numpy().tolist()

        return info
