"""Weighted Minimal attention probe for sequence-based evaluation."""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


class WeightedMinimalAttentionProbe(nn.Module):
    """A weighted minimal attention probe using single-layer MultiheadAttention.

    The probe extracts embeddings from specified layers of a **base_model** and
    processes them through a single attention mechanism. For multiple layer
    embeddings, it uses learned weights to combine them.

    Parameters
    ----------
    base_model
        Frozen backbone to extract embeddings from. Can be None when
        ``feature_mode`` is True.
    layers
        Layer names to extract embeddings from.
    num_classes
        Number of output classes.
    device
        Device string for module placement.
    feature_mode
        If True, the input to ``forward`` is already embeddings of shape
        ``(batch_size, time_steps, feature_dim)``.
    input_dim
        Feature dimension when in ``feature_mode``. If omitted and
        ``base_model`` is provided, a lightweight dummy pass is used to infer it.
    num_heads
        Number of attention heads for the MultiheadAttention layer.
    target_length
        Optional raw audio length in samples used for dummy inference when
        inferring dimensions.
    aggregation
        Aggregation method for embeddings. "none" extracts embeddings from
        multiple layers without aggregation, "mean" averages embeddings,
        "max" takes maximum values, "cls_token" uses CLS token.
    freeze_backbone
        Whether to freeze the backbone model and detach embeddings.

    Notes
    -----
    - If multiple layers are provided and the backbone returns a 4D tensor with
      a layer dimension, the probe averages across layers to obtain
      ``(B, T, D)``.
    - If embeddings are pooled (2D), an error is raised because sequence probes
      require a time dimension.
    - When aggregation="none", learned weights are used to combine embeddings.
    """

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
        num_heads: int = 1,
        target_length: Optional[int] = None,
        aggregation: str = "mean",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.layers = layers
        self.device = device
        self.feature_mode = feature_mode
        self.num_heads = num_heads
        self.target_length = target_length
        self.aggregation = aggregation
        self.freeze_backbone = freeze_backbone

        # Hooks are now registered in get_probe() after model mode is set

        inferred_dim: int = 0
        num_embeddings = 1  # Default for single embedding case

        if feature_mode:
            if input_dim is None:
                if base_model is None:
                    raise ValueError(
                        "input_dim must be provided when feature_mode=True and "
                        "base_model is None"
                    )
                # Infer using a dummy pass
                with torch.no_grad():
                    if self.target_length is not None:
                        computed_target_length = self.target_length
                    elif hasattr(base_model.audio_processor, "target_length_seconds"):
                        computed_target_length = (
                            base_model.audio_processor.target_length_seconds
                            * base_model.audio_processor.sr
                        )
                    elif hasattr(base_model.audio_processor, "target_length"):
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

                    assert isinstance(dummy_embeddings, torch.Tensor), (
                        "dummy_embeddings should be a tensor"
                    )
                    logger.info(
                        f"Input to MinimalAttention probe shape: "
                        f"{dummy_embeddings.shape}"
                    )
                    if dummy_embeddings.dim() == 3:
                        inferred_dim = dummy_embeddings.shape[-1]
                    else:
                        raise ValueError(
                            f"MinimalAttention probe expects 3D embeddings "
                            f"(batch_size, "
                            f"sequence_length, embedding_dim), got shape "
                            f"{dummy_embeddings.shape}"
                        )
            else:
                inferred_dim = input_dim

        else:
            # Extract embeddings from base_model to determine dimensions
            with torch.no_grad():
                if self.target_length is not None:
                    computed_target_length = self.target_length
                elif hasattr(base_model.audio_processor, "target_length_seconds"):
                    computed_target_length = (
                        base_model.audio_processor.target_length_seconds
                        * base_model.audio_processor.sr
                    )
                elif hasattr(base_model.audio_processor, "target_length"):
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

                    # For minimal attention probes, we expect 3D embeddings
                    # Check that all embeddings are 3D and have the same dimension
                    time_dims = []
                    embedding_dims = []
                    for i, emb in enumerate(dummy_embeddings):
                        logger.info(
                            f"Input to MinimalAttention probe shape: {emb.shape}"
                        )
                        if emb.dim() != 3:
                            raise ValueError(
                                f"MinimalAttention probe expects 3D embeddings "
                                f"(batch_size, "
                                f"sequence_length, embedding_dim), got shape "
                                f"{emb.shape} for layer {i}"
                            )
                        time_dims.append(emb.shape[1])
                        embedding_dims.append(emb.shape[-1])

                    # Assert all embeddings have the same dimension
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
                        f"WeightedMinimalAttentionProbe init "
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
                        f"WeightedMinimalAttentionProbe init "
                        f"(feature_mode=False, aggregation='none'): "
                        f"Only one layer extracted, treating as single embedding. "
                        f"dummy_embeddings type: {type(dummy_embeddings)}, "
                        f"shape: {dummy_embeddings.shape}"
                    )

                    if self.freeze_backbone:
                        dummy_embeddings = dummy_embeddings.detach()

                    # Handle single 4D tensor by reshaping to 3D (same as other probes)
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
                            f"MinimalAttention probe expects 2D, 3D or 4D embeddings, "
                            f"got shape {dummy_embeddings.shape}"
                        )
                else:
                    # Single tensor case
                    logger.debug(
                        f"MinimalAttention probe: Single tensor case - "
                        f"dummy_embeddings type: "
                        f"{type(dummy_embeddings)}, shape: {dummy_embeddings.shape}, "
                        f"aggregation: {self.aggregation}"
                    )

                    if self.freeze_backbone:
                        dummy_embeddings = dummy_embeddings.detach()
                    logger.info(
                        f"Input to MinimalAttention probe shape: "
                        f"{dummy_embeddings.shape}"
                    )

                    # Handle single 4D tensor by reshaping to 3D (same as other probes)
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
                        # For minimal attention probes, we expect 3D embeddings
                        # (batch_size, sequence_length, embedding_dim)
                        inferred_dim = dummy_embeddings.shape[-1]

                        logger.debug(
                            f"MinimalAttention probe: Using 3D tensor with "
                            f"inferred_dim: {inferred_dim}"
                        )

                    else:
                        logger.error(
                            f"MinimalAttention probe: Expected 2D, 3D or 4D embeddings "
                            f"but got shape "
                            f"{dummy_embeddings.shape}. This suggests the "
                            f"base_model.extract_embeddings did not respect "
                            f"aggregation='{self.aggregation}'"
                        )
                        raise ValueError(
                            f"MinimalAttention probe expects 2D, 3D or 4D embeddings, "
                            f"got shape {dummy_embeddings.shape}"
                        )

        # Create single attention mechanism for all cases
        self.attention = nn.MultiheadAttention(
            embed_dim=inferred_dim,
            num_heads=num_heads,
            dropout=0.0,  # Minimal attention doesn't use dropout
            batch_first=True,
        )

        # Final classification layer
        self.classifier = nn.Linear(inferred_dim, num_classes)

        self.to(device)

        # Log final probe parameters
        logger.info(
            f"WeightedMinimalAttentionProbe initialized with final parameters: "
            f"layers={self.layers}, feature_mode={self.feature_mode}, "
            f"aggregation={self.aggregation}, freeze_backbone={self.freeze_backbone}, "
            f"num_heads={self.num_heads}, target_length={self.target_length}, "
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
        """Forward pass through the weighted minimal attention probe.

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
                f"MinimalAttention probe forward: Calling "
                f"base_model.extract_embeddings "
                f"with aggregation='{self.aggregation}'"
            )
            embeddings = self.base_model.extract_embeddings(
                x,
                padding_mask=padding_mask,
                aggregation=self.aggregation,
                freeze_backbone=self.freeze_backbone,
            )
            logger.debug(
                f"MinimalAttention probe forward: Received embeddings type: "
                f"{type(embeddings)}, "
                f"shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'list'}"
            )

            if self.freeze_backbone:
                if isinstance(embeddings, list):
                    embeddings = [emb.detach() for emb in embeddings]
                else:
                    embeddings = embeddings.detach()

        # Handle the case where embeddings is a list (aggregation="none")
        if isinstance(embeddings, list):
            # Apply weighted sum to combine embeddings
            # Normalize weights to sum to 1
            weights = torch.softmax(self.layer_weights, dim=0)

            # Ensure all embeddings have the same shape for weighted sum
            # Take the minimum sequence length to ensure compatibility
            min_seq_len = min(emb.shape[1] for emb in embeddings)
            batch_size = embeddings[0].shape[0]
            embedding_dim = embeddings[0].shape[-1]

            # Truncate all embeddings to the same sequence length
            truncated_embeddings = []
            for emb in embeddings:
                truncated_emb = emb[:, :min_seq_len, :]
                truncated_embeddings.append(truncated_emb)

            # Apply weighted sum
            weighted_embeddings = torch.zeros(
                batch_size,
                min_seq_len,
                embedding_dim,
                device=embeddings[0].device,
                dtype=embeddings[0].dtype,
            )
            for _i, (emb, weight) in enumerate(
                zip(truncated_embeddings, weights, strict=False)
            ):
                weighted_embeddings += weight * emb

            embeddings = weighted_embeddings
            logger.debug(
                f"Applied weighted sum to {len(truncated_embeddings)} embeddings "
                f"with weights: {weights.detach().cpu().numpy()}"
            )

        # Handle single 4D tensor by reshaping to 3D (same as other probes)
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

        # Single tensor case - ensure it's 3D
        if embeddings.dim() != 3:
            raise ValueError(
                f"MinimalAttention probe expects 2D, 3D or 4D embeddings, "
                f"got shape {embeddings.shape}"
            )

        # Adjust padding mask to match embedding sequence length
        if padding_mask is not None:
            # If padding mask is provided, it should match the embedding sequence length
            if padding_mask.shape[1] != embeddings.shape[1]:
                # If the padding mask doesn't match, assume no padding for the
                # embeddings
                # This can happen when the original audio is much longer than the
                # processed embeddings
                padding_mask = None

        # Pass through attention
        attn_out, _ = self.attention(
            embeddings, embeddings, embeddings, key_padding_mask=padding_mask
        )

        # Global average pooling
        embeddings = attn_out.mean(dim=1)

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
            "probe_type": "weighted_minimal_attention",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "num_heads": self.num_heads,
            "target_length": self.target_length,
            "has_layer_weights": hasattr(self, "layer_weights"),
        }

        if hasattr(self, "layer_weights") and self.layer_weights is not None:
            info["layer_weights"] = self.layer_weights.detach().cpu().numpy().tolist()

        return info
