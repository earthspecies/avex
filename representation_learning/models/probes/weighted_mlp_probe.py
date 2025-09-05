"""Weighted MLP probe for representation learning evaluation."""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


class WeightedMLPProbe(torch.nn.Module):
    """
    Weighted MLP probe for representation learning evaluation.

    The probe extracts embeddings from specified layers of a **base_model** and
    processes them through a single MLP classifier. For multiple layer
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
        hidden_dims: List of hidden layer dimensions.
        dropout_rate: Dropout rate for regularization.
        activation: Activation function to use.
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
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = "relu",
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

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
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
                        dummy, aggregation=self.aggregation
                    ).detach()

                    # feature_mode=True assumes that the embeddings are not lists
                    assert isinstance(dummy_embeddings, torch.Tensor), (
                        "dummy_embeddings should be a tensor"
                    )
                    logger.info(f"Input to MLP probe shape: {dummy_embeddings.shape}")
                    if dummy_embeddings.dim() == 2:
                        # For MLP probes, we expect 2D embeddings
                        # (batch_size, embedding_dim)
                        inferred_dim = dummy_embeddings.shape[-1]
                    else:
                        raise ValueError(
                            f"MLP probe expects 2D embeddings (batch_size, "
                            f"embedding_dim), got shape {dummy_embeddings.shape}"
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
                    dummy, aggregation=self.aggregation
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
                            # 4D: (batch, channels, height, width) -> flatten to 2D
                            # For 2D probes, we flatten spatial dimensions
                            feature_dim = (
                                emb.shape[1] * emb.shape[2] * emb.shape[3]
                            )  # channels * height * width
                        elif emb.dim() == 3:
                            # 3D: (batch, seq_len, features) -> flatten to 2D
                            # For 2D probes, we flatten sequence dimension
                            feature_dim = (
                                emb.shape[1] * emb.shape[2]
                            )  # seq_len * features
                        elif emb.dim() == 2:
                            # 2D: (batch, features) -> already correct
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
                                "feature_dim": feature_dim,
                                "dim": emb.dim(),
                            }
                        )

                    # Find the most common feature dimension
                    from collections import Counter

                    feature_dims = [info["feature_dim"] for info in embedding_info]
                    dimension_counts = Counter(feature_dims)
                    most_common_dim = dimension_counts.most_common(1)[0]
                    most_common_feature_dim = most_common_dim[0]
                    most_common_count = most_common_dim[1]

                    # Check if more than 50% of embeddings have the same
                    # feature dimension
                    if most_common_count > len(embedding_info) / 2:
                        # Use the most common feature dimension as target
                        max_feature_dim = most_common_feature_dim
                        logger.info(
                            f"Using most common feature dimension as target: "
                            f"{max_feature_dim} (appears in {most_common_count}/"
                            f"{len(embedding_info)} embeddings)"
                        )
                    else:
                        # Use maximum feature dimension as target
                        max_feature_dim = max(
                            info["feature_dim"] for info in embedding_info
                        )
                        logger.info(
                            f"Using maximum feature dimension as target: "
                            f"{max_feature_dim}"
                        )

                    # Determine which embeddings need projection
                    needs_projection = False
                    for info in embedding_info:
                        if info["feature_dim"] != max_feature_dim:
                            needs_projection = True
                            break

                    logger.info(f"Target feature dimension: {max_feature_dim}")
                    logger.info(f"Needs projection: {needs_projection}")

                    if needs_projection:
                        # Create individual linear projectors for each embedding
                        # For 2D probes, we flatten all embeddings and apply
                        # linear projection
                        self.embedding_projectors = nn.ModuleList()

                        for info in embedding_info:
                            i = info["index"]
                            emb = info["embedding"]
                            feature_dim = info["feature_dim"]

                            # Create a simple linear projector for 2D output
                            projector = nn.Linear(feature_dim, max_feature_dim)
                            self.embedding_projectors.append(projector)

                            # Log whether this embedding needs dimension changes
                            if feature_dim == max_feature_dim:
                                logger.info(
                                    f"Created linear projector for embedding {i} "
                                    f"(shape: {emb.shape}) - already matches target "
                                    f"dimensions"
                                )
                            else:
                                logger.info(
                                    f"Created linear projector for embedding {i} "
                                    f"(shape: {emb.shape}) -> target: {max_feature_dim}"
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
                            # Flatten embedding to 2D first
                            if emb.dim() == 4:
                                # 4D: (batch, channels, height, width) ->
                                # (batch, channels*height*width)
                                batch_size = emb.shape[0]
                                emb_2d = emb.reshape(batch_size, -1)
                            elif emb.dim() == 3:
                                # 3D: (batch, seq_len, features) ->
                                # (batch, seq_len*features)
                                batch_size = emb.shape[0]
                                emb_2d = emb.reshape(batch_size, -1)
                            elif emb.dim() == 2:
                                # 2D: already correct
                                emb_2d = emb
                            else:
                                raise ValueError(
                                    f"Unsupported embedding dimension: {emb.dim()}"
                                )

                            # Apply linear projection
                            projected_emb = projector(emb_2d)
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
                        # All embeddings are already 2D with same dimensions
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
                        f"WeightedMLPProbe init "
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
                        f"WeightedMLPProbe init "
                        f"(feature_mode=False, aggregation='none'): "
                        f"Only one layer extracted, treating as single embedding. "
                        f"dummy_embeddings type: {type(dummy_embeddings)}, "
                        f"shape: {dummy_embeddings.shape}"
                    )

                    if self.freeze_backbone:
                        dummy_embeddings = dummy_embeddings.detach()

                    if dummy_embeddings.dim() == 2:
                        inferred_dim = dummy_embeddings.shape[-1]
                        # No layer_weights needed for single embedding
                        self.layer_weights = None
                        logger.info(
                            f"Single embedding case: inferred_dim={inferred_dim}, "
                            f"no layer_weights needed"
                        )
                    else:
                        raise ValueError(
                            f"MLP probe expects 2D embeddings (batch_size, "
                            f"embedding_dim), got shape {dummy_embeddings.shape}"
                        )
                else:
                    # Single tensor case
                    logger.debug(
                        f"MLP probe: Single tensor case - dummy_embeddings type: "
                        f"{type(dummy_embeddings)}, shape: {dummy_embeddings.shape}, "
                        f"aggregation: {self.aggregation}"
                    )

                    if self.freeze_backbone:
                        dummy_embeddings = dummy_embeddings.detach()
                    logger.info(f"Input to MLP probe shape: {dummy_embeddings.shape}")

                    if dummy_embeddings.dim() == 2:
                        # For MLP probes, we expect 2D embeddings
                        # (batch_size, embedding_dim)
                        inferred_dim = dummy_embeddings.shape[-1]

                        logger.debug(
                            f"MLP probe: Using 2D tensor with "
                            f"inferred_dim: {inferred_dim}"
                        )

                    else:
                        logger.error(
                            f"MLP probe: Expected 2D embeddings but got shape "
                            f"{dummy_embeddings.shape}. This suggests the "
                            f"base_model.extract_embeddings did not respect "
                            f"aggregation='{self.aggregation}'"
                        )
                        raise ValueError(
                            f"MLP probe expects 2D embeddings (batch_size, "
                            f"embedding_dim), got shape {dummy_embeddings.shape}"
                        )

        # Create single MLP for all cases
        layers_list = []
        current_dim = inferred_dim

        for hidden_dim in self.hidden_dims:
            layers_list.append(nn.Linear(current_dim, hidden_dim))

            # Add activation
            if self.activation == "relu":
                layers_list.append(nn.ReLU())
            elif self.activation == "gelu":
                layers_list.append(nn.GELU())
            elif self.activation == "tanh":
                layers_list.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")

            # Add dropout
            if self.dropout_rate > 0:
                layers_list.append(nn.Dropout(self.dropout_rate))

            current_dim = hidden_dim

        # Final classification layer
        layers_list.append(nn.Linear(current_dim, num_classes))

        self.mlp = nn.Sequential(*layers_list)

        self.to(device)

        # Log final probe parameters
        logger.info(
            f"WeightedMLPProbe initialized with final parameters: "
            f"layers={self.layers}, feature_mode={self.feature_mode}, "
            f"aggregation={self.aggregation}, freeze_backbone={self.freeze_backbone}, "
            f"hidden_dims={self.hidden_dims}, dropout_rate={self.dropout_rate}, "
            f"activation={self.activation}, target_length={self.target_length}, "
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
        """Forward pass through the weighted MLP probe.

        Args:
            x: Input tensor. If feature_mode=True, this should be embeddings.
                If feature_mode=False, this should be raw audio.
            padding_mask: Optional padding mask for the input. Currently not used
                by MLP probes but included for compatibility with training loops.

        Returns:
            Classification logits

        Raises:
            ValueError: If embeddings are not 2D (batch_size, embedding_dim)
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
                f"MLP probe forward: Calling base_model.extract_embeddings "
                f"with aggregation='{self.aggregation}'"
            )
            embeddings = self.base_model.extract_embeddings(
                x,
                padding_mask=padding_mask,
                aggregation=self.aggregation,
                freeze_backbone=self.freeze_backbone,
            )
            logger.debug(
                f"MLP probe forward: Received embeddings type: {type(embeddings)}, "
                f"shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'list'}"
            )

            if self.freeze_backbone:
                if isinstance(embeddings, list):
                    embeddings = [emb.detach() for emb in embeddings]
                else:
                    embeddings = embeddings.detach()

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
                    # Flatten embedding to 2D first
                    if emb.dim() == 4:
                        # 4D: (batch, channels, height, width) ->
                        # (batch, channels*height*width)
                        batch_size = emb.shape[0]
                        emb_2d = emb.reshape(batch_size, -1)
                    elif emb.dim() == 3:
                        # 3D: (batch, seq_len, features) -> (batch, seq_len*features)
                        batch_size = emb.shape[0]
                        emb_2d = emb.reshape(batch_size, -1)
                    elif emb.dim() == 2:
                        # 2D: already correct
                        emb_2d = emb
                    else:
                        raise ValueError(
                            f"Unsupported embedding dimension: {emb.dim()}"
                        )

                    # Apply linear projection
                    projected_emb = projector(emb_2d)
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
            embedding_dim = embeddings[0].shape[-1]

            # Verify all embeddings have the same shape
            for i, emb in enumerate(embeddings):
                if emb.shape != (batch_size, embedding_dim):
                    raise ValueError(
                        f"Embedding {i} has shape {emb.shape}, expected "
                        f"{(batch_size, embedding_dim)}. "
                        f"Projectors should ensure consistent shapes."
                    )

            # Apply weighted sum
            weighted_embeddings = torch.zeros(
                batch_size,
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

        # Single tensor case - ensure it's 2D
        if embeddings.dim() != 2:
            raise ValueError(
                f"MLP probe expects 2D embeddings (batch_size, "
                f"embedding_dim), got shape {embeddings.shape}"
            )

        # Pass through MLP
        logits = self.mlp(embeddings)

        return logits

    def print_learned_weights(self) -> None:
        """Print the learned weights for layer embeddings.
        This function prints the raw weights and normalized weights (softmax)
        for each layer when using list embeddings with aggregation='none'.
        """
        if not hasattr(self, "layer_weights") or self.layer_weights is None:
            print(
                "No learned weights found. This probe does not use weighted sum of "
                "embeddings."
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
            "probe_type": "weighted_mlp",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "target_length": self.target_length,
            "has_layer_weights": hasattr(self, "layer_weights"),
            "has_embedding_projectors": (
                hasattr(self, "embedding_projectors")
                and getattr(self, "embedding_projectors", None) is not None
            ),
        }

        if hasattr(self, "layer_weights") and self.layer_weights is not None:
            info["layer_weights"] = self.layer_weights.detach().cpu().numpy().tolist()

        return info
