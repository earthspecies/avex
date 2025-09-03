"""MLP probe for flexible probing system."""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


class MLPProbe(torch.nn.Module):
    """MLP probe for classification tasks.

    Args:
        base_model: Backbone network to pull embeddings from. Can be None if
            feature_mode=True.
        layers: List of layer names to extract embeddings from.
        num_classes: Number of output classes.
        device: Device to run on.
        feature_mode: Whether to use the input directly as embeddings.
        input_dim: Input dimension when in feature_mode=True.
            Required if base_model is None.
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
        projection_dim: Target dimension for each layer projection when
            aggregation='none'. If None, uses the minimum embedding dimension
            across all layers.
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
        projection_dim: Optional[int] = None,
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
        self.projection_dim = projection_dim

        # Register hooks for the specified layers if base_model is provided
        if self.base_model is not None and not self.feature_mode:
            self.base_model.register_hooks_for_layers(self.layers)

        # Initialize variables
        inferred_dim = None
        classifier_input_dim = None

        # Determine classifier input dimension
        if self.feature_mode:
            # Embeddings will be fed directly – base_model may be None.
            if input_dim is not None:
                inferred_dim = input_dim
                classifier_input_dim = inferred_dim
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
                    logger.info(f"Input to MLP probe shape: {dummy_embeddings.shape}")
                    if dummy_embeddings.dim() == 4:
                        # Collapse 4D to 2D by taking mean across dimensions 1 and 2
                        dummy_embeddings = dummy_embeddings.mean(dim=(1, 2))
                        logger.info(
                            f"Collapsed 4D embedding to: {dummy_embeddings.shape}"
                        )
                    elif dummy_embeddings.dim() == 3:
                        # Collapse 3D to 2D by taking mean across sequence dimension
                        dummy_embeddings = dummy_embeddings.mean(dim=1)
                        logger.info(
                            f"Collapsed 3D embedding to: {dummy_embeddings.shape}"
                        )
                    elif dummy_embeddings.dim() == 2:
                        # Already 2D, use as is
                        logger.info(
                            f"Using 2D embedding as is: {dummy_embeddings.shape}"
                        )
                    else:
                        raise ValueError(
                            f"MLP probe expects 2D, 3D or 4D embeddings, got shape "
                            f"{dummy_embeddings.shape}"
                        )
                    inferred_dim = dummy_embeddings.shape[1]
                    classifier_input_dim = inferred_dim

                logger.info(
                    f"MLPProbe init: dummy_embeddings shape: "
                    f"{dummy_embeddings.shape}, "
                    f"layers: {layers}, aggregation: {self.aggregation}, "
                    f"inferred_dim: {inferred_dim}"
                )
        else:
            # We will compute embeddings inside forward – need base_model.
            if base_model is None:
                raise ValueError("base_model must be provided when feature_mode=False")
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
                    logger.info(
                        f"MLP probe (none agg): {len(dummy_embeddings)} tensors"
                    )
                    # Get embedding dimensions for each layer, collapsing 3D to 2D
                    layer_dims = []
                    for i, emb in enumerate(dummy_embeddings):
                        if emb.dim() == 4:
                            # Collapse 4D to 2D by taking mean across dimensions 1 and 2
                            dummy_embeddings[i] = emb.mean(dim=(1, 2))
                            logger.info(
                                f"Collapsed 4D embedding {i} to: "
                                f"{dummy_embeddings[i].shape}"
                            )
                            layer_dims.append(dummy_embeddings[i].shape[1])
                        elif emb.dim() == 3:
                            # Collapse 3D to 2D by taking mean across sequence dimension
                            dummy_embeddings[i] = emb.mean(dim=1)
                            logger.info(
                                f"Collapsed 3D embedding {i} to: "
                                f"{dummy_embeddings[i].shape}"
                            )
                            layer_dims.append(dummy_embeddings[i].shape[1])
                        elif emb.dim() == 2:
                            # Already 2D, use as is
                            logger.info(f"Using 2D embedding {i} as is: {emb.shape}")
                            layer_dims.append(emb.shape[1])
                        else:
                            raise ValueError(
                                f"MLP probe expects 2D, 3D or 4D embeddings "
                                f"for layer {i}, "
                                f"got shape {emb.shape}"
                            )

                    # Determine projection dimension
                    if self.projection_dim is None:
                        # Use minimum dimension across all layers
                        self.projection_dim = min(layer_dims)

                    # Create projection heads for each layer
                    self.layer_projections = torch.nn.ModuleList(
                        [
                            torch.nn.Linear(dim, self.projection_dim).to(device)
                            for dim in layer_dims
                        ]
                    )

                    # Final MLP input dimension is projection_dim * num_layers
                    inferred_dim = self.projection_dim * len(dummy_embeddings)
                    classifier_input_dim = inferred_dim

                else:
                    logger.info(f"Input to MLP probe shape: {dummy_embeddings.shape}")
                    if dummy_embeddings.dim() == 4:
                        # Collapse 4D to 2D by taking mean across dimensions 1 and 2
                        dummy_embeddings = dummy_embeddings.mean(dim=(1, 2))
                        logger.info(
                            f"Collapsed 4D embedding to: {dummy_embeddings.shape}"
                        )
                    elif dummy_embeddings.dim() == 3:
                        # Collapse 3D to 2D by taking mean across sequence dimension
                        dummy_embeddings = dummy_embeddings.mean(dim=1)
                        logger.info(
                            f"Collapsed 3D embedding to: {dummy_embeddings.shape}"
                        )
                    elif dummy_embeddings.dim() == 2:
                        # Already 2D, use as is
                        logger.info(
                            f"Using 2D embedding as is: {dummy_embeddings.shape}"
                        )
                    else:
                        raise ValueError(
                            f"MLP probe expects 2D, 3D or 4D embeddings, got shape "
                            f"{dummy_embeddings.shape}"
                        )
                    inferred_dim = dummy_embeddings.shape[1]
                    classifier_input_dim = inferred_dim

                if isinstance(dummy_embeddings, list):
                    logger.info(
                        f"MLPProbe init (feature_mode=False, aggregation='none'): "
                        f"dummy_embeddings: list of {len(dummy_embeddings)} tensors, "
                        f"layers: {layers}, aggregation: {self.aggregation}, "
                        f"inferred_dim: {inferred_dim}"
                    )
                else:
                    logger.info(
                        f"MLPProbe init (feature_mode=False): "
                        f"dummy_embeddings shape: {dummy_embeddings.shape}, "
                        f"layers: {layers}, aggregation: {self.aggregation}, "
                        f"inferred_dim: {inferred_dim}"
                    )

        # Build MLP layers
        self.mlp = self._build_mlp(classifier_input_dim, num_classes)
        # Move MLP to the specified device
        self.mlp = self.mlp.to(self.device)

        # Log final probe parameters
        logger.info(
            f"MLPProbe initialized with final parameters: "
            f"layers={self.layers}, feature_mode={self.feature_mode}, "
            f"aggregation={self.aggregation}, freeze_backbone={self.freeze_backbone}, "
            f"hidden_dims={self.hidden_dims}, dropout_rate={self.dropout_rate}, "
            f"activation={self.activation}, target_length={self.target_length}, "
            f"projection_dim={self.projection_dim}, inferred_dim={inferred_dim}, "
            f"cls_dim={classifier_input_dim}, mlp_dim={self.mlp[0].in_features}, "
            f"n_classes={num_classes}, "
            f"has_layer_projections={hasattr(self, 'layer_projections')}"
        )

    def __del__(self) -> None:
        """Cleanup hooks when the probe is destroyed."""
        try:
            if hasattr(self, "base_model") and self.base_model is not None:
                if hasattr(self.base_model, "deregister_all_hooks"):
                    self.base_model.deregister_all_hooks()
        except Exception:
            pass  # Ignore errors during cleanup

    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build the MLP architecture.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension

        Returns:
            MLP module
        """
        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    self._get_activation(self.activation),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module.

        Args:
            activation: Activation function name

        Returns:
            Activation function module

        Raises:
            ValueError: If activation function is not supported
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "swish":
            return nn.SiLU()  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the MLP probe.

        Args:
            x: Input tensor of shape (batch_size, time_steps)
                or (batch_size, embedding_dim) in feature mode
            padding_mask: Optional padding mask tensor of shape (batch_size, time_steps)
        Returns:
            Classification logits of shape (batch_size, num_classes)
        Raises:
            ValueError: If base_model is None when not in feature mode
            RuntimeError: If embedding extraction fails
        """
        if self.feature_mode:
            assert isinstance(x, torch.Tensor), "x should be a tensor"
            embeddings = x

            # Handle different embedding dimensions in feature mode
            if embeddings.dim() == 4:
                # Collapse 4D to 2D by taking mean across dimensions 1 and 2
                embeddings = embeddings.mean(dim=(1, 2))
                logger.debug(
                    f"Feature mode: Collapsed 4D embedding to: {embeddings.shape}"
                )
            elif embeddings.dim() == 3:
                # Collapse 3D to 2D by taking mean across sequence dimension
                embeddings = embeddings.mean(dim=1)
                logger.debug(
                    f"Feature mode: Collapsed 3D embedding to: {embeddings.shape}"
                )
            elif embeddings.dim() == 2:
                # Already 2D, use as is
                logger.debug(
                    f"Feature mode: Using 2D embedding as is: {embeddings.shape}"
                )
            else:
                raise ValueError(
                    f"Feature mode expects 2D, 3D or 4D embeddings, got shape "
                    f"{embeddings.shape}"
                )
        else:
            if self.base_model is None:
                raise ValueError("base_model must be provided when feature_mode=False")

            try:
                embeddings = self.base_model.extract_embeddings(
                    x, padding_mask=padding_mask, aggregation=self.aggregation
                )

                # Detach embeddings if backbone is frozen to prevent gradient flow
                if self.freeze_backbone:
                    if isinstance(embeddings, list):
                        embeddings = [emb.detach() for emb in embeddings]
                    else:
                        embeddings = embeddings.detach()

            except Exception as e:
                raise RuntimeError(
                    f"Failed to extract embeddings from base_model: {e}. "
                    f"Input shape: {x.shape}, layers: {self.layers}"
                ) from e

            logger.debug(
                f"MLP probe forward: Received embeddings type: {type(embeddings)}, "
                f"shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'list'}"
            )

            # Log embeddings info for debugging
            if hasattr(self, "_debug_logged") and not self._debug_logged:
                logger.info(
                    f"MLPProbe: extracted embeddings shape: {embeddings.shape}, "
                    f"layers: {self.layers}, aggregation: {self.aggregation}"
                )
                self._debug_logged = True

            # Handle the case where embeddings is a list (aggregation="none")
            if isinstance(embeddings, list):
                # Apply projection heads to each layer's embeddings
                projected_embeddings = []
                for i, (emb, proj) in enumerate(
                    zip(embeddings, self.layer_projections, strict=False)
                ):
                    # Ensure embedding is 2D: (batch_size, embedding_dim)
                    if emb.dim() == 4:
                        # Collapse 4D to 2D by taking mean across dimensions 1 and 2
                        emb = emb.mean(dim=(1, 2))
                        logger.debug(f"Collapsed 4D embedding {i} to: {emb.shape}")
                    elif emb.dim() == 3:
                        # Collapse 3D to 2D by taking mean across sequence dimension
                        emb = emb.mean(dim=1)
                        logger.debug(f"Collapsed 3D embedding {i} to: {emb.shape}")
                    elif emb.dim() == 2:
                        # Already 2D, use as is
                        logger.debug(f"Using 2D embedding {i} as is: {emb.shape}")
                    else:
                        raise ValueError(
                            f"Expected 2D, 3D or 4D embeddings for layer {i}, got "
                            f"{emb.dim()}D. Shape: {emb.shape}"
                        )

                    projected_emb = proj(emb)  # (batch_size, projection_dim)
                    projected_embeddings.append(projected_emb)

                # Concatenate all projected embeddings
                # (batch_size, projection_dim * num_layers)
                embeddings = torch.cat(projected_embeddings, dim=1)

            else:
                # Validate embeddings shape for tensor case
                if embeddings.dim() == 4:
                    # Collapse 4D to 2D by taking mean across dimensions 1 and 2
                    embeddings = embeddings.mean(dim=(1, 2))
                    logger.debug(f"Collapsed 4D embedding to: {embeddings.shape}")
                elif embeddings.dim() == 3:
                    # Collapse 3D to 2D by taking mean across sequence dimension
                    embeddings = embeddings.mean(dim=1)
                    logger.debug(f"Collapsed 3D embedding to: {embeddings.shape}")
                elif embeddings.dim() == 2:
                    # Already 2D, use as is
                    logger.debug(f"Using 2D embedding as is: {embeddings.shape}")
                else:
                    raise ValueError(
                        f"Expected embeddings to have 2, 3 or 4 dimensions, "
                        f"got {embeddings.dim()}. Shape: {embeddings.shape}"
                    )

        # Debug logging for embeddings shape
        if hasattr(self, "_debug_logged") and not self._debug_logged:
            logger.info(
                f"MLPProbe: final embeddings shape: {embeddings.shape}, "
                f"MLP expects: {self.mlp[0].in_features}"
            )
            self._debug_logged = True

        # Final validation: ensure embeddings match MLP input dimension
        if embeddings.shape[1] != self.mlp[0].in_features:
            raise ValueError(
                f"Embeddings dimension {embeddings.shape[1]} does not match "
                f"MLP input dimension {self.mlp[0].in_features}. "
                f"Expected: {self.mlp[0].in_features}, got: {embeddings.shape[1]}. "
                f"Embeddings shape: {embeddings.shape}, layers: {self.layers}, "
                f"aggregation: {self.aggregation}"
            )

        return self.mlp(embeddings)

    def debug_info(self) -> dict:
        """Get debug information about the probe configuration.

        Returns
        -------
        dict
            Dictionary containing debug information about the probe
        """
        debug_info = {
            "feature_mode": self.feature_mode,
            "layers": self.layers,
            "aggregation": self.aggregation,
            "mlp_input_dim": self.mlp[0].in_features,
            "mlp_output_dim": self.mlp[-1].out_features,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "target_length": self.target_length,
            "base_model_type": (
                type(self.base_model).__name__ if self.base_model else None
            ),
            "device": self.device,
            "freeze_backbone": self.freeze_backbone,
        }

        # Add projection-specific info if using projections
        if hasattr(self, "layer_projections"):
            debug_info["projection_dim"] = self.projection_dim
            debug_info["num_projection_heads"] = len(self.layer_projections)
            if self.layer_projections:
                debug_info["projection_input_dims"] = [
                    proj.in_features for proj in self.layer_projections
                ]

        return debug_info
