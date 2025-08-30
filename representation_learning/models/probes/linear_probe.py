"""Linear probe for representation learning evaluation."""

import logging
from typing import List, Optional

import torch

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


class LinearProbe(torch.nn.Module):
    """
    Lightweight head for *linear probing* a frozen representation model.

    The probe extracts embeddings from specified layers of a **base_model** and
    feeds their concatenation into a single fully-connected classifier layer.

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
        target_length: Target length in samples for audio processing. If None, will be
            computed from base_model.audio_processor. Required if
            base_model.audio_processor does not have target_length or
            target_length_seconds attributes.
        projection_dim: Target dimension for each layer projection when
            aggregation=None. If None, uses the minimum embedding dimension
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
        self.target_length = target_length
        self.projection_dim = projection_dim
        self.freeze_backbone = freeze_backbone

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
                        # Take the mean across a dimension
                        dummy_embeddings = dummy_embeddings.squeeze(1)
                        if dummy_embeddings.dim() == 3:
                            dummy_embeddings = dummy_embeddings.mean(dim=1)
                    inferred_dim = dummy_embeddings.shape[1]

                logger.info(
                    f"LinearProbe init: dummy_embeddings shape: "
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
                    # Get embedding dimensions for each layer
                    layer_dims = [
                        emb.shape[1] if emb.dim() == 2 else emb.shape[-1]
                        for emb in dummy_embeddings
                    ]

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

                    # Final classifier input dimension is projection_dim * num_layers
                    inferred_dim = self.projection_dim * len(dummy_embeddings)

                else:
                    if dummy_embeddings.dim() == 3:
                        dummy_embeddings = dummy_embeddings.squeeze(1)
                        if dummy_embeddings.dim() == 3:
                            dummy_embeddings = dummy_embeddings.mean(dim=1)
                    inferred_dim = dummy_embeddings.shape[1]

                if isinstance(dummy_embeddings, list):
                    logger.info(
                        f"LinearProbe init (feature_mode=False, aggregation='none'): "
                        f"dummy_embeddings: list of {len(dummy_embeddings)} tensors, "
                        f"layers: {layers}, aggregation: {self.aggregation}, "
                        f"inferred_dim: {inferred_dim}"
                    )
                else:
                    logger.info(
                        f"LinearProbe init (feature_mode=False): "
                        f"dummy_embeddings shape: {dummy_embeddings.shape}, "
                        f"layers: {layers}, aggregation: {self.aggregation}, "
                        f"inferred_dim: {inferred_dim}"
                    )

        self.classifier = torch.nn.Linear(inferred_dim, num_classes).to(device)

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
            RuntimeError: If embedding extraction fails
        """
        if self.feature_mode:
            assert isinstance(x, torch.Tensor), "x should be a tensor"
            embeddings = x
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

            # Log embeddings info for debugging
            if hasattr(self, "_debug_logged") and not self._debug_logged:
                logger.info(
                    f"LinearProbe: extracted embeddings shape: {embeddings.shape}, "
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
                    if emb.dim() == 3:
                        emb = emb.squeeze(1)  # Remove feature dimension if present
                        if emb.dim() == 3:
                            emb = emb.mean(dim=1)
                    elif emb.dim() != 2:
                        raise ValueError(
                            f"Expected 2D or 3D embeddings for layer {i}, got "
                            f"{emb.dim()}D. Shape: {emb.shape}"
                        )

                    projected_emb = proj(emb)  # (batch_size, projection_dim)
                    projected_embeddings.append(projected_emb)

                # Concatenate all projected embeddings
                # (batch_size, projection_dim * num_layers)
                embeddings = torch.cat(projected_embeddings, dim=1)

            else:
                # Validate embeddings shape for tensor case
                if embeddings.dim() != 2 and embeddings.dim() != 3:
                    raise ValueError(
                        f"Expected embeddings to have 2 or 3 dimensions, "
                        f"got {embeddings.dim()}. Shape: {embeddings.shape}"
                    )

                if embeddings.dim() == 3:
                    embeddings = embeddings.squeeze(1)
                    if embeddings.dim() == 3:
                        embeddings = embeddings.mean(dim=1)

        # Debug logging for embeddings shape
        if hasattr(self, "_debug_logged") and not self._debug_logged:
            logger.info(
                f"LinearProbe: final embeddings shape: {embeddings.shape}, "
                f"classifier expects: {self.classifier.in_features}"
            )
            self._debug_logged = True

        # Final validation: ensure embeddings match classifier input dimension
        if embeddings.shape[1] != self.classifier.in_features:
            raise ValueError(
                f"Embeddings dimension {embeddings.shape[1]} does not match "
                f"classifier input dimension {self.classifier.in_features}. "
                f"Expected: {self.classifier.in_features}, got: {embeddings.shape[1]}. "
                f"Embeddings shape: {embeddings.shape}, layers: {self.layers}, "
                f"aggregation: {self.aggregation}"
            )

        return self.classifier(embeddings)

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
            "classifier_input_dim": self.classifier.in_features,
            "classifier_output_dim": self.classifier.out_features,
            "target_length": self.target_length,
            "base_model_type": (
                type(self.base_model).__name__ if self.base_model else None
            ),
            "device": self.device,
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
