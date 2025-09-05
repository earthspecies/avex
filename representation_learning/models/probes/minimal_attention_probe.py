"""Minimal attention probe for sequence-based evaluation.

This implementation is intentionally simple while remaining compatible with
the repository's embedding extraction flow and masking conventions.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


class MinimalAttentionProbe(nn.Module):
    """A minimal attention probe using single-layer MultiheadAttention.

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
    - When aggregation="none", each layer gets its own attention projection head.
    """

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: list[str],
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
        classifier_input_dim: int = 0

        if feature_mode:
            if input_dim is None:
                if base_model is None:
                    raise ValueError(
                        "input_dim must be provided when feature_mode=True and "
                        "base_model is None"
                    )
                # Infer using a dummy pass
                with torch.no_grad():
                    test_len = self._compute_target_length(base_model)
                    dummy = torch.randn(1, test_len, device=device)
                    emb = base_model.extract_embeddings(
                        dummy, aggregation="none", freeze_backbone=True
                    )
                    if isinstance(emb, list) and len(emb) > 0:
                        emb = emb[0]
                    logger.info(
                        f"MinimalAttn: {emb.shape if hasattr(emb, 'shape') else 'list'}"
                    )
                    if emb.dim() == 4:
                        # Handle 4D embeddings (B, C, H, W)
                        batch_size, channels, height, width = emb.shape
                        logger.debug(f"Processing 4D embedding: {emb.shape}")
                        # Reshape to (B, W, C*H) - treat width as sequence length,
                        # C*H as features. Match WeightedAttentionProbe pattern
                        emb = (
                            emb.permute(0, 3, 1, 2)
                            .contiguous()
                            .view(batch_size, width, channels * height)
                        )
                        logger.info(f"Reshaped 4D embedding to: {emb.shape}")

                    if emb.dim() == 2:
                        # Handle 2D embeddings by adding sequence dimension
                        emb = emb.unsqueeze(2)  # (B, D) -> (B, D, 1)
                        logger.info(f"Reshaped 2D embedding to: {emb.shape}")

                    if emb.dim() == 3:
                        inferred_dim = emb.shape[-1]
                        classifier_input_dim = inferred_dim
                    else:
                        raise ValueError(
                            "Feature mode requires 2D, 3D or 4D embeddings."
                        )
            else:
                inferred_dim = input_dim
                classifier_input_dim = inferred_dim

            # Find all divisors of embed_dim that are reasonable for
            # attention heads. We want heads that result in head_dim >= 64
            # (PyTorch recommendation)
            valid_heads = []
            for n in range(1, min(self.num_heads + 1, inferred_dim + 1)):
                if inferred_dim % n == 0 and inferred_dim // n >= 64:
                    valid_heads.append(n)

            if not valid_heads:
                # If no valid heads found, use 1 head and adjust
                # embed_dim if needed
                adjusted_num_heads = 1
                if inferred_dim < 64:
                    inferred_dim = 64
            else:
                # Use the largest valid number of heads
                # (closest to requested)
                adjusted_num_heads = max(valid_heads)

            # Create attention layer for feature mode
            self.attention = nn.MultiheadAttention(
                inferred_dim, num_heads=adjusted_num_heads, batch_first=True
            ).to(device)
        else:
            if base_model is None:
                raise ValueError("base_model must be provided when feature_mode=False")
            # Infer from a short dummy forward
            with torch.no_grad():
                test_len = self._compute_target_length(base_model)
                dummy = torch.randn(1, test_len, device=device)
                emb = base_model.extract_embeddings(
                    dummy, aggregation=self.aggregation, freeze_backbone=True
                )

                if self.aggregation == "none":
                    # Handle list of embeddings from multiple layers
                    if isinstance(emb, list):
                        logger.info(
                            f"MinimalAttention probe (none): {len(emb)} tensors"
                        )
                        # Create attention projection heads for each layer
                        self.layer_projections = nn.ModuleList(
                            [
                                nn.MultiheadAttention(
                                    embed_dim=emb.shape[-1],
                                    num_heads=min(self.num_heads, emb.shape[-1] // 64),
                                    dropout=0.0,  # No dropout for minimal probe
                                    batch_first=True,
                                )
                                for emb in emb
                            ]
                        )

                        # Set classifier input dimension for list case
                        # Each attention head outputs emb.shape[-1] features, concat
                        classifier_input_dim = sum(emb.shape[-1] for emb in emb)

                        # Log the setup
                        logger.info(
                            f"MinimalAttentionProbe init (feature_mode=False, "
                            f"aggregation='none'): dummy_embeddings: list of "
                            f"{len(emb)} tensors, layers: {layers}, "
                            f"aggregation: {self.aggregation}, "
                            f"classifier_input_dim: {classifier_input_dim}"
                        )
                    else:
                        raise ValueError(
                            f"Expected list of embeddings for aggregation='none', "
                            f"got {type(emb)}"
                        )
                else:
                    # Single tensor case
                    if isinstance(emb, list) and len(emb) > 0:
                        emb = emb[0]
                    logger.info(f"Input to MinimalAttention probe shape: {emb.shape}")
                    if emb.dim() == 4:
                        # Handle 4D embeddings (B, C, H, W)
                        batch_size, channels, height, width = emb.shape
                        logger.debug(f"Processing 4D embedding: {emb.shape}")
                        # Reshape to (B, W, C*H) - treat width as sequence length,
                        # C*H as features. Match WeightedAttentionProbe pattern
                        emb = (
                            emb.permute(0, 3, 1, 2)
                            .contiguous()
                            .view(batch_size, width, channels * height)
                        )
                        logger.info(f"Reshaped 4D embedding to: {emb.shape}")

                    if emb.dim() == 2:
                        # Handle 2D embeddings by adding sequence dimension
                        emb = emb.unsqueeze(2)  # (B, D) -> (B, D, 1)
                        logger.info(f"Reshaped 2D embedding to: {emb.shape}")

                    if emb.dim() == 3:
                        inferred_dim = emb.shape[-1]
                        classifier_input_dim = inferred_dim
                    else:
                        raise ValueError(
                            "MinimalAttentionProbe expects 2D, 3D or 4D embeddings."
                        )

                    # Find all divisors of embed_dim that are reasonable for
                    # attention heads. We want heads that result in head_dim >= 64
                    # (PyTorch recommendation)
                    valid_heads = []
                    for n in range(1, min(self.num_heads + 1, inferred_dim + 1)):
                        if inferred_dim % n == 0 and inferred_dim // n >= 64:
                            valid_heads.append(n)

                    if not valid_heads:
                        # If no valid heads found, use 1 head and adjust
                        # embed_dim if needed
                        adjusted_num_heads = 1
                        if inferred_dim < 64:
                            inferred_dim = 64
                    else:
                        # Use the largest valid number of heads
                        # (closest to requested)
                        adjusted_num_heads = max(valid_heads)

                    self.attention = nn.MultiheadAttention(
                        inferred_dim, num_heads=adjusted_num_heads, batch_first=True
                    ).to(device)

        # Ensure attention attribute exists for all cases
        if not hasattr(self, "attention"):
            self.attention = None

        self.classifier = nn.Linear(classifier_input_dim, num_classes).to(device)

        # Log final probe parameters
        logger.info(
            f"MinimalAttentionProbe initialized with final parameters: "
            f"layers={self.layers}, feature_mode={self.feature_mode}, "
            f"aggregation={self.aggregation}, freeze_backbone={self.freeze_backbone}, "
            f"num_heads={self.num_heads}, target_length={self.target_length}, "
            f"inferred_dim={inferred_dim}, classifier_dim={classifier_input_dim}, "
            f"num_classes={num_classes}, "
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

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x
            - If ``feature_mode`` is True: embeddings shaped ``(B,T,D)``.
            - Otherwise: raw audio ``(B, samples)`` to be embedded by the backbone.
        padding_mask
            Optional boolean mask of shape ``(B,T)`` where True indicates padded
            positions.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.

        Raises
        ------
        ValueError
            If embeddings are not 3D in feature mode, or if base_model is None when
            feature_mode=False, or if embeddings have unexpected dimensions.
        """
        if isinstance(x, dict):
            padding_mask = x.get("padding_mask")
            x = x["raw_wav"]
        else:
            assert padding_mask is not None, (
                "padding_mask must be provided if x is a tensor"
            )
        if self.feature_mode:
            embeddings = x
            if embeddings.dim() != 3:
                raise ValueError(
                    f"Expected 3D embeddings in feature mode, got {embeddings.shape}"
                )
        else:
            if self.base_model is None:
                raise ValueError("base_model must be provided when feature_mode=False")
            embeddings = self.base_model.extract_embeddings(
                x,
                padding_mask=padding_mask,
                aggregation=self.aggregation,
                freeze_backbone=self.freeze_backbone,
            )

            if self.freeze_backbone:
                if isinstance(embeddings, list):
                    embeddings = [emb.detach() for emb in embeddings]
                else:
                    embeddings = embeddings.detach()

        # Handle the case where embeddings is a list (aggregation="none")
        if isinstance(embeddings, list):
            # Apply attention projection heads to each layer's embeddings
            projected_embeddings = []
            for i, (emb, attn_proj) in enumerate(
                zip(embeddings, self.layer_projections, strict=False)
            ):
                # Ensure embeddings are 3D
                if emb.dim() == 3:
                    # Apply self-attention to each layer's embeddings
                    attn_out, _ = attn_proj(emb, emb, emb)
                    # Global average pooling over sequence dimension
                    projected_emb = attn_out.mean(dim=1)
                else:
                    raise ValueError(
                        f"MinimalAttentionProbe expects 3D embeddings "
                        f"(batch_size, sequence_length, embedding_dim), "
                        f"got shape {emb.shape} for layer {i}"
                    )
                projected_embeddings.append(projected_emb)

            # Concatenate along the feature dimension
            embeddings = torch.cat(projected_embeddings, dim=-1)

            # When using projection heads, embeddings are already processed
            # No need to pass through main attention - go directly to classifier
            pooled = embeddings
        else:
            # Single tensor case
            if isinstance(embeddings, list) and len(embeddings) > 0:
                embeddings = embeddings[0]

            if embeddings.dim() == 4:
                # Handle 4D embeddings (B, C, H, W)
                batch_size, channels, height, width = embeddings.shape
                logger.debug(f"Processing 4D embedding: {embeddings.shape}")
                # Reshape to (B, W, C*H) - treat width as sequence length,
                # C*H as features. Match WeightedAttentionProbe pattern
                embeddings = (
                    embeddings.permute(0, 3, 1, 2)
                    .contiguous()
                    .view(batch_size, width, channels * height)
                )
                # Result: (B, W, C*H) where W is sequence length, C*H is features
            elif embeddings.dim() == 2:
                # Handle 2D embeddings by adding sequence dimension
                embeddings = embeddings.unsqueeze(2)  # (B, D) -> (B, D, 1)
                embeddings = embeddings.transpose(1, 2)  # (B, D, 1) -> (B, 1, D)
                logger.debug(f"Reshaped 2D embedding to: {embeddings.shape}")
            elif embeddings.dim() == 3:
                pass
            else:
                raise ValueError(
                    f"Sequence probe requires (B,T,D) embeddings, got "
                    f"{embeddings.shape}"
                )

            # Ensure padding mask matches temporal dimension if provided
            if (
                padding_mask is not None
                and padding_mask.shape[1] != embeddings.shape[1]
            ):
                padding_mask = (
                    torch.nn.functional.interpolate(
                        padding_mask.float().unsqueeze(1),
                        size=embeddings.shape[1],
                        mode="nearest",
                    )
                    .squeeze(1)
                    .bool()
                )

            # Apply main attention
            attn_out, _ = self.attention(
                embeddings, embeddings, embeddings, key_padding_mask=padding_mask
            )
            pooled = attn_out.mean(dim=1)

        return self.classifier(pooled)

    def _compute_target_length(self, base_model: ModelBase) -> int:
        """Compute a reasonable target length for dummy inference.

        Parameters
        ----------
        base_model
            Backbone model whose audio processor defines target length.

        Returns
        -------
        int
            Target length in samples or frames depending on the processor.
        """
        if self.target_length is not None:
            return int(self.target_length)
        if hasattr(base_model.audio_processor, "target_length_seconds"):
            return int(
                base_model.audio_processor.target_length_seconds
                * base_model.audio_processor.sr
            )
        if hasattr(base_model.audio_processor, "target_length"):
            return int(base_model.audio_processor.target_length)
        # Fallback to one second at 16 kHz if nothing is available
        return 16000

    def debug_info(self) -> dict:
        """Get debug information about the probe.

        Returns:
            Dictionary containing debug information
        """
        return {
            "probe_type": "minimal_attention",
            "layers": self.layers,
            "feature_mode": self.feature_mode,
            "aggregation": self.aggregation,
            "freeze_backbone": self.freeze_backbone,
            "num_heads": self.num_heads,
            "target_length": self.target_length,
            "has_layer_projections": hasattr(self, "layer_projections"),
        }
