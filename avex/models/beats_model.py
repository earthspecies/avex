"""BEATs model implementation for audio representation learning.

This module provides BEATs Bidirectional Encoder representation from Audio Transformers
model implementation for audio representation learning tasks.
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn

from avex.configs import AudioConfig
from avex.models.base_model import ModelBase
from avex.models.beats.beats import BEATs, BEATsConfig
from avex.utils import universal_torch_load

logger = logging.getLogger(__name__)

BEATS_PRETRAINED_PATH_FT = "gs://foundation-models/beats_ckpts/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
BEATS_PRETRAINED_PATH_SSL = "gs://representation-learning/pretrained/BEATs_iter3_plus_AS2M.pt"

BEATS_PRETRAINED_PATH_NATURELM = (
    "gs://foundation-models/beats_ckpts/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2_rl_loaded.pt"
)

# Default config for BEATs iter3 plus AS2M (fine-tuned variant)
# This allows initialization without loading from GCS when pretrained=False
BEATS_ITER3_AS2M_CONFIG = {
    "input_patch_size": 16,
    "embed_dim": 512,
    "conv_bias": False,
    "encoder_layers": 12,
    "encoder_embed_dim": 768,
    "encoder_ffn_embed_dim": 3072,
    "encoder_attention_heads": 12,
    "activation_fn": "gelu",
    "layer_wise_gradient_decay_ratio": 1.0,
    "layer_norm_first": False,
    "deep_norm": False,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0.0,
    "encoder_layerdrop": 0.05,
    "dropout_input": 0.0,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 800,
    "gru_rel_pos": True,
    "finetuned_model": True,
    "predictor_dropout": 0.0,
    "predictor_class": 527,
}


class Model(ModelBase):
    """Wrapper that adapts the raw *BEATs* backbone for our training loop.

    This module follows the same conventions as the other model wrappers
    (e.g. ``efficientnet.py``) so that it can be selected via
    ``avex.models.utils.factory.build_model_from_spec``.

    The underlying BEATs implementation operates directly on raw‐waveform
    inputs.  We therefore do *not* apply the optional :class:`AudioProcessor`
    from :pymeth:`ModelBase.process_audio` unless an ``audio_config`` is
    explicitly supplied.

    Notes
    -----
    1.  BEATs extracts a sequence of frame-level embeddings with dimension
        ``cfg.encoder_embed_dim`` (default: ``768``).  We convert this
        variable-length sequence into a fixed-dimensional vector via masked
        mean-pooling before feeding it to a linear classifier.
    2.  When ``return_features_only=True`` the classifier layer is skipped and
        the unpooled frame-level features are returned directly (shape: B x T x D),
        which is handy for representation extraction / linear probing.
    """

    def __init__(
        self,
        *,
        num_classes: Optional[int] = None,
        pretrained: bool = False,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        use_naturelm: bool = False,
        fine_tuned: bool = False,
        disable_layerdrop: bool = False,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # If num_classes is not provided, always fall back to embedding mode.
        # This keeps BEATs usable as a pure backbone without requiring a head.
        if num_classes is None:
            if not return_features_only:
                logger.info(
                    "num_classes is None for BEATs; falling back to return_features_only=True "
                    "and disabling the classifier head."
                )
            return_features_only = True
            self.num_classes = None
        else:
            self.num_classes = num_classes

        # Store disable_layerdrop parameter
        self.disable_layerdrop = disable_layerdrop
        self.use_naturelm = use_naturelm
        self.fine_tuned = fine_tuned

        # ------------------------------------------------------------------
        # 1.  Build the BEATs backbone
        # ------------------------------------------------------------------

        if pretrained:
            # Load config and weights from GCS checkpoint
            if fine_tuned:
                beats_checkpoint_path = BEATS_PRETRAINED_PATH_FT
            else:
                beats_checkpoint_path = BEATS_PRETRAINED_PATH_SSL

            beats_ckpt = universal_torch_load(beats_checkpoint_path, cache_mode="use", map_location="cpu")
            beats_cfg = BEATsConfig(beats_ckpt["cfg"])
            logger.debug(f"Loaded BEATs config from checkpoint: {beats_cfg}")

            if use_naturelm:  # BEATs-NatureLM has no config, load from regular ckpt first.
                beats_ckpt_weights = universal_torch_load(BEATS_PRETRAINED_PATH_NATURELM, map_location="cpu")
            else:
                beats_ckpt_weights = beats_ckpt["model"]

            self.backbone = BEATs(beats_cfg)
            self.backbone.to(device)
            self.backbone.load_state_dict(beats_ckpt_weights, strict=False)
        else:
            # Use default config without loading pretrained weights
            # Weights can be loaded later via load_state_dict (e.g., from HuggingFace checkpoint)
            logger.info("Initializing BEATs with default config (pretrained=False)")
            beats_cfg = BEATsConfig(BEATS_ITER3_AS2M_CONFIG)
            self.backbone = BEATs(beats_cfg)
            self.backbone.to(device)

        # ------------------------------------------------------------------
        # 2.  Optional classifier for supervised training
        # ------------------------------------------------------------------
        self._return_features_only = return_features_only
        if not return_features_only:
            self.classifier = nn.Linear(768, num_classes)
        else:
            self.register_module("classifier", None)  # type: ignore[arg-type]

        # ------------------------------------------------------------------
        # 3.  Pre-discover MLP (fc1, fc2) layers for efficient hook management
        # ------------------------------------------------------------------
        # MLP layers will be discovered in _discover_linear_layers override

    def _discover_linear_layers(self) -> None:
        """
        Discover and cache only the BEATs layers that are useful for embeddings.
        This method is called when target_layers=["all"] is used.
        Specifically:
        - backbone.post_extract_proj
        - backbone.encoder.layers.{i}.fc2 (only fc2 layers from encoder blocks)
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            for name, _module in self.named_modules():
                # Keep the initial projection after conv frontend
                if name.endswith("post_extract_proj"):
                    self._layer_names.append(name)

                # Keep only the fc2 layers from transformer encoder blocks
                # Pattern: backbone.encoder.layers.{i}.fc2
                elif name.endswith(".fc2") and "backbone.encoder.layers." in name:
                    self._layer_names.append(name)

            logger.info(f"Discovered {len(self._layer_names)} embedding layers in BEATs: {self._layer_names}")

    def _discover_embedding_layers(self) -> None:
        """
        Discover and cache only the BEATs layers that are useful for embeddings.
        Specifically:
        - backbone.post_extract_proj
        - backbone.encoder.layers.{i}.fc2 (only fc2 layers from encoder blocks)
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            # # Discover standard linear layers
            # for name, module in self.named_modules():
            #     if isinstance(module, torch.nn.Linear):
            #         self._layer_names.append(name)

            for name, _module in self.named_modules():
                # # Keep the initial projection after conv frontend
                # if name.endswith("post_extract_proj"):
                #     self._layer_names.append(name)

                # Keep only the fc2 layers from transformer encoder blocks
                # Pattern: backbone.encoder.layers.{i}.fc2
                if name.endswith(".fc2") and "backbone.encoder.layers." in name:
                    if name not in self._layer_names:
                        self._layer_names.append(name)

            logger.info(f"Discovered {len(self._layer_names)} embedding layers in BEATs: {self._layer_names}")

    # ----------------------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # noqa: D401 – keep signature consistent
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw audio waveform with shape ``(batch, time)``.
        padding_mask : torch.Tensor, optional
            Boolean mask where *True* denotes padding elements.  Shape must be
            ``(batch, time)`` and match *x*.

        Returns
        -------
        torch.Tensor
            • When *return_features_only* is **False**: logits of shape
              ``(batch, num_classes)``
            • Otherwise: unpooled frame-level features of shape
              ``(batch, num_frames, encoder_embed_dim)``
        """
        # Optional audio pre-processing
        x = self.process_audio(x)

        features, frame_padding = self.backbone(x, padding_mask, disable_layerdrop=self.disable_layerdrop)

        # features: (B, T', D)
        # frame_padding: (B, T') or None

        if self._return_features_only:
            return features

        # ------------------------------------------------------------------
        # 3.  Masked mean-pooling over the temporal dimension
        # ------------------------------------------------------------------
        if frame_padding is not None and frame_padding.any():
            masked_features = features.clone()
            masked_features[frame_padding] = 0.0  # Zero-out padded frames
            valid_counts = (~frame_padding).sum(dim=1, keepdim=True).clamp(min=1)
            pooled = masked_features.sum(dim=1) / valid_counts
        else:
            pooled = features.mean(dim=1)

        return self.classifier(pooled)

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract embeddings from all registered hooks in the BEATs model.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav'
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (ignored for BEATs)
        aggregation : str
            Aggregation method for multiple layers ('mean', 'max', 'cls_token', 'none')
        freeze_backbone : bool
            Whether to freeze the backbone and use torch.no_grad()

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Model embeddings (tensor if aggregation!="none", list if False)

        Raises
        ------
        ValueError
            If input tensor is None or audio tensor is empty
        """
        # Validate input
        if x is None:
            raise ValueError("Input tensor cannot be None")

        # Check for empty audio
        if isinstance(x, dict):
            wav = x["raw_wav"]
        else:
            wav = x

        if wav.numel() == 0 or wav.shape[-1] == 0:
            raise ValueError("Audio tensor cannot be empty")

        # Check if hooks are registered
        if not self._hooks:
            raise ValueError("No hooks are registered in the model.")

        # Store original training state
        was_training = self.training
        mode_changed = False

        # Set model mode based on freeze_backbone parameter
        if freeze_backbone:
            # For frozen backbone: use eval mode for deterministic results
            if self.training:
                self.eval()
                mode_changed = True
            # Set deterministic behavior for CUDA if available
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        else:
            # For fine-tuning: keep current training state
            # Don't change model mode to allow proper training behavior
            # However, if disable_layerdrop=True, we need to ensure hooks work
            # by temporarily switching to eval mode for the forward pass
            if self.disable_layerdrop and self.training:
                self.eval()
                mode_changed = True

        try:
            # Clear previous hook outputs
            self._clear_hook_outputs()

            # Hooks are already registered in __init__ via base class

            # Process input
            if isinstance(x, dict):
                wav = x["raw_wav"]
                mask = x.get("padding_mask")
                expected_batch_size = wav.shape[0]
            else:
                wav = x
                mask = padding_mask
                expected_batch_size = wav.shape[0]

            # Forward pass to trigger hooks (conditionally use torch.no_grad based on
            # freeze_backbone)
            if freeze_backbone:
                with torch.no_grad():
                    self.forward(wav, mask)
            else:
                self.forward(wav, mask)

            logger.debug(f"Forward pass completed. Hook outputs: {list(self._hook_outputs.keys())}")

            # Collect embeddings from hook outputs
            embeddings = []
            for layer_name in self._hook_outputs.keys():
                embedding = self._hook_outputs[layer_name]
                embeddings.append(embedding)
                logger.debug(f"Found embedding for {layer_name}: {embedding.shape}")

            logger.debug(f"Collected {len(embeddings)} embeddings")

            # Check if we got any embeddings
            if not embeddings:
                raise ValueError(f"No layers found matching: {self._hook_outputs.keys()}")

            # First, ensure all embeddings are in batch-first format
            for i in range(len(embeddings)):
                if embeddings[i].shape[0] != expected_batch_size:
                    # Transpose to batch-first format
                    embeddings[i] = embeddings[i].transpose(0, 1)

            # Process embeddings based on aggregation parameter
            if aggregation == "none":
                if len(embeddings) == 1:
                    return embeddings[0]
                else:
                    return embeddings
            else:
                for i in range(len(embeddings)):
                    if embeddings[i].dim() == 2:
                        # Already in correct shape
                        pass
                    elif embeddings[i].dim() == 3:
                        if aggregation == "mean":
                            embeddings[i] = torch.mean(embeddings[i], dim=1)
                        elif aggregation == "max":
                            embeddings[i] = torch.max(embeddings[i], dim=1)[0]  # max returns (values, indices)
                        elif aggregation == "cls_token":
                            embeddings[i] = embeddings[i][:, 0, :]
                        else:
                            raise ValueError(f"Unsupported aggregation method: {aggregation}")
                    else:
                        raise ValueError(f"Unexpected embedding dimension: {embeddings[i].dim()}. Expected 2 or 3.")

                # Concatenate all embeddings
                if len(embeddings) == 1:
                    return embeddings[0]
                else:
                    return torch.cat(embeddings, dim=1)
        finally:
            # Clear hook outputs for next call
            self._clear_hook_outputs()
            # Restore original training state only if we changed it
            if mode_changed and was_training:
                self.train()

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        audio = super().process_audio(x)
        if self.use_naturelm:
            audio = torch.clamp(audio, -1.0, 1.0)
        return audio
