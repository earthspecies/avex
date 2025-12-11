"""BEATs model implementation for audio representation learning.

This module provides BEATs (Bidirectional Encoder representation from Audio Transformers)
model implementation for audio representation learning tasks.

Supports both original BEATs and OpenBEATs checkpoints:
- BEATs: Original Microsoft checkpoints (base size only)
- OpenBEATs: Open-source checkpoints from HuggingFace (base and large sizes)

Based on:
- BEATs Paper: https://arxiv.org/abs/2212.09058
- OpenBEATs Paper: https://arxiv.org/abs/2507.14129
- HuggingFace Collection: https://huggingface.co/collections/shikhar7ssu/openbeats
"""

import logging
import os
import warnings
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase
from representation_learning.models.beats.beats import (
    BEATS_OUTPUT_DIMS,
    BEATS_SIZE_CONFIGS,
    BEATs,
    BEATsConfig,
)
from representation_learning.utils import universal_torch_load

logger = logging.getLogger(__name__)

# Suppress torch.nn.utils.weight_norm deprecation warning
warnings.filterwarnings(
    "ignore",
    message=".*torch.nn.utils.weight_norm.*",
    category=FutureWarning,
)
# Suppress torch.load weights_only deprecation warning
warnings.filterwarnings(
    "ignore",
    message=".*torch.load.*weights_only.*",
    category=FutureWarning,
)
# Suppress torch.cuda.amp.autocast deprecation warning
warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.autocast.*is deprecated.*",
    category=FutureWarning,
)

# ============================================================================ #
#  Checkpoint paths for original BEATs (GCS)
# ============================================================================ #
BEATS_PRETRAINED_PATH_FT = (
    "gs://foundation-models/beats_ckpts/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
)
BEATS_PRETRAINED_PATH_SSL = (
    "gs://representation-learning/pretrained/BEATs_iter3_plus_AS2M.pt"
)
BEATS_PRETRAINED_PATH_NATURELM = "gs://foundation-models/beats_ckpts/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2_rl_loaded.pt"

# ============================================================================ #
#  HuggingFace model IDs for OpenBEATs
# ============================================================================ #
OPENBEATS_HF_MODELS: Dict[str, str] = {
    "openbeats-large-i3": "shikhar7ssu/OpenBEATs-Large-i3",
    "openbeats-base-i3": "shikhar7ssu/OpenBEATs-Base-i3",
}


def load_weights_from_huggingface(
    model_id: str,
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Load BEATs/OpenBEATs weights from HuggingFace.

    Args:
        model_id: HuggingFace model ID or short name (e.g., 'openbeats-large-i3')
        cache_dir: Optional cache directory for downloaded files

    Returns:
        State dict with weights
    """
    # Map short names to full model IDs
    if model_id.lower() in OPENBEATS_HF_MODELS:
        model_id = OPENBEATS_HF_MODELS[model_id.lower()]

    logger.info(f"Loading BEATs from HuggingFace: {model_id}")

    try:
        # Try different potential checkpoint locations
        checkpoint_path = None
        potential_paths = ["epoch_latest.pt", "model.pth", "work/exp/epoch_latest.pt"]

        for path in potential_paths:
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=model_id,
                    filename=path,
                    cache_dir=cache_dir,
                )
                break
            except Exception:
                continue

        if checkpoint_path is None:
            # Try to find the checkpoint by listing files
            from huggingface_hub import list_repo_files

            files = list_repo_files(model_id)
            for f in files:
                if f.endswith(".pt") or f.endswith(".pth"):
                    if "epoch" in f or "model" in f or "best" in f:
                        try:
                            checkpoint_path = hf_hub_download(
                                repo_id=model_id,
                                filename=f,
                                cache_dir=cache_dir,
                            )
                            break
                        except Exception:
                            continue

        if checkpoint_path is None:
            raise ValueError(f"Could not find checkpoint in {model_id}")

        logger.info(f"Downloaded checkpoint from: {checkpoint_path}")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint structures
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Handle encoder prefix (ESPnet checkpoints)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = key[8:]  # Remove "encoder."
            elif key.startswith("model.encoder."):
                new_key = key[14:]  # Remove "model.encoder."
            else:
                new_key = key
            new_state_dict[new_key] = value

        return new_state_dict

    except Exception as e:
        logger.error(f"Error loading from HuggingFace: {e}")
        raise


class Model(ModelBase):
    """Wrapper that adapts the raw *BEATs* backbone for our training loop.

    This module follows the same conventions as the other model wrappers
    (e.g. ``efficientnet.py``) so that it can be selected via
    ``representation_learning.models.get_model.get_model``.

    The underlying BEATs implementation operates directly on raw‐waveform
    inputs.  We therefore do *not* apply the optional :class:`AudioProcessor`
    from :pymeth:`ModelBase.process_audio` unless an ``audio_config`` is
    explicitly supplied.

    Supports two variants:
    - "beats": Original Microsoft BEATs (base size only, loads from GCS)
    - "openbeats": Open-source OpenBEATs (base/large sizes, loads from HuggingFace)

    Notes
    -----
    1.  BEATs extracts a sequence of frame-level embeddings with dimension
        ``cfg.encoder_embed_dim`` (768 for base, 1024 for large).  We convert this
        variable-length sequence into a fixed-dimensional vector via masked
        mean-pooling before feeding it to a linear classifier.
    2.  When ``return_features_only=True`` the classifier layer is skipped and
        the pooled embedding is returned directly, which is handy for
        representation extraction / linear probing.
    """

    def __init__(
        self,
        *,
        num_classes: Optional[int] = None,
        pretrained: bool = False,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        # Original BEATs options (variant="beats")
        use_naturelm: bool = False,
        fine_tuned: bool = False,
        # New unified options
        model_variant: Literal["beats", "openbeats"] = "beats",
        model_size: Literal["base", "large"] = "base",
        model_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        disable_layerdrop: bool = False,
    ) -> None:
        """Initialize BEATs model.

        Args:
            num_classes: Number of output classes (required if return_features_only=False)
            pretrained: Whether to load pretrained weights
            device: Device to run on ('cuda' or 'cpu')
            audio_config: Optional audio processing configuration
            return_features_only: If True, return embeddings instead of class predictions
            use_naturelm: (beats variant only) Use NatureLM checkpoint
            fine_tuned: (beats variant only) Use fine-tuned checkpoint
            model_variant: Which variant to use ('beats' or 'openbeats')
            model_size: Model size ('base' or 'large'). Only 'large' available for openbeats.
            model_id: HuggingFace model ID (e.g., 'openbeats-large-i3') for openbeats variant
            checkpoint_path: Optional local path to a checkpoint file
            disable_layerdrop: Whether to disable layer dropout during forward pass
        """
        super().__init__(device=device, audio_config=audio_config)

        # Validate num_classes
        if not return_features_only and num_classes is None:
            raise ValueError(
                "num_classes must be provided when return_features_only=False"
            )

        # Store parameters
        self.disable_layerdrop = disable_layerdrop
        self.model_variant = model_variant
        self.model_size = (model_size or "base").lower()  # Default to base if None
        self.use_naturelm = use_naturelm
        self.fine_tuned = fine_tuned
        self._return_features_only = return_features_only

        # Determine output dimension based on model size
        self._output_dim = BEATS_OUTPUT_DIMS.get(self.model_size, 768)

        # ------------------------------------------------------------------
        # Build the BEATs backbone based on variant
        # ------------------------------------------------------------------
        if model_variant == "openbeats" or model_id is not None:
            # OpenBEATs: Use size configs and load from HuggingFace or local
            self._init_openbeats(
                pretrained=pretrained,
                model_id=model_id,
                checkpoint_path=checkpoint_path,
                device=device,
            )
        else:
            # Original BEATs: Load from GCS checkpoints
            self._init_beats(
                pretrained=pretrained,
                fine_tuned=fine_tuned,
                use_naturelm=use_naturelm,
                device=device,
            )

        # ------------------------------------------------------------------
        # Optional classifier for supervised training
        # ------------------------------------------------------------------
        if not return_features_only:
            self.classifier = nn.Linear(self._output_dim, num_classes)
        else:
            self.register_module("classifier", None)

    def _init_beats(
        self,
        pretrained: bool,
        fine_tuned: bool,
        use_naturelm: bool,
        device: str,
    ) -> None:
        """Initialize original BEATs model from GCS checkpoints."""
        if fine_tuned:
            beats_checkpoint_path = BEATS_PRETRAINED_PATH_FT
        else:
            beats_checkpoint_path = BEATS_PRETRAINED_PATH_SSL

        beats_ckpt = universal_torch_load(
            beats_checkpoint_path, cache_mode="use", map_location="cpu"
        )
        beats_cfg = BEATsConfig(beats_ckpt["cfg"])
        logger.info(f"BEATs Config: {beats_cfg.__dict__}")

        if use_naturelm:
            # BEATs-NatureLM has no config, load from regular ckpt first
            state_dict = universal_torch_load(
                BEATS_PRETRAINED_PATH_NATURELM, map_location="cpu"
            )
        else:
            state_dict = beats_ckpt["model"]

        self.backbone = BEATs(beats_cfg)
        self.backbone.to(device)
        self.backbone.load_state_dict(state_dict, strict=False)

        # Update output dim based on loaded config
        self._output_dim = beats_cfg.encoder_embed_dim

    def _init_openbeats(
        self,
        pretrained: bool,
        model_id: Optional[str],
        checkpoint_path: Optional[str],
        device: str,
    ) -> None:
        """Initialize OpenBEATs model from HuggingFace or local checkpoint."""
        # Get config for model size
        if self.model_size not in BEATS_SIZE_CONFIGS:
            raise ValueError(
                f"Unknown model size: {self.model_size}. Supported: {list(BEATS_SIZE_CONFIGS.keys())}"
            )

        config_dict = BEATS_SIZE_CONFIGS[self.model_size].copy()
        beats_cfg = BEATsConfig(config_dict)
        logger.info(f"OpenBEATs Config ({self.model_size}): {beats_cfg.__dict__}")

        # Build backbone
        self.backbone = BEATs(beats_cfg)
        self.backbone.to(device)

        # Load weights
        if checkpoint_path is not None:
            # Load from local checkpoint
            logger.info(f"Loading BEATs from local checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Handle prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("encoder."):
                    new_key = key[8:]
                elif key.startswith("model.encoder."):
                    new_key = key[14:]
                else:
                    new_key = key
                new_state_dict[new_key] = value

            load_info = self.backbone.load_state_dict(new_state_dict, strict=False)
            if load_info.missing_keys or load_info.unexpected_keys:
                logger.debug(
                    f"Loaded checkpoint. "
                    f"Missing keys: {len(load_info.missing_keys)}, "
                    f"Unexpected keys: {len(load_info.unexpected_keys)}"
                )

        elif pretrained or model_id is not None:
            # Default model ID if pretrained but no specific model specified
            if model_id is None:
                model_id = f"openbeats-{self.model_size}-i3"

            state_dict = load_weights_from_huggingface(model_id)
            load_info = self.backbone.load_state_dict(state_dict, strict=False)
            if load_info.missing_keys or load_info.unexpected_keys:
                logger.debug(
                    f"Loaded from HuggingFace. "
                    f"Missing keys: {len(load_info.missing_keys)}, "
                    f"Unexpected keys: {len(load_info.unexpected_keys)}"
                )

        # Update output dim
        self._output_dim = beats_cfg.encoder_embed_dim

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

            logger.info(
                f"Discovered {len(self._layer_names)} embedding layers in BEATs: {self._layer_names}"
            )

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

            logger.info(
                f"Discovered {len(self._layer_names)} embedding layers in BEATs: {self._layer_names}"
            )

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
            • Otherwise: pooled embeddings of shape
              ``(batch, encoder_embed_dim)``
        """
        # Optional audio pre-processing
        x = self.process_audio(x)

        features, frame_padding = self.backbone(
            x, padding_mask, disable_layerdrop=self.disable_layerdrop
        )

        # features: (B, T', D)
        # frame_padding: (B, T') or None

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

        if self._return_features_only:
            return pooled
        else:
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

            logger.debug(
                f"Forward pass completed. Hook outputs: {list(self._hook_outputs.keys())}"
            )

            # Collect embeddings from hook outputs
            embeddings = []
            for layer_name in self._hook_outputs.keys():
                embedding = self._hook_outputs[layer_name]
                embeddings.append(embedding)
                logger.debug(f"Found embedding for {layer_name}: {embedding.shape}")

            logger.debug(f"Collected {len(embeddings)} embeddings")

            # Check if we got any embeddings
            if not embeddings:
                raise ValueError(
                    f"No layers found matching: {self._hook_outputs.keys()}"
                )

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
                            embeddings[i] = torch.max(embeddings[i], dim=1)[
                                0
                            ]  # max returns (values, indices)
                        elif aggregation == "cls_token":
                            embeddings[i] = embeddings[i][:, 0, :]
                        else:
                            raise ValueError(
                                f"Unsupported aggregation method: {aggregation}"
                            )
                    else:
                        raise ValueError(
                            f"Unexpected embedding dimension: {embeddings[i].dim()}. Expected 2 or 3."
                        )

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
