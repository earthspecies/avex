"""OpenBEATs model implementation for audio representation learning.

This module provides OpenBEATs model wrapper for integration with the
representation learning framework.

Based on:
- Paper: https://arxiv.org/abs/2507.14129 (OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder)
- HuggingFace Collection: https://huggingface.co/collections/shikhar7ssu/openbeats
"""

import logging
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase
from representation_learning.models.openbeats.openbeats import (
    OPENBEATS_BASE_CONFIG,
    OPENBEATS_GIANT_CONFIG,
    OPENBEATS_LARGE_CONFIG,
    OPENBEATS_TITAN_CONFIG,
    OpenBEATs,
    OpenBEATsConfig,
)

logger = logging.getLogger(__name__)

# Default HuggingFace model IDs for OpenBEATs
OPENBEATS_HF_MODELS = {
    "openbeats-large-i1": "shikhar7ssu/OpenBEATs-Large-i1",
    "openbeats-large-i2": "shikhar7ssu/OpenBEATs-Large-i2",
    "openbeats-large-i3": "shikhar7ssu/OpenBEATs-Large-i3",
    "openbeats-base-i1": "shikhar7ssu/OpenBEATs-Base-i1",
    "openbeats-base-i2": "shikhar7ssu/OpenBEATs-Base-i2",
    "openbeats-base-i3": "shikhar7ssu/OpenBEATs-Base-i3",
}

# ESPnet checkpoint path within the HuggingFace repository
ESPNET_CHECKPOINT_PATH = "work/nvme/bbjs/sbharadwaj/7Msounds/exp"


def find_checkpoint_path(repo_dir: str) -> Optional[str]:
    """Find the checkpoint file within the downloaded repository.

    Args:
        repo_dir: Path to the downloaded repository

    Returns:
        Path to the checkpoint file if found, None otherwise
    """
    # Common checkpoint patterns
    checkpoint_patterns = [
        "epoch_latest.pt",
        "valid.acc.best.pth",
        "model.pth",
    ]

    # Walk through the directory to find checkpoint files
    for root, dirs, files in os.walk(repo_dir):
        for pattern in checkpoint_patterns:
            if pattern in files:
                return os.path.join(root, pattern)

    return None


def load_openbeats_from_huggingface(
    model_id: str,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> tuple:
    """Load OpenBEATs model from HuggingFace.

    Args:
        model_id: HuggingFace model ID or path
        device: Device to load the model on
        cache_dir: Optional cache directory for downloaded files

    Returns:
        Tuple of (state_dict, config)
    """
    # Map short names to full model IDs
    if model_id.lower() in OPENBEATS_HF_MODELS:
        model_id = OPENBEATS_HF_MODELS[model_id.lower()]

    logger.info(f"Loading OpenBEATs from HuggingFace: {model_id}")

    # Determine model size from the model ID
    is_large = "large" in model_id.lower()
    base_config = OPENBEATS_LARGE_CONFIG if is_large else OPENBEATS_BASE_CONFIG

    try:
        # Try to download the checkpoint file
        # ESPnet models store checkpoints in a nested directory structure
        checkpoint_path = None

        # Try different potential checkpoint locations
        potential_paths = [
            # Direct checkpoint
            "epoch_latest.pt",
            "model.pth",
            # Nested ESPnet structure (simplified)
            "work/exp/epoch_latest.pt",
        ]

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
            # If direct paths don't work, try to find the checkpoint by listing files
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

        # ESPnet checkpoints have different structures
        if isinstance(checkpoint, dict):
            # Try different key patterns
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "encoder" in checkpoint:
                # Some ESPnet checkpoints store encoder separately
                state_dict = checkpoint
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Handle ESPnet encoder prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove common prefixes from ESPnet
            if key.startswith("encoder."):
                new_key = key[8:]  # Remove "encoder."
            elif key.startswith("model.encoder."):
                new_key = key[14:]  # Remove "model.encoder."
            else:
                new_key = key
            new_state_dict[new_key] = value

        return new_state_dict, base_config

    except Exception as e:
        logger.error(f"Error loading from HuggingFace: {e}")
        raise


class Model(ModelBase):
    """Wrapper that adapts the OpenBEATs backbone for the training loop.

    This module follows the same conventions as other model wrappers
    (e.g. ``beats_model.py``) so that it can be selected via
    ``representation_learning.models.get_model.get_model``.

    The underlying OpenBEATs implementation operates directly on raw‐waveform
    inputs. We therefore do *not* apply the optional :class:`AudioProcessor`
    from :pymeth:`ModelBase.process_audio` unless an ``audio_config`` is
    explicitly supplied.

    Notes
    -----
    1.  OpenBEATs extracts a sequence of frame-level embeddings with dimension
        ``cfg.encoder_embed_dim`` (default: ``768`` for base, ``1024`` for large).
        We convert this variable-length sequence into a fixed-dimensional vector
        via masked mean-pooling before feeding it to a linear classifier.
    2.  When ``return_features_only=True`` the classifier layer is skipped and
        the pooled embedding is returned directly.
    """

    def __init__(
        self,
        *,
        num_classes: Optional[int] = None,
        pretrained: bool = False,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        model_id: Optional[str] = None,
        model_size: str = "large",
        use_flash_attn: bool = False,
        disable_layerdrop: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Initialize OpenBEATs model.

        Args:
            num_classes: Number of output classes (required if return_features_only=False)
            pretrained: Whether to load pretrained weights from HuggingFace
            device: Device to run on ('cuda' or 'cpu')
            audio_config: Optional audio processing configuration
            return_features_only: If True, return embeddings instead of class predictions
            model_id: HuggingFace model ID (e.g., 'openbeats-large-i2' or full path)
            model_size: Model size ('base', 'large', 'giant', or 'titan'), used when model_id is not specified.
                        Giant (~1B params) and Titan (~1.9B params) are new in OpenBEATs.
            use_flash_attn: Whether to use flash attention (requires PyTorch >= 2.0)
            disable_layerdrop: Whether to disable layer dropout during forward pass
            checkpoint_path: Optional local path to a checkpoint file
        """
        super().__init__(device=device, audio_config=audio_config)

        # Validate num_classes
        if not return_features_only and num_classes is None:
            raise ValueError("num_classes must be provided when return_features_only=False")

        self.disable_layerdrop = disable_layerdrop
        self._return_features_only = return_features_only
        self.model_size = model_size

        # Determine config based on model size
        # OpenBEATs supports: base (~90M), large (~317M), giant (~1B), titan (~1.9B)
        model_size_lower = model_size.lower()
        if model_size_lower == "titan":
            base_config = OPENBEATS_TITAN_CONFIG.copy()
            self._output_dim = 1664
        elif model_size_lower == "giant":
            base_config = OPENBEATS_GIANT_CONFIG.copy()
            self._output_dim = 1408
        elif model_size_lower == "large":
            base_config = OPENBEATS_LARGE_CONFIG.copy()
            self._output_dim = 1024
        else:  # base or any other value defaults to base
            base_config = OPENBEATS_BASE_CONFIG.copy()
            self._output_dim = 768

        # Add flash attention setting
        base_config["use_flash_attn"] = use_flash_attn

        # Build the backbone
        openbeats_cfg = OpenBEATsConfig(base_config)
        self.backbone = OpenBEATs(openbeats_cfg)
        self.backbone.to(device)

        # Load pretrained weights
        if pretrained or model_id is not None or checkpoint_path is not None:
            if checkpoint_path is not None:
                # Load from local checkpoint
                logger.info(f"Loading OpenBEATs from local checkpoint: {checkpoint_path}")
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
                logger.info(f"Loaded checkpoint. Missing keys: {load_info.missing_keys}")
                logger.info(f"Unexpected keys: {load_info.unexpected_keys}")

            elif model_id is not None or pretrained:
                # Default model ID if pretrained but no specific model specified
                if model_id is None:
                    model_id = f"openbeats-{model_size.lower()}-i2"

                state_dict, loaded_config = load_openbeats_from_huggingface(model_id, device)
                load_info = self.backbone.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded from HuggingFace. Missing keys: {load_info.missing_keys}")
                logger.info(f"Unexpected keys: {load_info.unexpected_keys}")

        # Optional classifier for supervised training
        if not return_features_only:
            self.classifier = nn.Linear(self._output_dim, num_classes)
        else:
            self.register_module("classifier", None)

    def _discover_linear_layers(self) -> None:
        """Discover and cache only the OpenBEATs layers useful for embeddings."""
        if len(self._layer_names) == 0:
            self._layer_names = []

            for name, _module in self.named_modules():
                # Keep the initial projection after conv frontend
                if name.endswith("post_extract_proj"):
                    self._layer_names.append(name)

                # Keep only the fc2 layers from transformer encoder blocks
                elif name.endswith(".fc2") and "backbone.encoder.layers." in name:
                    self._layer_names.append(name)

            logger.info(f"Discovered {len(self._layer_names)} embedding layers in OpenBEATs: {self._layer_names}")

    def _discover_embedding_layers(self) -> None:
        """Discover and cache only the OpenBEATs layers useful for embeddings."""
        if len(self._layer_names) == 0:
            self._layer_names = []

            for name, _module in self.named_modules():
                # Keep only the fc2 layers from transformer encoder blocks
                if name.endswith(".fc2") and "backbone.encoder.layers." in name:
                    if name not in self._layer_names:
                        self._layer_names.append(name)

            logger.info(f"Discovered {len(self._layer_names)} embedding layers in OpenBEATs: {self._layer_names}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw audio waveform with shape ``(batch, time)``.
        padding_mask : torch.Tensor, optional
            Boolean mask where *True* denotes padding elements.

        Returns
        -------
        torch.Tensor
            - When *return_features_only* is **False**: logits of shape
              ``(batch, num_classes)``
            - Otherwise: pooled embeddings of shape ``(batch, encoder_embed_dim)``
        """
        # Optional audio pre-processing
        x = self.process_audio(x)

        features, frame_padding = self.backbone(x, padding_mask, disable_layerdrop=self.disable_layerdrop)

        # Masked mean-pooling over the temporal dimension
        if frame_padding is not None and frame_padding.any():
            masked_features = features.clone()
            masked_features[frame_padding] = 0.0
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
        """Extract embeddings from all registered hooks in the OpenBEATs model.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav'
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input
        aggregation : str
            Aggregation method for multiple layers ('mean', 'max', 'cls_token', 'none')
        freeze_backbone : bool
            Whether to freeze the backbone and use torch.no_grad()

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Model embeddings
        """
        # Validate input
        if x is None:
            raise ValueError("Input tensor cannot be None")

        if isinstance(x, dict):
            wav = x["raw_wav"]
        else:
            wav = x

        if wav.numel() == 0 or wav.shape[-1] == 0:
            raise ValueError("Audio tensor cannot be empty")

        if not self._hooks:
            raise ValueError("No hooks are registered in the model.")

        was_training = self.training
        mode_changed = False

        if freeze_backbone:
            if self.training:
                self.eval()
                mode_changed = True
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        else:
            if self.disable_layerdrop and self.training:
                self.eval()
                mode_changed = True

        try:
            self._clear_hook_outputs()

            if isinstance(x, dict):
                wav = x["raw_wav"]
                mask = x.get("padding_mask")
                expected_batch_size = wav.shape[0]
            else:
                wav = x
                mask = padding_mask
                expected_batch_size = wav.shape[0]

            if freeze_backbone:
                with torch.no_grad():
                    self.forward(wav, mask)
            else:
                self.forward(wav, mask)

            embeddings = []
            for layer_name in self._hook_outputs.keys():
                embedding = self._hook_outputs[layer_name]
                embeddings.append(embedding)

            if not embeddings:
                raise ValueError(f"No layers found matching: {self._hook_outputs.keys()}")

            # Ensure batch-first format
            for i in range(len(embeddings)):
                if embeddings[i].shape[0] != expected_batch_size:
                    embeddings[i] = embeddings[i].transpose(0, 1)

            if aggregation == "none":
                if len(embeddings) == 1:
                    return embeddings[0]
                else:
                    return embeddings
            else:
                for i in range(len(embeddings)):
                    if embeddings[i].dim() == 2:
                        pass
                    elif embeddings[i].dim() == 3:
                        if aggregation == "mean":
                            embeddings[i] = torch.mean(embeddings[i], dim=1)
                        elif aggregation == "max":
                            embeddings[i] = torch.max(embeddings[i], dim=1)[0]
                        elif aggregation == "cls_token":
                            embeddings[i] = embeddings[i][:, 0, :]
                        else:
                            raise ValueError(f"Unsupported aggregation method: {aggregation}")
                    else:
                        raise ValueError(f"Unexpected embedding dimension: {embeddings[i].dim()}")

                if len(embeddings) == 1:
                    return embeddings[0]
                else:
                    return torch.cat(embeddings, dim=1)
        finally:
            self._clear_hook_outputs()
            if mode_changed and was_training:
                self.train()

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio through the audio processor if configured."""
        return super().process_audio(x)
