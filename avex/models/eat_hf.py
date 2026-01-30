"""EAT (Data2Vec) HuggingFace model implementation.

This module provides HuggingFace-compatible EAT (Data2Vec) model implementation
for audio representation learning tasks.

.. warning::
    **Transformers 5.0.0+ Compatibility Issue**

    The upstream EAT model (worstchan/EAT-base_epoch30_pretrain) is not yet compatible
    with transformers >= 5.0.0. The remote model's `EATModel` class is missing the
    `_tied_weights_keys` class attribute required by the newer transformers library.

    Error: ``AttributeError: 'EATModel' object has no attribute 'all_tied_weights_keys'``

    **Workaround**: Pin transformers to version < 5.0.0 in your dependencies:
    ``transformers>=4.40.0,<5.0.0``

    This issue needs to be fixed in the upstream HuggingFace repository by adding
    ``_tied_weights_keys = []`` to the EATModel class in modeling_eat.py.

    See: https://huggingface.co/worstchan/EAT-base_epoch30_pretrain
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel

from avex.models.base_model import ModelBase
from avex.models.eat.audio_processor import (
    EATAudioProcessor,
)
from avex.utils.utils import universal_torch_load

logger = logging.getLogger(__name__)


def load_fairseq_weights(model: AutoModel, weights_path: str) -> None:
    """Load fairseq weights into HuggingFace model.

    Temporary function to load weights from fairseq checkpoint format
    into a HuggingFace model.

    Args:
        model: HuggingFace model to load weights into
        weights_path: Path to the fairseq checkpoint file
    """

    def _rename_key(key: str) -> str:
        """Rename fairseq keys to match HuggingFace naming convention.

        Args:
            key: Original fairseq key name

        Returns:
            str: Renamed key for HuggingFace model
        """
        if key == "modality_encoders.IMAGE.context_encoder.norm.weight":
            # return "model.fc_norm.weight"
            return "model.pre_norm.weight"
        if key == "modality_encoders.IMAGE.context_encoder.norm.bias":
            # return "model.fc_norm.bias"
            return "model.pre_norm.bias"
        img_prefix = "modality_encoders.IMAGE."
        if key.startswith(img_prefix):
            key = "model." + key[len(img_prefix) :]
        elif not key.startswith("model."):
            key = "model." + key
        return key

    alt_model = universal_torch_load(weights_path)["model"]

    hf_keys = set(model.state_dict().keys())
    mapped_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

    for k, v in alt_model.items():
        # Skip EMA / optimizer statistics, etc.
        if k.startswith("_ema"):
            continue
        new_k = _rename_key(k)
        if new_k in hf_keys:
            mapped_state_dict[new_k] = v
        else:
            print(f"[skip] {k:<70s} -> {new_k} (not in HF model)")

    # ------------------------------------------------------------------
    # Load the remapped weights
    # ------------------------------------------------------------------
    missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)

    if missing:
        print("\n[Warning] Missing keys after loading:")
        for k in missing:
            print("   ", k)
    if unexpected:
        print("\n[Warning] Unexpected keys after loading:")
        for k in unexpected:
            print("   ", k)


class EATHFModel(ModelBase):
    """Wrapper exposing HuggingFace EAT checkpoints.

    This class converts raw waveforms to
    **128-bin Mel FBanks** exactly like the original EAT pipeline and feeds
    the resulting spectrogram image to the Data2Vec-multi backbone obtained
    from :pyfunc:`transformers.AutoModel.from_pretrained`.

    Parameters
    ----------
    model_name
        HuggingFace repository ID or local path.  Defaults to the official
        pre-training checkpoint.
    num_classes
        If >0, a linear classifier is appended on top of the pooled backbone
        representation allowing end-to-end fine-tuning.  When set to 0 the
        model returns unpooled patch embeddings (shape: B x L x D).
    device
        PyTorch device string (e.g. ``"cuda"``).
    audio_config
        Kept for API parity with :class:`ModelBase` but ignored because we
        always employ the dedicated :class:`EATAudioProcessor` below.
    target_length
        Required spectrogram length (time frames).  The official checkpoints
        expect **1024**.
    pooling
        One of ``"cls"`` or ``"mean"`` determining how patch-level features
        are aggregated into a clip-level embedding.
    trust_remote_code
        Passed through to :pyfunc:`transformers.AutoModel.from_pretrained`.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        model_name: str = "worstchan/EAT-base_epoch30_pretrain",
        num_classes: Optional[int] = None,
        device: str = "cuda",
        audio_config: Optional[Dict[str, Any]] = None,
        target_length: int = 1024,
        pooling: str = "cls",
        fairseq_weights_path: Optional[str] = None,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        return_features_only: bool = False,
    ) -> None:
        """Initialize EATHFModel.

        Args:
            model_name: HuggingFace repository ID or local path
            num_classes: Number of output classes (None or 0 for feature extraction only)
            device: PyTorch device string
            audio_config: Audio configuration (ignored, kept for API compatibility)
            target_length: Required spectrogram length in time frames
            pooling: Pooling method ("cls" or "mean")
            fairseq_weights_path: Optional path to fairseq checkpoint
            norm_mean: Normalization mean for mel spectrograms
            norm_std: Normalization std for mel spectrograms

        Raises:
            ValueError: If num_classes is 0 or None when return_features_only=False
        """
        super().__init__(device=device, audio_config=audio_config)

        # Treat None the same as 0 (feature extraction only)
        if num_classes is None:
            num_classes = 0

        # Validate num_classes: required when return_features_only=False
        if not return_features_only and num_classes == 0:
            raise ValueError("num_classes must be > 0 when return_features_only=False")

        self.pooling = pooling
        self.num_classes = num_classes
        self.return_features_only = return_features_only
        self.audio_config = audio_config

        # -------------------------------------------------------------- #
        #  Audio pre-processing – Mel FBanks identical to EAT reference  #
        # -------------------------------------------------------------- #
        self.audio_processor = EATAudioProcessor(
            sample_rate=16_000,
            target_length=target_length,
            n_mels=128,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        # -------------------------------------------------------------- #
        #  Backbone: HuggingFace Data2Vec-multi                        #
        # -------------------------------------------------------------- #
        logger.info("Loading EAT backbone from '%s' …", model_name)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

        # Conditionally load fairseq weights if path is provided
        if fairseq_weights_path is not None:
            logger.info("Loading fairseq weights from '%s' …", fairseq_weights_path)
            load_fairseq_weights(self.backbone, fairseq_weights_path)

        # Only create classifier when return_features_only=False and num_classes > 0
        if not return_features_only and num_classes > 0:
            self.classifier = nn.Linear(768, num_classes)
            self.classifier = self.classifier.to(self.device)
        else:
            self.classifier = None

        # -------------------------------------------------------------- #
        #  Pre-discover MLP layers for efficient hook management        #
        # -------------------------------------------------------------- #
        # MLP layers will be discovered in _discover_linear_layers override

    def _discover_linear_layers(self) -> None:
        """Discover and cache only the EAT layers that are useful for embeddings.
        This method is called when target_layers=["all"] is used.
        Specifically:
        - backbone.model.blocks.{i}.mlp.fc2 (only fc2 layers from transformer blocks)
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            # Discover standard linear layers
            # for name, module in self.named_modules():
            #     if isinstance(module, torch.nn.Linear):
            #         self._layer_names.append(name)

            for name, _module in self.named_modules():
                # Keep only the fc2 layers from transformer blocks
                # Pattern: backbone.model.blocks.{i}.mlp.fc2
                if name.endswith("attn.proj") and "backbone.model.blocks." in name:
                    if name not in self._layer_names:
                        self._layer_names.append(name)

            logger.info(f"Discovered {len(self._layer_names)} embedding layers in EAT model: {self._layer_names}")

    def _discover_embedding_layers(self) -> None:
        """
        Discover and cache only the EAT layers that are useful for embeddings.
        Specifically:
        - backbone.model.blocks.{i}.mlp.fc2 (only fc2 layers from transformer blocks)
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            # Discover standard linear layers
            # for name, module in self.named_modules():
            #     if isinstance(module, torch.nn.Linear):
            #         self._layer_names.append(name)

            for name, _module in self.named_modules():
                # Keep only the fc2 layers from transformer blocks
                # Pattern: backbone.model.blocks.{i}.mlp.fc2
                if name.endswith("attn.proj") and "backbone.model.blocks." in name:
                    if name not in self._layer_names:
                        self._layer_names.append(name)
            logger.info(f"Discovered {len(self._layer_names)} embedding layers in EAT model: {self._layer_names}")

    # ------------------------------------------------------------------ #
    #  Forward pass                                                    #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # noqa: D401 – keep signature consistent
        """Forward pass through the EAT model.

        Parameters
        ----------
        x
            Raw waveform tensor of shape ``(B, T)``.
        padding_mask
            Not used (kept for interface compatibility).

        Returns
        -------
        torch.Tensor
            • When *return_features_only* is **True** (set at init): unpooled patch
              embeddings of shape ``(B, L, D)``
            • Otherwise: classification logits of shape ``(B, num_classes)``

        Raises
        ------
        ValueError
            If pooling method is not 'cls' or 'mean'
        """
        # 1) Waveform → Mel FBanks  (B, F, T)
        spec = self.process_audio(x)

        # 2) Add channel dimension expected by the EAT image encoder
        spec = spec.unsqueeze(1)  # (B, 1, F, T)

        # 3) Backbone – we only need features (classification handled below)
        feats = self.backbone.extract_features(spec)

        # Return unpooled features if set at init or if no classifier exists
        if self.return_features_only or self.classifier is None:
            return feats

        # 4) Pool patch embeddings → clip-level vector
        if self.pooling == "cls":
            pooled = feats[:, 0]
        elif self.pooling == "mean":
            pooled = feats.mean(dim=1)
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")

        # 5) Classification head
        return self.classifier(pooled)

    # ------------------------------------------------------------------ #
    #  Embedding extractor                                              #
    # ------------------------------------------------------------------ #
    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        pooling: str = "cls",
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # type: ignore[override]
        """Extract embeddings from all registered hooks in the EAT model.

        Args:
            x: Input tensor or dictionary containing 'raw_wav'
            padding_mask: Optional padding mask (unused)
            pooling: Pooling method ("cls" or "mean")
            aggregation: Aggregation method for multiple layers ('mean', 'max',
                'cls_token', 'none')
            freeze_backbone: Whether to freeze the backbone and use torch.no_grad()

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Model embeddings (tensor if
                aggregation!="none", list if False)

        Raises:
            ValueError: If no hooks are registered or no outputs are captured
        """
        # Ensure hooks are present (self-heal if they were cleared externally)
        self.ensure_hooks_registered()
        if not self._hooks:
            raise ValueError("No hooks are registered in the model.")

        # Clear previous hook outputs
        self._clear_hook_outputs()

        # Hooks are already registered in __init__ via base class

        try:
            # Process input
            if isinstance(x, dict):
                wav = x["raw_wav"]
                expected_batch_size = wav.shape[0]
            else:
                wav = x
                expected_batch_size = wav.shape[0]

            # Store original pooling method
            prev_pooling = self.pooling
            self.pooling = pooling

            # Forward pass to trigger hooks (conditionally use torch.no_grad based on
            # freeze_backbone)
            if freeze_backbone:
                with torch.no_grad():
                    self.forward(wav, padding_mask)
            else:
                self.forward(wav, padding_mask)

            # Restore original pooling method
            self.pooling = prev_pooling

            logger.debug(f"Forward pass completed. Hook outputs: {list(self._hook_outputs.keys())}")

            # Collect embeddings from hook outputs
            embeddings = []

            for layer_name in self._hook_outputs.keys():
                embeddings.append(self._hook_outputs[layer_name])
                logger.debug(f"Found embedding for {layer_name}: {self._hook_outputs[layer_name].shape}")

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


# Public alias for consistency with other model modules
Model = EATHFModel
