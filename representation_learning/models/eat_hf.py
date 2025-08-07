from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from representation_learning.models.base_model import ModelBase
from representation_learning.models.eat.audio_processor import EATAudioProcessor

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

    alt_model = torch.load(weights_path)["model"]

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
        model returns features only.
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
        num_classes: int = 0,
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
            num_classes: Number of output classes (0 for feature extraction only)
            device: PyTorch device string
            audio_config: Audio configuration (ignored, kept for API compatibility)
            target_length: Required spectrogram length in time frames
            pooling: Pooling method ("cls" or "mean")
            fairseq_weights_path: Optional path to fairseq checkpoint
            norm_mean: Normalization mean for mel spectrograms
            norm_std: Normalization std for mel spectrograms
        """
        super().__init__(device=device, audio_config=audio_config)

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
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)

        # Conditionally load fairseq weights if path is provided
        if fairseq_weights_path is not None:
            logger.info("Loading fairseq weights from '%s' …", fairseq_weights_path)
            load_fairseq_weights(self.backbone, fairseq_weights_path)

        self.classifier = nn.Linear(768, num_classes)
        self.classifier = self.classifier.to(self.device)

        # -------------------------------------------------------------- #
        #  Pre-discover MLP layers for efficient hook management        #
        # -------------------------------------------------------------- #
        self._mlp_layer_names: List[str] = []
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear) and (
                "mlp.fc1" in name or "mlp.fc2" in name
            ):
                self._mlp_layer_names.append(name)
        logger.info(
            f"Discovered {len(self._mlp_layer_names)} MLP layers for hook management"
        )

        # For backward compatibility with tests
        self._mlp_layers = self._mlp_layer_names

    # ------------------------------------------------------------------ #
    #  Forward pass                                                    #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        framewise_embeddings: bool = False,
        return_features_only: bool = False,
    ) -> torch.Tensor:  # noqa: D401 – keep signature consistent
        """Forward pass through the EAT model.

        Parameters
        ----------
        x
            Raw waveform tensor of shape ``(B, T)``.
        padding_mask
            Not used (kept for interface compatibility).
        framewise_embeddings
            If True, return frame-wise embeddings instead of pooled features.
        return_features_only
            If True, return features instead of classification logits.
            Defaults to False, but automatically True if num_classes=0.

        Returns
        -------
        torch.Tensor
            Either pooled feature embeddings or classification logits

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
        # backbone_out: Dict[str, torch.Tensor] = self.backbone(
        #     spec, mask=False, features_only=True
        # )  # type: ignore[arg-type]
        # feats: torch.Tensor = backbone_out["x"]  # (B, L, D)
        feats = self.backbone.extract_features(spec)
        if framewise_embeddings:
            return feats[:, 1:]  # drop the cls embedding

        # 4) Pool patch embeddings → clip-level vector
        if self.pooling == "cls":
            pooled = feats[:, 0]
        elif self.pooling == "mean":
            pooled = feats.mean(dim=1)
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")

        # 5) Optional classification head
        # Return features if explicitly requested or if no classifier exists
        if return_features_only or self.classifier is None:
            return pooled
        return self.classifier(pooled)

    # ------------------------------------------------------------------ #
    #  Embedding extractor                                              #
    # ------------------------------------------------------------------ #
    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        layers: List[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        pooling: str = "cls",
        average_over_time: bool = True,
    ) -> torch.Tensor:  # type: ignore[override]
        """Extract embeddings from specified layers of the EAT model.

        Args:
            x: Input tensor or dictionary containing 'raw_wav'
            layers: List of layer names to extract embeddings from. If 'all' is
                   included, MLP layers (fc1 and fc2 outputs) will be used as they
                   provide rich intermediate representations from the transformer
                   blocks.
            padding_mask: Optional padding mask (unused)
            pooling: Pooling method ("cls" or "mean")
            average_over_time: Whether to average embeddings over time dimension

        Returns:
            torch.Tensor: Concatenated embeddings from the requested layers

        Raises:
            ValueError: If none of the supplied layers are found in the model
        """
        # Handle empty layers list - return main features
        if not layers:
            if isinstance(x, dict):
                wav = x["raw_wav"]
            else:
                wav = x

            prev_pooling = self.pooling
            self.pooling = pooling

            # Temporarily set return_features_only to get main features
            prev_return_features_only = self.return_features_only
            self.return_features_only = True

            with torch.no_grad():
                emb = self.forward(wav, padding_mask, return_features_only=True)

            # Restore original settings
            self.pooling = prev_pooling
            self.return_features_only = prev_return_features_only
            return emb

        # Clear previous hook outputs
        self._clear_hook_outputs()

        # Handle 'all' case - use MLP layers (fc1 and fc2 outputs) for rich
        # representations
        # If 'all' is not in layers, use the exact layers specified
        target_layers = layers.copy()
        if "all" in layers:
            logger.info(
                "'all' specified in layers, using pre-discovered MLP layers "
                "for EAT model..."
            )

            if self._mlp_layer_names:
                logger.info(
                    f"Using {len(self._mlp_layer_names)} pre-discovered MLP layers"
                )
                # Get specific layers (excluding 'all')
                specific_layers = [layer for layer in layers if layer != "all"]
                # Combine specific layers with MLP layers, avoiding duplicates
                all_layers = specific_layers + self._mlp_layer_names
                target_layers = list(
                    dict.fromkeys(all_layers)
                )  # Remove duplicates while preserving order
                logger.info(
                    f"Target layers after 'all' expansion: {len(target_layers)} layers"
                )
            else:
                logger.warning("No MLP layers found in EAT model")
                # Fallback to classifier if available
                if self.classifier is not None:
                    target_layers = [layer for layer in layers if layer != "all"] + [
                        "classifier"
                    ]
                else:
                    # For features_only mode, return main features when no MLP
                    # layers found
                    if self.return_features_only:
                        if isinstance(x, dict):
                            wav = x["raw_wav"]
                        else:
                            wav = x

                        prev_pooling = self.pooling
                        self.pooling = pooling
                        with torch.no_grad():
                            emb = self.forward(wav, padding_mask)
                        self.pooling = prev_pooling
                        return emb
                    else:
                        target_layers = [layer for layer in layers if layer != "all"]

        # Register hooks for requested layers (only if not already registered)
        self._register_hooks_for_layers(target_layers)

        try:
            # Process input
            if isinstance(x, dict):
                wav = x["raw_wav"]
            else:
                wav = x

            # Store original pooling method
            prev_pooling = self.pooling
            self.pooling = pooling

            logger.debug(f"Starting forward pass with target layers: {target_layers}")

            # Forward pass to trigger hooks
            with torch.no_grad():
                self.forward(wav, padding_mask)

            # Restore original pooling method
            self.pooling = prev_pooling

            logger.debug(
                f"Forward pass completed. Hook outputs: "
                f"{list(self._hook_outputs.keys())}"
            )

            # Collect embeddings from hook outputs
            embeddings = []
            logger.debug(
                f"Collecting embeddings from {len(target_layers)} target layers"
            )
            for layer_name in target_layers:
                if layer_name in self._hook_outputs:
                    embeddings.append(self._hook_outputs[layer_name])
                    logger.debug(
                        f"Found embedding for {layer_name}: "
                        f"{self._hook_outputs[layer_name].shape}"
                    )
                else:
                    logger.warning(f"No output captured for layer: {layer_name}")

            logger.debug(f"Collected {len(embeddings)} embeddings")

            # Check if we got any embeddings
            if not embeddings:
                raise ValueError(f"No layers found matching: {target_layers}")

            if average_over_time:
                result = []
                for emb in embeddings:
                    if emb.dim() == 2:
                        # Already in correct shape, just append
                        result.append(emb)
                    elif emb.dim() == 3:
                        aggregated = torch.mean(emb, dim=1)
                        result.append(aggregated)
                    else:
                        raise ValueError(
                            f"Unexpected embedding dimension: {emb.dim()}. "
                            f"Expected 2 or 3."
                        )
                return torch.cat(result, dim=1)
            else:
                return embeddings

        finally:
            # Clear hook outputs for next call
            self._clear_hook_outputs()


# Public alias for consistency with other model modules
Model = EATHFModel
