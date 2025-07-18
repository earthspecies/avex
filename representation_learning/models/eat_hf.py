from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional
import torch.nn as nn
import torch
from transformers import AutoModel

from representation_learning.models.base_model import ModelBase
from representation_learning.models.eat.audio_processor import EATAudioProcessor

logger = logging.getLogger(__name__)

# Removed hardcoded VARIANT - now configurable via parameters


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
        # load_fairseq_weights(self.backbone, "../EAT/EAT-base_epoch30_pt.pt")
        # load_fairseq_weights(
        #     self.backbone,
        #     # "../EAT/multirun/2025-06-04/05-29-23/0/eat_animalspeak/"
        #     # "checkpoint_22_920000.pt"
        #     "../EAT/multirun/2025-06-03/05-59-45/0/eat_animalspeak/
        #     checkpoint_last.pt",
        # )
        # load_fairseq_weights(
        #     self.backbone,
        #     "../EAT/multirun/2025-05-31/09-19-15/0/eat_animalspeak/checkpoint_last.pt"
        # )
        # load_fairseq_weights(
        #     self.backbone,
        #     "../EAT/multirun/2025-06-20/05-07-14/0/eat_animalspeak/checkpoint30.pt"
        # ) # 48khz
        # load_fairseq_weights(
        #     self.backbone,
        #     "../EAT/multirun/2025-07-04/07-50-52/0/eat_audioset/checkpoint15.pt"
        # ) # AudioSet
        # load_fairseq_weights(
        #     self.backbone,
        #     "../EAT/multirun/2025-07-08/14-54-53/0/eat_animalspeak/checkpoint15.pt"
        # ) # AnimalSpeak

        # Conditionally load fairseq weights if path is provided
        if fairseq_weights_path is not None:
            logger.info("Loading fairseq weights from '%s' …", fairseq_weights_path)
            load_fairseq_weights(self.backbone, fairseq_weights_path)
        
        self.classifier = nn.Linear(768, num_classes)

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
        layers: List[str],  # ignored – kept for API parity
        *,
        padding_mask: Optional[torch.Tensor] = None,
        pooling: str = "cls",
    ) -> torch.Tensor:  # type: ignore[override]
        """Return a clip-level embedding (CLS or mean-pooled).

        Args:
            x: Input tensor or dictionary containing 'raw_wav'
            layers: Layer names (ignored, kept for API compatibility)
            padding_mask: Optional padding mask (unused)
            pooling: Pooling method ("cls" or "mean")

        Returns:
            torch.Tensor: Clip-level embedding tensor
        """
        if isinstance(x, dict):
            wav = x["raw_wav"]
        else:
            wav = x

        prev_pooling = self.pooling
        self.pooling = pooling
        try:
            with torch.no_grad():
                emb = self.forward(wav, padding_mask, return_features_only=True)
        finally:
            self.pooling = prev_pooling
        return emb


# Public alias for consistency with other model modules
Model = EATHFModel
