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
    def _rename_key(key: str) -> str:
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
    """LWrapper exposing HuggingFace EAT checkpoints.

    This class converts raw waveforms to
    **128-bin Mel FBanks** exactly like the original EAT pipeline and feeds
    `the resulting spectrogram image to the Data2Vec-multi backbone obtained
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
    return_features_only
        Force feature mode even when ``num_classes>0``.
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
        return_features_only: bool = True,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        self.pooling = pooling
        self.return_features_only = return_features_only or num_classes == 0

        # -------------------------------------------------------------- #
        #  Audio pre-processing – Mel FBanks identical to EAT reference  #
        # -------------------------------------------------------------- #
        self.audio_processor = EATAudioProcessor(
            sample_rate=16_000,
            target_length=target_length,
            n_mels=128,
        )

        # -------------------------------------------------------------- #
        #  Backbone: HuggingFace Data2Vec-multi                        #
        # -------------------------------------------------------------- #
        logger.info("Loading EAT backbone from '%s' …", model_name)
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        # load_fairseq_weights(self.backbone, "../EAT/EAT-base_epoch30_pt.pt")
        # load_fairseq_weights(self.backbone, "../EAT/multirun/2025-06-03/05-59-45/0/eat_animalspeak/checkpoint_last.pt")
        # load_fairseq_weights(self.backbone, "../EAT/multirun/2025-05-31/09-19-15/0/eat_animalspeak/checkpoint_last.pt")

        embed_dim = getattr(self.backbone.config, "hidden_size", 768)

        # Optional linear classifier for downstream tasks
        if self.return_features_only:
            self.classifier = None  # type: ignore[assignment]
            # self.register_module("classifier", None)  # satisfies mypy
        else:
            self.classifier = nn.Linear(embed_dim, num_classes).to(self.device)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                    #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # noqa: D401 – keep signature consistent
        """Forward pass.

        Parameters
        ----------
        x
            Raw waveform tensor of shape ``(B, T)``.
        padding_mask
            Not used (kept for interface compatibility).
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

        # 4) Pool patch embeddings → clip-level vector
        if self.pooling == "cls":
            pooled = feats[:, 0]
        elif self.pooling == "mean":
            pooled = feats.mean(dim=1)
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")

        # 5) Optional classification head
        if self.return_features_only:
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
        """Return a clip-level embedding (CLS or mean-pooled)."""
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


# Public alias for consistency with other model modules
Model = EATHFModel
