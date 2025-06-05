"""
Bird-AVES wrapper: BirdAVES backbone + optional classifier head.

This file provides two classes:

* **AvesEmbedding** – a lightweight wrapper around the BirdAVES W2V-2.0 encoder
  that returns per-frame embeddings (and an updated padding mask).

* **Model** – the high-level class that plugs this backbone into the ESP
  training pipeline (`ModelBase`).  It can operate either in
  *feature-extraction* mode (returning frame-level features) or with a linear
  classification head.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import wav2vec2_model

from representation_learning.models.base_model import ModelBase

CFG_PATH = "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.model_config.json"


# --------------------------------------------------------------------------- #
#  Low-level backbone
# --------------------------------------------------------------------------- #
class AvesEmbedding(nn.Module):
    """Light wrapper around a BirdAVES Wav2Vec 2.0 encoder.

    Parameters
    ----------
    sr : int
        Input sample-rate expected by downstream preprocessing.
    large : bool, default ``False``
        Reserved for potential larger backbones (currently unused).
    """

    def __init__(self, sr: int, large: bool = False) -> None:  # noqa: D401
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        config = self._load_config(CFG_PATH)
        self.model = wav2vec2_model(**config, aux_num_out=None)

        state_dict = torch.hub.load_state_dict_from_url(
            "https://storage.googleapis.com/esp-public-files/"
            "birdaves/birdaves-biox-base.torchaudio.pt",
            map_location=device,
        )
        self.model.load_state_dict(state_dict)

        # Allow fine-tuning of feature extractor by default.
        self.model.feature_extractor.requires_grad_(True)

        self.sr: int = sr
        self.large: bool = large

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load the JSON config used to instantiate the W2V-2.0 backbone.

        The helper seamlessly supports both local file paths and remote URLs.

        Parameters
        ----------
        config_path : str
            Either a local filesystem path or an HTTP(S) URL pointing to the
            *torchaudio* JSON configuration file.

        Returns
        -------
        Dict[str, Any]
            Parsed configuration dictionary suitable for
            ``torchaudio.models.wav2vec2_model``.
        """

        # ------------------------------------------------------------------ #
        #  Remote (HTTP[S]) path
        # ------------------------------------------------------------------ #
        if config_path.startswith(("http://", "https://")):
            import requests  # Delayed import to avoid unnecessary dependency at load-time

            response = requests.get(config_path, timeout=30)
            response.raise_for_status()
            return json.loads(response.text)

        # ------------------------------------------------------------------ #
        #  Local path
        # ------------------------------------------------------------------ #
        with open(config_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def forward(
        self, sig: torch.Tensor, padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract per-frame embeddings.

        Parameters
        ----------
        sig : torch.Tensor
            Raw waveform tensor of shape ``(B, T)``.
        padding_mask : torch.Tensor
            Boolean tensor of shape ``(B, T)`` where **True** denotes padded
            (invalid) samples.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            **features**
                Tensor of shape ``(B, T', C)`` – output of the final W2V layer.
            **new_mask**
                Boolean mask of shape ``(B, T')`` where **True** denotes padded
                frames after striding / pooling inside the encoder.
        """
        # Torchaudio returns a list of layer outputs – pick the last one.
        features = self.model.extract_features(sig.float())[0][-1]

        # Down-sample the original sample-level padding mask to frame level.
        frame_mask = ~padding_mask.bool().unsqueeze(1).float()
        frame_mask = F.max_pool1d(frame_mask, kernel_size=320, stride=320) > 0
        new_mask = ~frame_mask.squeeze(1)

        return features, new_mask

    # ------------------------------------------------------------------ #
    #  (Un)freezing convenience
    # ------------------------------------------------------------------ #
    def freeze(self) -> None:
        """Freeze encoder parameters (useful for linear-probe training)."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.model.feature_extractor.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unfreeze encoder parameters to allow fine-tuning."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        self.model.feature_extractor.requires_grad_(True)


# --------------------------------------------------------------------------- #
#  High-level wrapper compatible with the ESP training pipeline
# --------------------------------------------------------------------------- #
class Model(ModelBase):  # pylint: disable=abstract-method
    """BirdAVES backbone wrapped in the generic ESP ``ModelBase`` interface.

    Parameters
    ----------
    num_classes : int
        Number of target classes.  Ignored when *return_features_only* is
        ``True``.
    pretrained : bool, default ``True``
        Present for API parity – BirdAVES weights are always downloaded.
    device : str, default ``"cuda"``
        Device on which to run the model.
    audio_config : Optional[Dict[str, Any]]
        Audio-preprocessing parameters.  Must include *sample_rate*; if omitted
        defaults to 16 kHz (standard for BirdAVES).
    return_features_only : bool, default ``False``
        When ``True`` the forward pass returns frame-level embeddings without
        applying the classifier head.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,  # kept for API parity
        device: str = "cuda",
        audio_config: Optional[Dict[str, Any]] = None,
        return_features_only: bool = False,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # ------------------------------------------------------------------ #
        #  Backbone
        # ------------------------------------------------------------------ #
        sr: int = (
            (audio_config or {}).get("sample_rate", 16000)
            if isinstance(audio_config, dict)
            else getattr(audio_config, "sample_rate", 16000)
        )
        self.backbone = AvesEmbedding(sr=sr).to(device)

        # Hidden size is defined in the backbone's encoder config
        hidden_dim: int = getattr(self.backbone.model.encoder, "embed_dim", 768)

        self.return_features_only: bool = return_features_only
        if not self.return_features_only:
            self.classifier = nn.Linear(hidden_dim, num_classes).to(device)

    # ------------------------------------------------------------------ #
    #  Utility functions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool a sequence while ignoring padded frames.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, T, C)``.
        mask : torch.Tensor
            Boolean tensor of shape ``(B, T)`` where **True** denotes frames to
            *keep* (i.e. non-padded).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, C)`` – mean of the unmasked frames.
        """
        mask = mask.unsqueeze(-1).type_as(x)  # (B, T, 1)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # noqa: D401
        """Forward pass through backbone (and optional classifier).

        Parameters
        ----------
        x : torch.Tensor
            Raw waveform tensor of shape ``(B, T)``.
        padding_mask : Optional[torch.Tensor], default ``None``
            Boolean tensor of shape ``(B, T)`` where **True** denotes padding.
            If ``None``, a zero mask is created internally.

        Returns
        -------
        torch.Tensor
            * If ``return_features_only=True`` – per-frame features of shape
              ``(B, T', C)``.
            * Otherwise – per-frame logits of shape ``(B, T', num_classes)``.
        """
        x = self.process_audio(x)

        if padding_mask is None:
            padding_mask = torch.zeros(
                x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
            )

        feats, frame_mask = self.backbone(x, padding_mask)

        if self.return_features_only:
            return feats  # (B, T', C)

        return self.classifier(feats)  # (B, T', num_classes)

    # ------------------------------------------------------------------ #
    #  encode_audio alias (required by parts of the codebase)
    # ------------------------------------------------------------------ #
    def encode_audio(
        self, audio: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return a *pooled* AVES embedding (no classifier head).

        Parameters
        ----------
        audio : torch.Tensor
            Raw waveform tensor of shape ``(B, T)``.
        padding_mask : torch.Tensor
            Boolean tensor of shape ``(B, T)`` where **True** denotes padding.

        Returns
        -------
        torch.Tensor
            Clip-level embedding of shape ``(B, C)``.
        """
        feats, frame_mask = self.backbone(audio, padding_mask)
        return self._masked_mean(feats, ~frame_mask)

    # ------------------------------------------------------------------ #
    #  Override ModelBase's audio preprocessing
    # ------------------------------------------------------------------ #
    def process_audio(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return waveform unchanged (AVES consumes raw audio).

        Parameters
        ----------
        x : torch.Tensor
            Raw waveform tensor.

        Returns
        -------
        torch.Tensor
            The same tensor placed on the model's device.
        """
        return x.to(next(self.parameters()).device)
