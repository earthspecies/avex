from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from typing import List

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase

from .eat import (
    D2vModalitiesConfig,
    Data2VecMultiConfig,
    Data2VecMultiModel,
    Modality,
)
from .image import D2vImageConfig

# Mask helpers
from representation_learning.data.audio_utils import (
    waveform_to_frame_mask,
    sync_crop_or_pad_time,
    frame_mask_to_patch_mask,
)

# ------------------------------------------------------------------ #
#  Local imports
# ------------------------------------------------------------------ #

from .audio_processor import EATAudioProcessor

logger = logging.getLogger(__name__)


class Model(ModelBase):
    """Audio-spectrogram classifier built on an EAT backbone."""

    # --------------------------------------------------------------------- #
    # Initialisation
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool | None = None,  # kept for API parity (currently unused)
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        embed_dim: int = 768,
        patch_size: int = 16,
        target_length: int = 1024,
        return_features_only: bool = False,
        enable_ema: bool = False,
        pretraining_mode: bool = False,
        skip_padding_logic: bool = True,
        eat_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Initialise the generic ESP model wrapper
        super().__init__(device=device, audio_config=audio_config)

        # ------------------------------------------------------------------ #
        #  Config & flags
        # ------------------------------------------------------------------ #
        self.return_features_only: bool = return_features_only
        self.pretraining_mode: bool = pretraining_mode
        self.skip_padding_logic: bool = skip_padding_logic

        # Auto-enable EMA teacher when in pre-training mode (unless user overrode)
        if self.pretraining_mode and not enable_ema:
            logger.warning(
                "Pre-training mode requested but enable_ema=False; "
                "automatically enabling EMA teacher."
            )
            enable_ema = True

        # ------------------------------------------------------------------ #
        #  Build EAT backbone
        # ------------------------------------------------------------------ #
        image_cfg = D2vImageConfig(
            in_chans=1,
            patch_size=patch_size,
            embed_dim=embed_dim,
            target_length=target_length,
            max_length=target_length,
        )
        modalities_cfg = D2vModalitiesConfig(image=image_cfg)
        eat_cfg_obj = Data2VecMultiConfig(
            embed_dim=embed_dim,
            modalities=modalities_cfg,
            supported_modality=Modality.IMAGE,
            ema_decay=0.999,
            skip_ema=not enable_ema,
        )

        # Apply YAML / dict overrides if supplied
        if eat_cfg:
            logger.info("Applying overrides to EAT config")
            self._apply_overrides(eat_cfg_obj, eat_cfg)

        logger.debug("Instantiating EAT backbone")
        self.backbone: Data2VecMultiModel = Data2VecMultiModel(
            eat_cfg_obj, modalities=[Modality.IMAGE]
        ).to(self.device)

        # Simple linear classifier for clip-level logits
        self.classifier: nn.Linear = nn.Linear(embed_dim, num_classes).to(self.device)

        # ------------------------------------------------------------------ #
        #  Audio pre-processing                                             #
        # ------------------------------------------------------------------ #

        self.audio_processor = EATAudioProcessor(
            sample_rate=16_000,
            target_length=target_length,
            n_mels=128,
        )

    # --------------------------------------------------------------------- #
    # Utility: pad or crop the time dimension to a fixed length
    # --------------------------------------------------------------------- #
    @staticmethod
    def _pad_or_crop_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Center-crop or zero-pad the time dimension of a spectrogram.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, F)``  (time, freq).
        target_len : int
            Desired number of time frames.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(B, target_len, F)``.
        """
        bsz, t, feat = x.shape
        if t == target_len:
            return x
        if t > target_len:  # centre crop
            start = (t - target_len) // 2
            return x[:, start : start + target_len, :]
        pad = x.new_zeros(bsz, target_len - t, feat)
        return torch.cat([x, pad], dim=1)

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:  # type: ignore[override]
        """Forward pass.

        * **Pre-training mode** (`pretraining_mode=True`) returns the full
          backbone loss/output dict.
        * **Fine-tuning / inference** mode returns either:
            - Per-clip logits (``return_features_only=False``), shape ``(B, C)``
            - Per-clip feature vectors (``return_features_only=True``),
              shape ``(B, D)``

        Parameters
        ----------
        x : torch.Tensor
            Raw waveform (B, T) – will be converted to spectrogram by
            ``ModelBase.process_audio``.
        padding_mask : Optional[torch.Tensor]
            Not used here; kept for interface compatibility.

        Returns
        -------
        Union[torch.Tensor, Dict[str, Any]]
            See description above.

        Raises
        ------
        RuntimeError
            If ``ModelBase.process_audio`` does not return a 3-D tensor with
            shape ``(B, F, T)``.

        """
        # 1) Spectrogram extraction via AudioProcessor in ModelBase
        spec = super().process_audio(x)
        if spec.dim() != 3:
            raise RuntimeError(
                "AudioProcessor must return a 3-D tensor with shape (B, F, T)"
            )

        # 2) Re-arrange to (B, T, F)
        # spec = spec.permute(0, 2, 1)  # -> (B, T, F)

        # ------------------------------------------------------------------ #
        #  Optional: propagate padding mask (waveform → frame → patch)        #
        # ------------------------------------------------------------------ #
        patch_mask = None

        # 3) Add channel dimension expected by ImageEncoder
        spec = spec.unsqueeze(1)  # (B, 1, T, F)

        # 4) Backbone
        if self.pretraining_mode:
            # Backbone handles masking & loss internally
            return self.backbone(
                spec,
                padding_mask=patch_mask,
                mask=True,
                features_only=False,
            )

        # Features-only path for fine-tuning / inference
        backbone_out: Dict[str, Any] = self.backbone(
            spec,
            padding_mask=patch_mask,
            mask=False,
            features_only=True,
        )
        features: torch.Tensor = backbone_out["x"].mean(dim=1)  # global average pool

        if self.return_features_only:
            return features  # (B, embed_dim)

        # 5) Classification head
        return self.classifier(features)  # (B, num_classes)

    # --------------------------------------------------------------------- #
    # Recursive override helper (dataclass / namespace traversal)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _apply_overrides(obj: Any, overrides: Dict[str, Any]) -> None:  # noqa: ANN401
        """Recursively set attributes on a *dataclass-like* object.

        Unknown keys are ignored to preserve forward compatibility.

        Parameters
        ----------
        obj : Any
            The object whose attributes should be updated.
        overrides : Dict[str, Any]
            Mapping of attribute names → new values.  Nested dictionaries are
            applied recursively.
        """
        for key, val in overrides.items():
            if not hasattr(obj, key):
                continue  # silently ignore unknown fields

            current_val = getattr(obj, key)

            # Descend into nested configs
            if isinstance(val, dict) and not isinstance(
                current_val, (int, float, str, bool, type(None))
            ):
                Model._apply_overrides(current_val, val)
                continue

            # Attempt to cast numeric strings to the original numeric type
            if isinstance(current_val, (int, float)) and isinstance(val, str):
                try:
                    casted_val = type(current_val)(float(val))
                    setattr(obj, key, casted_val)
                    continue
                except ValueError:
                    pass  # fall through and set raw string

            setattr(obj, key, val)

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        layers: List[str],  # currently unused for EAT
        *,
        padding_mask: Optional[torch.Tensor] = None,
        pooling: str = "cls",  # one of: "cls", "mean"
    ) -> torch.Tensor:
        """Return a clip-level embedding using *pooling* strategy.

        Parameters
        ----------
        x : Tensor | dict
            Either a raw waveform tensor (B, T) or a dict with ``{"raw_wav",
            "padding_mask"}``.
        layers : List[str]
            Ignored for the moment – kept for interface parity with other
            models.
        padding_mask : Tensor | None, optional
            Explicit mask when *x* is provided as a raw tensor.
        pooling : {"cls", "mean"}
            • ``"cls"`` – return the CLS token (default).
            • ``"mean"`` – mean-pool patch embeddings along the time axis.
        """
        # ------------------------------------------------------------------ #
        #  Flexible input handling                                           #
        # ------------------------------------------------------------------ #
        if isinstance(x, dict):
            wav = x["raw_wav"]
            padding_mask = x.get("padding_mask")
        else:
            wav = x  # type: ignore[assignment]

        # 1) Spectrogram extraction via AudioProcessor
        spec = super().process_audio(wav)
        if spec.dim() != 3:
            raise RuntimeError(
                "AudioProcessor must return a 3-D tensor with shape (B, F, T)"
            )

        # 2) Re-arrange to (B, T, F)
        # spec = spec.permute(0, 2, 1)

        spec = spec.unsqueeze(1)  # (B, 1, T, F)

        remove_extra = pooling != "cls"

        backbone_out: Dict[str, Any] = self.backbone(
            spec,
            # padding_mask=patch_mask,
            mask=False,
            features_only=True,
            remove_extra_tokens=remove_extra,
        )

        feats: torch.Tensor = backbone_out["x"]  # (B, L, D)
        print(f"feats shape: {feats.shape} pooling: {pooling}")
        if pooling == "cls":
            return feats[:, 0]  # CLS token
        elif pooling == "mean":
            return feats.mean(dim=1)  # global average over patches
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")

