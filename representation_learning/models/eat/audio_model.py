from __future__ import annotations

import logging
from typing import Optional, Any

import torch
import torch.nn as nn

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase
from .eat import Data2VecMultiModel, Data2VecMultiConfig, Modality, D2vModalitiesConfig
from .image import D2vImageConfig

logger = logging.getLogger(__name__)


class Model(ModelBase):
    """Audio-spectrogram classifier on EAT backbone."""

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool | None = None,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        embed_dim: int = 768,
        patch_size: int = 16,
        target_length: int = 256,
        return_features_only: bool = False,
        enable_ema: bool = False,
        pretraining_mode: bool = False,
        eat_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        if audio_config is None:
            audio_config = AudioConfig()

        self.return_features_only = return_features_only
        self.pretraining_mode = pretraining_mode

        # TODO: Changed – auto-enable EMA teacher when in pretraining mode and
        # caller did not explicitly enable it.
        if self.pretraining_mode and not enable_ema:
            logger.warning("Pretraining mode requested but enable_ema=False; auto-enabling EMA teacher.")
            enable_ema = True

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

        # ----------------------------------------------------------
        # Apply YAML-provided overrides recursively so users can set
        # any Data2VecMultiConfig attribute without code changes.
        # ----------------------------------------------------------
        logger.info(f"Applying overrides to EAT config object")
        if eat_cfg:
            self._apply_overrides(eat_cfg_obj, eat_cfg)
        print(f"[DEBUG] EAT config object: {eat_cfg_obj}")
        logger.info(f"[DEBUG] Making Backbone")
        self.backbone = Data2VecMultiModel(eat_cfg_obj, modalities=[Modality.IMAGE])
        logger.info(f"[DEBUG] Backbone: {self.backbone}")
        self.backbone.to(self.device)
        print(f"[DEBUG] Backbone: {self.backbone}")

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.to(self.device)
        logger.info(f"[DEBUG] Classifier: {self.classifier}")

    # --------------------------------------------------------------
    @staticmethod
    def _pad_or_crop_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
        B, T, F = x.shape
        if T == target_len:
            return x
        if T > target_len:
            start = (T - target_len) // 2
            return x[:, start : start + target_len, :]
        pad = x.new_zeros(B, target_len - T, F)
        return torch.cat([x, pad], dim=1)

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        # Debug: raw input shape
        print(f"[DEBUG] Input wav shape: {x.shape}")
        logger.warning(f"[DEBUG] Input wav shape: {x.shape}")

        spec = super().process_audio(x)
        print(f"[DEBUG] Mel-spec after process_audio: {spec.shape}")
        if spec.dim() != 3:
            raise RuntimeError("AudioProcessor must return (B,F,T) spectrogram")
        spec = spec.permute(0, 2, 1)
        print(f"[DEBUG] After permute (B,T,F): {spec.shape}")
        spec = self._pad_or_crop_time(spec, self.backbone.cfg.modalities.image.target_length)
        print(f"[DEBUG] After pad/crop to target length: {spec.shape}")
        spec = spec.unsqueeze(1)
        print(f"[DEBUG] After unsqueeze (add channel dim): {spec.shape}")

        if self.pretraining_mode:
            # Full pretraining path inside backbone (computes loss dict)
            out = self.backbone(spec, mask=True, features_only=False)
            print(f"[DEBUG] Backbone output keys: {list(out.keys())}")
            if 'x' in out:
                print(f"[DEBUG] Backbone 'x' shape: {out['x'].shape}")
            if 'losses' in out:
                print("[DEBUG] Loss component shapes:")
                for k,v in out['losses'].items():
                    print(f"    {k}: {v.shape}")
            return out

        out = self.backbone(spec, mask=False, features_only=True)
        print(f"[DEBUG] Backbone features shape: {out['x'].shape}")
        features = out["x"].mean(dim=1)

        if self.return_features_only:
            return features
        return self.classifier(features)

    # --------------------------------------------------------------
    @staticmethod
    def _apply_overrides(obj, overrides: dict[str, Any]):  # noqa: ANN401
        """Recursively set attributes on a dataclass or object.

        Parameters
        ----------
        obj : Any
            The dataclass / namespace whose attributes should be updated.
        overrides : dict[str, Any]
            Mapping of attribute names → new values.  Nested dictionaries are
            applied recursively.
        """
        for k, v in overrides.items():
            if not hasattr(obj, k):
                # Silently ignore unknown keys – keeps forward-compatibility
                continue
            current_val = getattr(obj, k)
            if isinstance(v, dict) and not isinstance(current_val, (int, float, str, bool, type(None))):
                # Recurse into sub-objects (e.g. modalities.image.decoder)
                Model._apply_overrides(current_val, v)  # type: ignore[arg-type]
            else:
                # If the existing attribute is numeric and the override comes as
                # a string (e.g. YAML "1e-6"), attempt to cast to the original
                # type so downstream code receives the expected dtype.
                if isinstance(current_val, (int, float)) and isinstance(v, str):
                    try:
                        casted = type(current_val)(float(v))
                        setattr(obj, k, casted)
                    except ValueError:
                        # Fallback to raw string if conversion fails
                        setattr(obj, k, v)
                else:
                    setattr(obj, k, v)
