from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase
from representation_learning.models.beats.beats import BEATs, BEATsConfig
from representation_learning.utils import universal_torch_load

BEATS_PRETRAINED_PATH_FT = (
    "gs://foundation-models/beats_ckpts/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
)
BEATS_PRETRAINED_PATH_SSL = (
    "gs://representation-learning/pretrained/BEATs_iter3_plus_AS2M.pt"
)

BEATS_PRETRAINED_PATH_NATURELM = "gs://foundation-models/beats_ckpts/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2_rl_loaded.pt"


class Model(ModelBase):
    """Wrapper that adapts the raw *BEATs* backbone for our training loop.

    This module follows the same conventions as the other model wrappers
    (e.g. ``efficientnet.py``) so that it can be selected via
    ``representation_learning.models.get_model.get_model``.

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
        the pooled embedding is returned directly, which is handy for
        representation extraction / linear probing.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        pretrained: bool = False,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        use_naturelm: bool = False,
        fine_tuned: bool = False,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # ------------------------------------------------------------------
        # 1.  Build the BEATs backbone
        # ------------------------------------------------------------------

        # Determine which checkpoint to load based on configuration
        if use_naturelm:
            beats_checkpoint_path = BEATS_PRETRAINED_PATH_NATURELM
        elif fine_tuned:
            beats_checkpoint_path = BEATS_PRETRAINED_PATH_FT
        else:
            beats_checkpoint_path = BEATS_PRETRAINED_PATH_SSL

        beats_ckpt = universal_torch_load(
            beats_checkpoint_path, cache_mode="use", map_location="cpu"
        )
        self.use_naturelm = use_naturelm
        self.fine_tuned = fine_tuned
        beats_cfg = BEATsConfig(beats_ckpt["cfg"])
        print(beats_cfg)
        if use_naturelm:
            beats_ckpt_naturelm = universal_torch_load(
                BEATS_PRETRAINED_PATH_NATURELM, map_location="cpu"
            )
        else:
            beats_ckpt_naturelm = beats_ckpt["model"]
        # beats_ckpt_naturelm = beats_ckpt
        self.backbone = BEATs(beats_cfg)
        self.backbone.to(device)
        self.backbone.load_state_dict(beats_ckpt_naturelm)

        # ------------------------------------------------------------------
        # 2.  Optional classifier for supervised training
        # ------------------------------------------------------------------
        self._return_features_only = return_features_only
        if not return_features_only:
            self.classifier = nn.Linear(768, num_classes)
        else:
            self.register_module("classifier", None)  # type: ignore[arg-type]

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

        features, frame_padding = self.backbone(x, padding_mask)

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

    def extract_embeddings(self, x: torch.Tensor, layers: List[str]) -> torch.Tensor:
        self._return_features_only = True
        if isinstance(x, dict):
            return self.forward(x["raw_wav"], x["padding_mask"])
        else:
            return self.forward(x)

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        audio = super().process_audio(x)
        if self.use_naturelm:
            audio = torch.clamp(audio, -1.0, 1.0)
        return audio
