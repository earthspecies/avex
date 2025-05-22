from typing import Optional

import torch
import torch.nn as nn

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase
from representation_learning.models.beats.beats import BEATs, BEATsConfig


class Model(ModelBase):
    """Wrapper that adapts the raw *BEATs* backbone for our training loop.

    This module follows the same conventions as the other model wrappers
    (e.g. ``efficientnetb0.py``) so that it can be selected via
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
        pretrained: bool = False,  # Currently unused; placeholder for future.
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # ------------------------------------------------------------------
        # 1.  Build the BEATs backbone
        # ------------------------------------------------------------------
        cfg = BEATsConfig()
        # A small, square patch (4×4) works well for the 128-bin log-mel FBanks
        # produced by the internal pre-processing.  The original BEATs model
        # also uses a 4×4 patch when the input resolution is 128×1024.
        cfg.input_patch_size = 4
        # We do **not** enable the internal predictor head as we attach our own
        # classifier below, hence ``finetuned_model`` stays *False*.

        self.backbone = BEATs(cfg)
        self.backbone.to(device)

        # ------------------------------------------------------------------
        # 2.  Optional classifier for supervised training
        # ------------------------------------------------------------------
        self._return_features_only = return_features_only
        if not return_features_only:
            self.classifier = nn.Linear(cfg.encoder_embed_dim, num_classes)
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
