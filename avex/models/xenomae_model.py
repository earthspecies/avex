"""XC_1M MAE (Moritz) -- ViT-B/16 masked autoencoder pretrained on Xeno-canto.

Thin avex adapter around the vendored, unmodified `avex/models/xenomae/xenomae.py`
(the MVP handed over in /mnt/home/moritz_mvp). Everything model-specific is
reused from that file; this adapter only adds what avex needs:

  * raw audio in -> embeddings out, batched on the GPU;
  * 5 s windowing: the encoder takes exactly 5 s at 32 kHz. Longer clips are cut
    into non-overlapping 5 s windows (a tail > 1.25 s is kept, zero-padded), each
    window is preprocessed independently exactly as the README prescribes
    ("Split longer recordings into five-second clips before preprocessing"), and
    window embeddings are averaged per clip -- the same scheme as PupuJEPA's
    10.24 s windows;
  * `.backbone` + classifier head, so avex's two-stage post-training (freeze,
    then unfreeze at epoch 3) works unchanged.

PREPROCESSING is the MVP's `preprocess_audio`, applied to a batch: per-window peak
normalisation, pad to 5 s, power mel (128 bins, n_fft/win 1024, hop 320, fmin 50),
dB with an 80 dB floor (per window), `2 * (dB + 80) / 80 - 1`, pad to 512 frames.
The mel/dB transforms are deep copies of the MVP's own module-level objects, so the
parameters cannot drift. Parity with the reference function is tested in
scripts/check_xenomae_adapter.py.

POOLING (extra_config.pooling):
  "mean" (default) -- mean of the 256 patch tokens after the final norm: the
                      paper's "time-averaged final-layer embedding" (S3.4), and the
                      choice made for PupuJEPA.
  "cls"            -- the CLS token, the MVP's default output. In MAE pre-training
                      the CLS token gets little training signal, so this is checked
                      against "mean" before full benchmarking.

forward() returns (B, 1, 768) through an Identity marker ("output_features"), like
PupuJEPA, so the probe's `aggregation: mean` is a no-op. Keep evaluation in float32
(README); post-training uses the shared avex recipe.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from safetensors.torch import load_file

from avex.models.base_model import ModelBase
from avex.models.xenomae import xenomae as _mvp

logger = logging.getLogger(__name__)

EMBED_DIM = 768


class XenoMAEModel(ModelBase):
    """Moritz's XC_1M MAE encoder behind the avex ModelBase interface."""

    def __init__(
        self,
        device: str = "cuda",
        audio_config: Optional[Any] = None,
        num_classes: int = 0,
        return_features_only: bool = False,
        extra_config: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)
        extra_config = extra_config or {}

        self.pooling = str(extra_config.get("pooling", "mean"))
        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"extra_config.pooling must be 'mean' or 'cls', got {self.pooling!r}")
        self.return_features_only = return_features_only

        self.backbone = _mvp.XenoMAE()
        ckpt = extra_config.get("checkpoint_path")
        if pretrained:
            if not ckpt:
                raise ValueError("XenoMAEModel needs extra_config.checkpoint_path (model.safetensors)")
            # strict=True: every encoder tensor must be present and nothing extra.
            self.backbone.load_state_dict(load_file(str(ckpt), device="cpu"), strict=True)
            logger.info("Loaded XenoMAE weights (strict) from %s", ckpt)
        self.backbone = self.backbone.float().to(self.device)

        # The MVP's own transforms, copied so they can live on the GPU.
        self.mel = copy.deepcopy(_mvp._MEL).to(self.device)
        self.to_db = copy.deepcopy(_mvp._TO_DB).to(self.device)

        self.target_sr = _mvp.SAMPLE_RATE
        self.clip_samples = _mvp.CLIP_SAMPLES
        self.input_sr = int(getattr(audio_config, "sample_rate", self.target_sr) or self.target_sr)

        self.output_features = nn.Identity()
        self.classifier: Optional[nn.Module] = None
        if not return_features_only and num_classes > 0:
            self.classifier = nn.Linear(EMBED_DIM, num_classes).to(self.device)
        logger.info("XenoMAE: pooling=%s, input_sr=%d", self.pooling, self.input_sr)

    # -- avex layer discovery: one layer, the pooled final output ------------
    def _discover_embedding_layers(self) -> None:
        if len(self._layer_names) == 0:
            self._layer_names = ["output_features"]

    # -- audio -> 5 s windows -------------------------------------------------
    def _windows(self, audio: torch.Tensor) -> torch.Tensor:
        """(T,) at 32 kHz -> (n_windows, 5 s): non-overlapping, tail > 1/4 kept."""
        n_len = self.clip_samples
        length = audio.shape[-1]
        if length <= n_len:
            return F.pad(audio, (0, n_len - length)).unsqueeze(0)
        n = length // n_len
        chunks = [audio[c * n_len : (c + 1) * n_len] for c in range(n)]
        rem = audio[n * n_len :]
        if rem.shape[-1] > n_len // 4:
            chunks.append(F.pad(rem, (0, n_len - rem.shape[-1])))
        return torch.stack(chunks)

    def preprocess(self, windows: torch.Tensor) -> torch.Tensor:
        """(N, 5 s) waveforms at 32 kHz -> (N, 1, 128, 512), the MVP's preprocess_audio
        applied per window. Zero padding does not change a window's peak, so padding
        before normalising matches the MVP (normalise, then pad)."""
        peak = windows.abs().amax(dim=-1, keepdim=True)
        windows = torch.where(peak > 0, windows / peak.clamp_min(1e-30), windows)
        spec = self.mel(windows).unsqueeze(1)                      # (N, 1, 128, frames)
        spec = 2 * (self.to_db(spec) + 80) / 80 - 1               # top_db is per window (4-D input)
        return F.pad(spec, (0, 512 - spec.shape[-1]))

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """(N, 257, D) final-norm tokens -> (N, D)."""
        return tokens[:, 0] if self.pooling == "cls" else tokens[:, 1:].mean(dim=1)

    # -- forward ----------------------------------------------------------------
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T) raw waveform at input_sr. Returns (B, 1, D) features, or logits.

        All windows of all clips are encoded in one batch, then averaged per clip.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 3:                                            # (B, C, T) -> mono
            x = x.mean(dim=1)
        x = x.float().to(self.device)
        if self.input_sr != self.target_sr:
            x = AF.resample(x, self.input_sr, self.target_sr)

        per_clip = [self._windows(x[i]) for i in range(x.shape[0])]
        counts = [w.shape[0] for w in per_clip]
        tokens = self.backbone(self.preprocess(torch.cat(per_clip)), return_tokens=True)
        win_emb = self._pool(tokens)                                # (sum(counts), D)
        feats = torch.stack([c.mean(dim=0) for c in torch.split(win_emb, counts)])
        feats = self.output_features(feats.unsqueeze(1))            # (B, 1, D)

        if self.return_features_only or self.classifier is None:
            return feats
        return self.classifier(feats.squeeze(1))
