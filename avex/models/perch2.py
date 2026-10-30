"""Google Perch v2 bird-vocalization classifier as a native PyTorch model.

The architecture is a hand-written port (:mod:`avex.models.perch2_net`) of the
Perch v2 ONNX export; weights are converted from the pinned ONNX file on first
use and cached (see :mod:`avex.models.perch2_net.weights`). The model runs on
any torch device, exposes named layers for probing
(``backbone.stem``, ``backbone.blocks.{stage}.{idx}``, ``backbone.head``,
``backbone.pool``) and supports gradients.

Input contract: mono waveform at **32 kHz**, no normalisation. Clips are
centre-cropped or zero-padded to ``window_seconds`` (5 s, 160 000 samples),
the length Perch v2 was trained on.

Batch norm is folded into the convolutions, so the backbone has no
``BatchNorm`` modules; fine-tuning updates the folded weights, and train/eval
mode does not change the backbone's output.

Source: https://huggingface.co/justinchuby/Perch-onnx (converted from Google's
Kaggle release ``google/bird-vocalization-classifier``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from avex.models.base_model import ModelBase
from avex.models.perch2_net.network import Perch2Config, Perch2Network
from avex.models.perch2_net.weights import load_weights, resolve_weights

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 1536


class Perch2Model(ModelBase):
    """Google Perch v2 (EfficientNet-B3 + ProtoPNet head, 14 795 classes).

    ``forward`` returns the pooled 1536-d embedding, or the logits of the
    optional avex linear classifier when ``num_classes > 0``. Perch's own
    species logits are available from :meth:`species_logits`.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        device: str = "cpu",
        audio_config: Optional[Dict[str, Any]] = None,
        target_sample_rate: int = 32_000,
        window_seconds: float = 5.0,
        freeze_backbone: bool = True,
        return_features_only: bool = False,
        pretrained: bool = True,
        weights_path: Optional[Union[str, Path]] = None,
        onnx_path: Optional[Union[str, Path]] = None,
        allow_download: bool = True,
    ) -> None:
        """Build the model and load the Perch v2 weights.

        Parameters
        ----------
        num_classes : Optional[int]
            Classes of an optional avex linear classifier on the embedding.
        device : str
            Torch device.
        audio_config : Optional[Dict[str, Any]]
            Kept for API compatibility; Perch v2 has its own frontend.
        target_sample_rate : int
            Expected input sample rate (32 kHz).
        window_seconds : float
            Clip length the input is cropped or padded to.
        freeze_backbone : bool
            Unused; kept for backwards compatibility. Freezing is controlled by
            ``extract_embeddings(freeze_backbone=...)`` and the probe.
        return_features_only : bool
            If True, never attach the avex classifier.
        pretrained : bool
            If True, load the converted Perch v2 weights (converting on first
            use). If False, keep random weights (e.g. to load a checkpoint).
            Note that ``load_model`` does not forward this flag, so a
            ``checkpoint_path`` load still resolves the Perch v2 weights first.
        weights_path : Optional[Union[str, Path]]
            Converted safetensors file to load instead of the cache.
        onnx_path : Optional[Union[str, Path]]
            Pre-downloaded ``perch_v2.onnx`` to convert from if the cache is
            empty (also ``$AVEX_PERCH2_ONNX``).
        allow_download : bool
            If False, fail instead of downloading the ONNX when nothing is cached.
        """
        super().__init__(device, audio_config)
        self.target_sr = target_sample_rate
        self.window_samples = int(window_seconds * self.target_sr)
        self.embedding_dim = _EMBEDDING_DIM
        self.return_features_only = return_features_only

        if pretrained:
            path = resolve_weights(weights_path=weights_path, onnx_path=onnx_path, allow_download=allow_download)
            # Build on the meta device and adopt the loaded tensors: one copy in memory.
            with torch.device("meta"):
                self.backbone = Perch2Network(Perch2Config())
            self.backbone.load_state_dict(load_weights(path), strict=True, assign=True)
        else:
            self.backbone = Perch2Network(Perch2Config())

        if not return_features_only and num_classes is not None and num_classes > 0:
            self.num_classes = num_classes
            self.classifier = nn.Linear(self.embedding_dim, num_classes)
        else:
            self.num_classes = None
            self.classifier = None

        self.to(device)

    # ------------------------------------------------------------------ #
    #  Input and outputs
    # ------------------------------------------------------------------ #
    def _prepare_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        """Downmix to mono and centre-crop or right-pad to ``window_samples``.

        Returns
        -------
        torch.Tensor
            Float waveform of shape (B, window_samples) on the model's device.

        Raises
        ------
        ValueError
            If ``wav`` is not (B, N) or (B, C, N).
        """
        if wav.dim() == 3:
            wav = wav.mean(dim=1)
        if wav.dim() != 2:
            raise ValueError("Audio must be (batch, channels, samples) or (batch, samples) waveform.")
        n = wav.size(-1)
        if n > self.window_samples:
            start = (n - self.window_samples) // 2
            wav = wav[:, start : start + self.window_samples]
        elif n < self.window_samples:
            wav = nn.functional.pad(wav, (0, self.window_samples - n))
        return wav.to(device=self.backbone.frontend.window.device, dtype=torch.float32)

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Pooled Perch v2 embedding.

        Returns
        -------
        torch.Tensor
            Shape (B, 1536).
        """
        return self.backbone(self._prepare_waveform(audio), with_logits=False).embedding

    def spatial_embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Unpooled embedding over time and frequency.

        Returns
        -------
        torch.Tensor
            Shape (B, 16, 4, 1536) for 5 s clips: (time, frequency, channels).
        """
        return self.backbone(self._prepare_waveform(audio), with_logits=False).spatial_embedding

    def species_logits(self, audio: torch.Tensor) -> torch.Tensor:
        """Perch v2's own classifier output (ProtoPNet head).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 14795).
        """
        return self.backbone(self._prepare_waveform(audio), with_logits=True).label

    def extract_features(self, audio: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pooled embedding (kept for backwards compatibility).

        Returns
        -------
        torch.Tensor
            Shape (B, 1536).
        """
        return self.embed(audio)

    def forward(self, audio: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Embedding, or avex classifier logits when ``num_classes > 0``.

        Returns
        -------
        torch.Tensor
            (B, 1536) embedding or (B, num_classes) logits.
        """
        feats = self.embed(audio)
        return self.classifier(feats) if self.classifier is not None else feats

    # ------------------------------------------------------------------ #
    #  Probing
    # ------------------------------------------------------------------ #
    def _discover_embedding_layers(self) -> None:
        """List hookable layers in forward order; ``last_layer`` is ``backbone.pool``."""
        if not self._layer_names:
            names = ["backbone.stem"]
            for s, stage in enumerate(self.backbone.blocks):
                names.extend(f"backbone.blocks.{s}.{i}" for i in range(len(stage)))
            names.extend(["backbone.head", "backbone.pool"])
            self._layer_names = names

    @staticmethod
    def _aggregate(tensor: torch.Tensor, aggregation: str) -> torch.Tensor:
        """Reduce one hooked output.

        Feature maps (B, C, T, F) are pooled over time and frequency for
        "mean"/"max"; "cls_token" behaves like "mean" (there is no class
        token); "none" returns a (B, T, C*F) sequence. Pooled outputs (B, C)
        become (B, 1, C) for "none".

        Returns
        -------
        torch.Tensor
            Aggregated tensor.

        Raises
        ------
        ValueError
            If ``aggregation`` is unsupported.
        """
        if aggregation not in ("mean", "max", "cls_token", "none"):
            raise ValueError(f"Unsupported aggregation method: {aggregation}")
        if tensor.dim() == 2:
            return tensor.unsqueeze(1) if aggregation == "none" else tensor
        if aggregation == "none":
            b, c, t, f = tensor.shape
            return tensor.permute(0, 2, 1, 3).reshape(b, t, c * f)
        if aggregation == "max":
            return tensor.amax(dim=(2, 3))
        return tensor.mean(dim=(2, 3))

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean",
        freeze_backbone: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Embeddings from the hooked layers, or the pooled embedding if none are hooked.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Waveform (B, N) / (B, C, N) or a dict with ``raw_wav``.
        padding_mask : Optional[torch.Tensor]
            Ignored: clips are a fixed 5 s window.
        aggregation : str
            "mean", "max", "cls_token" (same as "mean") or "none".
        freeze_backbone : bool
            If True, run without gradients.

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            For "none", a list with one (B, T, D) tensor per layer. Otherwise
            one (B, D) tensor, concatenated over layers.

        Raises
        ------
        ValueError
            If a hooked layer (e.g. the classifier) does not run on this path.
        """
        wav = x["raw_wav"] if isinstance(x, dict) else x
        self.ensure_hooks_registered()
        with torch.set_grad_enabled(not freeze_backbone and torch.is_grad_enabled()):
            if not self._hooks:
                outputs = [self.embed(wav)]
            else:
                self._clear_hook_outputs()
                try:
                    self.backbone(self._prepare_waveform(wav), with_logits=False)
                    missing = [name for name in self._hook_layers if name not in self._hook_outputs]
                    if missing:
                        raise ValueError(
                            f"Layers {missing} do not run when extracting embeddings. "
                            f"Hookable layers: {self.get_model_layers()}"
                        )
                    outputs = [self._hook_outputs[name] for name in self._hook_layers]
                finally:
                    self._clear_hook_outputs()
        aggregated = [self._aggregate(t, aggregation) for t in outputs]
        if aggregation == "none":
            return aggregated
        return aggregated[0] if len(aggregated) == 1 else torch.cat(aggregated, dim=1)


# Alias for auto-discovery consistency
Model = Perch2Model
