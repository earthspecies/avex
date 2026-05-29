"""Perch2 model via ONNX runtime — no TensorFlow dependency.

https://huggingface.co/justinchuby/Perch-onnx
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from avex.models.base_model import ModelBase

logger = logging.getLogger(__name__)

_HF_REPO_ID = "justinchuby/Perch-onnx"
_HF_FILENAME = "perch_v2.onnx"
_EMBEDDING_DIM = 1536

_ort_session: Any = None
_ort_session_key: tuple[str, str] | None = None


def _load_ort_session(repo_id: str = _HF_REPO_ID, filename: str = _HF_FILENAME) -> object:
    global _ort_session, _ort_session_key
    key = (repo_id, filename)
    if _ort_session is None or _ort_session_key != key:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as e:
            raise ImportError(
                "onnxruntime is required for the Perch2 model.\n"
                "pip install onnxruntime   # CPU\n"
                "pip install onnxruntime-gpu  # CUDA"
            ) from e

        logger.info("Downloading Perch2 ONNX model from %s …", repo_id)
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Prefer CUDA if available, fall back to CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _ort_session = ort.InferenceSession(model_path, providers=providers)
        _ort_session_key = key
        logger.info("Perch2 ONNX session ready (providers: %s)", _ort_session.get_providers())
    return _ort_session


class Perch2Model(ModelBase):
    """Google Perch v2 bird-vocalization classifier via ONNX Runtime.

    Input: raw waveform, 160 000 samples (32 kHz × 5 s).
    The ONNX model only cares about the sample count; pass audio at 32 kHz.
    Shorter clips are zero-padded; longer clips are centre-cropped.

    ONNX outputs (perch_v2.onnx):
      • embedding          — shape [B, 1536]
      • spatial_embedding  — shape [B, 16, 4, 1536]
      • spectrogram        — shape [B, 500, 128]
      • label              — shape [B, 14795]
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
        hf_repo_id: str = _HF_REPO_ID,
        hf_filename: str = _HF_FILENAME,
    ) -> None:
        super().__init__(device, audio_config)

        self.target_sr = target_sample_rate
        self.window_samples = int(window_seconds * self.target_sr)
        self.embedding_dim = _EMBEDDING_DIM
        self.return_features_only = return_features_only
        self._hf_repo_id = hf_repo_id
        self._hf_filename = hf_filename

        _load_ort_session(hf_repo_id, hf_filename)

        if not return_features_only and num_classes is not None and num_classes > 0:
            self.num_classes = num_classes
            self.classifier = nn.Linear(self.embedding_dim, num_classes).to(device)
        else:
            self.num_classes = None
            self.classifier = None

        self.to(device)

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #
    def _prepare_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3 and wav.size(1) == 1:
            wav = wav.squeeze(1)
        if wav.dim() != 2:
            raise ValueError("Audio must be (batch, samples) waveform.")
        n = wav.size(-1)
        if n > self.window_samples:
            start = (n - self.window_samples) // 2
            wav = wav[:, start : start + self.window_samples]
        elif n < self.window_samples:
            wav = nn.functional.pad(wav, (0, self.window_samples - n))
        return wav

    def _ort_forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Run ONNX inference and return embeddings [B, embedding_dim].

        perch_v2.onnx input/output contract:
          in:  "inputs"          [B, 160000] float32
          out[0]: "embedding"    [B, 1536]
          out[1]: "spatial_embedding" [B, 16, 4, 1536]
          out[2]: "spectrogram"  [B, 500, 128]
          out[3]: "label"        [B, 14795]

        Returns
        -------
        torch.Tensor
            Embeddings with shape [B, embedding_dim].
        """
        session = _load_ort_session(self._hf_repo_id, self._hf_filename)
        audio_np = audio.detach().cpu().float().numpy()
        # output index 0 is always "embedding" per the perch_v2 ONNX contract
        outputs = session.run(["embedding"], {"inputs": audio_np})
        return torch.from_numpy(outputs[0])

    # ------------------------------------------------------------------ #
    #  ModelBase interface
    # ------------------------------------------------------------------ #
    def _discover_linear_layers(self) -> None:
        if not self._layer_names:
            self._layer_names = [name for name, mod in self.named_modules() if isinstance(mod, nn.Linear)]

    _discover_embedding_layers = _discover_linear_layers  # type: ignore[assignment]

    def extract_features(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        audio = self._prepare_waveform(audio)
        return self._ort_forward(audio).to(self.device)

    def forward(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = self.extract_features(audio, padding_mask)
        return self.classifier(feats) if self.classifier else feats

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean",
    ) -> torch.Tensor | list[torch.Tensor]:
        audio = x["raw_wav"] if isinstance(x, dict) else x
        embeddings = self.extract_features(audio, padding_mask)

        if aggregation == "none":
            return [embeddings.unsqueeze(1)]  # (B, 1, D) for sequence probes
        elif aggregation in ("mean", "max", "cls_token"):
            return embeddings
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")


# Alias for auto-discovery consistency
Model = Perch2Model
