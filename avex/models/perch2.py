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
                "onnxruntime is required for the Perch2 model: "
                "uv pip install onnxruntime (CPU) or uv pip install 'onnxruntime-gpu>=1.27' (CUDA 13). "
                "Install one of them, not both (see docs/supported_models.md)."
            ) from e

        logger.info("Downloading Perch2 ONNX model from %s …", repo_id)
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Prefer CUDA when this onnxruntime build has it (onnxruntime-gpu), else CPU.
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        _ort_session = ort.InferenceSession(model_path, providers=providers)
        _ort_session_key = key
        logger.info("Perch2 ONNX session ready (providers: %s)", _ort_session.get_providers())
    return _ort_session


class _OnnxBackbone(nn.Module):
    """Thin wrapper around the ONNX Runtime call; adds no computation.

    It exists only so probes have a named layer (``backbone``) to hook, which
    is what ``target_layers=["last_layer"]`` resolves to. No parameters, no
    gradients: ONNX Runtime runs outside autograd.
    """

    def __init__(self, repo_id: str, filename: str) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.filename = filename
        _load_ort_session(repo_id, filename)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Embed a ``(B, 160000)`` 32 kHz waveform batch.

        Returns
        -------
        torch.Tensor
            Embeddings ``(B, 1536)`` on the device of ``wav``.
        """
        session = _load_ort_session(self.repo_id, self.filename)
        # ONNX Runtime takes numpy, so the batch always goes through host memory,
        # even with the CUDA provider. A 5 s clip is 640 KB; the copy is cheap
        # next to the model, so IOBinding is not worth the complexity here.
        audio_np = wav.detach().cpu().float().numpy()
        # "embedding" is output 0 of the perch_v2 ONNX contract.
        (embedding,) = session.run(["embedding"], {"inputs": audio_np})
        return torch.from_numpy(embedding).to(wav.device)


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

    Only the embedding is exposed. It is produced by the ``backbone``
    submodule, which is what ``target_layers=["last_layer"]`` resolves to.
    ONNX Runtime runs outside autograd, so the backbone is always frozen,
    whatever ``freeze_backbone`` says; a classifier on top still trains.
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
        self.backbone = _OnnxBackbone(hf_repo_id, hf_filename)
        if not freeze_backbone:
            self._warn_cannot_unfreeze()
        providers = _load_ort_session(hf_repo_id, hf_filename).get_providers()
        if str(device).startswith("cuda") and "CUDAExecutionProvider" not in providers:
            logger.warning(
                "Perch2 runs on CPU (ONNX Runtime providers: %s); outputs are copied to %s. "
                "For GPU inference replace onnxruntime with onnxruntime-gpu>=1.27 "
                "(see docs/supported_models.md).",
                providers,
                device,
            )

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
    def _warn_cannot_unfreeze(self) -> None:
        logger.warning(
            "freeze_backbone=False has no effect on Perch2: ONNX Runtime runs outside autograd, "
            "so the backbone cannot be fine-tuned. Only layers on top of it train."
        )

    def _prepare_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            # (B, C, N) → (B, N): downmix any channel count to mono
            wav = wav.mean(dim=1)
        if wav.dim() != 2:
            raise ValueError("Audio must be (batch, channels, samples) or (batch, samples) waveform.")
        n = wav.size(-1)
        if n > self.window_samples:
            start = (n - self.window_samples) // 2
            wav = wav[:, start : start + self.window_samples]
        elif n < self.window_samples:
            wav = nn.functional.pad(wav, (0, self.window_samples - n))
        return wav

    # ------------------------------------------------------------------ #
    #  ModelBase interface
    # ------------------------------------------------------------------ #
    def _discover_embedding_layers(self) -> None:
        # The ONNX graph is opaque, so the embedding (``backbone``) is the only
        # layer worth hooking; ``last_layer`` resolves to it.
        if not self._layer_names:
            self._layer_names = ["backbone"]

    def extract_features(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        audio = self._prepare_waveform(audio)
        return self.backbone(audio).to(self.device)

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
        freeze_backbone: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Return the Perch v2 embedding for a waveform batch.

        Registered hooks do not change the output: ``backbone`` is the only
        embedding layer, and its output is this embedding. Hooks on any other
        submodule are ignored, with a warning.

        Args:
            x: Waveform ``(B, N)`` or ``(B, C, N)`` at 32 kHz, or a dict with
                ``"raw_wav"``.
            padding_mask: Unused; accepted for interface compatibility.
            aggregation: ``"mean"``, ``"max"`` or ``"cls_token"`` return
                ``(B, 1536)``; ``"none"`` returns ``[(B, 1, 1536)]``.
            freeze_backbone: Accepted for probe compatibility and ignored:
                ONNX Runtime is never differentiable.

        Returns:
            torch.Tensor | list[torch.Tensor]: The embeddings.

        Raises:
            ValueError: If ``aggregation`` is not supported.
        """
        if not freeze_backbone and not getattr(self, "_warned_unfreeze", False):
            self._warned_unfreeze = True
            self._warn_cannot_unfreeze()
        ignored = [name for name in self._hook_layers if name != "backbone"]
        if ignored:
            logger.warning("Perch2 only exposes the 'backbone' embedding; ignoring hooks on %s.", ignored)

        audio = x["raw_wav"] if isinstance(x, dict) else x
        embeddings = self.extract_features(audio, padding_mask)
        self._clear_hook_outputs()

        if aggregation == "none":
            return [embeddings.unsqueeze(1)]  # (B, 1, D) for sequence probes
        elif aggregation in ("mean", "max", "cls_token"):
            return embeddings
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")


# Alias for auto-discovery consistency
Model = Perch2Model
