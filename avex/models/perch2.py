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

    Only the embedding is exposed, as the single non-hookable output
    ``ModelBase.SINGLE_OUTPUT_LAYER`` that ``target_layers=["last_layer"]``
    resolves to. ONNX Runtime runs outside autograd, so the backbone is
    always frozen, whatever ``freeze_backbone`` says; a classifier on top
    still trains.
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
        if not freeze_backbone:
            self._warn_cannot_unfreeze()

        # Warm up the session (downloads the model on first use).
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

    def _ort_forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Run the ONNX session and return the ``(B, 1536)`` embedding on ``self.device``.

        Returns
        -------
        torch.Tensor
            Embeddings with shape ``(B, 1536)``.
        """
        session = _load_ort_session(self._hf_repo_id, self._hf_filename)
        # ONNX Runtime takes numpy, so the batch always goes through host memory,
        # even with the CUDA provider. A 5 s clip is 640 KB; the copy is cheap
        # next to the model, so IOBinding is not worth the complexity here.
        audio_np = audio.detach().cpu().float().numpy()
        # "embedding" is output 0 of the perch_v2 ONNX contract.
        (embedding,) = session.run(["embedding"], {"inputs": audio_np})
        return torch.from_numpy(embedding).to(self.device)

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
    def register_hooks_for_layers(self, target_layers: list[str | int]) -> list[str]:
        """Accept generic layer selectors for the single ONNX embedding output.

        The ONNX graph has no PyTorch modules to hook, so this only validates
        ``target_layers`` and reports :attr:`ModelBase.SINGLE_OUTPUT_LAYER`.

        Args:
            target_layers: ``"last_layer"``, ``"all"`` or ``"embeddings"``.

        Returns:
            list[str]: ``["embeddings"]``.
        """
        return self._register_single_output_layer(target_layers)

    def enable_gradient_checkpointing(self) -> None:
        logger.warning("Gradient checkpointing not supported for Perch2 ONNX.")

    def extract_features(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        audio = self._prepare_waveform(audio)
        return self._ort_forward(audio)

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

        The embedding is the model's only output for probes, so registered
        layer selectors do not change it.

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
        """
        if not freeze_backbone and not getattr(self, "_warned_unfreeze", False):
            self._warned_unfreeze = True
            self._warn_cannot_unfreeze()

        audio = x["raw_wav"] if isinstance(x, dict) else x
        embeddings = self.extract_features(audio, padding_mask)
        return self._aggregate_single_embedding(embeddings, aggregation)


# Alias for auto-discovery consistency
Model = Perch2Model
