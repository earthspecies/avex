"""
SurfPerch model implementation for underwater fish-audio classification and embedding extraction
using locally saved model weights from the SurfPerch_v1.0 folder
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Utility: lazy TensorFlow import (keeps PyTorch-only envs light-weight)
# --------------------------------------------------------------------------- #
_SURFPERCH_MODEL_PATH = "SurfPerch_v1.0/savedmodel"
_surfperch_model = None  # loaded on first use


def _load_surfperch_model(model_path: str = _SURFPERCH_MODEL_PATH) -> Any:  # noqa: ANN401
    """Load the SurfPerch TensorFlow SavedModel on first use.

    Args:
        model_path: Path to the SurfPerch SavedModel directory

    Returns:
        Any: The loaded SurfPerch TensorFlow model

    Raises:
        ImportError: If TensorFlow is not installed
        FileNotFoundError: If the model path doesn't exist
    """
    global _surfperch_model
    if _surfperch_model is None:
        try:
            import tensorflow as tf
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ImportError(
                "TensorFlow is required for the SurfPerch model\n"
                "pip install tensorflow>=2.12"
            ) from e

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SurfPerch model not found at {model_path}. "
                "Please ensure the SurfPerch_v1.0/savedmodel directory exists."
            )

        logger.info(f"Loading SurfPerch model from {model_path} …")
        _surfperch_model = tf.saved_model.load(model_path)
    return _surfperch_model


# --------------------------------------------------------------------------- #
#  Wrapped SurfPerch model
# --------------------------------------------------------------------------- #
class SurfPerchModel(ModelBase):
    """
    Wrapper that exposes the SurfPerch (Underwater Fish Audio Classifier) through
    the same interface as other `ModelBase` subclasses.

    *Input expected*: raw waveform, **32 kHz**, 5 s (160 000 samples) or longer.
      If longer than 5 s, the audio is centre-cropped; if shorter, it is
      right-padded with zeros.

    The SurfPerch model returns embeddings and logits for fish sound classification.
    We expose embeddings directly and, if `num_classes>0`, learn a small PyTorch
    linear classifier on top of them (kept on the PyTorch device).
    """

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        num_classes: int,
        device: str = "cpu",
        audio_config: Optional[Dict[str, Any]] = None,
        target_sample_rate: int = 32_000,
        window_seconds: float = 5.0,
        freeze_backbone: bool = True,
        model_path: str = _SURFPERCH_MODEL_PATH,
    ) -> None:
        """Initialize SurfPerchModel.

        Args:
            num_classes: Number of output classes for classification head
            device: PyTorch device to use
            audio_config: Optional audio configuration (kept for API compatibility)
            target_sample_rate: Expected sample rate in Hz
            window_seconds: Duration of audio window in seconds
            freeze_backbone: Whether to freeze the backbone (currently unused)
            model_path: Path to the SurfPerch SavedModel directory
        """
        super().__init__(device, audio_config)

        self.num_classes = num_classes
        self.target_sr = target_sample_rate
        self.window_samples = int(window_seconds * self.target_sr)
        self.model_path = model_path

        # ↳ initialise SurfPerch backbone and infer embedding dimension once
        _load_surfperch_model(self.model_path)  # Ensure model is loaded
        dummy = torch.zeros(1, self.window_samples)
        emb_dim = int(self._tf_forward(dummy).shape[-1])
        self.embedding_dim = emb_dim

        # Optional classification head (learnable in PyTorch)
        self.classifier: Optional[nn.Linear] = (
            nn.Linear(self.embedding_dim, num_classes) if num_classes > 0 else None
        )

        self.to(device)

    # --------------------------------------------------------------------- #
    #  Private helpers
    # --------------------------------------------------------------------- #
    def _prepare_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        """Prepare waveform for Perch model input.

        • Ensures correct shape [B, N] and sample-rate.
        • Pads / crops to exactly 5 s (160 k samples).

        Args:
            wav: Input waveform tensor

        Returns:
            torch.Tensor: Prepared waveform of shape [B, window_samples]

        Raises:
            ValueError: If audio shape is not compatible
        """
        if wav.dim() == 3 and wav.size(1) == 1:  # (B,1,N) → (B,N)
            wav = wav.squeeze(1)

        if wav.dim() != 2:
            raise ValueError("Audio must be (batch, samples) waveform.")

        if wav.size(-1) != self.window_samples:
            if wav.size(-1) > self.window_samples:  # centre-crop
                start = (wav.size(-1) - self.window_samples) // 2
                wav = wav[:, start : start + self.window_samples]
            else:  # right-pad
                pad = self.window_samples - wav.size(-1)
                wav = torch.nn.functional.pad(wav, (0, pad))

        return wav

    @torch.inference_mode()
    def _tf_forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Run the SurfPerch TensorFlow backbone and return embeddings.

        Runs the SurfPerch backbone and returns embeddings as a *torch* tensor.

        Args:
            audio: Input audio tensor of shape [B, N]

        Returns:
            torch.Tensor: Embeddings tensor from SurfPerch model
        """
        import tensorflow as tf  # local import after verification

        tf_model = _load_surfperch_model(self.model_path)

        # → NumPy float32, shape [B, N]
        audio_np = audio.detach().cpu().float().numpy()
        tf_audio = tf.convert_to_tensor(audio_np, dtype=tf.float32)

        # The SurfPerch SavedModel exposes a serving_default signature
        # expecting a float32 tensor under the keyword argument *inputs* 
        # of shape [B, 160000]. It returns embeddings and logits.
        outputs = tf_model.signatures["serving_default"](inputs=tf_audio)

        # Extract embeddings from the model output
        # Note: We need to check what keys the SurfPerch model uses
        if "output_1" in outputs:
            emb = outputs["output_1"].numpy()
        elif "embeddings" in outputs:
            emb = outputs["embeddings"].numpy()
        else:
            # Fallback: use the first output if standard keys are not found
            keys = list(outputs.keys())
            logger.warning(f"Using fallback key '{keys[0]}' for embeddings. Available keys: {keys}")
            emb = outputs[keys[0]].numpy()

        return torch.from_numpy(emb)

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def extract_features(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,  # unused
    ) -> torch.Tensor:
        """Extract features from audio using the SurfPerch backbone.

        Args:
            audio: Input audio tensor
            padding_mask: Unused, kept for API compatibility

        Returns:
            torch.Tensor: Extracted feature embeddings
        """
        audio = self._prepare_waveform(audio).to(self.device)
        emb = self._tf_forward(audio)
        return emb.to(audio.device)

    def forward(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the SurfPerch model.

        Args:
            audio: Input audio tensor
            padding_mask: Optional padding mask (unused)

        Returns:
            torch.Tensor: Model output (features or classification logits)
        """
        feats = self.extract_features(audio, padding_mask)
        return self.classifier(feats) if self.classifier else feats

    # ----------------------------------------------------------------- #
    #  Compatibility helper for evaluation pipeline
    # ----------------------------------------------------------------- #
    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        layers: list[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        """Return SurfPerch embeddings irrespective of *layers* argument.

        The generic :py:meth:`ModelBase.extract_embeddings` relies on PyTorch
        forward hooks which do not apply to the SurfPerch TensorFlow backbone.  
        Instead we ignore *layers* (only logging a warning if the caller requested a
        specific one) and return the features produced by
        :py:meth:`extract_features`.

        Args:
            x: Input audio tensor or dictionary containing 'raw_wav' key
            layers: Layer names (ignored for SurfPerch model)
            padding_mask: Optional padding mask (unused)

        Returns:
            torch.Tensor: Extracted embeddings from the SurfPerch backbone
        """

        if layers and layers != ["last_layer"]:
            logger.warning(
                "SurfPerchModel ignores layer selection; returning backbone "
                "embeddings regardless (requested layers=%s).",
                layers,
            )

        if isinstance(x, dict):
            audio = x["raw_wav"]
        else:
            audio = x

        return self.extract_features(audio, padding_mask)


# Alias for consistency
Model = SurfPerchModel
