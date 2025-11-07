"""
Perch model implementation for bird-audio classification and embedding extraction
using the official TF-Hub export
https://tfhub.dev/google/bird-vocalization-classifier/4
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Utility: lazy TensorFlow import (keeps PyTorch-only envs light-weight)
# --------------------------------------------------------------------------- #
_TF_HUB_HANDLE = "https://tfhub.dev/google/bird-vocalization-classifier/4"
_tf_hub_model = None  # loaded on first use


def _load_tf_model() -> Any:  # noqa: ANN401
    """Load the TensorFlow Hub model on first use.

    Returns:
        Any: The loaded TensorFlow Hub model

    Raises:
        ImportError: If TensorFlow or TensorFlow Hub are not installed
    """
    global _tf_hub_model
    if _tf_hub_model is None:
        try:
            import tensorflow_hub as hub
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ImportError(
                "TensorFlow and tensorflow-hub are required for the Perch model\n"
                "pip install tensorflow>=2.12 tensorflow-hub"
            ) from e

        logger.info("Downloading Perch model from TF-Hub …")
        _tf_hub_model = hub.load(_TF_HUB_HANDLE)
    return _tf_hub_model


# --------------------------------------------------------------------------- #
#  Wrapped Perch model
# --------------------------------------------------------------------------- #
class PerchModel(ModelBase):
    """
    Wrapper that exposes Google's Perch (Bird-Vocalization-Classifier) through
    the same interface as other `ModelBase` subclasses.

    *Input expected*: raw waveform, **32 kHz**, 5 s (160 000 samples) or longer.
      If longer than 5 s, the audio is centre-cropped; if shorter, it is
      right-padded with zeros.

    The TF model returns:
      • `outputs["output_1"]`  – embeddings, shape `[B, 1280]`
      • `outputs["output_0"]`  – logits,     shape `[B, 10 932]`

    We expose embeddings directly and, if `num_classes>0`, learn a small PyTorch
    linear classifier on top of them (kept on the PyTorch device).
    """

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        num_classes: Optional[int] = None,
        device: str = "cpu",
        audio_config: Optional[Dict[str, Any]] = None,
        target_sample_rate: int = 32_000,
        window_seconds: float = 5.0,
        freeze_backbone: bool = True,
    ) -> None:
        """Initialize PerchModel.

        Args:
            num_classes: Number of output classes for classification head (None or 0 for feature extraction only)
            device: PyTorch device to use
            audio_config: Optional audio configuration (kept for API compatibility)
            target_sample_rate: Expected sample rate in Hz
            window_seconds: Duration of audio window in seconds
            freeze_backbone: Whether to freeze the backbone (currently unused)
        """
        super().__init__(device, audio_config)

        # Treat None as 0 (feature extraction only)
        if num_classes is None:
            num_classes = 0

        self.num_classes = num_classes
        self.target_sr = target_sample_rate
        self.window_samples = int(window_seconds * self.target_sr)

        # ↳ initialise TF-Hub backbone and infer embedding dimension once
        _load_tf_model()  # Ensure model is loaded
        dummy = torch.zeros(1, self.window_samples)
        emb_dim = int(self._tf_forward(dummy).shape[-1])
        self.embedding_dim = emb_dim

        # Optional classification head (learnable in PyTorch)
        self.classifier: Optional[nn.Linear] = nn.Linear(self.embedding_dim, num_classes) if num_classes > 0 else None

        self.to(device)

    def _discover_linear_layers(self) -> None:
        """Discover and cache layers for Perch model.

        Note: Perch is a TensorFlow Hub model with no accessible intermediate layers.
        The TensorFlow Hub model only exposes the final embedding output.
        This method is implemented for API consistency but will only find the optional
        PyTorch classifier layer.
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            # Perch is a TensorFlow Hub model with no accessible intermediate layers
            # Only the optional PyTorch classifier layer can be discovered
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Linear):
                    self._layer_names.append(name)

            logger.info(f"Discovered {len(self._layer_names)} layers in Perch model: {self._layer_names}")
            if len(self._layer_names) == 0:
                logger.info(
                    "Perch is a TensorFlow Hub model with no accessible "
                    "intermediate layers. "
                    "Only the optional PyTorch classifier layer can be discovered."
                )

    def _discover_embedding_layers(self) -> None:
        """Discover and cache layers for Perch model.

        Note: Perch is a TensorFlow Hub model with no accessible intermediate layers.
        The TensorFlow Hub model only exposes the final embedding output.
        This method is implemented for API consistency but will only find the optional
        PyTorch classifier layer.
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            # Perch is a TensorFlow Hub model with no accessible intermediate layers
            # Only the optional PyTorch classifier layer can be discovered
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Linear):
                    self._layer_names.append(name)

            logger.info(f"Discovered {len(self._layer_names)} layers in Perch model: {self._layer_names}")
            if len(self._layer_names) == 0:
                logger.info(
                    "Perch is a TensorFlow Hub model with no accessible "
                    "intermediate layers. "
                    "Only the optional PyTorch classifier layer can be discovered."
                )

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
        """Run the TensorFlow backbone and return embeddings.

        Runs the TF backbone and returns embeddings as a *torch* tensor.

        Args:
            audio: Input audio tensor of shape [B, N]

        Returns:
            torch.Tensor: Embeddings tensor of shape [B, 1280]
        """
        import tensorflow as tf  # local import after verification

        tf_model = _load_tf_model()

        # → NumPy float32, shape [B, N]
        audio_np = audio.detach().cpu().float().numpy()
        tf_audio = tf.convert_to_tensor(audio_np, dtype=tf.float32)

        # The TF-Hub module exposes a SavedModel with a single signature
        # called "serving_default" expecting a float32 tensor under the
        # keyword argument *inputs* of shape [B, 160000].  It returns two
        # tensors:
        #   • output_1 – embedding, shape [B, 1280]
        #   • output_0 – logits,    shape [B, 10 932]

        outputs = tf_model.signatures["serving_default"](inputs=tf_audio)

        # Embeddings are stored under key "output_1"
        emb = outputs["output_1"].numpy()  # np.ndarray [B, 1280]

        return torch.from_numpy(emb)

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def extract_features(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,  # unused
    ) -> torch.Tensor:
        """Extract features from audio using the Perch backbone.

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
        """Forward pass through the Perch model.

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
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean",
    ) -> torch.Tensor | list[torch.Tensor]:  # type: ignore[override]
        """Return Perch embeddings with support for different aggregation methods.

        The generic :py:meth:`ModelBase.extract_embeddings` relies on PyTorch
        forward hooks which do not apply to the TF-Hub backbone.  Instead we
        return the features produced by :py:meth:`extract_features`.

        Args:
            x: Input audio tensor or dictionary containing 'raw_wav' key
            padding_mask: Optional padding mask (unused)
            aggregation: Aggregation method - "mean", "max", "cls_token", or "none"

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Extracted embeddings based on
            aggregation method

        Raises:
            ValueError: If unsupported aggregation method is provided.
        """

        if isinstance(x, dict):
            audio = x["raw_wav"]
        else:
            audio = x

        # Extract base embeddings
        embeddings = self.extract_features(audio, padding_mask)

        # Handle different aggregation methods
        if aggregation == "none":
            # For sequence probes, return list of embeddings
            # Since Perch is a single-layer model, we return a list with one element
            # Reshape from (B, 1280) to (B, 1, 1280) to make it 3D for sequence probes
            batch_size, embed_dim = embeddings.shape
            reshaped_embeddings = embeddings.unsqueeze(1)  # (B, 1, 1280)
            return [reshaped_embeddings]
        elif aggregation == "mean":
            # Default behavior - return embeddings as is
            return embeddings
        elif aggregation == "max":
            # Max pooling over time dimension (if applicable)
            return embeddings
        elif aggregation == "cls_token":
            # For Perch, treat as mean since no CLS token
            return embeddings
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")


# Alias for consistency
Model = PerchModel
