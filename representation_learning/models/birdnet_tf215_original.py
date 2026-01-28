"""BirdNet model implementation for bioacoustic classification (TF 2.15 baseline).

This module is a direct copy of the original BirdNet wrapper from `main`
prior to the TensorFlow 2.17.0+ embedding bug fix.

It is used as a reference implementation when running with
TensorFlow 2.15.x, so we can compare embeddings against the new
implementation in `birdnet.py`.
"""

# NOTE:
# This file is intentionally kept as close as possible to the original
# implementation from `origin/main:representation_learning/models/birdnet.py`.
# Do not modify it except to sync with that baseline for comparison tests.

import contextlib
import logging
import os
import sys
import tempfile
from typing import Any, Dict, Generator, Optional

import numpy as np
import soundfile as sf  # lightweight; writes the tmp .wav we feed in
import torch
import torch.nn as nn

# NOTE: birdnetlib imports are LAZY to avoid TensorFlow setting CUDA_VISIBLE_DEVICES=""
# which would break PyTorch CUDA availability. Imports happen in __init__ and forward methods.
from representation_learning.models.base_model import (
    ModelBase,
)  # Add missing import

logger = logging.getLogger(__name__)

# Suppress verbose logging from birdnetlib
logging.getLogger("birdnetlib").setLevel(logging.WARNING)
# Also suppress any TensorFlow logging if present
logging.getLogger("tensorflow").setLevel(logging.WARNING)
# Suppress any other potential verbose loggers
for logger_name in ["absl", "h5py", "soundfile"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


@contextlib.contextmanager
def suppress_stdout_stderr() -> Generator[None, None, None]:
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class Model(ModelBase):
    """
    Thin wrapper around *birdnetlib* that exposes:
        • forward()  – returns clip-level class probabilities (rarely needed)
        • extract_embeddings() – 1024-d BirdNET feature vectors
    Everything else (batching, device placement, …) comes from ModelBase.
    """

    SAMPLE_RATE = 48_000
    CHUNK_SEC = 3.0  # fixed window length inside the model

    def __init__(
        self,
        num_classes: Optional[int] = None,
        device: str = "cpu",
        audio_config: Optional[Dict[str, Any]] = None,
        *,
        language: str = "en_us",
        apply_sigmoid: bool = True,
        freeze_backbone: bool = True,
        return_features_only: bool = False,
        **kwargs,  # Accept additional config parameters  # noqa: ANN003
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # Preserve CUDA_VISIBLE_DEVICES before importing TensorFlow
        # TensorFlow sets it to "" which breaks PyTorch CUDA
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # Lazy import to avoid TensorFlow setting CUDA_VISIBLE_DEVICES=""
        from birdnetlib.analyzer import Analyzer

        self._analyzer = Analyzer()  # classification helper

        # Restore CUDA_VISIBLE_DEVICES after TensorFlow import
        if cuda_visible_devices is None:
            # Remove the empty string that TensorFlow set
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            # Restore the original value
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self._interpreter = self._analyzer.interpreter  # TFLite interpreter
        self.species = self._analyzer.labels  # 6 522 labels
        self.num_species = len(self.species)

        self.return_features_only = return_features_only
        self.language = language
        self.apply_sigmoid = apply_sigmoid
        self.freeze_backbone = freeze_backbone

        # Handle return_features_only: if True, force num_classes to None
        if return_features_only:
            self.num_classes = None
        else:
            self.num_classes = num_classes

        if (
            not return_features_only
            and self.num_classes is not None
            and self.num_classes > 0
            and self.num_classes != self.num_species
        ):
            self.classifier = nn.Linear(self.num_species, self.num_classes)
            # Move classifier to the specified device
            if device != "cpu" and torch.cuda.is_available():
                self.classifier = self.classifier.to(device)
        else:
            self.classifier = None

        logger.info(
            "BirdNetTFLite ready – v2.4 • %d species • embeddings dim = 1024 • num_classes = %s • device = %s",
            self.num_species,
            self.num_classes if self.num_classes is not None else "None",
            device,
        )

    def _discover_linear_layers(self) -> None:
        """Discover and cache layers for BirdNET model.

        Note
        ----
        BirdNET is a TensorFlow Lite model with no accessible intermediate layers.
        The TensorFlow Lite model only exposes the final output (species predictions).
        This method is implemented for API consistency but will only find the optional
        PyTorch classifier layer.
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            # BirdNET is a TensorFlow Lite model with no accessible intermediate layers
            # Only the optional PyTorch classifier layer can be discovered
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Linear):
                    self._layer_names.append(name)

            logger.info(f"Discovered {len(self._layer_names)} layers in BirdNET model: {self._layer_names}")
            if len(self._layer_names) == 0:
                logger.info(
                    "BirdNET is a TensorFlow Lite model with no accessible "
                    "intermediate layers. "
                    "Only the optional PyTorch classifier layer can be discovered."
                )

    def _discover_embedding_layers(self) -> None:
        """Discover and cache layers for BirdNET model.

        Note
        ----
        BirdNET is a TensorFlow Lite model with no accessible intermediate layers.
        The TensorFlow Lite model only exposes the final output (species predictions).
        This method is implemented for API consistency but will only find the optional
        PyTorch classifier layer.
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            # BirdNET is a TensorFlow Lite model with no accessible intermediate layers
            # Only the optional PyTorch classifier layer can be discovered
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Linear):
                    self._layer_names.append(name)

            logger.info(f"Discovered {len(self._layer_names)} layers in BirdNET model: {self._layer_names}")
            if len(self._layer_names) == 0:
                logger.info(
                    "BirdNET is a TensorFlow Lite model with no accessible "
                    "intermediate layers. "
                    "Only the optional PyTorch classifier layer can be discovered."
                )

    # --------------------------------------------------------------------- #
    #                       NN.Module / ModelBase hooks                     #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        wav: torch.Tensor,  # (B, T) mono @48 kHz
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for BirdNET.

        Parameters
        ----------
        wav :
            Input waveform tensor with shape ``(B, T)``.
        padding_mask :
            Optional padding mask (unused, for API compatibility).

        Returns
        -------
        torch.Tensor
            Averaged logits for the audio clip.
        """
        wav = wav.detach().cpu()
        probs = []
        for clip in wav:
            probs.append(self._infer_clip(clip.numpy()))

        result = torch.stack(probs)  # (B, Nclasses)

        # Move to appropriate device - BirdNet core runs on CPU,
        # but classifier might be on GPU
        if self.classifier is not None:
            # Move to classifier's device for processing
            result = result.to(next(self.classifier.parameters()).device)
            result = self.classifier(result)
        else:
            # No classifier, move to model's device
            result = result.to(self.device)

        return result

    # ------------------------------------------------------------------ #
    #                  ***  Embedding extraction API  ***                #
    # ------------------------------------------------------------------ #
    def extract_embeddings(
        self,
        x: torch.Tensor | Dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean",
    ) -> torch.Tensor | list[torch.Tensor]:
        """Extract embeddings for the given audio input.

        Parameters
        ----------
        x :
            Input tensor or mapping containing ``\"raw_wav\"``.
        padding_mask :
            Optional padding mask (unused, for API compatibility).
        aggregation :
            Aggregation method. One of ``\"mean\"``, ``\"max\"``, ``\"none\"``,
            or ``\"cls_token\"``.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            Aggregated embeddings based on the specified method.

        Raises
        ------
        ValueError
            If an unsupported aggregation method is provided.
        """
        if isinstance(x, dict):  # allow {'raw_wav': …}
            wav = x["raw_wav"]
        else:
            wav = x
        wav = wav.detach().cpu()

        batch_out = []
        for clip in wav:
            emb_np = self._embedding_for_clip(clip.numpy())  # (N,1024) or (1024,)
            emb = torch.from_numpy(emb_np).float()
            # Ensure embeddings are at least 2D: (N, 1024) or (1, 1024)
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)  # (1024,) -> (1, 1024)
            batch_out.append(emb)

        # Handle different aggregation methods
        if aggregation == "none":
            # For sequence probes, return list of embeddings per batch item
            # Each batch item gets a list of chunk embeddings
            result = []
            for clip_embeddings in batch_out:
                # Reshape from (N, 1024) to (1, N, 1024) to make it 3D for
                # sequence probes
                # N is the number of chunks, 1024 is the embedding dimension
                reshaped = clip_embeddings.unsqueeze(0)  # (1, N, 1024)
                result.append(reshaped)
            return result
        if aggregation == "mean":
            # Average over time dimension for each clip
            result = torch.stack([emb.mean(0) for emb in batch_out])  # (B, 1024)
        elif aggregation == "max":
            # Max pooling over time dimension for each clip
            result = torch.stack([emb.max(0)[0] for emb in batch_out])  # (B, 1024)
        elif aggregation == "cls_token":
            # For BirdNET, treat as mean since no CLS token
            result = torch.stack([emb.mean(0) for emb in batch_out])  # (B, 1024)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        # Move to appropriate device (BirdNet embeddings are computed on CPU)
        if self.classifier is not None and hasattr(self.classifier, "weight"):
            # If we have a classifier, move to its device
            result = result.to(next(self.classifier.parameters()).device)
        else:
            # Otherwise move to model device
            result = result.to(self.device)

        return result

    # ------------------------------------------------------------------ #
    #                          Helper functions                          #
    # ------------------------------------------------------------------ #
    def _infer_clip(self, mono_wave: np.ndarray) -> torch.Tensor:
        """Infer clip-level probabilities from mono audio.

        Parameters
        ----------
        mono_wave :
            Mono audio waveform with sample rate ``SAMPLE_RATE``.

        Returns
        -------
        torch.Tensor
            Probability vector for the audio clip.
        """
        # Lazy import to avoid TensorFlow setting CUDA_VISIBLE_DEVICES=""
        from birdnetlib import Recording

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, mono_wave, self.SAMPLE_RATE)
            with suppress_stdout_stderr():
                rec = Recording(self._analyzer, tmp.name, min_conf=0.0)
                rec.analyze()
            # build a species-probability vector from detections
            scores = np.zeros(self.num_species, dtype=np.float32)
            for det in rec.detections:
                idx = self.species.index(det["label"])
                scores[idx] = max(scores[idx], det["confidence"])
            return torch.from_numpy(scores)

    def _embedding_for_clip(self, mono_wave: np.ndarray) -> np.ndarray:
        """Extract an embedding vector for a single clip.

        This directly grabs the **embedding output tensor** from BirdNET's
        underlying TFLite *Interpreter* – the tensor right before logits.

        Parameters
        ----------
        mono_wave :
            Audio waveform data.

        Returns
        -------
        np.ndarray
            Embedding vector for the audio clip.

        Raises
        ------
        ValueError
            If embedding output tensor cannot be found in model.
        """
        # Use the analyzer's built-in embedding extraction method if available
        if hasattr(self._analyzer, "extract_embeddings_for_recording"):
            # Lazy import to avoid TensorFlow setting CUDA_VISIBLE_DEVICES=""
            from birdnetlib import Recording

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, mono_wave, self.SAMPLE_RATE)
                try:
                    # Create Recording object, analyze, and extract embeddings
                    with suppress_stdout_stderr():
                        recording = Recording(self._analyzer, tmp.name)
                        recording.analyze()
                        recording.extract_embeddings()

                    # Get embeddings from the recording
                    embeddings_data = recording.embeddings

                    if embeddings_data and len(embeddings_data) > 0:
                        # Extract embeddings from the first (and typically only) segment
                        first_segment = embeddings_data[0]
                        if isinstance(first_segment, dict) and "embeddings" in first_segment:
                            embeddings_list = first_segment["embeddings"]
                            if isinstance(embeddings_list, list) and len(embeddings_list) > 0:
                                # Convert to numpy array and ensure 2D shape (1, 1024)
                                embeddings = np.array(embeddings_list, dtype=np.float32)
                                if embeddings.ndim == 1:
                                    embeddings = embeddings.reshape(1, -1)  # (1024,) -> (1, 1024)
                                return embeddings

                    # If we get here, the built-in method didn't work as expected
                    logger.debug("Built-in embedding extraction returned unexpected format")
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"Built-in embedding extraction failed: {exc}")
                    # Fall back to manual extraction

        # Manual extraction as fallback
        interp = self._interpreter

        # Lazy import to avoid TensorFlow setting CUDA_VISIBLE_DEVICES=""
        from birdnetlib import RecordingBuffer

        # Feed through RecordingBuffer which handles resampling + spectrograms
        with suppress_stdout_stderr():
            buf = RecordingBuffer(self._analyzer, mono_wave, self.SAMPLE_RATE)
            buf.analyze()

        # Ensure tensors are allocated
        try:
            interp.allocate_tensors()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to allocate tensors: {exc}")
            raise

        # Try different approaches to find embeddings
        embeddings: np.ndarray | None = None

        # First try: check if there are multiple outputs (old BirdNet format)
        out_info = interp.get_output_details()
        if len(out_info) > 1:
            try:
                emb_idx = out_info[1]["index"]
                embeddings = interp.get_tensor(emb_idx)
            except Exception:  # noqa: BLE001
                embeddings = None

        # Second try: search tensor details for embedding layer
        if embeddings is None:
            tensor_details = interp.get_tensor_details()

            # Look for the global average pooling layer first (most reliable)
            for detail in tensor_details:
                name = detail.get("name", "")
                shape = detail.get("shape", [])
                if "GLOBAL_AVG_POOL" in name and len(shape) == 2 and 1024 in shape:
                    try:
                        embeddings = interp.get_tensor(detail.get("index"))
                        break
                    except Exception:  # noqa: BLE001
                        continue

            # Fallback: look for any 1024-dimensional tensor
            if embeddings is None:
                for detail in tensor_details:
                    shape = detail.get("shape", [])
                    name = detail.get("name", "")
                    # Normalize shape to tuple for comparison (handles both list, tuple, and numpy array)
                    # Check if shape exists and has elements before converting
                    if shape is not None and len(shape) > 0:
                        shape_tuple = tuple(shape)
                    else:
                        shape_tuple = ()
                    if (
                        len(shape) >= 1
                        and 1024 in shape
                        and (
                            "embedding" in name.lower()
                            or "feature" in name.lower()
                            or shape_tuple == (1024,)
                            or shape_tuple == (1, 1024)
                        )
                    ):
                        try:
                            embeddings = interp.get_tensor(detail.get("index"))
                            break
                        except Exception:  # noqa: BLE001
                            continue

        if embeddings is None:
            raise ValueError(
                f"Could not find embedding tensor in BirdNet model. "
                f"Available outputs: {len(out_info)} with shapes: "
                f"{[detail.get('shape') for detail in out_info]}. "
                f"This may be a newer BirdNet model format that does not "
                f"expose embeddings."
            )

        embeddings = embeddings.astype(np.float32)
        # Ensure embeddings are at least 2D: (N, 1024) or (1, 1024)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)  # (1024,) -> (1, 1024)
        return embeddings

    # ------------------------------------------------------------------ #
    #                 OPTIONALS EXPECTED BY ModelBase                    #
    # ------------------------------------------------------------------ #
    def enable_gradient_checkpointing(self) -> None:
        """Gradient checkpointing is not supported for BirdNET."""
        logger.warning("Gradient checkpointing is not supported for BirdNET.")

    def to(self, device: torch.device | str) -> "Model":
        """Move model (classifier) to the specified device.

        Parameters
        ----------
        device :
            Target device or device string.

        Returns
        -------
        Model
            Self for method chaining.
        """
        # Update internal device tracking
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        # Move PyTorch components (classifier) to the device
        if self.classifier is not None:
            if str(device).startswith("cuda") and torch.cuda.is_available():
                self.classifier = self.classifier.to(device)
            else:
                self.classifier = self.classifier.cpu()

        # TensorFlow Lite interpreter stays on CPU - no need to move
        return self

    def cpu(self) -> "Model":
        """Move model to CPU.

        Returns
        -------
        Model
            Self for method chaining.
        """
        return self.to("cpu")

    def cuda(self, device: int | None = None) -> "Model":
        """Move model to a CUDA device.

        Parameters
        ----------
        device :
            Optional CUDA device index.

        Returns
        -------
        Model
            Self for method chaining.
        """
        device_str = f"cuda:{device}" if device is not None else "cuda"
        return self.to(device_str)

    # ------------------------------------------------------------------ #
    # Convenience to expose BirdNET's species mapping
    # ------------------------------------------------------------------ #
    def idx_to_species(self, idx: int) -> str:
        """Map class index to species name.

        Parameters
        ----------
        idx :
            Class index.

        Returns
        -------
        str
            Species name corresponding to the given index.
        """
        return self.species[idx]

    def species_to_idx(self, name: str) -> int:
        """Map species name to class index.

        Parameters
        ----------
        name :
            Species name.

        Returns
        -------
        int
            Class index corresponding to the given species name.
        """
        return self.species.index(name)
