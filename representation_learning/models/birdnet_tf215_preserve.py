"""BirdNet model: TF 2.15-style implementation with interpreter preservation flag.

This is a copy of `birdnet_tf215_original.Model` but with a single change:
after constructing the birdnetlib `Analyzer`, we try to recreate the
underlying TFLite interpreter with
`experimental_preserve_all_tensors=True`.

The embedding extraction path (using birdnetlib's Recording / RecordingBuffer)
is otherwise unchanged. This lets us test whether the TensorFlow >=2.17
embedding bug can be fixed *only* by setting this flag, without changing
spectrogram computation or where we tap the graph.
"""

from __future__ import annotations

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

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)

# Suppress verbose logging from birdnetlib
logging.getLogger("birdnetlib").setLevel(logging.WARNING)
# Also suppress any TensorFlow logging if present
logging.getLogger("tensorflow").setLevel(logging.WARNING)
# Suppress any other potential verbose loggers
for _logger_name in ["absl", "h5py", "soundfile"]:
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


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
    """BirdNet wrapper matching the original implementation plus preserve-all-tensors."""

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
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # Preserve CUDA_VISIBLE_DEVICES before importing TensorFlow
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # Lazy import to avoid TensorFlow setting CUDA_VISIBLE_DEVICES=""
        from birdnetlib.analyzer import Analyzer

        self._analyzer = Analyzer()  # classification helper

        # Restore CUDA_VISIBLE_DEVICES after TensorFlow import
        if cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        # Try to recreate interpreter with experimental_preserve_all_tensors=True
        try:
            import tensorflow as tf

            model_path: Optional[str] = None
            if hasattr(self._analyzer, "model_path"):
                model_path = self._analyzer.model_path
            elif hasattr(self._analyzer, "_model_path"):
                model_path = self._analyzer._model_path  # type: ignore[assignment]
            elif hasattr(self._analyzer, "interpreter") and hasattr(
                self._analyzer.interpreter,
                "_model_path",
            ):
                model_path = self._analyzer.interpreter._model_path  # type: ignore[attr-defined]

            if model_path:
                logger.info(
                    "Recreating BirdNet interpreter with experimental_preserve_all_tensors=True (TF >= 2.17 fix).",
                )
                self._interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_preserve_all_tensors=True,
                )
            else:
                logger.warning(
                    "Could not access model path to recreate BirdNet interpreter. "
                    "Using birdnetlib's interpreter directly.",
                )
                self._interpreter = self._analyzer.interpreter
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to recreate BirdNet interpreter with "
                "experimental_preserve_all_tensors=True: %s. "
                "Using original interpreter.",
                exc,
            )
            self._interpreter = self._analyzer.interpreter

        self.species = self._analyzer.labels  # 6 522 labels
        self.num_species = len(self.species)

        self.return_features_only = return_features_only
        self.language = language
        self.apply_sigmoid = apply_sigmoid
        self.freeze_backbone = freeze_backbone

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
            if device != "cpu" and torch.cuda.is_available():
                self.classifier = self.classifier.to(device)
        else:
            self.classifier = None

        logger.info(
            "BirdNetTFLite (preserve-all-tensors) – v2.4 • %d species • "
            "embeddings dim = 1024 • num_classes = %s • device = %s",
            self.num_species,
            self.num_classes if self.num_classes is not None else "None",
            device,
        )

    # The rest of the implementation is identical to birdnet_tf215_original:

    def _discover_linear_layers(self) -> None:
        """Discover and cache layers for BirdNET model."""
        if len(self._layer_names) == 0:
            self._layer_names = []
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Linear):
                    self._layer_names.append(name)
            logger.info(
                "Discovered %d layers in BirdNET model: %s",
                len(self._layer_names),
                self._layer_names,
            )

    def _discover_embedding_layers(self) -> None:
        """Discover and cache layers for BirdNET model (classifier only)."""
        if len(self._layer_names) == 0:
            self._layer_names = []
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Linear):
                    self._layer_names.append(name)
            logger.info(
                "Discovered %d layers in BirdNET model: %s",
                len(self._layer_names),
                self._layer_names,
            )

    def forward(
        self,
        wav: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning clip-level logits or probabilities.

        Returns
        -------
        torch.Tensor
            Batch of clip-level logits or probabilities with shape
            ``(batch_size, num_classes)`` when a classifier head is present,
            otherwise ``(batch_size, num_species)``.
        """
        del padding_mask
        wav = wav.detach().cpu()
        probs = []
        for clip in wav:
            probs.append(self._infer_clip(clip.numpy()))
        result = torch.stack(probs)
        if self.classifier is not None:
            result = result.to(next(self.classifier.parameters()).device)
            result = self.classifier(result)
        else:
            result = result.to(self.device)
        return result

    def extract_embeddings(
        self,
        x: torch.Tensor | Dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean",
    ) -> torch.Tensor | list[torch.Tensor]:
        """Extract embeddings using the same path as the original BirdNet wrapper.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            If ``aggregation`` is ``\"none\"``, returns a list of per-clip
            embeddings with shape ``(1, embedding_dim)`` each. Otherwise,
            returns a tensor of aggregated embeddings for the batch with shape
            ``(batch_size, embedding_dim)``.

        Raises
        ------
        ValueError
            If an unsupported aggregation method is provided.
        """
        del padding_mask
        if isinstance(x, dict):
            wav = x["raw_wav"]
        else:
            wav = x
        wav = wav.detach().cpu()

        batch_out = []
        for clip in wav:
            emb_np = self._embedding_for_clip(clip.numpy())
            emb = torch.from_numpy(emb_np).float()
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            batch_out.append(emb)

        if aggregation == "none":
            result_list: list[torch.Tensor] = []
            for clip_embeddings in batch_out:
                result_list.append(clip_embeddings.unsqueeze(0))
            return result_list
        if aggregation == "mean":
            result = torch.stack([emb.mean(0) for emb in batch_out])
        elif aggregation == "max":
            result = torch.stack([emb.max(0)[0] for emb in batch_out])
        elif aggregation == "cls_token":
            result = torch.stack([emb.mean(0) for emb in batch_out])
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        if self.classifier is not None and hasattr(self.classifier, "weight"):
            result = result.to(next(self.classifier.parameters()).device)
        else:
            result = result.to(self.device)
        return result

    def _infer_clip(self, mono_wave: np.ndarray) -> torch.Tensor:
        """Use Analyzer to get per-clip probabilities (unchanged).

        Returns
        -------
        torch.Tensor
            Tensor of per-species probabilities with shape ``(num_species,)``.
        """
        from birdnetlib import Recording

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, mono_wave, self.SAMPLE_RATE)
            with suppress_stdout_stderr():
                rec = Recording(self._analyzer, tmp.name, min_conf=0.0)
                rec.analyze()
            scores = np.zeros(self.num_species, dtype=np.float32)
            for det in rec.detections:
                idx = self.species.index(det["label"])
                scores[idx] = max(scores[idx], det["confidence"])
            return torch.from_numpy(scores)

    def _embedding_for_clip(self, mono_wave: np.ndarray) -> np.ndarray:
        """Embedding extraction using birdnetlib Recording + interpreter, unchanged.

        Returns
        -------
        numpy.ndarray
            Array of embeddings for the clip with shape ``(num_segments,
            embedding_dim)``.

        Raises
        ------
        ValueError
            If no suitable embedding tensor can be found in the BirdNet model.
        """
        # Use the analyzer's built-in embedding extraction method if available
        if hasattr(self._analyzer, "extract_embeddings_for_recording"):
            from birdnetlib import Recording

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, mono_wave, self.SAMPLE_RATE)
                try:
                    with suppress_stdout_stderr():
                        recording = Recording(self._analyzer, tmp.name)
                        recording.analyze()
                        recording.extract_embeddings()
                    embeddings_data = recording.embeddings
                    if embeddings_data and len(embeddings_data) > 0:
                        first_segment = embeddings_data[0]
                        if isinstance(first_segment, dict) and "embeddings" in first_segment:
                            embeddings_list = first_segment["embeddings"]
                            if isinstance(embeddings_list, list) and len(embeddings_list) > 0:
                                embeddings = np.array(embeddings_list, dtype=np.float32)
                                if embeddings.ndim == 1:
                                    embeddings = embeddings.reshape(1, -1)
                                return embeddings
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Built-in embedding extraction failed: %s", exc)

        # Manual interpreter-based fallback (unchanged from original)
        interp = self._interpreter
        from birdnetlib import RecordingBuffer

        with suppress_stdout_stderr():
            buf = RecordingBuffer(self._analyzer, mono_wave, self.SAMPLE_RATE)
            buf.analyze()

        try:
            interp.allocate_tensors()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to allocate tensors: %s", exc)
            raise

        embeddings: Optional[np.ndarray] = None
        out_info = interp.get_output_details()
        if len(out_info) > 1:
            try:
                emb_idx = out_info[1]["index"]
                embeddings = interp.get_tensor(emb_idx)
            except Exception:  # noqa: BLE001
                embeddings = None

        if embeddings is None:
            tensor_details = interp.get_tensor_details()
            for detail in tensor_details:
                name = detail.get("name", "")
                shape = detail.get("shape", [])
                if "GLOBAL_AVG_POOL" in name and len(shape) == 2 and 1024 in shape:
                    try:
                        embeddings = interp.get_tensor(detail.get("index"))
                        break
                    except Exception:  # noqa: BLE001
                        continue

            if embeddings is None:
                for detail in tensor_details:
                    shape = detail.get("shape", [])
                    name = detail.get("name", "")
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
                "Could not find embedding tensor in BirdNet model. "
                f"Available outputs: {len(out_info)} with shapes: "
                f"{[detail.get('shape') for detail in out_info]}. "
                "This may be a newer BirdNet model format that does not expose embeddings.",
            )

        embeddings = embeddings.astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def enable_gradient_checkpointing(self) -> None:
        """Gradient checkpointing is not supported for BirdNET."""
        logger.warning("Gradient checkpointing is not supported for BirdNET.")

    def to(self, device: torch.device | str) -> "Model":
        """Move model (classifier) to the specified device.

        Returns
        -------
        Model
            The model instance on the requested device.
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if self.classifier is not None:
            if str(device).startswith("cuda") and torch.cuda.is_available():
                self.classifier = self.classifier.to(device)
            else:
                self.classifier = self.classifier.cpu()
        return self

    def cpu(self) -> "Model":
        """Move model to CPU.

        Returns
        -------
        Model
            The model instance moved to CPU.
        """
        return self.to("cpu")

    def cuda(self, device: int | None = None) -> "Model":
        """Move model to a CUDA device.

        Parameters
        ----------
        device
            CUDA device index. If ``None``, uses the default CUDA device.

        Returns
        -------
        Model
            The model instance moved to the requested CUDA device.
        """
        device_str = f"cuda:{device}" if device is not None else "cuda"
        return self.to(device_str)

    def idx_to_species(self, idx: int) -> str:
        """Map class index to species name.

        Parameters
        ----------
        idx
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
        name
            Species name.

        Returns
        -------
        int
            Class index corresponding to the given species name.
        """
        return self.species.index(name)
