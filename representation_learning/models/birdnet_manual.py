"""BirdNet model implementation without birdnetlib embedding fallbacks.

This variant is identical in spirit to ``representation_learning.models.birdnet``,
but it **never** calls ``Analyzer.extract_embeddings_for_recording`` or
``RecordingBuffer.analyze`` to obtain embeddings. Instead, it always:

- Uses ``AudioProcessor`` (via ``ModelBase``) to compute mel spectrograms.
- Applies BirdNet-style post-processing (log-dB, normalization).
- Feeds the result directly into the TensorFlow Lite interpreter that is
  recreated with ``experimental_preserve_all_tensors=True`` when possible.

This makes the behavior independent of birdnetlib's internal embedding
pipeline and TensorFlow 2.17+/2.18 bugs related to intermediate tensors.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from typing import Any, Dict, Generator, Optional

import librosa
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
    """BirdNet wrapper that always uses manual embedding extraction.

    This model:

    - Uses birdnetlib's ``Analyzer`` only for loading the model and labels.
    - Recreates the TFLite interpreter with
      ``experimental_preserve_all_tensors=True`` when possible.
    - Computes mel spectrograms via ``AudioProcessor`` from ``ModelBase``.
    - Feeds processed features into the interpreter and searches for
      a 1024-d embedding tensor.

    It never calls ``extract_embeddings_for_recording`` or
    ``RecordingBuffer.analyze`` to obtain embeddings.
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
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        # Create default audio_config for BirdNet if not provided
        # BirdNet expects 48 kHz audio, mel spectrogram with specific parameters
        if audio_config is None:
            from representation_learning.configs import AudioConfig

            audio_config = AudioConfig(
                sample_rate=self.SAMPLE_RATE,
                representation="mel_spectrogram",
                n_fft=2048,
                hop_length=320,
                n_mels=128,
                target_length_seconds=self.CHUNK_SEC,
                normalize=False,  # BirdNet-specific normalization applied later
            )

        super().__init__(device=device, audio_config=audio_config)

        # Preserve CUDA_VISIBLE_DEVICES before importing TensorFlow
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # Lazy import to avoid TensorFlow setting CUDA_VISIBLE_DEVICES=""
        from birdnetlib.analyzer import Analyzer

        self._analyzer = Analyzer()

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
                    "Recreating BirdNet interpreter with "
                    "experimental_preserve_all_tensors=True for TF 2.17.0+ compatibility",
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

        self.species = self._analyzer.labels
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
            "BirdNetTFLite (manual embeddings) ready – v2.4 • %d species • "
            "embeddings dim = 1024 • num_classes = %s • device = %s",
            self.num_species,
            self.num_classes if self.num_classes is not None else "None",
            device,
        )

    # --------------------------------------------------------------------- #
    #                       NN.Module / ModelBase hooks                     #
    # --------------------------------------------------------------------- #
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

        This always uses the manual AudioProcessor + interpreter path.

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

    # ------------------------------------------------------------------ #
    #                          Helper functions                          #
    # ------------------------------------------------------------------ #
    def _infer_clip(self, mono_wave: np.ndarray) -> torch.Tensor:
        """Use Analyzer to get per-clip probabilities (not embeddings).

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
        """Manual embedding extraction using AudioProcessor and TFLite interpreter.

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
        interp = self._interpreter

        try:
            interp.allocate_tensors()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "BirdNet manual: allocate_tensors() failed: %s. Interpreter may already be allocated.",
                exc,
            )

        input_details = interp.get_input_details()
        if not input_details:
            raise ValueError("BirdNet interpreter has no input details.")

        input_idx = input_details[0]["index"]
        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]

        # Compute mel spectrogram using librosa to better match birdnetlib's
        # internal preprocessing under TensorFlow 2.15.x.
        mel_spec = librosa.feature.melspectrogram(
            y=mono_wave.astype(np.float32),
            sr=self.SAMPLE_RATE,
            n_fft=2048,
            hop_length=320,
            n_mels=128,
            power=2.0,
            center=True,
            window="hann",
        )

        # Convert power mel spectrogram to dB, then normalize similarly to BirdNET:
        # original BirdNET caps at -80 dB and scales into [0, 1].
        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
        mel_spec_db = np.clip(mel_spec_db, -80.0, 0.0)
        mel_spec_db = (mel_spec_db + 80.0) / 80.0

        processed_input = mel_spec_db.T

        if processed_input.shape != tuple(input_shape):
            if processed_input.size >= np.prod(input_shape):
                processed_input = processed_input.flatten()[: np.prod(input_shape)].reshape(input_shape)
            else:
                padded = np.zeros(input_shape, dtype=processed_input.dtype)
                flat_input = processed_input.flatten()
                padded.flat[: len(flat_input)] = flat_input
                processed_input = padded

        input_data = processed_input.astype(input_dtype)

        if np.all(input_data == 0):
            logger.warning("BirdNet manual: input tensor is all zeros.")
        if np.any(np.isnan(input_data)):
            logger.warning("BirdNet manual: input tensor contains NaN values.")

        interp.set_tensor(input_idx, input_data)
        interp.invoke()

        embeddings: Optional[np.ndarray] = None

        out_info = interp.get_output_details()
        if len(out_info) > 1:
            try:
                emb_idx = out_info[1]["index"]
                embeddings = interp.get_tensor(emb_idx).copy()
            except Exception:  # noqa: BLE001
                embeddings = None

        if embeddings is None:
            tensor_details = interp.get_tensor_details()
            for detail in tensor_details:
                name = detail.get("name", "")
                shape = detail.get("shape", [])
                if "GLOBAL_AVG_POOL" in name and len(shape) == 2 and 1024 in shape:
                    try:
                        tensor_idx = detail.get("index")
                        tensor_data = interp.get_tensor(tensor_idx)
                        if tensor_data is not None and tensor_data.size > 0:
                            embeddings = tensor_data.copy()
                            break
                    except ValueError as exc:  # noqa: BLE001
                        if "Tensor data is null" in str(exc):
                            logger.debug(
                                "BirdNet manual: tensor %s (%s) data is null (TF 2.17+ bug).",
                                detail.get("index"),
                                name,
                            )
                    except Exception:  # noqa: BLE001
                        continue

        if embeddings is None:
            tensor_details = interp.get_tensor_details()
            for detail in tensor_details:
                shape = detail.get("shape", [])
                name = detail.get("name", "")
                shape_tuple = tuple(shape) if shape is not None and len(shape) > 0 else ()
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
                        tensor_idx = detail.get("index")
                        tensor_data = interp.get_tensor(tensor_idx)
                        if tensor_data is not None and tensor_data.size > 0:
                            embeddings = tensor_data.copy()
                            break
                    except ValueError as exc:  # noqa: BLE001
                        if "Tensor data is null" in str(exc):
                            logger.debug(
                                "BirdNet manual: tensor %s (%s) data is null (TF 2.17+ bug).",
                                detail.get("index"),
                                name,
                            )
                    except Exception:  # noqa: BLE001
                        continue

        if embeddings is None:
            raise ValueError(
                "BirdNet manual: could not find embedding tensor in model. "
                f"Outputs: {len(out_info)}, shapes: {[d.get('shape') for d in out_info]}.",
            )

        embeddings = embeddings.astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    # ------------------------------------------------------------------ #
    #                 OPTIONALS EXPECTED BY ModelBase                    #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Convenience to expose BirdNET's species mapping                    #
    # ------------------------------------------------------------------ #
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
