# representation_learning/models/birdnet.py
import logging
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf  # lightweight; writes the tmp .wav we feed in
import torch
import torch.nn as nn
from birdnetlib import Recording, RecordingBuffer

# --- external deps (≈ 30 MB wheels) -----------------------------------------
#   pip install birdnetlib tflite-runtime
from birdnetlib.analyzer import Analyzer  # downloads + wraps *.tflite

from representation_learning.models.base_model import ModelBase  # Add missing import

logger = logging.getLogger(__name__)


class Model(ModelBase):
    """
    Thin wrapper around *birdnetlib* that exposes:
        • forward()  – returns clip-level class probabilities (rarely needed)
        • extract_embeddings() – 1024-d BirdNET feature vectors
    Everything else (batching, .prepare_train(), …) comes from ModelBase.
    """

    SAMPLE_RATE = 48_000
    CHUNK_SEC = 3.0  # fixed window length inside the model

    def __init__(
        self,
        num_classes: int = 0,
        device: str = "cpu",
        audio_config: Optional[Dict[str, Any]] = None,
        *,
        language: str = "en_us",
        apply_sigmoid: bool = True,
        freeze_backbone: bool = True,
        **kwargs,  # Accept additional config parameters  # noqa: ANN003
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        self._analyzer = Analyzer()  # classification helper
        self._interpreter = self._analyzer.interpreter  # TFLite interpreter
        self.species = self._analyzer.labels  # 6 522 labels
        self.num_species = len(self.species)

        self.num_classes = num_classes
        self.language = language
        self.apply_sigmoid = apply_sigmoid
        self.freeze_backbone = freeze_backbone

        if num_classes > 0 and num_classes != self.num_species:
            self.classifier = nn.Linear(self.num_species, num_classes)
            # Move classifier to the specified device
            if device != "cpu" and torch.cuda.is_available():
                self.classifier = self.classifier.to(device)
        else:
            self.classifier = None

        logger.info(
            "BirdNetTFLite ready – v2.4 • %d species • embeddings dim = 1024 • "
            "num_classes = %d • device = %s",
            self.num_species,
            num_classes,
            device,
        )

    # --------------------------------------------------------------------- #
    #                       NN.Module / ModelBase hooks                     #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        wav: torch.Tensor,  # (B, T) mono @48 kHz
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience wrapper that averages per-chunk logits into one clip score.
        (Mostly here so batch_inference() keeps working.)

        Returns:
            torch.Tensor: Averaged logits for the audio clip.
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
    def extract_embeddings(  # ← the piece you asked for
        self,
        x: torch.Tensor | Dict[str, torch.Tensor],
        layers=None,  # ignored – BirdNET has no named layers  # noqa: ANN001
        *,
        padding_mask: Optional[torch.Tensor] = None,
        average_over_time: bool = True,
    ) -> torch.Tensor:
        """
        Returns:
            • (B, 1024) if *average_over_time*=True   – one vector / clip
            • list[Tensor] with shape (n_chunks,1024) per clip otherwise
        """
        if isinstance(x, dict):  # allow {'raw_wav': …}
            wav = x["raw_wav"]
        else:
            wav = x
        wav = wav.detach().cpu()

        batch_out = []
        for clip in wav:
            emb_np = self._embedding_for_clip(clip.numpy())  # (N,1024)
            emb = torch.from_numpy(emb_np).float()
            batch_out.append(emb.mean(0, keepdim=True) if average_over_time else emb)

        result = torch.cat(batch_out, dim=0)

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
        """
        Uses birdnetlib's Analyzer to get a per-clip probability vector.
        (Not required for embeddings, but handy for quick tests.)

        Returns:
            torch.Tensor: Probability vector for the audio clip.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, mono_wave, self.SAMPLE_RATE)
            rec = Recording(self._analyzer, tmp.name, min_conf=0.0)
            rec.analyze()
            # build a species-probability vector from detections
            scores = np.zeros(self.num_species, dtype=np.float32)
            for det in rec.detections:
                idx = self.species.index(det["label"])
                scores[idx] = max(scores[idx], det["confidence"])
            return torch.from_numpy(scores)

    def _embedding_for_clip(self, mono_wave: np.ndarray) -> np.ndarray:
        """
        Directly grabs the **embedding output tensor** from BirdNET's
        underlying TFLite *Interpreter* – the tensor right before logits.

        Returns:
            np.ndarray: Embedding vector for the audio clip.
        """
        # Use the analyzer's built-in embedding extraction method if available
        if hasattr(self._analyzer, "extract_embeddings_for_recording"):
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, mono_wave, self.SAMPLE_RATE)
                # Try using the built-in embedding extraction
                try:
                    embeddings = self._analyzer.extract_embeddings_for_recording(
                        tmp.name
                    )
                    return embeddings.astype(np.float32)
                except Exception:
                    # Fall back to manual extraction
                    pass

        interp = self._interpreter

        buf = RecordingBuffer(self._analyzer, mono_wave, self.SAMPLE_RATE)
        buf.analyze()

        out_info = interp.get_output_details()
        emb_idx = (
            out_info[1]["index"] if len(out_info) > 1 else out_info[0]["index"] - 1
        )
        embeddings = interp.get_tensor(emb_idx)  # (Nchunks,1024)
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------ #
    #                 OPTIONALS EXPECTED BY ModelBase                    #
    # ------------------------------------------------------------------ #
    def enable_gradient_checkpointing(self) -> None:
        logger.warning("Gradient checkpointing is not supported for BirdNET.")

    def to(self, device: torch.device | str) -> "Model":
        """Override to handle device movement properly for BirdNet.

        Returns
        -------
        Model
            Self for method chaining.
        """
        # Update internal device tracking
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        # Move PyTorch components (classifier) to the device
        if self.classifier is not None:
            if str(device).startswith("cuda") and torch.cuda.is_available():
                self.classifier = self.classifier.to(device)
            else:
                self.classifier = self.classifier.cpu()

        # TensorFlow Lite interpreter stays on CPU - no need to move
        return self

    def cpu(self) -> "Model":
        """Move to CPU.

        Returns
        -------
        Model
            Self for method chaining.
        """
        return self.to("cpu")

    def cuda(self, device: int | None = None) -> "Model":
        """Move to CUDA device.

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
        return self.species[idx]

    def species_to_idx(self, name: str) -> int:
        return self.species.index(name)
