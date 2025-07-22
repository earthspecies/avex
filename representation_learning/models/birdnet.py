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


class Model(ModelBase):  # Changed from BirdNetTFLite to Model for factory compatibility
    """
    Thin wrapper around *birdnetlib* that exposes:
        • forward()  – returns clip-level class probabilities (rarely needed)
        • extract_embeddings() – 1024-d BirdNET feature vectors
    Everything else (batching, .prepare_train(), …) comes from ModelBase.
    """

    SAMPLE_RATE = 48_000  # what BirdNET expects
    CHUNK_SEC = 3.0  # fixed window length inside the model

    def __init__(
        self,
        num_classes: int = 0,  # Add back for factory compatibility
        device: str = "cpu",
        audio_config: Optional[Dict[str, Any]] = None,
        *,
        language: str = "en_us",  # Add back for factory compatibility
        apply_sigmoid: bool = True,  # Add back for factory compatibility
        freeze_backbone: bool = True,  # Add back for factory compatibility
        **kwargs,  # Accept additional config parameters  # noqa: ANN003
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # birdnetlib downloads ~/.cache/birdnet/analyzer/BirdNET_…_V2.4_FP32.tflite
        # on first construction – no manual model handling required.
        self._analyzer = Analyzer()  # classification helper
        self._interpreter = self._analyzer.interpreter  # TFLite interpreter
        self.species = self._analyzer.labels  # 6 522 labels
        self.num_species = len(self.species)

        # Store factory parameters for compatibility
        self.num_classes = num_classes
        self.language = language
        self.apply_sigmoid = apply_sigmoid
        self.freeze_backbone = freeze_backbone

        # Optional classification head if num_classes is specified and different
        # from BirdNET's default
        if num_classes > 0 and num_classes != self.num_species:
            self.classifier = nn.Linear(self.num_species, num_classes)
        else:
            self.classifier = None

        # Remove the circular reference that was causing recursion error
        # self.model = self  # This line caused infinite recursion in .to(device)

        logger.info(
            "BirdNetTFLite ready – v2.4 • %d species • embeddings dim = 1024 • "
            "num_classes = %d",
            self.num_species,
            num_classes,
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

        result = torch.stack(probs).to(self.device)  # (B, Nclasses)

        # Apply additional classification head if specified
        if self.classifier is not None:
            result = self.classifier(result)

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
        return torch.cat(batch_out, dim=0).to(self.device)

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

        Raises
        ------
        ValueError
            If no embedding tensor can be found in the model.

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

        # Manual extraction as fallback
        interp = self._interpreter
        logger.info(
            f"DEBUG: Starting manual embedding extraction, interpreter: {type(interp)}"
        )

        # The neat trick: feed through RecordingBuffer which handles
        # resampling + spectrograms the same way the CLI tool does.
        buf = RecordingBuffer(self._analyzer, mono_wave, self.SAMPLE_RATE)
        logger.info("DEBUG: Created RecordingBuffer, about to analyze...")
        buf.analyze()
        logger.info("DEBUG: RecordingBuffer.analyze() completed")

        # Ensure tensors are allocated before accessing tensor data
        # This is required in newer versions of TensorFlow Lite
        logger.info("DEBUG: About to allocate tensors...")
        try:
            interp.allocate_tensors()
            logger.info("DEBUG: allocate_tensors() successful")
        except Exception as e:
            logger.error(f"DEBUG: allocate_tensors() failed: {e}")

        # After .analyze() the Interpreter holds tensors for each 3-s chunk.
        # Modern BirdNet models may only have logits output,
        # so we need to find the embedding layer
        out_info = interp.get_output_details()
        logger.info(f"DEBUG: Output details: {len(out_info)} outputs")
        for i, detail in enumerate(out_info):
            logger.info(
                f"DEBUG: Output {i}: shape={detail.get('shape')}, "
                f"dtype={detail.get('dtype')}, index={detail.get('index')}"
            )

        # Try to find embeddings in different ways
        embeddings = None

        if len(out_info) > 1:
            # Old BirdNet format with separate embedding output
            emb_idx = out_info[1]["index"]
            logger.info(f"DEBUG: Using embedding index from output[1]: {emb_idx}")
            try:
                embeddings = interp.get_tensor(emb_idx)
                logger.info(
                    f"DEBUG: Found embeddings from output[1], shape: {embeddings.shape}"
                )
            except Exception as e:
                logger.error(f"DEBUG: Failed to get embeddings from output[1]: {e}")

        if embeddings is None:
            # Modern BirdNet format - need to find embedding layer
            # by searching tensor details
            logger.info("DEBUG: Searching for embedding layer in tensor details...")
            tensor_details = interp.get_tensor_details()
            logger.info(f"DEBUG: Found {len(tensor_details)} total tensors")

            # Look for a tensor with shape containing 1024 (typical embedding dimension)
            embedding_candidates = []
            for i, detail in enumerate(tensor_details):
                shape = detail.get("shape", [])
                name = detail.get("name", "")
                if len(shape) >= 1 and (
                    1024 in shape
                    or "embedding" in name.lower()
                    or "feature" in name.lower()
                ):
                    embedding_candidates.append((i, detail))
                    logger.info(
                        f"DEBUG: Embedding candidate {i}: name='{name}', "
                        f"shape={shape}, index={detail.get('index')}"
                    )

            # Try the most promising candidates
            for i, detail in embedding_candidates:
                try:
                    emb_idx = detail.get("index")
                    logger.info(
                        f"DEBUG: Trying embedding candidate {i} with index {emb_idx}"
                    )
                    embeddings = interp.get_tensor(emb_idx)
                    logger.info(
                        f"DEBUG: Successfully extracted embeddings, "
                        f"shape: {embeddings.shape}"
                    )
                    break
                except Exception as e:
                    logger.error(f"DEBUG: Failed to get tensor from candidate {i}: {e}")
                    continue

        if embeddings is None:
            # Last resort: try to find any tensor with reasonable embedding dimensions
            logger.error("DEBUG: No embedding layer found, trying fallback approach...")
            # For now, raise the original error but with more context
            raise ValueError(
                f"Could not find embedding tensor in BirdNet model. "
                f"Available outputs: {len(out_info)} with shapes: "
                f"{[detail.get('shape') for detail in out_info]}. "
                f"This may be a newer BirdNet model format "
                f"that doesn't expose embeddings."
            )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------ #
    #                 OPTIONALS EXPECTED BY ModelBase                    #
    # ------------------------------------------------------------------ #
    def enable_gradient_checkpointing(self) -> None:
        logger.warning("Gradient checkpointing is not supported for BirdNET.")

    # ------------------------------------------------------------------ #
    # Convenience to expose BirdNET's species mapping
    # ------------------------------------------------------------------ #
    def idx_to_species(self, idx: int) -> str:
        return self.species[idx]

    def species_to_idx(self, name: str) -> int:
        return self.species.index(name)
