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
            
            if average_over_time:
                # Ensure we're averaging over the correct dimension
                if emb.ndim == 1:
                    # Already a 1D tensor, just add batch dimension
                    processed_emb = emb.unsqueeze(0)  # (1024,) -> (1, 1024)
                    logger.debug(f"BirdNet: Reshaped 1D embedding {emb.shape} -> {processed_emb.shape}")
                else:
                    # Average over time dimension (first dimension)
                    processed_emb = emb.mean(0, keepdim=True)  # (N, 1024) -> (1, 1024)
            else:
                processed_emb = emb
                
            batch_out.append(processed_emb)
            
        result = torch.cat(batch_out, dim=0).to(self.device)
        
        # Validate output shape
        if result.ndim != 2 or result.shape[1] != 1024:
            logger.warning(f"BirdNet embedding shape warning: expected (B, 1024), got {result.shape}")
            
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
                try:
                    # Create Recording object, analyze, and extract embeddings
                    recording = Recording(self._analyzer, tmp.name)
                    recording.analyze()
                    recording.extract_embeddings()

                    # Get embeddings from the recording
                    embeddings_data = recording.embeddings

                    if embeddings_data and len(embeddings_data) > 0:
                        # Extract embeddings from the first (and typically only) segment
                        first_segment = embeddings_data[0]
                        if (
                            isinstance(first_segment, dict)
                            and "embeddings" in first_segment
                        ):
                            embeddings_list = first_segment["embeddings"]
                            if (
                                isinstance(embeddings_list, list)
                                and len(embeddings_list) > 0
                            ):
                                # Convert to numpy array and ensure 2D shape (n_chunks, 1024)
                                embeddings = np.array(embeddings_list, dtype=np.float32)
                                if embeddings.ndim == 1:
                                    embeddings = embeddings.reshape(
                                        1, -1
                                    )  # (1024,) -> (1, 1024)
                                elif embeddings.ndim > 2:
                                    # Flatten extra dimensions but keep batch and feature dimensions
                                    embeddings = embeddings.reshape(-1, embeddings.shape[-1])
                                return embeddings

                    # If we get here, the built-in method didn't work as expected
                    logger.debug(
                        "Built-in embedding extraction returned unexpected format"
                    )
                except Exception as e:
                    logger.debug(f"Built-in embedding extraction failed: {e}")
                    # Fall back to manual extraction
                    pass

        # Manual extraction as fallback
        interp = self._interpreter

        # Feed through RecordingBuffer which handles resampling + spectrograms
        buf = RecordingBuffer(self._analyzer, mono_wave, self.SAMPLE_RATE)
        buf.analyze()

        # Ensure tensors are allocated
        try:
            interp.allocate_tensors()
        except Exception as e:
            logger.error(f"Failed to allocate tensors: {e}")
            raise

        # Extract embeddings from the global average pooling layer
        embeddings = None
        tensor_details = interp.get_tensor_details()

        # Priority 1: Look for the global average pooling layer (most reliable for embeddings)
        for detail in tensor_details:
            name = detail.get("name", "")
            shape = detail.get("shape", [])
            if "GLOBAL_AVG_POOL" in name and len(shape) == 2 and shape[-1] == 1024:
                try:
                    embeddings = interp.get_tensor(detail.get("index"))
                    logger.debug(f"BirdNet: Using GLOBAL_AVG_POOL tensor: {name}, shape: {shape}, index: {detail.get('index')}")
                    break
                except Exception as e:
                    logger.debug(f"BirdNet: Failed to get GLOBAL_AVG_POOL tensor: {e}")
                    continue

        # If we didn't find GLOBAL_AVG_POOL, try the manual search by index
        if embeddings is None:
            logger.debug("BirdNet: GLOBAL_AVG_POOL not found, trying manual search...")
            # Manually try index 545 which we know exists
            try:
                test_tensor = interp.get_tensor(545)
                if test_tensor.shape == (1, 1024):
                    embeddings = test_tensor
                    logger.debug(f"BirdNet: Successfully using manual index 545, shape: {test_tensor.shape}")
            except Exception as e:
                logger.debug(f"BirdNet: Manual index 545 failed: {e}")

        # Priority 2: Look for other 1024-dimensional tensors as fallback
        if embeddings is None:
            for detail in tensor_details:
                name = detail.get("name", "")
                shape = detail.get("shape", [])
                if (
                    len(shape) == 2 
                    and shape[-1] == 1024
                    and ("embedding" in name.lower() or "feature" in name.lower() or "dense" in name.lower())
                ):
                    try:
                        embeddings = interp.get_tensor(detail.get("index"))
                        logger.debug(f"BirdNet: Using fallback tensor: {name}, shape: {shape}")
                        break
                    except Exception:
                        continue

        # Priority 3: Check outputs for legacy models
        if embeddings is None:
            out_info = interp.get_output_details()
            if len(out_info) > 1:
                try:
                    emb_idx = out_info[1]["index"]
                    embeddings = interp.get_tensor(emb_idx)
                    logger.debug(f"BirdNet: Using output tensor {emb_idx}")
                except Exception:
                    pass

        if embeddings is None:
            raise ValueError(
                f"Could not find embedding tensor in BirdNet model. "
                f"Available outputs: {len(out_info)} with shapes: "
                f"{[detail.get('shape') for detail in out_info]}. "
                f"This may be a newer BirdNet model format that doesn't "
                f"expose embeddings."
            )

        # Ensure embeddings are always 2D with shape (n_chunks, 1024)
        embeddings = embeddings.astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)  # (1024,) -> (1, 1024)
        elif embeddings.ndim > 2:
            # Flatten extra dimensions but keep batch and feature dimensions
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        return embeddings

    # ------------------------------------------------------------------ #
    #                 OPTIONALS EXPECTED BY ModelBase                    #
    # ------------------------------------------------------------------ #
    def prepare_inference(self) -> None:
        """Prepare model for inference - BirdNet doesn't need special setup."""
        self.eval()
        # BirdNet models are typically run on CPU, but allow device transfer
        if hasattr(self, 'classifier') and self.classifier is not None:
            self.classifier = self.classifier.to(self.device)

    def prepare_train(self) -> None:
        """Prepare model for training - BirdNet doesn't support training."""
        self.train()
        if hasattr(self, 'classifier') and self.classifier is not None:
            self.classifier = self.classifier.to(self.device)

    def enable_gradient_checkpointing(self) -> None:
        logger.warning("Gradient checkpointing is not supported for BirdNET.")

    # ------------------------------------------------------------------ #
    # Convenience to expose BirdNET's species mapping
    # ------------------------------------------------------------------ #
    def idx_to_species(self, idx: int) -> str:
        return self.species[idx]

    def species_to_idx(self, name: str) -> int:
        return self.species.index(name)
