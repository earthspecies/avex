"""BirdNet v2.4 model via ONNX runtime.

Source model: BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite
ONNX export: https://huggingface.co/justinchuby/BirdNET-onnx
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from avex.models.base_model import ModelBase

logger = logging.getLogger(__name__)

_HF_REPO_ID = "justinchuby/BirdNET-onnx"
_MODEL_FILENAME = "birdnet.onnx"
_LABELS_FILENAME = "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
_SAMPLE_RATE = 48_000
_CHUNK_SAMPLES = 144_000  # 3 s @ 48 kHz
_EMBEDDING_DIM = 1024

# Module-level session cache
_session: Any = None
_session_key: tuple[str, str] | None = None
_input_name: str | None = None
_output_name: str | None = None
_embedding_output_name: str | None = None  # None if not found in graph


# --------------------------------------------------------------------------- #
#  ONNX graph helpers
# --------------------------------------------------------------------------- #


def _find_embedding_tensor_name(model_proto: object) -> str | None:
    """Traverse ONNX graph backwards from output to find the 1024-d embedding.

    Returns
    -------
    str | None
        Tensor name of the nearest ancestor with shape [*, 1024], or None.
    """
    # Build producer map: tensor_name → node
    output_to_node: dict[str, Any] = {}
    for node in model_proto.graph.node:
        for out in node.output:
            if out:
                output_to_node[out] = node

    # Build shape lookup from value_info (populated after shape inference)
    name_to_shape: dict[str, list[int]] = {}
    for vi in model_proto.graph.value_info:
        try:
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
                shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                name_to_shape[vi.name] = shape
        except Exception:
            pass

    # BFS backwards from classification output
    visited: set[str] = set()
    queue: list[str] = [model_proto.graph.output[0].name]

    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)

        shape = name_to_shape.get(name, [])
        # Match [batch, 1024] or [1024] — 2-D or 1-D with correct last dim
        if _EMBEDDING_DIM in shape and len(shape) <= 2:
            return name

        node = output_to_node.get(name)
        if node:
            for inp in node.input:
                if inp and inp not in visited:
                    queue.append(inp)

    return None


def _build_session(repo_id: str, model_filename: str) -> tuple[Any, str, str, str | None]:
    """Download + load ONNX model, returning (session, input_name, output_name, embedding_name).

    Modifies the graph in-memory to expose the 1024-d embedding tensor as an
    extra output.  Falls back to model.onnx if not found in birdnet.onnx.

    Returns
    -------
    tuple[Any, str, str, str | None]
        ONNX Runtime session, input name, classifier output name, and optional
        embedding output name.

    Raises
    ------
    ImportError
        If onnxruntime or onnx is not installed.
    """
    from huggingface_hub import hf_hub_download

    try:
        import onnx as onnx_lib
        import onnx.shape_inference
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("onnxruntime and onnx are required.\nuv pip install onnxruntime onnx") from e

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _try_load(filename: str) -> tuple[Any, str, str, str | None]:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.info("Loading BirdNet ONNX: %s", model_path)

        model_proto = onnx_lib.load(model_path)
        model_proto = onnx.shape_inference.infer_shapes(model_proto)

        inp_name = model_proto.graph.input[0].name
        clf_out_name = model_proto.graph.output[0].name
        emb_name = _find_embedding_tensor_name(model_proto)

        if emb_name:
            logger.info("BirdNet embedding tensor found: %s", emb_name)
            model_proto.graph.output.append(
                onnx_lib.helper.make_tensor_value_info(emb_name, onnx_lib.TensorProto.FLOAT, None)
            )
            session = ort.InferenceSession(model_proto.SerializeToString(), providers=providers)
        else:
            logger.warning("No 1024-d tensor found in %s", filename)
            session = ort.InferenceSession(model_path, providers=providers)

        return session, inp_name, clf_out_name, emb_name

    session, inp_name, clf_out_name, emb_name = _try_load(model_filename)

    # If the optimised model lacks shape info, fall back to the raw tf2onnx export
    if emb_name is None and model_filename == _MODEL_FILENAME:
        logger.info("Falling back to model.onnx for embedding extraction")
        session, inp_name, clf_out_name, emb_name = _try_load("model.onnx")

    return session, inp_name, clf_out_name, emb_name


def _load_session(
    repo_id: str = _HF_REPO_ID,
    model_filename: str = _MODEL_FILENAME,
) -> tuple[Any, str, str, str | None]:
    global _session, _session_key, _input_name, _output_name, _embedding_output_name

    key = (repo_id, model_filename)
    if _session is not None and _session_key == key:
        return _session, _input_name, _output_name, _embedding_output_name

    _session, _input_name, _output_name, _embedding_output_name = _build_session(repo_id, model_filename)
    _session_key = key
    logger.info(
        "BirdNet ONNX ready — providers: %s — input: %s — embedding: %s",
        _session.get_providers(),
        _input_name,
        _embedding_output_name,
    )
    return _session, _input_name, _output_name, _embedding_output_name


# --------------------------------------------------------------------------- #
#  Model class
# --------------------------------------------------------------------------- #


class Model(ModelBase):
    """BirdNet v2.4 bird-sound classifier via ONNX Runtime.

    Input: raw waveform at any sample rate.  Audio is resampled to 48 kHz and
    split into 3-second (144 000-sample) chunks before inference.

    Outputs:
      • forward()           → [B, num_species] logits (or [B, num_classes] with head)
      • extract_embeddings()→ [B, 1024] embeddings (penultimate layer)
    """

    SAMPLE_RATE = _SAMPLE_RATE
    CHUNK_SAMPLES = _CHUNK_SAMPLES

    def __init__(
        self,
        num_classes: Optional[int] = None,
        device: str = "cpu",
        audio_config: Optional[Dict[str, Any]] = None,
        *,
        freeze_backbone: bool = True,
        return_features_only: bool = False,
        hf_repo_id: str = _HF_REPO_ID,
        hf_model_filename: str = _MODEL_FILENAME,
        hf_labels_filename: str = _LABELS_FILENAME,
        **kwargs: object,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        self._hf_repo_id = hf_repo_id
        self._hf_model_filename = hf_model_filename
        self._hf_labels_filename = hf_labels_filename
        self.freeze_backbone = freeze_backbone
        self.return_features_only = return_features_only

        # Warm up session (downloads model on first use)
        _load_session(hf_repo_id, hf_model_filename)  # noqa: F841

        # Load species labels from HF repo
        from huggingface_hub import hf_hub_download

        labels_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_labels_filename)
        with open(labels_path) as f:
            self.species: list[str] = [line.strip() for line in f if line.strip()]
        self.num_species = len(self.species)

        if return_features_only:
            self.num_classes = None
            self.classifier = None
        elif num_classes is not None and num_classes > 0 and num_classes != self.num_species:
            self.num_classes = num_classes
            self.classifier = nn.Linear(self.num_species, num_classes).to(device)
        else:
            self.num_classes = num_classes
            self.classifier = None

        logger.info(
            "BirdNet ONNX ready — %d species — embedding_dim=%d — device=%s",
            self.num_species,
            _EMBEDDING_DIM,
            device,
        )

    # ------------------------------------------------------------------ #
    #  ModelBase overrides
    # ------------------------------------------------------------------ #

    @property
    def input_sr(self) -> int:
        if self.audio_processor is not None:
            return self.audio_processor.sr
        return self.SAMPLE_RATE

    def _discover_linear_layers(self) -> None:
        if not self._layer_names:
            self._layer_names = [name for name, mod in self.named_modules() if isinstance(mod, nn.Linear)]

    _discover_embedding_layers = _discover_linear_layers  # type: ignore[assignment]

    def enable_gradient_checkpointing(self) -> None:
        logger.warning("Gradient checkpointing not supported for BirdNet ONNX.")

    def to(self, device: torch.device | str) -> "Model":
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if self.classifier is not None:
            if str(device).startswith("cuda") and torch.cuda.is_available():
                self.classifier = self.classifier.to(device)
            else:
                self.classifier = self.classifier.cpu()
        return self

    def cpu(self) -> "Model":
        return self.to("cpu")

    def cuda(self, device: int | None = None) -> "Model":
        return self.to(f"cuda:{device}" if device is not None else "cuda")

    # ------------------------------------------------------------------ #
    #  Audio helpers
    # ------------------------------------------------------------------ #

    def _to_chunks(self, wav: torch.Tensor, src_sr: int) -> torch.Tensor:
        """Resample to 48 kHz and split into fixed-length chunks.

        Returns
        -------
        torch.Tensor
            Float32 tensor with shape [N, CHUNK_SAMPLES].
        """
        if src_sr != self.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, src_sr, self.SAMPLE_RATE)

        wav = wav.float()

        # Ensure at least one full chunk
        if wav.shape[-1] < self.CHUNK_SAMPLES:
            wav = torch.nn.functional.pad(wav, (0, self.CHUNK_SAMPLES - wav.shape[-1]))

        n_full = wav.shape[-1] // self.CHUNK_SAMPLES
        remainder = wav.shape[-1] % self.CHUNK_SAMPLES

        chunks = [wav[i * self.CHUNK_SAMPLES : (i + 1) * self.CHUNK_SAMPLES] for i in range(n_full)]

        # Keep partial tail if it contains > half a chunk of signal
        if remainder > self.CHUNK_SAMPLES // 2:
            tail = wav[n_full * self.CHUNK_SAMPLES :]
            tail = torch.nn.functional.pad(tail, (0, self.CHUNK_SAMPLES - tail.shape[-1]))
            chunks.append(tail)

        return torch.stack(chunks)  # [N, CHUNK_SAMPLES]

    # ------------------------------------------------------------------ #
    #  ONNX inference
    # ------------------------------------------------------------------ #

    def _run_onnx(self, chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run ONNX inference on fixed-length chunks.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Logits with shape [N, S] and optional embeddings with shape [N, 1024].
        """
        session, inp_name, clf_out, emb_out = _load_session(self._hf_repo_id, self._hf_model_filename)
        chunks_np = chunks.detach().cpu().numpy().astype(np.float32)

        if emb_out:
            logits_np, emb_np = session.run([clf_out, emb_out], {inp_name: chunks_np})
            return torch.from_numpy(logits_np), torch.from_numpy(emb_np)
        else:
            (logits_np,) = session.run([clf_out], {inp_name: chunks_np})
            return torch.from_numpy(logits_np), None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def forward(
        self,
        wav: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return mean-pooled logits.

        Returns
        -------
        torch.Tensor
            Logits with shape [B, num_species] or [B, num_classes].
        """
        src_sr = self.input_sr
        clip_logits = []
        for clip in wav:
            chunks = self._to_chunks(clip, src_sr)
            logits, _ = self._run_onnx(chunks)
            clip_logits.append(logits.mean(0))  # [num_species]

        result = torch.stack(clip_logits).to(self.device)
        if self.classifier is not None:
            result = self.classifier(result)
        return result

    def extract_embeddings(
        self,
        x: torch.Tensor | Dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean",
    ) -> torch.Tensor | list[torch.Tensor]:
        wav = x["raw_wav"] if isinstance(x, dict) else x
        src_sr = self.input_sr

        batch: list[torch.Tensor] = []
        for clip in wav:
            chunks = self._to_chunks(clip, src_sr)
            _, embs = self._run_onnx(chunks)
            if embs is None:
                raise RuntimeError(
                    "BirdNet ONNX model does not expose a 1024-d embedding output. "
                    "Could not find the embedding tensor in the ONNX graph."
                )
            if embs.ndim == 1:
                embs = embs.unsqueeze(0)
            batch.append(embs)  # [N_chunks, 1024]

        if aggregation == "none":
            # Return list of [1, N_chunks, 1024] per clip
            return [emb.unsqueeze(0) for emb in batch]
        elif aggregation == "mean":
            result = torch.stack([emb.mean(0) for emb in batch])
        elif aggregation == "max":
            result = torch.stack([emb.max(0).values for emb in batch])
        elif aggregation == "cls_token":
            result = torch.stack([emb.mean(0) for emb in batch])
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        return result.to(self.device)

    # ------------------------------------------------------------------ #
    #  Species mapping
    # ------------------------------------------------------------------ #

    def idx_to_species(self, idx: int) -> str:
        return self.species[idx]

    def species_to_idx(self, name: str) -> int:
        try:
            return self.species.index(name)
        except ValueError as err:
            raise ValueError(f"Species '{name}' not found in BirdNet label list") from err
