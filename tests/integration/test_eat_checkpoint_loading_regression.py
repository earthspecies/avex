"""Regression tests for raw SSL EAT checkpoint loading.

These tests target the same class of issue discussed during BEATs loading
regression checks: checkpoints that appear to load but effectively leave the
model at default/base weights.

The checkpoint in this module is a remote `gs://` artifact and requires network
access. Like other remote-artifact tests, failures to access remote storage are
reported as skips to keep CI robust across environments.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import pytest
import torch
from transformers import AutoModel

from avex.evaluation.clustering import eval_clustering
from avex.models.eat_hf import EATHFModel
from avex.utils.utils import universal_torch_load

_BASE_MODEL_ID = "worstchan/EAT-base_epoch30_pretrain"
_REPRESENTATIVE_SSL_EAT_CHECKPOINT = "gs://representation-learning/models/eat_all.pt"


def _build_synthetic_audio_batch(
    *,
    samples_per_class: int = 8,
    sample_rate: int = 16_000,
    seconds: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a small labeled audio batch for clustering sanity checks.

    Args:
        samples_per_class: Number of examples per synthetic class.
        sample_rate: Audio sample rate.
        seconds: Duration of each synthetic clip.

    Returns:
        Tuple of `(audio, labels)` where `audio` has shape `(B, T)` and labels
        has shape `(B,)`.
    """
    length = sample_rate * seconds
    t = torch.linspace(0.0, float(seconds), steps=length, dtype=torch.float32)
    class_freqs = (220.0, 660.0, 1320.0)
    clips: list[torch.Tensor] = []
    labels: list[int] = []
    generator = torch.Generator(device="cpu").manual_seed(1234)

    for class_idx, frequency in enumerate(class_freqs):
        base_wave = torch.sin(2.0 * torch.pi * frequency * t)
        for _ in range(samples_per_class):
            # Small perturbations keep class structure but avoid exact duplicates.
            noise = 0.02 * torch.randn(length, generator=generator, dtype=torch.float32)
            amplitude = 0.8 + 0.4 * torch.rand((), generator=generator, dtype=torch.float32).item()
            clips.append((amplitude * base_wave + noise).to(torch.float32))
            labels.append(class_idx)

    audio = torch.stack(clips, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return audio, label_tensor


def _extract_pooled_embeddings(model: EATHFModel, audio: torch.Tensor) -> torch.Tensor:
    """Extract pooled clip-level embeddings from an EAT feature model.

    Args:
        model: EAT feature-extractor model.
        audio: Raw waveform batch `(B, T)`.

    Returns:
        Pooled embeddings of shape `(B, D)`.

    Raises:
        ValueError: If model output is not a 3D tensor shaped `(B, L, D)`.
    """
    with torch.no_grad():
        feats = model(audio)
    if feats.dim() != 3:
        raise ValueError(f"Expected 3D features (B, L, D), got shape={tuple(feats.shape)}")
    return feats.mean(dim=1).cpu()


def _pairwise_cosine_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine-similarity matrix for clip embeddings.

    Args:
        embeddings: Embedding matrix of shape `(N, D)`.

    Returns:
        Pairwise cosine-similarity matrix of shape `(N, N)`.
    """
    norms = embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = embeddings / norms
    return normalized @ normalized.T


def _rename_fairseq_key(key: str) -> str:
    """Map fairseq EAT keys to HuggingFace EAT keys.

    Args:
        key: Raw fairseq state dict key.

    Returns:
        Remapped key matching the HuggingFace model naming convention.
    """
    if key == "modality_encoders.IMAGE.context_encoder.norm.weight":
        return "model.pre_norm.weight"
    if key == "modality_encoders.IMAGE.context_encoder.norm.bias":
        return "model.pre_norm.bias"
    prefix = "modality_encoders.IMAGE."
    if key.startswith(prefix):
        return "model." + key[len(prefix) :]
    if not key.startswith("model."):
        return "model." + key
    return key


def _iter_source_tensors(fairseq_state: dict[str, torch.Tensor]) -> Iterable[tuple[str, torch.Tensor]]:
    """Iterate relevant tensors from fairseq checkpoint state.

    Args:
        fairseq_state: Fairseq model state dictionary.

    Yields:
        Tuples of original key and tensor value for loadable tensors.
    """
    for key, value in fairseq_state.items():
        if key.startswith("_ema"):
            continue
        if isinstance(value, torch.Tensor):
            yield key, value


def _load_fairseq_model_state(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load raw fairseq-style state dict from checkpoint.

    Args:
        checkpoint_path: Path or URI to checkpoint.

    Returns:
        Fairseq model state dict from checkpoint["model"].

    Raises:
        ValueError: If checkpoint structure is not a fairseq-style dictionary with
            a dictionary under the `"model"` key.
    """
    checkpoint = universal_torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected dict checkpoint, got {type(checkpoint)}")
    if "model" not in checkpoint:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' is not fairseq-style (missing 'model' key). "
            f"Available keys sample: {list(checkpoint.keys())[:10]}"
        )
    model_state = checkpoint["model"]
    if not isinstance(model_state, dict):
        raise ValueError(f"Expected dict in checkpoint['model'], got {type(model_state)}")
    return model_state


def _compute_changed_tensors(
    before_state: dict[str, torch.Tensor],
    after_state: dict[str, torch.Tensor],
) -> tuple[int, int]:
    """Count changed tensors between two model state dict snapshots.

    Args:
        before_state: State dict before loading checkpoint.
        after_state: State dict after loading checkpoint.

    Returns:
        Tuple of (changed_count, compared_count).
    """
    changed = 0
    compared = 0
    for key, before_tensor in before_state.items():
        if key not in after_state:
            continue
        compared += 1
        if not torch.equal(before_tensor, after_state[key]):
            changed += 1
    return changed, compared


@pytest.mark.slow
def test_raw_ssl_eat_checkpoint_changes_backbone_weights() -> None:
    """Ensure each raw SSL EAT checkpoint materially changes backbone weights.

    This is a regression guard against fallback-like behavior where loading
    silently keeps the default HuggingFace EAT checkpoint.

    Args:
        None.
    """
    model = AutoModel.from_pretrained(_BASE_MODEL_ID, trust_remote_code=True).to("cpu")
    model.eval()
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    hf_state = model.state_dict()

    try:
        fairseq_state = _load_fairseq_model_state(_REPRESENTATIVE_SSL_EAT_CHECKPOINT)
    except Exception as exc:
        pytest.skip(f"Unable to read checkpoint '{_REPRESENTATIVE_SSL_EAT_CHECKPOINT}': {exc}")

    remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    source_count = 0
    key_match_count = 0
    shape_match_count = 0
    for key, value in _iter_source_tensors(fairseq_state):
        source_count += 1
        mapped_key = _rename_fairseq_key(key)
        if mapped_key not in hf_state:
            continue
        key_match_count += 1
        if tuple(hf_state[mapped_key].shape) != tuple(value.shape):
            continue
        shape_match_count += 1
        remapped[mapped_key] = value

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    after = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    changed_count, compared_count = _compute_changed_tensors(before, after)

    assert source_count > 0, "Checkpoint should contain loadable tensors."
    assert key_match_count > 0, "At least some checkpoint keys should map to HF keys."
    assert shape_match_count > 0, "At least some mapped keys should have matching shape."
    assert changed_count > 0, (
        "Checkpoint load changed zero tensors. This indicates fallback-like behavior to base/default HF EAT weights."
    )
    assert compared_count > 0, "Should compare at least one parameter tensor."
    # In healthy load paths for these checkpoints this is expected to be 0/0.
    assert len(unexpected) == 0, f"Unexpected keys found: {unexpected[:10]}"
    assert len(missing) == 0, f"Missing keys found: {missing[:10]}"


@pytest.mark.slow
def test_raw_ssl_eat_checkpoint_changes_clustering_behavior() -> None:
    """Ensure checkpoint-loaded EAT differs from base model on clustering sanity set.

    This test approximates the regression signal discussed for BEATs: if loading
    silently falls back to base weights, clustering behavior and embeddings should
    remain effectively identical to the base model.

    Args:
        None.
    """
    audio, labels = _build_synthetic_audio_batch()

    base_model = EATHFModel(
        model_name=_BASE_MODEL_ID,
        num_classes=None,
        device="cpu",
        return_features_only=True,
    )
    loaded_model = EATHFModel(
        model_name=_BASE_MODEL_ID,
        num_classes=None,
        device="cpu",
        return_features_only=True,
        fairseq_weights_path=_REPRESENTATIVE_SSL_EAT_CHECKPOINT,
    )
    base_model.eval()
    loaded_model.eval()

    base_embeddings = _extract_pooled_embeddings(base_model, audio)
    loaded_embeddings = _extract_pooled_embeddings(loaded_model, audio)
    mean_abs_embedding_delta = float((loaded_embeddings - base_embeddings).abs().mean().item())

    base_metrics = eval_clustering(base_embeddings, labels)
    loaded_metrics = eval_clustering(loaded_embeddings, labels)
    metric_delta_sum = float(sum(abs(float(loaded_metrics[k]) - float(base_metrics[k])) for k in base_metrics.keys()))

    base_pairwise = _pairwise_cosine_matrix(base_embeddings)
    loaded_pairwise = _pairwise_cosine_matrix(loaded_embeddings)
    pairwise_delta_mean = float((loaded_pairwise - base_pairwise).abs().mean().item())

    assert mean_abs_embedding_delta > 1e-6, (
        "Loaded checkpoint produced embeddings nearly identical to base model. "
        "This suggests fallback-like loading behavior."
    )
    # Clustering metrics can be identical for simple synthetic data even when
    # embeddings differ substantially. We therefore use embedding-geometry
    # change as the primary regression signal and keep metric delta as optional.
    assert pairwise_delta_mean > 1e-6 or metric_delta_sum > 1e-9, (
        "Loaded checkpoint produced embeddings too close to base model, including "
        "near-identical pairwise structure and clustering metrics. "
        "This suggests fallback-like loading behavior."
    )
