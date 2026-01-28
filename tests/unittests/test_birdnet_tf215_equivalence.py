"""Tests comparing old and new BirdNet implementations under TensorFlow 2.15.

This verifies that, when running with TensorFlow 2.15.x (where the
`experimental_preserve_all_tensors` bug is not present), the new BirdNet
implementation produces embeddings that closely match the original
BirdNet wrapper from `main`.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
import torch

from representation_learning.models.birdnet import Model as BirdNetNew
from representation_learning.models.birdnet_tf215_original import (
    Model as BirdNetOriginal,
)


def _make_sine_wave(
    duration_seconds: float = 3.0,
    sample_rate: int = 48_000,
    frequency_hz: float = 440.0,
    amplitude: float = 0.1,
) -> torch.Tensor:
    """Create a simple mono sine-wave tensor.

    Returns
    -------
    torch.Tensor
        Mono waveform tensor with shape ``(num_samples,)``.
    """
    num_samples = int(duration_seconds * sample_rate)
    t = torch.linspace(0.0, duration_seconds, num_samples, dtype=torch.float32)
    waveform = amplitude * torch.sin(2.0 * math.pi * frequency_hz * t)
    return waveform


def _make_sine_mixture(
    duration_seconds: float = 3.0,
    sample_rate: int = 48_000,
    components: Tuple[Tuple[float, float, float], ...] | None = None,
) -> torch.Tensor:
    """Create a mixture of sine waves.

    Each component is a tuple of ``(frequency_hz, amplitude, phase_radians)``.

    Returns
    -------
    torch.Tensor
        Mono waveform tensor containing the sine mixture with shape
        ``(num_samples,)``.
    """
    if components is None:
        components = (
            (440.0, 0.08, 0.0),
            (880.0, 0.05, math.pi / 4.0),
            (1760.0, 0.03, math.pi / 2.0),
        )

    num_samples = int(duration_seconds * sample_rate)
    t = torch.linspace(0.0, duration_seconds, num_samples, dtype=torch.float32)
    waveform = torch.zeros_like(t)
    for freq, amp, phase in components:
        waveform = waveform + amp * torch.sin(2.0 * math.pi * freq * t + phase)
    return waveform


def _get_birdnet_models(device: torch.device) -> Tuple[BirdNetNew, BirdNetOriginal]:
    """Instantiate new and original BirdNet models on the given device.

    Parameters
    ----------
    device
        Device on which to place the models.

    Returns
    -------
    tuple[BirdNetNew, BirdNetOriginal]
        Tuple containing the new and original BirdNet models.
    """
    model_new = BirdNetNew(
        num_classes=None,
        device=str(device),
        return_features_only=True,
    )
    model_original = BirdNetOriginal(
        num_classes=None,
        device=str(device),
        return_features_only=True,
    )
    model_new.eval()
    model_original.eval()
    return model_new, model_original


@pytest.mark.skipif(
    pytest.importorskip("tensorflow").__version__.split(".")[0:2] != ["2", "15"],  # type: ignore[call-arg]
    reason="This equivalence test is intended to run only with TensorFlow 2.15.x",
)
def test_birdnet_embeddings_match_on_sine_wave_tf215() -> None:
    """Compare embeddings from new and original BirdNet for a sine-wave signal."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create synthetic mono waveform at BirdNet's expected sample rate (48 kHz)
    waveform = _make_sine_wave(duration_seconds=3.0, sample_rate=48_000)
    waveform_batch = waveform.unsqueeze(0).to(device)  # (B=1, T)

    model_new, model_original = _get_birdnet_models(device=device)

    with torch.no_grad():
        emb_new = model_new.extract_embeddings(waveform_batch, aggregation="mean")
        emb_old = model_original.extract_embeddings(waveform_batch, aggregation="mean")

    assert emb_new.shape == emb_old.shape, f"Shape mismatch: {emb_new.shape} vs {emb_old.shape}"

    # Move to CPU for comparison
    emb_new_np = emb_new.detach().cpu().numpy().astype(np.float32)
    emb_old_np = emb_old.detach().cpu().numpy().astype(np.float32)

    # Compute relative L2 difference
    diff = np.linalg.norm(emb_new_np - emb_old_np)
    denom = np.linalg.norm(emb_old_np) + 1e-8
    rel_diff = float(diff / denom)

    # We expect very close agreement under TF 2.15.x
    assert rel_diff < 1e-3, f"Relative difference too large: {rel_diff}"


@pytest.mark.skipif(
    pytest.importorskip("tensorflow").__version__.split(".")[0:2] != ["2", "15"],  # type: ignore[call-arg]
    reason="This equivalence test is intended to run only with TensorFlow 2.15.x",
)
def test_birdnet_embeddings_match_on_sine_mixture_tf215() -> None:
    """Compare embeddings from new and original BirdNet for a sine-mixture signal."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create more complex synthetic mono waveform (mixture of sines)
    waveform = _make_sine_mixture(duration_seconds=3.0, sample_rate=48_000)
    waveform_batch = waveform.unsqueeze(0).to(device)  # (B=1, T)

    model_new, model_original = _get_birdnet_models(device=device)

    with torch.no_grad():
        emb_new = model_new.extract_embeddings(waveform_batch, aggregation="mean")
        emb_old = model_original.extract_embeddings(waveform_batch, aggregation="mean")

    assert emb_new.shape == emb_old.shape, f"Shape mismatch: {emb_new.shape} vs {emb_old.shape}"

    # Move to CPU for comparison
    emb_new_np = emb_new.detach().cpu().numpy().astype(np.float32)
    emb_old_np = emb_old.detach().cpu().numpy().astype(np.float32)

    # Compute relative L2 difference
    diff = np.linalg.norm(emb_new_np - emb_old_np)
    denom = np.linalg.norm(emb_old_np) + 1e-8
    rel_diff = float(diff / denom)

    # We expect very close agreement under TF 2.15.x for more complex signals too
    assert rel_diff < 1e-3, f"Relative difference too large for sine mixture: {rel_diff}"
