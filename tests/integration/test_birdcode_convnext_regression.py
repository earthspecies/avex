"""Output regression for the BirdCODE ConvNeXt encoder on in-contract audio.

The shared official-model regression feeds every model 1 s of 16 kHz audio,
which is outside this model's contract (32 kHz, 5 s clips) and collapses its
output to a single time frame. This test uses deterministic bird-like signals at
the model's real input format and checks the output against stored reference
values with a tolerance, so it works across PyTorch versions without
per-version hashes.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
import torch

from avex import load_model

_MODEL_NAME: Final[str] = "esp_aves2_sed_birdcode_convnext_encoder"
_SR: Final[int] = 32_000
_SECONDS: Final[float] = 5.0
_N: Final[int] = int(_SR * _SECONDS)


def _tone_burst(t: np.ndarray, start: float, dur: float, f0: float, f1: float) -> np.ndarray:
    """Linear chirp from ``f0`` to ``f1`` Hz between ``start`` and ``start + dur`` with a Hann envelope.

    Returns
    -------
    np.ndarray
        Signal of the same length as ``t``.
    """
    inside = (t >= start) & (t < start + dur)
    tau = np.where(inside, t - start, 0.0)
    phase = 2 * np.pi * (f0 * tau + 0.5 * (f1 - f0) / dur * tau**2)
    env = np.where(inside, np.sin(np.pi * tau / dur) ** 2, 0.0)
    return env * np.sin(phase)


def _realistic_batch() -> torch.Tensor:
    """Four deterministic 5 s clips at 32 kHz with bird-like content.

    0. Repeated down-sweeping notes (6 to 3 kHz), like a simple song.
    1. Harmonic call at 2.5 kHz with 5 Hz amplitude modulation.
    2. Trill: 20 short up-sweeps (3 to 5 kHz) at 4 per second.
    3. A faint 4 kHz note and a loud 7 kHz note in one clip, 40 dB apart
       (exercises the per-clip dB floor and level handling).

    Returns
    -------
    torch.Tensor
        Shape ``(4, 160000)``, float32.
    """
    t = np.arange(_N, dtype=np.float64) / _SR
    song = sum(_tone_burst(t, 0.3 + 0.9 * k, 0.35, 6000.0, 3000.0) for k in range(5))
    harmonics = np.sin(2 * np.pi * 2500 * t) + 0.4 * np.sin(2 * np.pi * 5000 * t)
    call = harmonics * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)) * ((t > 1.0) & (t < 4.0))
    trill = sum(_tone_burst(t, 0.1 + 0.25 * k, 0.08, 3000.0, 5000.0) for k in range(20))
    pair = 0.01 * _tone_burst(t, 0.5, 1.0, 4000.0, 4000.0) + _tone_burst(t, 2.5, 1.0, 7000.0, 7000.0)
    clips = np.stack([0.5 * song, 0.3 * call, 0.4 * trill, 0.5 * pair])
    return torch.from_numpy(clips.astype(np.float32))


# Reference values from torch 2.11 on CPU: L2 norm of the time/frequency-pooled
# embedding per clip, and its first 8 channels. Regenerate by running this
# module's helpers against the pinned checkpoint if the checkpoint or the
# front-end changes intentionally.
_EXPECTED_NORMS: Final[list[float]] = [56.0662, 60.9024, 52.6827, 73.2704]
_EXPECTED_FIRST_CHANNELS: Final[list[list[float]]] = [
    [0.7609, -0.6977, 0.9955, 1.762, -1.4337, 0.0772, -0.5231, 1.3657],
    [0.6968, -0.3092, 0.814, 3.4814, -0.5053, 1.1454, -0.7669, 2.3563],
    [0.2856, 0.1421, 0.7041, 3.1169, -1.4514, 1.2728, -1.1282, 1.0132],
    [1.0948, -0.6343, 0.3206, 3.8477, -0.781, 1.3506, -1.2637, 2.6567],
]


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    """Load the official BirdCODE encoder once.

    Returns
    -------
    torch.nn.Module
        Model in eval mode on CPU.
    """
    try:
        m = load_model(_MODEL_NAME, device="cpu")
    except (OSError, ConnectionError, TimeoutError) as exc:  # pragma: no cover - network
        pytest.skip(f"Unable to load {_MODEL_NAME}: {exc}")
    return m.eval()


@pytest.mark.slow
class TestBirdcodeConvnextRealisticInput:
    """In-contract output checks for the BirdCODE ConvNeXt encoder."""

    def test_output_shape_and_values(self, model: torch.nn.Module) -> None:
        """5 s at 32 kHz gives a (B, 1024, 8, 19) map whose pooled values match the reference."""
        with torch.no_grad():
            out = model(_realistic_batch())
        assert out.shape == (4, 1024, 8, 19)
        assert torch.isfinite(out).all()

        pooled = out.mean(dim=(2, 3))
        np.testing.assert_allclose(pooled.norm(dim=1).numpy(), _EXPECTED_NORMS, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(pooled[:, :8].numpy(), _EXPECTED_FIRST_CHANNELS, rtol=1e-3, atol=1e-3)

    def test_batch_composition_does_not_change_outputs(self, model: torch.nn.Module) -> None:
        """Each clip must embed the same alone as in a batch (per-clip dB floor)."""
        audio = _realistic_batch()
        with torch.no_grad():
            batched = model(audio)
            single = torch.cat([model(audio[i : i + 1]) for i in range(audio.shape[0])])
        # Float rounding in batched convolutions gives ~1e-4; a batch-wide dB
        # floor would change the faint note in clip 3 by whole units.
        torch.testing.assert_close(batched, single, rtol=1e-3, atol=1e-3)
