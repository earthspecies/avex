"""Deterministic test clips for Perch v2 parity checks.

The clips are pure functions of time (no RNG), so the parity script, the test
fixtures and the first-use self-check all see identical input.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 32_000
NUM_SAMPLES = 160_000
CLIP_NAMES = ("chirp", "harmonic_call", "trill", "silence")


def golden_clips() -> np.ndarray:
    """Return four 5 s clips at 32 kHz: chirp, harmonic call, trill and silence.

    Returns
    -------
    np.ndarray
        Float32 array of shape (4, 160000).
    """
    t = np.arange(NUM_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    chirp = 0.3 * np.sin(2 * np.pi * (2000 * t + 600 * t**2))  # 2 -> 8 kHz
    am = 0.5 * (1 + np.sin(2 * np.pi * 8 * t))
    harmonic = 0.2 * am * sum(np.sin(2 * np.pi * 1500 * k * t) / k for k in (1, 2, 3))
    gate = (np.floor(t / 0.02) % 2 == 0).astype(np.float64)
    trill = 0.25 * gate * np.sin(2 * np.pi * 4000 * t)
    silence = np.zeros_like(t)
    return np.stack([chirp, harmonic, trill, silence]).astype(np.float32)


# Reference values from ONNX Runtime (CPU, fp32) on ``perch_v2.onnx`` at Hub
# commit da88bf8c, for clip 0 ("chirp"). Used to self-check a fresh conversion.
# Regenerate with ``scripts/perch2_parity.py --write-fixtures``.
SELF_CHECK_EMBEDDING_HEAD = (-0.007501, -0.086165, 0.072711, 0.021453, 0.008744, -0.036102, 0.0769, 0.003377)
SELF_CHECK_EMBEDDING_NORM = 4.576394
SELF_CHECK_TOP_LABEL = 335
SELF_CHECK_TOP_LOGIT = 8.067737
SELF_CHECK_ATOL = 1e-4
