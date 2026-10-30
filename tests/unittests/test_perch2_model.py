"""Tests for the Perch v2 wrapper (issue #208) with a fake ONNX Runtime session.

No model download: ``_load_ort_session`` is patched to return a session whose
"embedding" output is a deterministic function of the input.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest
import torch

import avex.models.perch2 as perch2
from avex.configs import ProbeConfig
from avex.models.perch2 import Perch2Model
from avex.models.probes.utils.factory import build_probe_from_config
from avex.models.utils.load import load_model

_DIM = 1536
_SAMPLES = 160_000


class _FakeSession:
    """Mimics the perch_v2 ONNX contract for the "embedding" output."""

    def run(self, output_names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert output_names == ["embedding"]
        audio = feeds["inputs"]
        assert audio.dtype == np.float32 and audio.shape[1] == _SAMPLES
        return [audio[:, :_DIM] * 2.0]


@pytest.fixture(autouse=True)
def fake_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replace the ONNX Runtime session loader with a fake.

    Yields:
        None
    """
    monkeypatch.setattr(perch2, "_load_ort_session", lambda *args, **kwargs: _FakeSession())
    yield


def _audio(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, _SAMPLES)


def test_last_layer_resolves_to_backbone() -> None:
    """``target_layers=["last_layer"]`` used to fail: no hookable layers."""
    model = Perch2Model(num_classes=3)
    assert model.register_hooks_for_layers(["last_layer"]) == ["backbone"]


def test_extract_embeddings_accepts_freeze_backbone() -> None:
    """Probes always pass ``freeze_backbone``; it used to raise TypeError."""
    model = Perch2Model()
    audio = _audio()
    for freeze in (True, False):
        emb = model.extract_embeddings(audio, aggregation="mean", freeze_backbone=freeze)
        assert emb.shape == (2, _DIM)
        torch.testing.assert_close(emb, audio[:, :_DIM] * 2.0)


def test_hooked_extraction_matches_and_does_not_leak() -> None:
    model = Perch2Model()
    model.register_hooks_for_layers(["last_layer"])
    audio = _audio()
    emb = model.extract_embeddings({"raw_wav": audio}, aggregation="none")
    assert isinstance(emb, list) and emb[0].shape == (2, 1, _DIM)
    assert model._hook_outputs == {}


def test_online_linear_probe_trains() -> None:
    """End to end through the registry and the probe factory, as in the issue."""
    model = load_model("perch2", device="cpu")
    probe = build_probe_from_config(
        probe_config=ProbeConfig(
            probe_type="linear",
            target_layers=["last_layer"],
            aggregation="mean",
            freeze_backbone=True,
            online_training=True,
        ),
        base_model=model,
        num_classes=4,
        device="cpu",
    )
    logits = probe(_audio(), padding_mask=None)
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert probe.classifier.weight.grad is not None


def test_registered_in_load_model() -> None:
    model = load_model("perch2", device="cpu")
    assert isinstance(model, Perch2Model)
    assert model(_audio(1)).shape == (1, _DIM)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_output_follows_model_device() -> None:
    model = Perch2Model(device="cuda")
    assert model.extract_embeddings(_audio().cuda()).device.type == "cuda"
