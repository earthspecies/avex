"""Tests for the shared ONNX Runtime helpers used by the BirdNet and Perch v2 wrappers."""

from __future__ import annotations

import pytest
import torch

from avex.models.utils import onnx_runtime as oru


def test_ort_providers_keeps_only_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Ort:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["AzureExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(oru, "import_onnxruntime", lambda: _Ort)
    assert oru.ort_providers() == ["CPUExecutionProvider"]

    _Ort.get_available_providers = staticmethod(lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert oru.ort_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_warn_if_cpu_fallback_only_for_cuda_without_provider(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger=oru.__name__):
        oru.warn_if_cpu_fallback("M", "cpu", ["CPUExecutionProvider"])
        oru.warn_if_cpu_fallback("M", "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert caplog.text == ""
        oru.warn_if_cpu_fallback("M", torch.device("cuda:1"), ["CPUExecutionProvider"])
    assert "M runs on CPU" in caplog.text and "cuda:1" in caplog.text


def test_warn_cannot_unfreeze_fires_once_per_model(caplog: pytest.LogCaptureFixture) -> None:
    model = torch.nn.Identity()
    with caplog.at_level("WARNING", logger=oru.__name__):
        oru.warn_cannot_unfreeze(model)
        oru.warn_cannot_unfreeze(model)
        oru.warn_cannot_unfreeze(torch.nn.Identity())
    assert caplog.text.count("has no effect on Identity") == 2
