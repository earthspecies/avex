"""Unit tests for shared logic in base_probes.py.

Covers:
- print_learned_weights messaging and table
- _sum behavior with/without learned weights
- _get_embeddings behavior in feature_mode and with base_model
"""

from __future__ import annotations

import io
import sys
from typing import Any, Optional

import torch
import torch.nn as nn

from avex.models.base_model import ModelBase
from avex.models.probes.base_probes import BaseProbe2D, BaseProbe3D


class _Dummy2DProbe(BaseProbe2D):
    def build_head(self, inferred_dim: int) -> None:
        self.head = nn.Identity()

    def forward(self, x: torch.Tensor | dict, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401
        emb = self._get_embeddings(x, padding_mask)
        emb = self._combine_or_reshape_embeddings(emb)
        return self.head(emb)


class _Dummy3DProbe(BaseProbe3D):
    def build_head(self, inferred_dim: int) -> None:
        self.head = nn.Identity()

    def forward(self, x: torch.Tensor | dict, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401
        emb = self._get_embeddings(x, padding_mask)
        emb = self._combine_or_reshape_embeddings(emb)
        return self.head(emb)


class _MockAudioProcessor:
    def __init__(self, target_length: int = 1000, sr: int = 16000) -> None:
        self.target_length = target_length
        self.sr = sr
        self.target_length_seconds = target_length / sr


class _MockBaseModel(ModelBase):
    def __init__(self, out_shape: torch.Size, device: str = "cpu") -> None:
        super().__init__(device=device)
        self._shape = out_shape
        self.device = device
        self.audio_processor = _MockAudioProcessor()
        self.called: dict[str, Any] = {}

    def extract_embeddings(
        self,
        x: torch.Tensor | dict,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean",
        freeze_backbone: bool = True,
    ) -> torch.Tensor:
        self.called = {"aggregation": aggregation, "freeze_backbone": freeze_backbone}
        b = x.shape[0]
        return torch.randn(b, *self._shape, device=self.device)


def _capture_print(fn: callable) -> str:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn()
        return buf.getvalue()
    finally:
        sys.stdout = old


def test_sum_without_weights_is_simple_sum() -> None:
    probe = _Dummy2DProbe(
        base_model=None,
        layers=[],
        num_classes=2,
        device="cpu",
        feature_mode=True,
        input_dim=4,
    )
    t1 = torch.ones(2, 4)
    t2 = 2 * torch.ones(2, 4)
    summed = probe._sum([t1, t2])
    assert torch.allclose(summed, 3 * torch.ones(2, 4))


def test_sum_with_weights_uses_softmax() -> None:
    probe = _Dummy2DProbe(
        base_model=None,
        layers=[],
        num_classes=2,
        device="cpu",
        feature_mode=True,
        input_dim=4,
    )
    probe.layer_weights = nn.Parameter(torch.tensor([0.0, 2.0], dtype=torch.float))
    t1 = torch.ones(1, 4)
    t2 = 2 * torch.ones(1, 4)
    w = torch.softmax(probe.layer_weights, dim=0)
    expected = w[0] * t1 + w[1] * t2
    got = probe._sum([t1, t2])
    assert torch.allclose(got, expected)


def test_get_embeddings_feature_mode_dict_raw_wav_priority() -> None:
    probe = _Dummy2DProbe(
        base_model=None,
        layers=[],
        num_classes=2,
        device="cpu",
        feature_mode=True,
        input_dim=4,
    )
    raw = torch.randn(3, 4)
    out = probe._get_embeddings({"raw_wav": raw, "other": torch.randn(3, 4)}, None)
    assert out is raw


def test_get_embeddings_feature_mode_single_key_tensor() -> None:
    probe = _Dummy2DProbe(
        base_model=None,
        layers=[],
        num_classes=2,
        device="cpu",
        feature_mode=True,
        input_dim=4,
    )
    emb = torch.randn(2, 4)
    out = probe._get_embeddings({"layer1": emb}, None)
    assert torch.allclose(out, emb)


def test_get_embeddings_feature_mode_multi_key_list() -> None:
    probe = _Dummy2DProbe(
        base_model=None,
        layers=[],
        num_classes=2,
        device="cpu",
        feature_mode=True,
        input_dim=4,
    )
    emb1 = torch.randn(2, 4)
    emb2 = torch.randn(2, 4)
    out = probe._get_embeddings({"a": emb1, "b": emb2}, None)
    assert isinstance(out, list)
    assert len(out) == 2


def test_get_embeddings_non_feature_mode_calls_base_model_and_detaches() -> None:
    base = _MockBaseModel(out_shape=torch.Size([6]))
    probe = _Dummy2DProbe(
        base_model=base,
        layers=["l1"],
        num_classes=2,
        device="cpu",
        feature_mode=False,
        aggregation="mean",
    )
    x = torch.randn(2, 1000)
    out = probe._get_embeddings(x, None)
    assert out.shape == (2, 6)
    assert base.called.get("aggregation") == "mean"
