"""Regression tests for the BaseProbe3D 2-D-embedding semantic change.

PR #184 changed how 3-D probes (LSTM/attention/transformer) interpret a 2-D
``(batch, features)`` embedding:

- old: treated as ``features`` timesteps each with a single scalar feature
  (``_format_to_seq_feat`` -> ``(B, F, 1)``; ``_infer_single_tensor_dim`` -> 1)
- new: treated as a single timestep with the full feature dim
  (``_format_to_seq_feat`` -> ``(B, 1, F)``; ``_infer_single_tensor_dim`` -> F)

The new behavior is the correct reading of a pooled embedding, but it silently
changes the architecture/results of any existing 3-D probe fed 2-D embeddings
(e.g. any backbone with ``aggregation="mean"``). These tests lock in the new
semantics at both the unit level and end-to-end across all three real 3-D
probe types.
"""

from __future__ import annotations

from typing import List, Optional, Union

import pytest
import torch
import torch.nn as nn

from avex.models.base_model import ModelBase
from avex.models.probes.attention_probe import AttentionProbe
from avex.models.probes.base_probes import BaseProbe3D
from avex.models.probes.lstm_probe import LSTMProbe
from avex.models.probes.transformer_probe import TransformerProbe


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
class _Dummy3DProbe(BaseProbe3D):
    """Minimal concrete 3-D probe with an identity head (shape-only checks)."""

    def build_head(self, inferred_dim: int) -> None:
        self.head = nn.Identity()

    def forward(self, x: torch.Tensor | dict, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self._get_embeddings(x, padding_mask)
        emb = self._combine_or_reshape_embeddings(emb)
        return self.head(emb)


class _MockAudioProcessor:
    def __init__(self, target_length: int = 1000, sr: int = 16000) -> None:
        self.target_length = target_length
        self.sr = sr
        self.target_length_seconds = target_length / sr


class _Pooled2DBaseModel(ModelBase):
    """Base model that returns a 2-D pooled embedding (B, F), as aggregation='mean' does."""

    def __init__(self, feat_dim: int, device: str = "cpu") -> None:
        super().__init__(device=device)
        self.device = device
        self.feat_dim = feat_dim
        self.audio_processor = _MockAudioProcessor()

    def extract_embeddings(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean",
        freeze_backbone: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        return torch.randn(x.shape[0], self.feat_dim, device=self.device)


_THREE_D_PROBES = [LSTMProbe, AttentionProbe, TransformerProbe]


def _built_feature_dim(probe: nn.Module) -> int:
    # The feature dim the probe head was actually built with.
    if isinstance(probe, LSTMProbe):
        return probe.lstm.input_size
    return probe.classifier.in_features


# --------------------------------------------------------------------------- #
#  Unit-level: the changed shape primitives
# --------------------------------------------------------------------------- #
def test_format_to_seq_feat_2d_is_single_timestep() -> None:
    """2-D (B, F) must become (B, 1, F): one timestep, full feature dim."""
    probe = _Dummy3DProbe(base_model=None, layers=[], num_classes=2, device="cpu", feature_mode=True, input_dim=8)
    emb = torch.randn(3, 8)
    out = probe._format_to_seq_feat(emb)
    assert out.shape == (3, 1, 8)
    # The single timestep is exactly the original embedding.
    assert torch.allclose(out[:, 0, :], emb)


def test_infer_single_tensor_dim_2d_is_feature_dim() -> None:
    """2-D (B, F) must infer feature dim F, not the old scalar 1."""
    probe = _Dummy3DProbe(base_model=None, layers=[], num_classes=2, device="cpu", feature_mode=True, input_dim=8)
    assert probe._infer_single_tensor_dim(torch.randn(3, 16)) == 16


def test_format_to_seq_feat_3d_passthrough() -> None:
    probe = _Dummy3DProbe(base_model=None, layers=[], num_classes=2, device="cpu", feature_mode=True, input_dim=8)
    emb = torch.randn(3, 5, 8)
    assert probe._format_to_seq_feat(emb).shape == (3, 5, 8)


def test_format_to_seq_feat_4d_reshape() -> None:
    probe = _Dummy3DProbe(base_model=None, layers=[], num_classes=2, device="cpu", feature_mode=True, input_dim=8)
    emb = torch.randn(2, 4, 3, 7)  # (B, C, H, W) -> (B, W, C*H)
    assert probe._format_to_seq_feat(emb).shape == (2, 7, 12)
    assert probe._infer_single_tensor_dim(emb) == 12


def test_analyze_projectors_2d_uses_feature_dim_as_target() -> None:
    """A list with a single 2-D embedding targets feat=F, seq=1 (no projector needed)."""
    probe = _Dummy3DProbe(base_model=None, layers=[], num_classes=2, device="cpu", feature_mode=True, input_dim=8)
    target_feat = probe._analyze_and_create_projectors([torch.randn(1, 8)])
    assert target_feat == 8


# --------------------------------------------------------------------------- #
#  End-to-end across all three real 3-D probes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("probe_cls", _THREE_D_PROBES)
def test_3d_probe_feature_mode_2d_input_builds_full_feature_dim(probe_cls: type) -> None:
    """feature_mode with a 2-D pooled input (F,) builds the head with dim F (not 1)."""
    feat_dim = 24  # divisible by attention/transformer head counts after adjustment
    num_classes = 5
    probe = probe_cls(
        base_model=None,
        layers=[],
        num_classes=num_classes,
        device="cpu",
        feature_mode=True,
        input_dim=(feat_dim,),  # 2-D pooled embedding shape
    )
    # Regression guard: old behavior would have built with feature dim 1.
    assert _built_feature_dim(probe) == feat_dim

    probe.eval()
    with torch.no_grad():
        out = probe(torch.randn(4, feat_dim))
    assert out.shape == (4, num_classes)


@pytest.mark.parametrize("probe_cls", _THREE_D_PROBES)
def test_3d_probe_with_pooled_base_model_2d_embeddings(probe_cls: type) -> None:
    """Non-feature-mode: a backbone returning 2-D (B, F) builds and runs correctly."""
    feat_dim = 24
    num_classes = 3
    base = _Pooled2DBaseModel(feat_dim=feat_dim)
    probe = probe_cls(
        base_model=base,
        layers=["layer1"],
        num_classes=num_classes,
        device="cpu",
        feature_mode=False,
        aggregation="mean",
    )
    assert _built_feature_dim(probe) == feat_dim

    probe.eval()
    with torch.no_grad():
        out = probe(torch.randn(4, 1000))
    assert out.shape == (4, num_classes)


@pytest.mark.parametrize("probe_cls", _THREE_D_PROBES)
def test_3d_probe_2d_input_batch_independence(probe_cls: type) -> None:
    """Each row's output must not depend on its batch neighbors (no leakage across seq=1)."""
    feat_dim = 24
    probe = probe_cls(
        base_model=None,
        layers=[],
        num_classes=4,
        device="cpu",
        feature_mode=True,
        input_dim=(feat_dim,),
    )
    probe.eval()
    x = torch.randn(6, feat_dim)
    with torch.no_grad():
        batched = probe(x)
        rows = torch.cat([probe(x[i : i + 1]) for i in range(x.shape[0])], dim=0)
    assert torch.allclose(batched, rows, atol=1e-5)


@pytest.mark.parametrize("probe_cls", _THREE_D_PROBES)
def test_3d_probe_3d_input_still_works(probe_cls: type) -> None:
    """A genuine 3-D (B, T, F) sequence input is unaffected by the 2-D change."""
    feat_dim = 24
    num_classes = 3
    probe = probe_cls(
        base_model=None,
        layers=[],
        num_classes=num_classes,
        device="cpu",
        feature_mode=True,
        input_dim=(10, feat_dim),  # (T, F)
    )
    assert _built_feature_dim(probe) == feat_dim
    probe.eval()
    with torch.no_grad():
        out = probe(torch.randn(2, 10, feat_dim))
    assert out.shape == (2, num_classes)


@pytest.mark.parametrize("probe_cls", _THREE_D_PROBES)
def test_3d_probe_multilayer_mixed_2d_and_3d(probe_cls: type) -> None:
    """A multi-layer list mixing a 2-D pooled layer and a 3-D sequence layer works."""
    feat_dim = 24
    num_classes = 3
    probe = probe_cls(
        base_model=None,
        layers=[],
        num_classes=num_classes,
        device="cpu",
        feature_mode=True,
        input_dim=[(feat_dim,), (10, feat_dim)],  # one pooled (2-D), one sequence (3-D)
    )
    probe.eval()
    with torch.no_grad():
        out = probe([torch.randn(2, feat_dim), torch.randn(2, 10, feat_dim)])
    assert out.shape == (2, num_classes)
