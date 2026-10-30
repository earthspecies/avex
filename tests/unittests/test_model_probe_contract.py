"""Contract tests between ``ModelBase.extract_embeddings`` and the probes.

Probes call ``base_model.extract_embeddings(x, padding_mask=..., aggregation=...,
freeze_backbone=...)`` blindly, so every ``ModelBase`` subclass must accept that
keyword set. Regression tests for
https://github.com/earthspecies/avex/issues/215.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Iterator, Optional

import pytest
import torch
from torch import nn

import avex.models
from avex.configs import ProbeConfig
from avex.models.base_model import ModelBase
from avex.models.probes.linear_probe import LinearProbe
from avex.models.probes.utils.factory import build_probe_from_config

REQUIRED_KEYWORDS = ("padding_mask", "aggregation", "freeze_backbone")


def _iter_model_classes() -> Iterator[type[ModelBase]]:
    """Yield every concrete ``ModelBase`` subclass defined under ``avex.models``.

    Modules whose optional dependencies are missing (e.g. ``pytorch_lightning``
    for ATST) are skipped rather than failing the whole parametrization.

    Yields:
        type[ModelBase]: Each discovered subclass, deduplicated.
    """
    seen: set[type] = set()
    for info in pkgutil.walk_packages(avex.models.__path__, "avex.models."):
        if ".probes" in info.name:
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, ModelBase) and obj is not ModelBase and obj.__module__ == info.name and obj not in seen:
                seen.add(obj)
                yield obj


MODEL_CLASSES = sorted(_iter_model_classes(), key=lambda c: f"{c.__module__}.{c.__name__}")


@pytest.mark.parametrize("model_cls", [ModelBase, *MODEL_CLASSES], ids=lambda c: f"{c.__module__}.{c.__name__}")
def test_extract_embeddings_signature_matches_probe_contract(model_cls: type[ModelBase]) -> None:
    """Every ``extract_embeddings`` must take the keywords the probes pass."""
    sig = inspect.signature(model_cls.extract_embeddings)
    params = sig.parameters
    for name in REQUIRED_KEYWORDS:
        assert name in params, f"{model_cls.__qualname__}.extract_embeddings is missing `{name}`"
        assert params[name].default is not inspect.Parameter.empty, f"`{name}` must have a default"
    # Only `self` and the input may be positional; everything else is keyword-only.
    positional = [
        p.name
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == ["self", "x"], f"{model_cls.__qualname__}: unexpected positional params {positional}"


class _TinyBackbone(ModelBase):
    """Smallest possible backbone that relies on ``ModelBase.extract_embeddings``."""

    def __init__(self, target_length: int = 256) -> None:
        super().__init__(device="cpu", audio_config=None)
        self.encoder = nn.Linear(target_length, 32)
        self.head = nn.Linear(32, 4)
        self.target_length = target_length

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the two linear layers.

        Args:
            x: Waveform batch of shape (B, target_length).
            padding_mask: Unused.

        Returns:
            torch.Tensor: Logits of shape (B, 4).
        """
        return self.head(torch.relu(self.encoder(x)))


def _build_tiny_probe(freeze_backbone: bool) -> tuple[_TinyBackbone, LinearProbe]:
    backbone = _TinyBackbone()
    backbone.register_hooks_for_layers(["encoder"])
    probe = LinearProbe(
        base_model=backbone,
        layers=["encoder"],
        num_classes=3,
        device="cpu",
        aggregation="mean",
        target_length=backbone.target_length,
        freeze_backbone=freeze_backbone,
    )
    return backbone, probe


def test_default_extract_embeddings_works_with_online_probe() -> None:
    """A subclass that does not override ``extract_embeddings`` can host a probe."""
    backbone, probe = _build_tiny_probe(freeze_backbone=True)
    out = probe(torch.randn(2, backbone.target_length))
    assert out.shape == (2, 3)


def test_default_extract_embeddings_no_grad_when_frozen() -> None:
    """``freeze_backbone=True`` must not build a graph through the backbone."""
    backbone, probe = _build_tiny_probe(freeze_backbone=True)
    probe(torch.randn(2, backbone.target_length)).sum().backward()
    assert backbone.encoder.weight.grad is None
    assert probe.classifier.weight.grad is not None


def test_default_extract_embeddings_grad_when_unfrozen() -> None:
    """``freeze_backbone=False`` fine-tunes the backbone end to end."""
    backbone, probe = _build_tiny_probe(freeze_backbone=False)
    probe(torch.randn(2, backbone.target_length)).sum().backward()
    assert backbone.encoder.weight.grad is not None


def test_factory_builds_online_probe_on_default_backbone() -> None:
    """The config-driven factory path (what users call) works end to end."""
    backbone = _TinyBackbone()
    cfg = ProbeConfig(probe_type="linear", target_layers=["last_layer"], aggregation="mean", freeze_backbone=True)
    probe = build_probe_from_config(cfg, num_classes=3, device="cpu", base_model=backbone, target_length=256)
    assert probe(torch.randn(2, 256)).shape == (2, 3)


class _SingleOutputBackbone(ModelBase):
    """Stand-in for TF-Hub / ONNX wrappers: one fixed embedding, no hookable layers."""

    def __init__(self) -> None:
        super().__init__(device="cpu", audio_config=None)
        self.proj = nn.Linear(256, 16)

    def register_hooks_for_layers(self, target_layers: list[str | int]) -> list[str]:
        """Delegate to the shared single-output helper.

        Args:
            target_layers: Requested selectors.

        Returns:
            list[str]: ``["embeddings"]``.
        """
        return self._register_single_output_layer(target_layers)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project the waveform.

        Args:
            x: Waveform batch of shape (B, 256).
            padding_mask: Unused.

        Returns:
            torch.Tensor: Embedding of shape (B, 16).
        """
        return self.proj(x)

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Return the single embedding via the shared aggregation helper.

        Args:
            x: Waveform batch or dict with ``"raw_wav"``.
            padding_mask: Unused.
            aggregation: Aggregation method.
            freeze_backbone: Unused.

        Returns:
            torch.Tensor | list[torch.Tensor]: Aggregated embedding.
        """
        wav = x["raw_wav"] if isinstance(x, dict) else x
        return self._aggregate_single_embedding(self.forward(wav), aggregation)


@pytest.mark.parametrize("selector", [["last_layer"], ["all"], ["embeddings"], [-1], [0]])
def test_register_single_output_layer_accepts_generic_selectors(selector: list[str | int]) -> None:
    """Generic selectors resolve to the pseudo layer for non-hookable backbones."""
    model = _SingleOutputBackbone()
    assert model.register_hooks_for_layers(selector) == [ModelBase.SINGLE_OUTPUT_LAYER]
    assert model._hook_layers == [ModelBase.SINGLE_OUTPUT_LAYER]


@pytest.mark.parametrize("selector", [["layer4"], ["last_layer", "proj"], [True], [3]])
def test_register_single_output_layer_rejects_unknown_selectors(selector: list) -> None:
    """Anything that names a real layer is an error, with a message naming the model."""
    model = _SingleOutputBackbone()
    with pytest.raises(ValueError, match="_SingleOutputBackbone"):
        model.register_hooks_for_layers(selector)


def test_aggregate_single_embedding_contract() -> None:
    """``"none"`` yields a one-element (B, 1, D) list; pooling methods pass through."""
    emb = torch.randn(2, 16)
    none = ModelBase._aggregate_single_embedding(emb, "none")
    assert isinstance(none, list) and len(none) == 1 and none[0].shape == (2, 1, 16)
    for method in ("mean", "max", "cls_token"):
        assert ModelBase._aggregate_single_embedding(emb, method) is emb
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        ModelBase._aggregate_single_embedding(emb, "median")


@pytest.mark.parametrize("probe_type", ["linear", "lstm"])
def test_factory_builds_probe_on_single_output_backbone(probe_type: str) -> None:
    """``target_layers=["last_layer"]`` works through the factory on a hook-less wrapper."""
    cfg_kwargs: dict = {"probe_type": probe_type, "target_layers": ["last_layer"], "freeze_backbone": True}
    if probe_type == "lstm":
        cfg_kwargs.update(aggregation="none", input_processing="sequence", lstm_hidden_size=8, num_layers=1)
    else:
        cfg_kwargs.update(aggregation="mean")
    probe = build_probe_from_config(
        ProbeConfig(**cfg_kwargs), num_classes=3, device="cpu", base_model=_SingleOutputBackbone(), target_length=256
    )
    assert probe(torch.randn(2, 256)).shape == (2, 3)
