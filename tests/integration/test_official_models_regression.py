"""Regression tests for packaged `official_models`.

This file merges:
- a probe-style smoke regression for all official models
- BEATs-family pinned probe losses (fixtures)
- BEATs-family pinned config + weight sentinel hashes (fixtures)

The intent is to keep a single integration entrypoint that covers:
1) "does it load and run" across the whole official model set
2) "did BEATs behavior/weights/config wiring change" for the BEATs-family subset
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Generator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Final, Iterable

import pytest
import torch
import torch.nn.functional as F

from avex.io import anypath, exists
from avex.models.beats.beats import BEATsConfig
from avex.models.utils.load import load_model
from avex.models.utils.registry import get_checkpoint_path
from avex.utils import universal_torch_load

_OFFICIAL_MODELS_PKG: Final[str] = "avex.api.configs.official_models"


def _iter_official_model_keys() -> Iterable[str]:
    root = resources.files(_OFFICIAL_MODELS_PKG)
    for entry in root.iterdir():
        if entry.is_file() and entry.name.endswith(".yml"):
            yield Path(entry.name).stem


@dataclass(frozen=True)
class _ProbeResult:
    initial_loss: float
    final_loss: float


@contextlib.contextmanager
def _determinism_ctx(seed: int) -> Generator[None, None, None]:
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_threads = torch.get_num_threads()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_det)
        torch.set_num_threads(prev_threads)


def _pool_embeddings(feats: torch.Tensor) -> torch.Tensor:
    """Convert model outputs into a 2D (B, D) embedding tensor.

    Returns
    -------
    torch.Tensor
        Pooled embeddings with shape ``(B, D)``.

    Raises
    ------
    ValueError
        If the model output has an unsupported rank.
    """
    if feats.ndim == 3:
        return feats.mean(dim=1)
    if feats.ndim == 4:
        return feats.mean(dim=(2, 3))
    if feats.ndim == 2:
        return feats
    raise ValueError(f"Unexpected embedding shape {tuple(feats.shape)}")


def _run_linear_probe_smoke(*, model_key: str, device: str, seed: int) -> _ProbeResult:
    """Probe regression used for the full official_models set (no pinned fixtures).

    Returns
    -------
    _ProbeResult
        Initial and final probe losses.
    """
    with _determinism_ctx(seed):
        checkpoint_path = get_checkpoint_path(model_key)
        if checkpoint_path is None:
            pytest.skip(f"Checkpoint not available in registry for {model_key!r}")

        if not exists(anypath(checkpoint_path)):
            pytest.skip(f"Checkpoint path not reachable: {checkpoint_path!r}")

        model = load_model(model_key, device=device, return_features_only=True)
        model.eval()

        batch = 8
        num_samples = 16_000
        num_classes = 5
        audio = torch.randn(batch, num_samples, device=device)

        with torch.no_grad():
            feats = model(audio)
            pooled = _pool_embeddings(feats)

        teacher = torch.nn.Linear(pooled.shape[-1], num_classes, bias=True).to(device)
        torch.nn.init.normal_(teacher.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(teacher.bias)
        labels = teacher(pooled).argmax(dim=-1)

        probe = torch.nn.Linear(pooled.shape[-1], num_classes, bias=True).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=0.0)

        def loss_fn() -> torch.Tensor:
            logits = probe(pooled)
            return F.cross_entropy(logits, labels)

        initial = float(loss_fn().item())
        for _ in range(50):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            opt.step()
        final = float(loss_fn().item())
        return _ProbeResult(initial_loss=initial, final_loss=final)


@pytest.mark.parametrize("model_key", sorted(_iter_official_model_keys()))
def test_official_models_linear_probe_smoke(model_key: str) -> None:
    observed = _run_linear_probe_smoke(model_key=model_key, device="cpu", seed=123)
    assert observed.final_loss < observed.initial_loss, f"{model_key}: loss did not decrease"


# ---------------------------------------------------------------------------
# BEATs-family: pinned probe regression fixtures
# ---------------------------------------------------------------------------


_BEATS_PROBE_FIXTURES_DIR: Final[Path] = Path(__file__).parent / "fixtures" / "beats_probe"


def _load_probe_fixture(path: Path) -> _ProbeResult:
    text = path.read_text(encoding="utf-8").strip().splitlines()
    mapping: dict[str, float] = {}
    for line in text:
        if not line.strip() or line.strip().startswith("#"):
            continue
        key, value = line.split("=", maxsplit=1)
        mapping[key.strip()] = float(value.strip())
    return _ProbeResult(
        initial_loss=mapping["initial_loss"],
        final_loss=mapping["final_loss"],
    )


def _run_beats_probe_with_pinned_losses(*, model_key: str, device: str, seed: int) -> _ProbeResult:
    with _determinism_ctx(seed):
        checkpoint_path = get_checkpoint_path(model_key)
        if checkpoint_path is None:
            pytest.skip(f"Checkpoint not available in registry for {model_key!r}")

        model = load_model(model_key, device=device, return_features_only=True)
        model.eval()

        batch = 8
        num_samples = 16_000
        num_classes = 5
        audio = torch.randn(batch, num_samples, device=device)

        with torch.no_grad():
            feats = model(audio)
            if feats.ndim != 3:
                raise AssertionError(f"Expected embeddings of shape (B, T, D), got {tuple(feats.shape)}")
            pooled = feats.mean(dim=1)

        teacher = torch.nn.Linear(pooled.shape[-1], num_classes, bias=True).to(device)
        torch.nn.init.normal_(teacher.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(teacher.bias)
        labels = teacher(pooled).argmax(dim=-1)

        probe = torch.nn.Linear(pooled.shape[-1], num_classes, bias=True).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=0.0)

        def loss_fn() -> torch.Tensor:
            logits = probe(pooled)
            return F.cross_entropy(logits, labels)

        initial = float(loss_fn().item())
        for _ in range(50):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            opt.step()
        final = float(loss_fn().item())
    return _ProbeResult(initial_loss=initial, final_loss=final)


@pytest.mark.parametrize(
    ("model_key", "fixture_name"),
    [
        ("esp_aves2_naturelm_audio_v1_beats", "esp_aves2_naturelm_audio_v1_beats.txt"),
        ("esp_aves2_sl_beats_all", "esp_aves2_sl_beats_all.txt"),
        ("esp_aves2_sl_beats_bio", "esp_aves2_sl_beats_bio.txt"),
    ],
)
def test_official_beats_checkpoints_linear_probe_regression(model_key: str, fixture_name: str) -> None:
    fixture_path = _BEATS_PROBE_FIXTURES_DIR / fixture_name
    if not fixture_path.exists():
        pytest.skip(f"Missing fixture {fixture_path}")

    observed = _run_beats_probe_with_pinned_losses(model_key=model_key, device="cpu", seed=123)
    expected = _load_probe_fixture(fixture_path)

    assert observed.final_loss < observed.initial_loss, f"{model_key}: loss did not decrease"

    torch.testing.assert_close(
        torch.tensor([observed.initial_loss, observed.final_loss]),
        torch.tensor([expected.initial_loss, expected.final_loss]),
        atol=1e-4,
        rtol=0.0,
    )


# ---------------------------------------------------------------------------
# BEATs-family: pinned config + weight sentinel hashes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ExpectedBeats:
    config: dict[str, Any]
    tensors: dict[str, str]


_BEATS_FIXTURES_DIR: Final[Path] = Path(__file__).parent / "fixtures" / "beats_weights_and_config"

_TENSOR_SUBSTRINGS: Final[tuple[str, ...]] = (
    "patch_embedding.weight",
    "encoder.layers.0.self_attn.q_proj.weight",
    "encoder.layers.0.fc1.weight",
    "encoder.layers.0.fc2.weight",
    "post_extract_proj.weight",
)

_PINNED_CFG_KEYS: Final[tuple[str, ...]] = (
    "encoder_layers",
    "encoder_embed_dim",
    "encoder_ffn_embed_dim",
    "encoder_attention_heads",
    "activation_fn",
    "deep_norm",
    "layer_norm_first",
    "dropout",
    "attention_dropout",
    "activation_dropout",
    "encoder_layerdrop",
    "dropout_input",
    "relative_position_embedding",
    "num_buckets",
    "max_distance",
    "gru_rel_pos",
    "input_patch_size",
    "embed_dim",
    "conv_bias",
    "conv_pos",
    "conv_pos_groups",
    "layer_wise_gradient_decay_ratio",
    "finetuned_model",
    "predictor_class",
    "predictor_dropout",
)


def _sha256_tensor(t: torch.Tensor) -> str:
    b = t.detach().to("cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def _pick_tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    picked: dict[str, str] = {}
    for needle in _TENSOR_SUBSTRINGS:
        matches = [k for k in state.keys() if needle in k]
        if not matches:
            continue
        key = sorted(matches, key=len)[0]
        picked[key] = _sha256_tensor(state[key])
    if not picked:
        raise AssertionError("No sentinel tensors found to hash")
    return picked


def _load_expected_beats(path: Path) -> _ExpectedBeats:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _ExpectedBeats(config=data["config"], tensors=data["tensors"])


def _extract_beats_loader_config(model_key: str) -> dict[str, Any]:
    ckpt_path = get_checkpoint_path(model_key)
    if ckpt_path is None:
        pytest.skip(f"Checkpoint not available in registry for {model_key!r}")

    if ckpt_path.endswith(".pt"):
        ckpt = universal_torch_load(ckpt_path, cache_mode="use", map_location="cpu")
        cfg = BEATsConfig(**ckpt["cfg"])
        cfg_dump = cfg.model_dump()
        return {k: cfg_dump.get(k) for k in _PINNED_CFG_KEYS if k in cfg_dump}

    model = load_model(model_key, device="cpu", return_features_only=True)
    backbone_cfg = getattr(getattr(model, "backbone", model), "cfg", None)
    if backbone_cfg is None:
        pytest.skip(f"{model_key}: could not locate backbone cfg")
    if not isinstance(backbone_cfg, BEATsConfig):
        pytest.skip(f"{model_key}: unexpected cfg type {type(backbone_cfg)}")
    cfg_dump = backbone_cfg.model_dump()
    return {k: cfg_dump.get(k) for k in _PINNED_CFG_KEYS if k in cfg_dump}


def _extract_beats_loader_weight_hashes(model_key: str) -> dict[str, str]:
    model = load_model(model_key, device="cpu", return_features_only=True)
    return _pick_tensor_hashes(model.state_dict())


@pytest.mark.parametrize(
    ("model_key", "fixture_name"),
    [
        ("esp_aves2_naturelm_audio_v1_beats", "esp_aves2_naturelm_audio_v1_beats.json"),
        ("esp_aves2_sl_beats_all", "esp_aves2_sl_beats_all.json"),
        ("esp_aves2_sl_beats_bio", "esp_aves2_sl_beats_bio.json"),
    ],
)
def test_official_beats_weight_and_config_fixtures(model_key: str, fixture_name: str) -> None:
    fixture_path = _BEATS_FIXTURES_DIR / fixture_name
    if not fixture_path.exists():
        pytest.skip(f"Missing fixture {fixture_path}")

    expected = _load_expected_beats(fixture_path)
    observed_cfg = _extract_beats_loader_config(model_key)
    observed_hashes = _extract_beats_loader_weight_hashes(model_key)

    assert observed_cfg == expected.config, f"{model_key}: config mismatch"
    assert observed_hashes == expected.tensors, f"{model_key}: weight hash mismatch"
