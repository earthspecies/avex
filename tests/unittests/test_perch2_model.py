"""Perch v2 wrapper, probing and weight resolution, without the real weights."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import torch

from avex.configs import ProbeConfig
from avex.models.perch2 import Perch2Model
from avex.models.perch2_net import weights as w
from avex.models.perch2_net.convert import expected_shapes
from avex.models.probes.utils.factory import build_probe_from_config

N = 160_000


@pytest.fixture(scope="module")
def model() -> Perch2Model:
    """Randomly initialised Perch v2 (no weight download).

    Returns
    -------
    Perch2Model
        Model on CPU in eval mode.
    """
    torch.manual_seed(0)
    m = Perch2Model(pretrained=False, device="cpu")
    for p in m.backbone.parameters():  # zero-initialised by default; make outputs non-trivial
        torch.nn.init.normal_(p, std=0.05)
    m.backbone.frontend.window.fill_(1.0)
    torch.nn.init.uniform_(m.backbone.frontend.mel_matrix)
    return m.eval()


def test_output_shapes(model: Perch2Model) -> None:
    """Embedding, spatial embedding and species logits have Perch v2's shapes."""
    wav = torch.randn(2, N) * 0.1
    with torch.no_grad():
        assert model(wav).shape == (2, 1536)
        assert model.spatial_embed(wav).shape == (2, 16, 4, 1536)
        assert model.species_logits(wav).shape == (2, 14795)


def test_input_is_cropped_or_padded_to_five_seconds(model: Perch2Model) -> None:
    """Longer and shorter clips give the same embedding shape; stereo is downmixed."""
    with torch.no_grad():
        assert model(torch.randn(1, N + 999)).shape == (1, 1536)
        assert model(torch.randn(1, 2, N // 3)).shape == (1, 1536)


def test_last_layer_resolves_to_pooled_embedding(model: Perch2Model) -> None:
    """Issue #208: target_layers=['last_layer'] used to fail with 'No layers available'."""
    assert model.register_hooks_for_layers(["last_layer"]) == ["backbone.pool"]
    wav = torch.randn(2, N) * 0.1
    with torch.no_grad():
        hooked = model.extract_embeddings(wav, aggregation="mean", freeze_backbone=True)
        torch.testing.assert_close(hooked, model.embed(wav))
    model.deregister_all_hooks()
    model._hook_layers = []


def test_named_block_layers_are_hookable(model: Perch2Model) -> None:
    """Hooks on stable names return pooled, max-pooled or sequence features."""
    layers = model.register_hooks_for_layers(["backbone.blocks.4.2", "backbone.head"])
    assert layers == ["backbone.blocks.4.2", "backbone.head"]
    wav = torch.randn(2, N) * 0.1
    with torch.no_grad():
        mean = model.extract_embeddings(wav, aggregation="mean")
        seq = model.extract_embeddings(wav, aggregation="none")
    assert mean.shape == (2, 136 + 1536)
    assert [t.shape for t in seq] == [(2, 32, 136 * 8), (2, 16, 1536 * 4)]
    model.deregister_all_hooks()
    model._hook_layers = []


def test_no_hooks_returns_pooled_embedding(model: Perch2Model) -> None:
    """Issue #208: target_layers=[] with freeze_backbone used to raise TypeError."""
    wav = torch.randn(2, N) * 0.1
    model.register_hooks_for_layers([])
    with torch.no_grad():
        assert model.extract_embeddings(wav, freeze_backbone=True).shape == (2, 1536)
        assert [t.shape for t in model.extract_embeddings(wav, aggregation="none")] == [(2, 1, 1536)]


def test_unsupported_aggregation_raises(model: Perch2Model) -> None:
    """Unknown aggregation names are rejected."""
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        model.extract_embeddings(torch.randn(1, N), aggregation="median")


@pytest.mark.parametrize("target_layers", [["last_layer"], []])
def test_linear_probe_online(model: Perch2Model, target_layers: list[str]) -> None:
    """Issue #208: an online linear probe on Perch v2 builds and runs."""
    config = ProbeConfig(
        probe_type="linear",
        target_layers=target_layers,
        aggregation="mean",
        freeze_backbone=True,
        online_training=True,
    )
    probe = build_probe_from_config(config, num_classes=5, device="cpu", base_model=model, target_length=N)
    with torch.no_grad():
        assert probe(torch.randn(2, N) * 0.1, padding_mask=None).shape == (2, 5)
    model.deregister_all_hooks()
    model._hook_layers = []


def test_gradients_reach_the_backbone_through_the_frontend() -> None:
    """Fine-tuning: loss.backward() gives non-zero gradients on a block and the input."""
    torch.manual_seed(0)
    m = Perch2Model(pretrained=False, device="cpu")
    for p in m.backbone.parameters():
        torch.nn.init.normal_(p, std=0.05)
    m.backbone.frontend.window.fill_(1.0)
    torch.nn.init.uniform_(m.backbone.frontend.mel_matrix)
    wav = (torch.randn(1, N) * 0.1).requires_grad_()
    m.extract_embeddings(wav, freeze_backbone=False).pow(2).sum().backward()
    grad = m.backbone.blocks[3][0].depthwise.conv.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    assert wav.grad is not None and wav.grad.abs().sum() > 0

    m.zero_grad(set_to_none=True)
    assert not m.extract_embeddings(torch.randn(1, N), freeze_backbone=True).requires_grad


def test_avex_classifier_head() -> None:
    """num_classes adds a linear classifier on the embedding; Perch's head is separate."""
    m = Perch2Model(pretrained=False, device="cpu", num_classes=7)
    with torch.no_grad():
        assert m(torch.randn(1, N)).shape == (1, 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runs_on_cuda() -> None:
    """Issue #208: Perch v2 used to run only on CPU."""
    m = Perch2Model(pretrained=False, device="cuda")
    with torch.no_grad():
        out = m(torch.randn(2, N))
    assert out.device.type == "cuda"


# --------------------------------------------------------------------- #
#  Weight resolution and verification
# --------------------------------------------------------------------- #
def _random_state() -> dict[str, torch.Tensor]:
    return {k: torch.randn(shape) for k, shape in expected_shapes().items()}


def _write_cache(path: Path, state: dict[str, torch.Tensor], **metadata: str) -> None:
    from safetensors.torch import save_file

    meta = {"format_version": str(w.FORMAT_VERSION), "source_onnx_sha256": w.ONNX_SHA256}
    meta.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata=meta)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the avex cache at a temp dir and disable the HF cache lookup.

    Returns
    -------
    Path
        The Perch v2 cache file path.
    """
    monkeypatch.setenv("ESP_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv(w.ONNX_ENV_VAR, raising=False)
    monkeypatch.setattr(w, "_hf_cached_onnx", lambda: None)
    return w.cache_path()


def test_cache_path_is_versioned(cache: Path) -> None:
    """A mapping fix bumps FORMAT_VERSION and therefore the filename."""
    assert cache.name == f"perch2-v{w.FORMAT_VERSION}.safetensors"


def test_cached_weights_load_and_verify(cache: Path) -> None:
    """A valid cache file is returned without conversion and loads."""
    state = _random_state()
    _write_cache(cache, state)
    assert w.resolve_weights(allow_download=False) == cache
    loaded = w.load_weights(cache)
    torch.testing.assert_close(loaded["head.conv.weight"], state["head.conv.weight"])


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda s: s.pop("head.conv.bias"), "wrong keys"),
        (lambda s: s.__setitem__("extra", torch.zeros(1)), "wrong keys"),
        (lambda s: s.__setitem__("head.conv.bias", torch.zeros(3)), "expected"),
        (lambda s: s.__setitem__("head.conv.bias", torch.full((1536,), float("nan"))), "NaN"),
        (lambda s: s.__setitem__("head.conv.bias", torch.zeros(1536, dtype=torch.float16)), "float32"),
    ],
)
def test_structural_verification_rejects_bad_files(cache: Path, mutate: object, match: str) -> None:
    """Wrong keys, shapes, dtypes or non-finite values never load silently."""
    state = _random_state()
    mutate(state)
    _write_cache(cache, state)
    with pytest.raises(w.Perch2WeightsError, match=match):
        w.load_weights(cache)


@pytest.mark.parametrize(
    "metadata, match",
    [({"format_version": "0"}, "format version"), ({"source_onnx_sha256": "0" * 64}, "different ONNX")],
)
def test_metadata_mismatch_is_rejected(cache: Path, metadata: dict[str, str], match: str) -> None:
    """Files from another format version or source ONNX are not reused."""
    _write_cache(cache, _random_state(), **metadata)
    with pytest.raises(w.Perch2WeightsError, match=match):
        w.load_weights(cache)


def test_missing_cache_without_download_names_the_cache_path(cache: Path) -> None:
    """Offline nodes get a clear error naming the expected cache path and the fetch command."""
    with pytest.raises(w.Perch2WeightsError, match="avex fetch perch2") as err:
        w.resolve_weights(allow_download=False)
    assert str(cache) in str(err.value)


def test_wrong_onnx_is_rejected(cache: Path, tmp_path: Path) -> None:
    """Only the pinned ONNX (by SHA-256) is converted."""
    bad = tmp_path / "perch_v2.onnx"
    bad.write_bytes(b"not an onnx file")
    with pytest.raises(w.Perch2WeightsError, match="not the pinned"):
        w.resolve_weights(onnx_path=bad)


def test_concurrent_cold_start_converts_once(cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Several workers on a cold cache: one converts, the others wait and reuse it."""
    calls: list[int] = []
    source = tmp_path / "perch_v2.onnx"
    source.write_bytes(b"x")

    def fake_convert(onnx_path: Path, target: Path) -> Path:
        calls.append(1)
        time.sleep(0.3)
        _write_cache(target, _random_state())
        return target

    monkeypatch.setattr(w, "convert_to_cache", fake_convert)
    results: list[Path] = []
    threads = [threading.Thread(target=lambda: results.append(w.resolve_weights(onnx_path=source))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert results == [cache] * 4
