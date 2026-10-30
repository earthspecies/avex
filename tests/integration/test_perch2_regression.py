"""Perch v2 PyTorch port against ONNX Runtime reference outputs.

``test_matches_golden_fixtures`` compares the converted model with outputs
that ONNX Runtime produced on deterministic clips
(``tests/fixtures/perch2/golden.npz``, written by
``scripts/perch2_parity.py --write-fixtures``), so it needs neither onnx nor
onnxruntime at test time. It does need the converted weights: it reuses the
avex cache, or downloads and converts the pinned ONNX (~410 MB) once.

``test_live_layerwise_parity`` reruns the full layer-by-layer comparison
against ONNX Runtime.

Both are marked slow (excluded from CI's ``-m "not slow"`` run) because of
the weight download; run them with ``pytest -m slow``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from avex.models.perch2_net.golden import CLIP_NAMES, golden_clips

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "perch2" / "golden.npz"


@pytest.fixture(scope="module")
def outputs() -> dict[str, np.ndarray]:
    """Run ``load_model('perch2')`` on the golden clips in fp32.

    Returns
    -------
    dict[str, np.ndarray]
        Embedding, spatial embedding mean, subsampled spectrogram and logits.
    """
    from avex import load_model

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model("perch2", device="cpu").eval()
    clips = torch.from_numpy(golden_clips())
    with torch.no_grad():
        out = model.backbone(model._prepare_waveform(clips))
    return {
        "embedding": out.embedding.numpy(),
        "spatial_embedding_mean": out.spatial_embedding.mean(dim=(1, 2)).numpy(),
        "spectrogram_subsampled": out.spectrogram[:, ::25, ::8].numpy(),
        "label": out.label.numpy(),
    }


def test_matches_golden_fixtures(outputs: dict[str, np.ndarray]) -> None:
    """Embeddings and logits match ONNX Runtime within 1e-4; top-10 species agree."""
    golden = np.load(FIXTURE)
    assert tuple(golden["clip_names"]) == CLIP_NAMES
    np.testing.assert_allclose(outputs["embedding"], golden["embedding"], atol=1e-4, rtol=0)
    np.testing.assert_allclose(outputs["spatial_embedding_mean"], golden["spatial_embedding_mean"], atol=1e-4, rtol=0)
    # log(max(x, 1e-5)) amplifies float32 FFT rounding just above the floor.
    np.testing.assert_allclose(outputs["spectrogram_subsampled"], golden["spectrogram_subsampled"], atol=1e-3, rtol=0)
    top = golden["top_label_indices"]
    np.testing.assert_allclose(
        np.take_along_axis(outputs["label"], top, axis=1), golden["top_label_logits"], atol=1e-4, rtol=0
    )
    for clip in range(len(CLIP_NAMES) - 1):  # silence has near-tied logits
        assert int(outputs["label"][clip].argmax()) == int(top[clip, 0]), CLIP_NAMES[clip]


def test_live_layerwise_parity() -> None:
    """Layer-by-layer comparison with ONNX Runtime (needs onnxruntime and the ONNX file)."""
    pytest.importorskip("onnxruntime")
    spec = importlib.util.spec_from_file_location("perch2_parity", ROOT / "scripts" / "perch2_parity.py")
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)

    import onnx

    from avex.models.perch2_net import weights
    from avex.models.perch2_net.convert import convert_onnx_model
    from avex.models.perch2_net.network import Perch2Config, Perch2Network

    torch.backends.cuda.matmul.allow_tf32 = False
    onnx_path = weights.onnx_in_hf_cache()
    config = Perch2Config()
    net = Perch2Network(config).eval()
    state = convert_onnx_model(onnx.load(str(onnx_path)), config)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
    clips = golden_clips()
    reference = parity.run_onnx(onnx_path, clips, config)
    rows = parity.compare(reference, parity.run_torch_local(net, clips, reference), parity.run_torch(net, clips))
    failed = [(name, local_rel, cum_rel) for name, _, local_rel, cum_rel, ok in rows if not ok]
    assert not failed, failed
