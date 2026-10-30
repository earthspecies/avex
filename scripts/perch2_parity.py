"""Layer-by-layer parity between the Perch v2 ONNX export and the PyTorch port.

Exposes intermediate ONNX tensors (spectrogram, stem, every MBConv block,
head) as graph outputs, runs ONNX Runtime and :class:`Perch2Network` in fp32
(TF32 off) on deterministic clips, and reports the error per layer, so a
broken mapping points at the first layer that diverges.

Usage::

    uv run python scripts/perch2_parity.py                   # report, exit 1 on failure
    uv run python scripts/perch2_parity.py --write-fixtures  # also refresh test fixtures

``--write-fixtures`` writes ``tests/fixtures/perch2/golden.npz`` (ONNX Runtime
outputs on the golden clips, used by CI without onnxruntime) and prints the
self-check constants for ``avex/models/perch2_net/golden.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from avex.models.perch2_net import weights as perch2_weights
from avex.models.perch2_net.convert import convert_onnx_model
from avex.models.perch2_net.golden import CLIP_NAMES, golden_clips
from avex.models.perch2_net.network import Perch2Config, Perch2Network

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "perch2" / "golden.npz"

# Relative tolerance (max abs error / max abs reference) per layer kind. The
# spectrogram is looser because log(max(x, 1e-5)) amplifies float32 FFT
# rounding for mel bins just above the floor; FFT magnitudes agree to ~3e-8.
TOLERANCES = {"spectrogram": 1e-3, "default": 1e-5}
# End-to-end: intermediate feature maps inherit the spectrogram's rounding;
# final outputs must still be within 1e-5.
CUMULATIVE_TOLERANCES = {"default": 1e-3, "embedding": 1e-5, "label": 1e-5}


def _intermediate_names(model: object, config: Perch2Config) -> dict[str, str]:  # noqa: ANN001
    """Map torch layer names to ONNX tensor names (NHWC) for intermediates.

    Returns
    -------
    dict[str, str]
        e.g. ``{"stem": "mul_C_0", "blocks.0.0": "...", ...}``.

    Raises
    ------
    RuntimeError
        If an intermediate tensor cannot be located.
    """
    nodes = list(model.graph.node)
    consumers: dict[str, list] = {}
    for n in nodes:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def output_of(suffix: str) -> str:
        found = [n for n in nodes if n.name.endswith(suffix)]
        if len(found) != 1:
            raise RuntimeError(f"Expected one node '*{suffix}', found {len(found)}")
        return found[0].output[0]

    names = {"spectrogram": "spectrogram"}
    stem_bn = output_of("/backbone/Stem_0/BatchNorm_0/add1")
    stem_act = [n for n in consumers[stem_bn] if n.op_type == "Mul"]
    names["stem"] = stem_act[0].output[0]
    previous = names["stem"]
    for i in range(config.num_blocks):
        stage, idx = config.block_index(i)
        st = config.stages[stage]
        in_ch = st.in_channels if idx == 0 else st.out_channels
        is_residual = (st.stride if idx == 0 else 1) == 1 and in_ch == st.out_channels
        proj = output_of(f"/backbone/MBConv_{i}/ProjectBatchNorm/add1")
        if is_residual:
            # The shortcut Add consumes this block's projection and its input.
            adds = [n for n in consumers.get(proj, []) if n.op_type == "Add" and previous in n.input]
            if len(adds) != 1:
                raise RuntimeError(f"MBConv_{i}: expected one residual Add, found {len(adds)}")
            proj = adds[0].output[0]
        names[f"blocks.{stage}.{idx}"] = proj
        previous = proj
    names["head"] = "spatial_embedding"
    names["embedding"] = "embedding"
    names["label"] = "label"
    return names


def run_onnx(onnx_path: Path, clips: np.ndarray, config: Perch2Config) -> dict[str, np.ndarray]:
    """Run ONNX Runtime with intermediate outputs exposed.

    Returns
    -------
    dict[str, np.ndarray]
        Torch layer name -> ONNX output (NHWC for feature maps).
    """
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    names = _intermediate_names(model, config)
    existing = {o.name for o in model.graph.output}
    for tensor in names.values():
        if tensor not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(tensor, onnx.TensorProto.FLOAT, None))
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    outputs = session.run(list(names.values()), {"inputs": clips})
    return dict(zip(names.keys(), outputs, strict=True))


def run_torch(net: Perch2Network, clips: np.ndarray) -> dict[str, np.ndarray]:
    """Run the PyTorch port with hooks on stem, blocks and head.

    Returns
    -------
    dict[str, np.ndarray]
        Layer name -> output, feature maps permuted to NHWC like ONNX.
    """
    captured: dict[str, torch.Tensor] = {}
    handles = []
    modules = {"stem": net.stem, "head": net.head}
    for s, stage in enumerate(net.blocks):
        for i, block in enumerate(stage):
            modules[f"blocks.{s}.{i}"] = block
    for name, module in modules.items():
        handles.append(module.register_forward_hook(lambda _m, _i, out, name=name: captured.__setitem__(name, out)))
    try:
        with torch.no_grad():
            out = net(torch.from_numpy(clips))
    finally:
        for h in handles:
            h.remove()
    result = {k: v.permute(0, 2, 3, 1).numpy() for k, v in captured.items()}
    result.update(spectrogram=out.spectrogram.numpy(), embedding=out.embedding.numpy(), label=out.label.numpy())
    return result


def run_torch_local(net: Perch2Network, clips: np.ndarray, reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run each torch layer on ONNX Runtime's input for that layer.

    Isolates per-layer mapping errors from error accumulated upstream.

    Returns
    -------
    dict[str, np.ndarray]
        Layer name -> output, feature maps in NHWC like ONNX.
    """

    def nchw(name: str) -> torch.Tensor:
        return torch.from_numpy(reference[name]).permute(0, 3, 1, 2)

    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        out["spectrogram"] = net.frontend(torch.from_numpy(clips)).numpy()
        out["stem"] = net.stem(torch.from_numpy(reference["spectrogram"]).unsqueeze(1)).permute(0, 2, 3, 1).numpy()
        previous = "stem"
        for s, stage in enumerate(net.blocks):
            for i, block in enumerate(stage):
                name = f"blocks.{s}.{i}"
                out[name] = block(nchw(previous)).permute(0, 2, 3, 1).numpy()
                previous = name
        out["head"] = net.head(nchw(previous)).permute(0, 2, 3, 1).numpy()
        out["embedding"] = net.pool(nchw("head")).numpy()
        out["label"] = net.label_head(torch.from_numpy(reference["head"])).numpy()
    return out


def _rel_error(ref: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    """Max absolute error and the same relative to the reference's max magnitude.

    Returns
    -------
    tuple[float, float]
        (max abs error, relative error).
    """
    err = float(np.abs(ref - got).max())
    return err, err / max(float(np.abs(ref).max()), 1e-12)


def compare(
    reference: dict[str, np.ndarray], local: dict[str, np.ndarray], cumulative: dict[str, np.ndarray]
) -> list[tuple[str, float, float, float, bool]]:
    """Compare layer outputs in graph order.

    A layer passes if its local error (layer run on ONNX's own input) is
    within ``TOLERANCES`` and its cumulative error is within
    ``CUMULATIVE_TOLERANCES``.

    Returns
    -------
    list[tuple[str, float, float, float, bool]]
        (layer, local abs, local rel, cumulative rel, passed) per layer.
    """
    rows = []
    for name, ref in reference.items():
        local_abs, local_rel = _rel_error(ref, local[name])
        _, cum_rel = _rel_error(ref, cumulative[name])
        ok = local_rel <= TOLERANCES.get(name, TOLERANCES["default"]) and cum_rel <= CUMULATIVE_TOLERANCES.get(
            name, CUMULATIVE_TOLERANCES["default"]
        )
        rows.append((name, local_abs, local_rel, cum_rel, ok))
    return rows


def main() -> int:
    """Run the parity check.

    Returns
    -------
    int
        Process exit code: 0 if every layer is within tolerance.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onnx", type=Path, help="Pre-downloaded perch_v2.onnx (checksum-verified)")
    parser.add_argument("--write-fixtures", action="store_true", help="Refresh tests/fixtures/perch2/golden.npz")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(max(1, torch.get_num_threads()))

    import onnx

    config = Perch2Config()
    onnx_path = args.onnx or perch2_weights.onnx_in_hf_cache()
    perch2_weights.verify_onnx(onnx_path)
    net = Perch2Network(config).eval()
    state = convert_onnx_model(onnx.load(str(onnx_path)), config)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)

    clips = golden_clips()
    reference = run_onnx(onnx_path, clips, config)
    rows = compare(reference, run_torch_local(net, clips, reference), run_torch(net, clips))
    print(f"{'layer':16s} {'local abs':>10s} {'local rel':>10s} {'cumul. rel':>10s}")
    for name, local_abs, local_rel, cum_rel, ok in rows:
        print(f"{name:16s} {local_abs:10.3g} {local_rel:10.3g} {cum_rel:10.3g}{'' if ok else '  <-- FAIL'}")
    failed = [r[0] for r in rows if not r[4]]

    if args.write_fixtures:
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        label = reference["label"]
        top = np.argsort(-label, axis=1)[:, :10]
        np.savez_compressed(
            FIXTURE_PATH,
            clip_names=np.array(CLIP_NAMES),
            embedding=reference["embedding"],
            spatial_embedding_mean=reference["head"].mean(axis=(1, 2)),
            spectrogram_subsampled=reference["spectrogram"][:, ::25, ::8],
            top_label_indices=top,
            top_label_logits=np.take_along_axis(label, top, axis=1),
        )
        emb = reference["embedding"][0]
        print(f"\nWrote {FIXTURE_PATH}\nSelf-check constants for golden.py (clip 0):")
        print(f"SELF_CHECK_EMBEDDING_HEAD = {tuple(round(float(v), 6) for v in emb[:8])}")
        print(f"SELF_CHECK_EMBEDDING_NORM = {float(np.linalg.norm(emb)):.6f}")
        print(f"SELF_CHECK_TOP_LABEL = {int(label[0].argmax())}")
        print(f"SELF_CHECK_TOP_LOGIT = {float(label[0].max()):.6f}")

    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nAll layers within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
