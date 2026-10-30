"""Convert the Perch v2 ONNX export into a :class:`Perch2Network` state dict.

The ONNX graph is not executed. Weights are located by graph connectivity and
by the JAX scope names that the export kept on non-conv nodes
(``.../backbone/MBConv_5/DepthwiseBatchNorm/mul1`` etc.), and every
architectural assumption of :mod:`avex.models.perch2_net.network` is checked
against the graph, so a different export fails loudly instead of loading
wrong weights.

Batch norm appears in the graph as a per-channel ``Mul`` and ``Add`` after
each convolution; it is folded into the convolution here (in float64).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from avex.models.perch2_net.network import Perch2Config, tf_same_padding


class Perch2ConversionError(ValueError):
    """The ONNX graph does not match the expected Perch v2 structure."""


class _Graph:
    """Read-only view of an ONNX graph with producer/consumer lookups."""

    def __init__(self, model: Any) -> None:  # noqa: ANN401
        from onnx import numpy_helper

        self.nodes = list(model.graph.node)
        self.inits: dict[str, np.ndarray] = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        for node in self.nodes:
            if node.op_type == "Constant":
                self.inits[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
        self.consumers: dict[str, list[Any]] = defaultdict(list)
        for node in self.nodes:
            for name in node.input:
                self.consumers[name].append(node)
        self.by_name = {node.name: node for node in self.nodes}

    def attrs(self, node: Any) -> dict[str, Any]:  # noqa: ANN401
        """Return a node's attributes as Python values.

        Returns
        -------
        dict[str, Any]
            Attribute name to value.
        """
        from onnx import helper

        return {a.name: helper.get_attribute_value(a) for a in node.attribute}

    def consumer(self, tensor: str, op_type: str) -> Any:  # noqa: ANN401
        """Return the single node of ``op_type`` consuming ``tensor``.

        Returns
        -------
        Any
            The consuming ONNX node.

        Raises
        ------
        Perch2ConversionError
            If there is not exactly one such consumer.
        """
        found = [n for n in self.consumers.get(tensor, []) if n.op_type == op_type]
        if len(found) != 1:
            raise Perch2ConversionError(f"Expected one {op_type} consuming {tensor!r}, found {len(found)}")
        return found[0]

    def const_input(self, node: Any) -> np.ndarray:  # noqa: ANN401
        """Return the single constant (initializer) input of ``node``.

        Returns
        -------
        np.ndarray
            The constant value.

        Raises
        ------
        Perch2ConversionError
            If ``node`` does not have exactly one constant input.
        """
        values = [self.inits[x] for x in node.input if x in self.inits]
        if len(values) != 1:
            raise Perch2ConversionError(f"Node {node.name!r} should have one constant input, has {len(values)}")
        return values[0]

    def find(self, suffix: str) -> Any:  # noqa: ANN401
        """Return the single node whose name ends with ``suffix``.

        Returns
        -------
        Any
            The matching ONNX node.

        Raises
        ------
        Perch2ConversionError
            If there is not exactly one match.
        """
        found = [n for n in self.nodes if n.name.endswith(suffix)]
        if len(found) != 1:
            raise Perch2ConversionError(f"Expected one node named '*{suffix}', found {len(found)}")
        return found[0]


def _check(condition: bool, message: str) -> None:
    """Raise :class:`Perch2ConversionError` unless ``condition`` holds.

    Raises
    ------
    Perch2ConversionError
        If ``condition`` is False.
    """
    if not condition:
        raise Perch2ConversionError(message)


def _close(value: np.ndarray, expected: float) -> bool:
    """Check a scalar graph constant against an expected value.

    Returns
    -------
    bool
        True if ``value`` is a scalar equal to ``expected`` in float32.
    """
    return value.size == 1 and np.float32(value.item()) == np.float32(expected)


def _conv_with_bn(graph: _Graph, conv: Any) -> tuple[str, np.ndarray, np.ndarray]:  # noqa: ANN401
    """Fold the batch norm following ``conv`` into its weight and bias.

    In the graph: ``Conv -> Transpose(NHWC) -> Mul(scale) -> Add(shift)``.

    Returns
    -------
    tuple[str, np.ndarray, np.ndarray]
        The batch-norm scope name (e.g. ``MBConv_5/DepthwiseBatchNorm``),
        the folded weight and the folded bias, both float32.
    """
    transpose = graph.consumer(conv.output[0], "Transpose")
    _check(list(graph.attrs(transpose)["perm"]) == [0, 2, 3, 1], f"Unexpected layout after {conv.name}")
    mul = graph.consumer(transpose.output[0], "Mul")
    add = graph.consumer(mul.output[0], "Add")
    _check(mul.name.endswith("/mul1") and add.name.endswith("/add1"), f"No batch norm after {conv.name}")
    scope = mul.name.split("/backbone/", 1)[-1].rsplit("/", 1)[0]

    weight = graph.inits[conv.input[1]].astype(np.float64)
    bias = graph.inits[conv.input[2]].astype(np.float64) if len(conv.input) > 2 else np.zeros(weight.shape[0])
    scale = graph.const_input(mul).astype(np.float64).reshape(-1)
    shift = graph.const_input(add).astype(np.float64).reshape(-1)
    _check(scale.shape == shift.shape == (weight.shape[0],), f"Batch norm shape mismatch in {scope}")
    folded_w = weight * scale[:, None, None, None]
    folded_b = bias * scale + shift
    return scope, folded_w.astype(np.float32), folded_b.astype(np.float32)


def _dense(graph: _Graph, matmul: Any) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401
    """Return a ``MatMul -> Add`` pair as ``nn.Linear`` weight and bias.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Weight of shape (out, in) and bias of shape (out,).
    """
    weight = graph.const_input(matmul)
    bias = graph.const_input(graph.consumer(matmul.output[0], "Add")).reshape(-1)
    return np.ascontiguousarray(weight.T), bias


def _convert_frontend(graph: _Graph, config: Perch2Config, out: dict[str, np.ndarray]) -> None:
    """Extract the window and mel matrix and check every frontend constant."""
    cfg = config
    framing_shape = (cfg.frame_length, 1, cfg.frame_length, 1)
    framing = [n for n in graph.nodes if n.op_type == "Conv" and graph.inits[n.input[1]].shape == framing_shape]
    _check(len(framing) == 1, f"Expected one framing convolution, found {len(framing)}")
    framing = framing[0]
    a = graph.attrs(framing)
    half = cfg.hop_length // 2
    _check(list(a.get("strides", [])) == [cfg.hop_length, 1], "Unexpected framing hop")
    _check(list(a.get("pads", [])) == [half, 0, half, 0], "Unexpected framing padding")
    kernel = graph.inits[framing.input[1]].reshape(cfg.frame_length, cfg.frame_length)
    _check(np.array_equal(kernel, np.eye(cfg.frame_length, dtype=kernel.dtype)), "Framing kernel is not identity")

    window = graph.const_input(graph.find("frontend/mul2")).reshape(-1)
    _check(window.shape == (cfg.frame_length,), "Unexpected window shape")
    out["frontend.window"] = window.astype(np.float32)

    pad = next((n for n in graph.nodes if n.op_type == "Pad"), None)
    if pad is not None:  # perch_v2.onnx: zero-pad frames to fft_length, then DFT
        pads = graph.inits[pad.input[1]].tolist()
        _check(pads[-1] == cfg.fft_length - cfg.frame_length, "Unexpected FFT zero padding")
    mel = graph.const_input(graph.find("frontend/dot_general5"))
    _check(mel.shape == (cfg.fft_length // 2 + 1, cfg.n_mels), f"Unexpected mel matrix shape {mel.shape}")
    out["frontend.mel_matrix"] = mel.astype(np.float32)

    _check(_close(graph.const_input(graph.find("magnitude_scaling/max2")), cfg.log_floor), "Unexpected log floor")
    scale_node = graph.consumer(graph.find("magnitude_scaling/log").output[0], "Mul")
    _check(_close(graph.const_input(scale_node), cfg.log_scale), "Unexpected log scale")


def _convert_backbone(graph: _Graph, config: Perch2Config, out: dict[str, np.ndarray]) -> None:
    """Fold and map every backbone convolution and squeeze-excitation layer."""
    roles = {"ExpandBatchNorm": "expand", "DepthwiseBatchNorm": "depthwise", "ProjectBatchNorm": "project"}
    seen: set[str] = set()
    convs = [n for n in graph.nodes if n.op_type == "Conv" and len(n.input) > 2]
    for conv in convs:
        scope, weight, bias = _conv_with_bn(graph, conv)
        a = graph.attrs(conv)
        if scope == "Stem_0/BatchNorm_0":
            prefix = "stem.conv"
            _check(list(a.get("strides", [])) == [2, 2] and not any(a.get("pads", [0])), "Stem is not VALID/2")
        elif scope == "Head_0/BatchNorm_0":
            prefix = "head.conv"
        else:
            m = re.fullmatch(r"MBConv_(\d+)/(\w+)", scope)
            _check(m is not None and m.group(2) in roles, f"Unexpected conv scope {scope!r}")
            stage, idx = config.block_index(int(m.group(1)))
            prefix = f"blocks.{stage}.{idx}.{roles[m.group(2)]}.conv"
            if m.group(2) == "DepthwiseBatchNorm":
                _check_depthwise_padding(config, int(m.group(1)), stage, idx, a)
        _check(prefix not in seen, f"Two convolutions map to {prefix}")
        out[f"{prefix}.weight"], out[f"{prefix}.bias"] = weight, bias
        seen.add(prefix)

    for i in range(config.num_blocks):
        stage, idx = config.block_index(i)
        se = f"/backbone/MBConv_{i}/SqueezeAndExcitation_0"
        for part in ("Reduce", "Expand"):
            w, b = _dense(graph, graph.find(f"{se}/{part}/dot_general5"))
            out[f"blocks.{stage}.{idx}.se.{part.lower()}.weight"] = w
            out[f"blocks.{stage}.{idx}.se.{part.lower()}.bias"] = b
    _check("stem.conv" in seen and "head.conv" in seen, "Stem or head convolution missing")


def _check_depthwise_padding(config: Perch2Config, block: int, stage: int, idx: int, attrs: dict) -> None:
    """Check the graph's depthwise padding equals TF "SAME" for the configured input."""
    sizes = _feature_sizes(config)
    h, w = sizes[block]
    st = config.stages[stage]
    stride = st.stride if idx == 0 else 1
    _check(list(attrs.get("strides", [1, 1])) == [stride, stride], f"MBConv_{block}: unexpected stride")
    top, bottom = tf_same_padding(h, st.kernel_size, stride)
    left, right = tf_same_padding(w, st.kernel_size, stride)
    expected = [top, left, bottom, right]
    got = list(attrs.get("pads", [0, 0, 0, 0]))
    _check(got == expected, f"MBConv_{block}: graph padding {got} != TF SAME {expected}")


def _feature_sizes(config: Perch2Config) -> list[tuple[int, int]]:
    """Spatial size entering each MBConv block for the configured input length.

    Returns
    -------
    list[tuple[int, int]]
        (height, width) per global block index.
    """
    frames = config.num_samples // config.hop_length
    h, w = (frames - 3) // 2 + 1, (config.n_mels - 3) // 2 + 1
    sizes = []
    for stage in config.stages:
        for i in range(stage.repeats):
            sizes.append((h, w))
            if i == 0 and stage.stride > 1:
                h, w = _ceil_div(h, stage.stride), _ceil_div(w, stage.stride)
    return sizes


def _ceil_div(a: int, b: int) -> int:
    """Integer ceiling division.

    Returns
    -------
    int
        ``ceil(a / b)``.
    """
    return -(-a // b)


def _convert_label_head(graph: _Graph, config: Perch2Config, out: dict[str, np.ndarray]) -> None:
    """Map the ProtoPNet classifier and check its constants."""
    cfg = config
    prototypes = graph.const_input(graph.find("heads_protopnet_logits/dot_general7"))
    _check(
        prototypes.shape == (cfg.head_channels, cfg.num_classes * cfg.prototypes_per_class),
        f"Unexpected prototype shape {prototypes.shape}",
    )
    reshape = graph.find("heads_protopnet_logits/dot_general10")
    shape = graph.inits[reshape.input[1]].tolist()
    _check(shape[-2:] == [cfg.num_classes, cfg.prototypes_per_class], "Prototypes are not class-major")
    eps = graph.const_input(graph.find("heads_protopnet_logits/add1"))
    _check(_close(eps, cfg.prototype_eps), "Unexpected prototype normalisation epsilon")
    weights = graph.const_input(graph.find("heads_protopnet_logits/mul2"))
    bias = graph.const_input(graph.consumer(graph.find("heads_protopnet_logits/reduce_sum").output[0], "Add"))
    out["label_head.prototypes"] = prototypes.astype(np.float32)
    out["label_head.prototype_weights"] = weights.reshape(cfg.num_classes, cfg.prototypes_per_class).astype(np.float32)
    out["label_head.bias"] = bias.reshape(cfg.num_classes).astype(np.float32)


def convert_onnx_model(model: Any, config: Perch2Config | None = None) -> dict[str, np.ndarray]:  # noqa: ANN401
    """Convert a loaded Perch v2 ``onnx.ModelProto`` into state-dict arrays.

    Returns
    -------
    dict[str, np.ndarray]
        Float32 arrays keyed like :class:`Perch2Network`'s state dict.
    """
    cfg = config or Perch2Config()
    graph = _Graph(model)
    out: dict[str, np.ndarray] = {}
    _convert_frontend(graph, cfg, out)
    _convert_backbone(graph, cfg, out)
    _convert_label_head(graph, cfg, out)
    return {k: np.require(v, dtype=np.float32, requirements=["C", "W"]) for k, v in out.items()}


def convert_onnx(path: str | Path, config: Perch2Config | None = None) -> dict[str, np.ndarray]:
    """Load ``path`` (a Perch v2 ONNX file) and convert it.

    Peak memory is roughly three times the file size (graph, arrays, output).

    Returns
    -------
    dict[str, np.ndarray]
        Float32 arrays keyed like :class:`Perch2Network`'s state dict.
    """
    import onnx

    return convert_onnx_model(onnx.load(str(path)), config)


def expected_shapes(config: Perch2Config | None = None) -> dict[str, tuple[int, ...]]:
    """State-dict key -> shape that a converted file must contain exactly.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Shapes of every parameter and persistent buffer.
    """
    import torch

    from avex.models.perch2_net.network import Perch2Network

    with torch.device("meta"):
        net = Perch2Network(config)
    return {k: tuple(v.shape) for k, v in net.state_dict().items()}


def missing_or_unexpected(keys: Iterable[str], config: Perch2Config | None = None) -> tuple[set[str], set[str]]:
    """Compare ``keys`` with the expected state-dict keys.

    Returns
    -------
    tuple[set[str], set[str]]
        Missing keys and unexpected keys.
    """
    expected = set(expected_shapes(config))
    keys = set(keys)
    return expected - keys, keys - expected
