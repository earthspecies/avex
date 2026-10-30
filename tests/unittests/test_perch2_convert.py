"""Perch v2 ONNX -> PyTorch conversion, tested on a tiny synthetic graph.

The synthetic graph mirrors the node names and wiring of the real export
(``Conv -> Transpose -> Mul/Add`` batch norm, scoped SE dense layers,
ProtoPNet head) for a two-block network. It is never executed: the converter
only reads names, connectivity, attributes and initializers, so no download
or onnxruntime is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

onnx = pytest.importorskip("onnx")
from onnx import helper, numpy_helper  # noqa: E402

from avex.models.perch2_net.convert import (  # noqa: E402
    Perch2ConversionError,
    convert_onnx_model,
    missing_or_unexpected,
)
from avex.models.perch2_net.network import (  # noqa: E402
    Perch2Config,
    Perch2Network,
    StageConfig,
    tf_same_padding,
)

TINY = Perch2Config(
    num_samples=64,
    frame_length=8,
    hop_length=4,
    fft_length=16,
    n_mels=10,
    stem_channels=8,
    stages=(StageConfig(3, 1, 1, 8, 8, 1), StageConfig(5, 2, 2, 8, 16, 1)),
    head_channels=12,
    num_classes=3,
    prototypes_per_class=2,
)
EM = "jit(infer_fn)/MultiHeadClassifier/embedding_model"
HEAD = "jit(infer_fn)/MultiHeadClassifier/MultiHeadClassifier._call_model/heads_protopnet_logits"


class _Builder:
    """Accumulates nodes and initializers for a synthetic Perch-like graph."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.nodes: list = []
        self.inits: list = []
        self.raw: dict[str, np.ndarray] = {}
        self._n = 0

    def init(self, name: str, value: np.ndarray) -> str:
        """Add an initializer and return its name.

        Returns
        -------
        str
            The initializer name.
        """
        self.inits.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
        return name

    def tensor(self) -> str:
        """Fresh intermediate tensor name.

        Returns
        -------
        str
            A unique tensor name.
        """
        self._n += 1
        return f"t{self._n}"

    def node(self, op: str, name: str, inputs: list[str], **attrs: object) -> str:
        """Append a node and return its output name.

        Returns
        -------
        str
            The node's output tensor name.
        """
        out = self.tensor()
        self.nodes.append(helper.make_node(op, inputs, [out], name=name, **attrs))
        return out

    def conv_bn(
        self, x: str, scope: str, cin: int, cout: int, k: int, stride: int, groups: int, pads: list | None
    ) -> str:
        """Conv with bias followed by the export's NHWC batch-norm Mul/Add.

        Returns
        -------
        str
            Output tensor name (NHWC).
        """
        w = self.rng.normal(size=(cout, cin // groups, k, k)).astype(np.float32)
        b = self.rng.normal(size=cout).astype(np.float32)
        s = self.rng.uniform(0.5, 1.5, size=cout).astype(np.float32)
        t = self.rng.normal(size=cout).astype(np.float32)
        self.raw[scope] = (w, b, s, t)
        attrs: dict = {"kernel_shape": [k, k], "strides": [stride, stride], "group": groups}
        if pads is not None:
            attrs["pads"] = pads
        inputs = [x, self.init(f"{scope}/w", w), self.init(f"{scope}/b", b)]
        y = self.node("Conv", f"node_Conv_{len(self.nodes)}", inputs, **attrs)
        y = self.node("Transpose", f"Transpose_{len(self.nodes)}", [y], perm=[0, 2, 3, 1])
        y = self.node("Mul", f"{EM}/backbone/{scope}/mul1", [y, self.init(f"{scope}/s", s.reshape(1, 1, 1, -1))])
        return self.node("Add", f"{EM}/backbone/{scope}/add1", [y, self.init(f"{scope}/t", t.reshape(1, 1, 1, -1))])

    def dense(self, x: str, scope: str, add_name: str, cin: int, cout: int) -> str:
        """SE dense layer as MatMul + Add.

        Returns
        -------
        str
            Output tensor name.
        """
        w = self.rng.normal(size=(cin, cout)).astype(np.float32)
        b = self.rng.normal(size=(1, cout)).astype(np.float32)
        self.raw[scope] = (w, b)
        y = self.node("MatMul", f"{EM}/backbone/{scope}/dot_general5", [x, self.init(f"{scope}/w", w)])
        return self.node("Add", f"{EM}/backbone/{scope}/{add_name}", [y, self.init(f"{scope}/b", b)])

    def model(self) -> object:
        """Assemble the ONNX model.

        Returns
        -------
        object
            An ``onnx.ModelProto``.
        """
        graph = helper.make_graph(
            self.nodes,
            "synthetic_perch2",
            [helper.make_tensor_value_info("inputs", onnx.TensorProto.FLOAT, [None, TINY.num_samples])],
            [helper.make_tensor_value_info(self.nodes[-1].output[0], onnx.TensorProto.FLOAT, None)],
            initializer=self.inits,
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def build_synthetic(
    framing_kernel: np.ndarray | None = None,
    depthwise_pads: list | None = None,
    log_floor: float = 1e-5,
) -> tuple[object, _Builder]:
    """Build a tiny Perch-like ONNX graph for ``TINY``.

    Returns
    -------
    tuple[object, _Builder]
        The model and the builder holding the raw (unfolded) parameters.
    """
    cfg, b = TINY, _Builder()
    fl, half = cfg.frame_length, cfg.hop_length // 2
    kernel = np.eye(fl, dtype=np.float32) if framing_kernel is None else framing_kernel
    x = b.node("Conv", "node_Conv_0", ["inputs", b.init("framing", kernel.reshape(fl, 1, fl, 1))],
               kernel_shape=[fl, 1], strides=[cfg.hop_length, 1], pads=[half, 0, half, 0])  # fmt: skip
    b.raw["window"] = b.rng.uniform(size=(1, 1, fl)).astype(np.float32)
    x = b.node("Mul", f"{EM}/frontend/mul2", [b.init("window", b.raw["window"]), x])
    x = b.node("Pad", "pad", [x, b.init("pads", np.array([0, 0, 0, 0, 0, cfg.fft_length - fl]))])
    b.raw["mel"] = b.rng.uniform(size=(cfg.fft_length // 2 + 1, cfg.n_mels)).astype(np.float32)
    x = b.node("MatMul", f"{EM}/frontend/dot_general5", [x, b.init("mel", b.raw["mel"])])
    x = b.node("Max", f"{EM}/magnitude_scaling/max2", [x, b.init("floor", np.array(log_floor))])
    x = b.node("Log", f"{EM}/magnitude_scaling/log", [x])
    x = b.node("Mul", "StatefulPartitionedCall:3", [b.init("scale", np.array(cfg.log_scale)), x])

    x = b.conv_bn(x, "Stem_0/BatchNorm_0", 1, cfg.stem_channels, 3, 2, 1, None)
    # Block 0: expand_ratio 1 (no expand conv), stride 1, residual. Feature map 7x4.
    x = b.conv_bn(x, "MBConv_0/DepthwiseBatchNorm", 8, 8, 3, 1, 8, [1, 1, 1, 1])
    b.dense(x, "MBConv_0/SqueezeAndExcitation_0/Reduce", "add2", 8, 2)
    b.dense(x, "MBConv_0/SqueezeAndExcitation_0/Expand", "add1", 2, 8)
    x = b.conv_bn(x, "MBConv_0/ProjectBatchNorm", 8, 8, 1, 1, 1, None)
    # Block 1: expand 8 -> 16, 5x5 stride 2 on a 7x4 map: TF SAME pads width (1, 2).
    x = b.conv_bn(x, "MBConv_1/ExpandBatchNorm", 8, 16, 1, 1, 1, None)
    top, bottom = tf_same_padding(7, 5, 2)
    left, right = tf_same_padding(4, 5, 2)
    assert (left, right) == (1, 2)
    pads = depthwise_pads if depthwise_pads is not None else [top, left, bottom, right]
    x = b.conv_bn(x, "MBConv_1/DepthwiseBatchNorm", 16, 16, 5, 2, 16, pads)
    b.dense(x, "MBConv_1/SqueezeAndExcitation_0/Reduce", "add2", 16, 2)
    b.dense(x, "MBConv_1/SqueezeAndExcitation_0/Expand", "add1", 2, 16)
    x = b.conv_bn(x, "MBConv_1/ProjectBatchNorm", 16, 16, 1, 1, 1, None)
    x = b.conv_bn(x, "Head_0/BatchNorm_0", 16, cfg.head_channels, 1, 1, 1, None)

    n_proto = cfg.num_classes * cfg.prototypes_per_class
    b.raw["prototypes"] = b.rng.normal(size=(cfg.head_channels, n_proto)).astype(np.float32)
    b.raw["proto_w"] = b.rng.normal(size=(1, cfg.num_classes, cfg.prototypes_per_class)).astype(np.float32)
    b.raw["proto_b"] = b.rng.normal(size=(1, cfg.num_classes)).astype(np.float32)
    eps = b.node("Add", f"{HEAD}/add1", [x, b.init("eps", np.array(cfg.prototype_eps))])
    y = b.node("MatMul", f"{HEAD}/dot_general7", [eps, b.init("prototypes", b.raw["prototypes"])])
    shape = np.array([-1, 4, 1, cfg.num_classes, cfg.prototypes_per_class])
    y = b.node("Reshape", f"{HEAD}/dot_general10", [y, b.init("shape", shape)])
    y = b.node("Mul", f"{HEAD}/mul2", [y, b.init("proto_w", b.raw["proto_w"])])
    y = b.node("ReduceSum", f"{HEAD}/reduce_sum", [y])
    b.node("Add", "StatefulPartitionedCall:1", [y, b.init("proto_b", b.raw["proto_b"])])
    return b.model(), b


def test_conversion_matches_network_keys_and_loads_strict() -> None:
    """Every parameter and buffer is produced exactly once and loads strictly."""
    model, _ = build_synthetic()
    state = convert_onnx_model(model, TINY)
    assert missing_or_unexpected(state, TINY) == (set(), set())
    net = Perch2Network(TINY)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)


def test_batch_norm_is_folded_into_convolution() -> None:
    """conv(x, W', b') equals (conv(x, W) + b) * scale + shift."""
    model, b = build_synthetic()
    state = convert_onnx_model(model, TINY)
    w, bias, s, t = b.raw["MBConv_1/ExpandBatchNorm"]
    x = torch.randn(2, 8, 5, 3)
    expected = (F.conv2d(x, torch.from_numpy(w), torch.from_numpy(bias)) * torch.from_numpy(s)[:, None, None]
                + torch.from_numpy(t)[:, None, None])  # fmt: skip
    folded = F.conv2d(
        x,
        torch.from_numpy(state["blocks.1.0.expand.conv.weight"]),
        torch.from_numpy(state["blocks.1.0.expand.conv.bias"]),
    )
    torch.testing.assert_close(folded, expected, rtol=1e-5, atol=1e-5)


def test_dense_frontend_and_head_mapping() -> None:
    """SE layers are transposed into nn.Linear layout; frontend and head are copied."""
    model, b = build_synthetic()
    state = convert_onnx_model(model, TINY)
    w, bias = b.raw["MBConv_0/SqueezeAndExcitation_0/Reduce"]
    np.testing.assert_array_equal(state["blocks.0.0.se.reduce.weight"], w.T)
    np.testing.assert_array_equal(state["blocks.0.0.se.reduce.bias"], bias.reshape(-1))
    np.testing.assert_array_equal(state["frontend.window"], b.raw["window"].reshape(-1))
    np.testing.assert_array_equal(state["frontend.mel_matrix"], b.raw["mel"])
    np.testing.assert_array_equal(state["label_head.prototypes"], b.raw["prototypes"])
    np.testing.assert_array_equal(state["label_head.prototype_weights"], b.raw["proto_w"][0])
    np.testing.assert_array_equal(state["label_head.bias"], b.raw["proto_b"][0])
    assert "blocks.0.0.expand.conv.weight" not in state  # expand_ratio 1 has no expand conv


def test_rejects_non_identity_framing() -> None:
    """A framing kernel that is not the identity would silently change the frontend."""
    kernel = np.eye(TINY.frame_length, dtype=np.float32)
    kernel[0, 1] = 0.5
    model, _ = build_synthetic(framing_kernel=kernel)
    with pytest.raises(Perch2ConversionError, match="not identity"):
        convert_onnx_model(model, TINY)


def test_rejects_padding_that_is_not_tf_same() -> None:
    """Symmetric padding where TF SAME is asymmetric is the classic EfficientNet-port bug."""
    model, _ = build_synthetic(depthwise_pads=[2, 2, 2, 2])
    with pytest.raises(Perch2ConversionError, match="TF SAME"):
        convert_onnx_model(model, TINY)


def test_rejects_unexpected_frontend_constant() -> None:
    """Changed log floor is caught."""
    model, _ = build_synthetic(log_floor=1e-6)
    with pytest.raises(Perch2ConversionError, match="log floor"):
        convert_onnx_model(model, TINY)


@pytest.mark.parametrize(
    "size, kernel, stride, expected",
    [
        (249, 3, 2, (1, 1)),  # MBConv_2 time axis
        (32, 5, 2, (1, 2)),  # MBConv_5 frequency axis (asymmetric)
        (16, 3, 2, (0, 1)),  # MBConv_8 frequency axis (asymmetric)
        (32, 5, 2, (1, 2)),  # MBConv_18 time axis (asymmetric)
        (63, 5, 1, (2, 2)),  # stride 1
    ],
)
def test_tf_same_padding_matches_perch_graph(size: int, kernel: int, stride: int, expected: tuple[int, int]) -> None:
    """Padding values read from the real Perch v2 graph."""
    assert tf_same_padding(size, kernel, stride) == expected
