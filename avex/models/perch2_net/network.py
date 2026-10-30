"""PyTorch implementation of the Google Perch v2 bird-vocalization classifier.

A hand-written port of the architecture in the Perch v2 ONNX export
(``justinchuby/Perch-onnx``, itself converted from Google's Kaggle release):

* ``frontend`` - framing (640-sample Hann-style window, hop 320), 1024-point
  real FFT magnitude, 128-bin mel projection and ``0.1 * log(max(x, 1e-5))``.
* ``stem`` / ``blocks.{stage}.{idx}`` / ``head`` - an EfficientNet-B3 backbone
  (MBConv blocks with squeeze-and-excitation and SiLU).
* ``pool`` - global average over time and frequency (the 1536-d embedding).
* ``label_head`` - the ProtoPNet classifier (4 prototypes per class).

Batch normalisation is folded into the preceding convolutions when the
weights are converted, so there are no ``BatchNorm`` modules: convs carry
biases, and train/eval mode does not change the backbone's output.

Strided depthwise convolutions use TensorFlow "SAME" padding, which is
asymmetric for some input sizes; ``Conv2dSame`` reproduces it exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StageConfig:
    """One EfficientNet stage: ``repeats`` MBConv blocks sharing kernel and widths."""

    kernel_size: int
    stride: int
    expand_ratio: int
    in_channels: int
    out_channels: int
    repeats: int


# EfficientNet-B3 stages as found in the Perch v2 graph (MBConv_0 .. MBConv_25).
_PERCH2_STAGES = (
    StageConfig(3, 1, 1, 40, 24, 2),
    StageConfig(3, 2, 6, 24, 32, 3),
    StageConfig(5, 2, 6, 32, 48, 3),
    StageConfig(3, 2, 6, 48, 96, 5),
    StageConfig(5, 1, 6, 96, 136, 5),
    StageConfig(5, 2, 6, 136, 232, 6),
    StageConfig(3, 1, 6, 232, 384, 2),
)


@dataclass(frozen=True)
class Perch2Config:
    """Architecture hyper-parameters. Defaults reproduce Perch v2."""

    sample_rate: int = 32_000
    num_samples: int = 160_000
    frame_length: int = 640
    hop_length: int = 320
    fft_length: int = 1024
    n_mels: int = 128
    log_floor: float = 1e-5
    log_scale: float = 0.1
    stem_channels: int = 40
    stages: tuple[StageConfig, ...] = field(default=_PERCH2_STAGES)
    se_ratio: float = 0.25
    head_channels: int = 1536
    num_classes: int = 14_795
    prototypes_per_class: int = 4
    prototype_eps: float = 1e-5

    @property
    def num_blocks(self) -> int:
        """Total number of MBConv blocks.

        Returns
        -------
        int
            Sum of ``repeats`` over all stages.
        """
        return sum(s.repeats for s in self.stages)

    def block_index(self, global_index: int) -> tuple[int, int]:
        """Map a global MBConv index (as in ``MBConv_{i}``) to ``(stage, idx)``.

        Returns
        -------
        tuple[int, int]
            Stage index and position within the stage.

        Raises
        ------
        IndexError
            If ``global_index`` is out of range.
        """
        remaining = global_index
        for stage, cfg in enumerate(self.stages):
            if remaining < cfg.repeats:
                return stage, remaining
            remaining -= cfg.repeats
        raise IndexError(f"MBConv index {global_index} out of range for {self.num_blocks} blocks")


def tf_same_padding(size: int, kernel_size: int, stride: int) -> tuple[int, int]:
    """TensorFlow "SAME" padding (before, after) for one spatial dimension.

    Returns
    -------
    tuple[int, int]
        Padding before and after; the extra pixel, if any, goes after.
    """
    total = max((math.ceil(size / stride) - 1) * stride + kernel_size - size, 0)
    return total // 2, total - total // 2


class Conv2dSame(nn.Conv2d):
    """``nn.Conv2d`` with TensorFlow "SAME" padding computed from the input size."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad like TensorFlow, then convolve without padding.

        Returns
        -------
        torch.Tensor
            Convolution output.
        """
        top, bottom = tf_same_padding(x.shape[-2], self.kernel_size[0], self.stride[0])
        left, right = tf_same_padding(x.shape[-1], self.kernel_size[1], self.stride[1])
        if top or bottom or left or right:
            x = F.pad(x, (left, right, top, bottom))
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class ConvAct(nn.Module):
    """Convolution (batch norm folded in) followed by an optional SiLU."""

    def __init__(self, conv: nn.Conv2d, activation: bool = True) -> None:
        super().__init__()
        self.conv = conv
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution and activation.

        Returns
        -------
        torch.Tensor
            Activated feature map.
        """
        return self.act(self.conv(x))


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation: spatial mean -> reduce -> SiLU -> expand -> sigmoid gate."""

    def __init__(self, channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Linear(channels, squeeze_channels)
        self.expand = nn.Linear(squeeze_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate the channels of ``x`` (B, C, H, W).

        Returns
        -------
        torch.Tensor
            ``x`` scaled per channel.
        """
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.expand(F.silu(self.reduce(s))))
        return x * s[:, :, None, None]


class MBConv(nn.Module):
    """Mobile inverted bottleneck block (EfficientNet), batch norm folded."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
    ) -> None:
        super().__init__()
        mid = in_channels * expand_ratio
        self.expand = ConvAct(nn.Conv2d(in_channels, mid, 1, bias=True)) if expand_ratio != 1 else None
        self.depthwise = ConvAct(Conv2dSame(mid, mid, kernel_size, stride=stride, groups=mid, bias=True))
        self.se = SqueezeExcite(mid, max(1, int(in_channels * se_ratio)))
        self.project = ConvAct(nn.Conv2d(mid, out_channels, 1, bias=True), activation=False)
        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block.

        Returns
        -------
        torch.Tensor
            Block output, with the identity shortcut when shapes allow.
        """
        y = self.expand(x) if self.expand is not None else x
        y = self.project(self.se(self.depthwise(y)))
        return y + x if self.residual else y


class Frontend(nn.Module):
    """Waveform -> log-mel spectrogram, exactly as in the Perch v2 graph.

    The window and mel matrix are buffers loaded with the weights, so they
    follow ``.to(device)``. The frontend has no trainable parameters, but it
    is differentiable with respect to its input.
    """

    def __init__(self, config: Perch2Config) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("window", torch.zeros(config.frame_length))
        self.register_buffer("mel_matrix", torch.zeros(config.fft_length // 2 + 1, config.n_mels))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute the spectrogram of ``wav`` (B, N).

        Returns
        -------
        torch.Tensor
            Log-mel spectrogram of shape (B, frames, n_mels).
        """
        cfg = self.config
        half = cfg.hop_length // 2
        frames = F.pad(wav, (half, half)).unfold(-1, cfg.frame_length, cfg.hop_length)
        frames = frames * self.window
        magnitude = torch.fft.rfft(frames, n=cfg.fft_length).abs()
        mel = magnitude @ self.mel_matrix
        return cfg.log_scale * torch.log(torch.clamp(mel, min=cfg.log_floor))


class GlobalAvgPool(nn.Module):
    """Mean over time and frequency: (B, C, T, F) -> (B, C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool ``x``.

        Returns
        -------
        torch.Tensor
            Pooled tensor of shape (B, C).
        """
        return x.mean(dim=(2, 3))


class ProtoPNetHead(nn.Module):
    """Perch v2's prototype classifier.

    Each spatial embedding is L2-normalised and compared with
    ``prototypes_per_class`` prototypes per class; the best match over time and
    frequency is taken per prototype, and prototypes are combined with learned
    weights.
    """

    def __init__(self, config: Perch2Config) -> None:
        super().__init__()
        self.config = config
        n_proto = config.num_classes * config.prototypes_per_class
        self.prototypes = nn.Parameter(torch.zeros(config.head_channels, n_proto))
        self.prototype_weights = nn.Parameter(torch.zeros(config.num_classes, config.prototypes_per_class))
        self.bias = nn.Parameter(torch.zeros(config.num_classes))

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        """Score ``spatial`` (B, T, F, C) embeddings.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        cfg = self.config
        normed = spatial / (spatial.pow(2).sum(dim=-1, keepdim=True).sqrt() + cfg.prototype_eps)
        sims = normed @ self.prototypes
        sims = sims.reshape(*sims.shape[:3], cfg.num_classes, cfg.prototypes_per_class)
        best = sims.amax(dim=(1, 2))
        return (best * self.prototype_weights).sum(dim=-1) + self.bias


class Perch2Outputs(NamedTuple):
    """Outputs of :class:`Perch2Network`, named as in the ONNX export."""

    spectrogram: torch.Tensor
    spatial_embedding: torch.Tensor
    embedding: torch.Tensor
    label: Optional[torch.Tensor]


class Perch2Network(nn.Module):
    """Perch v2: frontend, EfficientNet-B3 backbone, pooling and ProtoPNet head."""

    def __init__(self, config: Optional[Perch2Config] = None) -> None:
        super().__init__()
        cfg = config or Perch2Config()
        self.config = cfg
        self.frontend = Frontend(cfg)
        # VALID padding: the graph's stem conv has no padding.
        self.stem = ConvAct(nn.Conv2d(1, cfg.stem_channels, 3, stride=2, bias=True))
        stages = []
        for stage in cfg.stages:
            blocks = []
            for i in range(stage.repeats):
                blocks.append(
                    MBConv(
                        stage.in_channels if i == 0 else stage.out_channels,
                        stage.out_channels,
                        stage.kernel_size,
                        stage.stride if i == 0 else 1,
                        stage.expand_ratio,
                        cfg.se_ratio,
                    )
                )
            stages.append(nn.ModuleList(blocks))
        self.blocks = nn.ModuleList(stages)
        self.head = ConvAct(nn.Conv2d(cfg.stages[-1].out_channels, cfg.head_channels, 1, bias=True))
        self.pool = GlobalAvgPool()
        self.label_head = ProtoPNetHead(cfg)

    def feature_map(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Run stem, blocks and head on a spectrogram (B, frames, n_mels).

        Returns
        -------
        torch.Tensor
            Feature map of shape (B, head_channels, T, F).
        """
        x = self.stem(spectrogram.unsqueeze(1))
        for stage in self.blocks:
            for block in stage:
                x = block(x)
        return self.head(x)

    def forward(self, wav: torch.Tensor, *, with_logits: bool = True) -> Perch2Outputs:
        """Run the full model on a waveform (B, num_samples) at 32 kHz.

        Parameters
        ----------
        wav : torch.Tensor
            Raw waveform, float32.
        with_logits : bool
            If False, skip the (large) prototype classifier.

        Returns
        -------
        Perch2Outputs
            Spectrogram (B, frames, n_mels), spatial embedding (B, T, F, C),
            pooled embedding (B, C) and logits (B, num_classes) or None.
        """
        spectrogram = self.frontend(wav)
        features = self.feature_map(spectrogram)
        embedding = self.pool(features)
        spatial = features.permute(0, 2, 3, 1)
        label = self.label_head(spatial) if with_logits else None
        return Perch2Outputs(spectrogram, spatial, embedding, label)
