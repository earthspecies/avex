# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
import warnings
from typing import Any, Callable, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class GradMultiply(torch.autograd.Function):
    """Gradient scaling function for layer-wise gradient decay."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Forward pass that passes input unchanged while storing scale factor.

        Args:
            ctx: Context object to store information for backward pass
            x: Input tensor
            scale: Scale factor for gradients

        Returns:
            torch.Tensor: The input tensor unchanged
        """
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass that scales gradients by stored factor.

        Args:
            ctx: Context object containing stored scale factor
            grad: Gradient tensor

        Returns:
            Tuple[torch.Tensor, None]: Scaled gradient and None for scale parameter
        """
        return grad * ctx.scale, None


class SamePad(nn.Module):
    """Padding module that ensures same output size for convolutions."""

    def __init__(self, kernel_size: int, causal: bool = False) -> None:
        """Initialize SamePad module.

        Args:
            kernel_size: Size of the convolution kernel
            causal: Whether to use causal padding (removes from end)
        """
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply padding removal to maintain same size.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Tensor with padding removed
        """
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def __init__(self) -> None:
        """Initialize Swish activation."""
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Swish-activated tensor
        """
        return x * self.act(x)


class GLU_Linear(nn.Module):
    """Gated Linear Unit (GLU) implementation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
    ) -> None:
        """Initialize GLU_Linear module.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            glu_type: Type of gating function ("sigmoid", "swish", "relu", "gelu",
                "bilinear")
            bias_in_glu: Whether to use bias in the linear layer
        """
        super(GLU_Linear, self).__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GLU transformation.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: GLU-transformed tensor
        """
        # to be consistent with GLU_Linear, we assume the input always has the
        # #channel (#dim) in the last dimension of the tensor, so need to switch
        # the dimension first for 1D-Conv case
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = (
                x[:, :, 0 : self.output_dim]
                * x[:, :, self.output_dim : self.output_dim * 2]
            )
        else:
            x = x[:, :, 0 : self.output_dim] * self.glu_act(
                x[:, :, self.output_dim : self.output_dim * 2]
            )

        return x


def gelu_accurate(x: torch.Tensor) -> torch.Tensor:
    """Accurate GELU activation function implementation.

    Args:
        x: Input tensor

    Returns:
        torch.Tensor: GELU-activated tensor
    """
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Standard GELU activation function.

    Args:
        x: Input tensor

    Returns:
        torch.Tensor: GELU-activated tensor
    """
    return torch.nn.functional.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get activation function by name.

    Args:
        activation: Name of the activation function

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: The activation function

    Raises:
        RuntimeError: If activation function is not supported
    """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        warnings.warn(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate", stacklevel=2
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def quant_noise(
    module: Union[nn.Linear, nn.Embedding, nn.Conv2d], p: float, block_size: int
) -> Union[nn.Linear, nn.Embedding, nn.Conv2d]:
    """Apply quantization noise to module weights.

    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        module: PyTorch module (Linear, Embedding, or Conv2d)
        p: Amount of quantization noise (probability)
        block_size: Size of the blocks for subsequent quantization with iPQ

    Returns:
        Union[nn.Linear, nn.Embedding, nn.Conv2d]: The module with quantization
            noise applied

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert module.weight.size(1) % block_size == 0, (
            "Input features must be a multiple of block sizes"
        )

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert module.in_channels % block_size == 0, (
                "Input channels must be a multiple of block sizes"
            )
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod: nn.Module, input: Any) -> None:
        """Pre-forward hook to apply quantization noise during training.

        Args:
            mod: The module
            input: The input to the module
        """
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        in_channels // block_size * out_channels, device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(
                        out_channels, in_channels
                    )
                else:
                    mask = torch.zeros(
                        weight.size(0) * weight.size(1) * weight.size(2) // block_size,
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(
                        weight.size(0), weight.size(1), weight.size(2)
                    )
            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module
