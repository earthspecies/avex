"""ATST Frame model implementation.

This module provides the Frame-level Attention-based Self-supervised Time-series (ATST)
model for audio representation learning with support for multi-crop wrapper and
Lightning integration.
"""

import argparse
import math
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops.layers.torch import Rearrange
from pytorch_lightning import LightningModule
from torch.nn.init import trunc_normal_
from torch.optim import AdamW
from torchvision import transforms

from avex.utils import universal_torch_load

N_BLOCKS = 12

__all__ = [
    "FrameAST",
    "FrameATST",
    "FrameATSTLightningModule",
    "FrameAST_small",
    "FrameAST_base",
    "FrameAST_large",
    "load_model",
    "get_scene_embedding",
    "get_timestamp_embedding",
]


class LinearRandomCropTransform:
    """Linear random crop transform for audio spectrograms."""

    def __init__(self, size: int) -> None:
        """Initialize the transform.

        Args:
            size: Target crop size
        """
        self.size = size

    def __repr__(self) -> str:
        """Return string representation of the transform.

        Returns:
            str: String representation of the transform class
        """
        return self.__class__.__name__ + "()"


class CustomAudioTransform:
    """Base class for custom audio transformations."""

    def __repr__(self) -> str:
        """Return string representation of the transform.

        Returns:
            str: String representation of the transform class
        """
        return self.__class__.__name__ + "()"


class Identity(CustomAudioTransform):
    """Identity transform that returns input unchanged."""

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply identity transform.

        Args:
            signal: Input audio signal tensor

        Returns:
            torch.Tensor: Unchanged input signal
        """
        return signal


class GaussianNoise(CustomAudioTransform):
    """Add Gaussian noise to audio signal."""

    def __init__(self, g: float) -> None:
        """Initialize Gaussian noise transform.

        Args:
            g: Noise scaling factor
        """
        self.g = g

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to signal.

        Args:
            signal: Input audio signal tensor

        Returns:
            torch.Tensor: Signal with added Gaussian noise
        """
        return signal + self.g * torch.randn_like(signal)


class PadToSize(CustomAudioTransform):
    """Pad audio signal to specified size."""

    def __init__(self, size: int) -> None:
        """Initialize pad to size transform.

        Args:
            size: Target size for padding
        """
        self.size = size

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Pad signal to target size.

        Args:
            signal: Input audio signal tensor

        Returns:
            torch.Tensor: Padded signal
        """
        if signal.shape[1] < self.size:
            signal = F.pad(signal, (0, self.size - signal.shape[1]))
        return signal


class ToSizeN(CustomAudioTransform):
    """Resize audio signal to multiple of specified size."""

    def __init__(self, size: int) -> None:
        """Initialize to size N transform.

        Args:
            size: Target size divisor
        """
        self.size = size

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Resize signal to multiple of target size.

        Args:
            signal: Input audio signal tensor

        Returns:
            torch.Tensor: Resized signal
        """
        n = signal.shape[1] // self.size
        m = signal.shape[1] % self.size
        if m > self.size // 2 or n == 0:
            signal = F.pad(signal, (0, self.size * (n + 1) - signal.shape[1]))
        else:
            signal = F.pad(signal, (0, self.size * n - signal.shape[1]))
        return signal


class CentralCrop(CustomAudioTransform):
    """Apply center cropping to audio signal."""

    def __init__(self, size: int, pad: bool = True) -> None:
        """Initialize central crop transform.

        Args:
            size: Target crop size
            pad: Whether to pad if signal is too short
        """
        self.size = size
        self.pad = pad

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply central crop to signal.

        Args:
            signal: Input audio signal tensor

        Returns:
            torch.Tensor: Center-cropped signal
        """
        if signal.shape[-1] < self.size:
            if self.pad:
                signal = F.pad(signal, (0, self.size - signal.shape[-1]))
            return signal

        start = (signal.shape[-1] - self.size) // 2
        if len(signal.shape) > 1:
            return signal[:, start : start + self.size]
        else:
            return signal[start : start + self.size]


class RandomCrop(CustomAudioTransform):
    """Apply random cropping to audio signal."""

    def __init__(self, size: int, pad: bool = True) -> None:
        """Initialize random crop transform.

        Args:
            size: Target crop size
            pad: Whether to pad if signal is too short
        """
        self.size = size
        self.pad = pad

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply random crop to signal.

        Args:
            signal: Input audio signal tensor

        Returns:
            torch.Tensor: Randomly cropped signal
        """
        if signal.shape[1] < self.size:
            if self.pad:
                signal = F.pad(signal, (0, self.size - signal.shape[-1]))
            return signal
        start = np.random.randint(0, signal.shape[-1] - self.size + 1)
        return signal[:, start : start + self.size]


class Normalize(CustomAudioTransform):
    """Normalize audio signal using mean and standard deviation."""

    def __init__(
        self,
        std_mean: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        reduce_dim: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> None:
        """Initialize normalize transform.

        Args:
            std_mean: Optional precomputed (std, mean) tuple
            reduce_dim: Dimensions along which to compute statistics
        """
        self.std_mean = std_mean
        self.reduce_dim = reduce_dim

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Apply normalization to input.

        Args:
            input: Input tensor with shape [batch, nmels, time]

        Returns:
            torch.Tensor: Normalized tensor
        """
        std, mean = None, None
        if self.std_mean is None:
            if self.reduce_dim is not None:
                std, mean = torch.std_mean(input, dim=self.reduce_dim, keepdim=True)
            else:
                std, mean = torch.std_mean(input)
        else:
            std, mean = self.std_mean
        output = input - mean
        output = output / (std + 1e-6)
        return output


class MinMax(CustomAudioTransform):
    """Apply min-max normalization to audio signal."""

    def __init__(
        self,
        min: Optional[torch.Tensor],
        max: Optional[torch.Tensor],
    ) -> None:
        """Initialize min-max normalization.

        Args:
            min: Minimum value for normalization
            max: Maximum value for normalization
        """
        self.min = min
        self.max = max

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Apply min-max normalization.

        Args:
            input: Input tensor

        Returns:
            torch.Tensor: Min-max normalized tensor in range [-1, 1]
        """
        min_, max_ = None, None
        if self.min is None:
            min_ = torch.min(input)
            max_ = torch.max(input)
        else:
            min_ = self.min
            max_ = self.max
        input = (input - min_) / (max_ - min_) * 2.0 - 1.0
        return input


class div(CustomAudioTransform):
    """Divide input by a constant value."""

    def __init__(self, value: float = 100) -> None:
        """Initialize division transform.

        Args:
            value: Divisor value
        """
        self.value = value

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Apply division to input.

        Args:
            input: Input tensor

        Returns:
            torch.Tensor: Divided tensor
        """
        input /= 100
        return input


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample for residual blocks.

    Args:
        x: Input tensor
        drop_prob: Drop probability
        training: Whether in training mode

    Returns:
        torch.Tensor: Output tensor with dropped paths
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    When applied in main path of residual blocks.
    """

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        """Initialize DropPath module.

        Args:
            drop_prob: Drop probability
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with drop path.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor with dropped paths
        """
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Initialize MLP.

        Args:
            in_features: Number of input features
            hidden_features: Number of hidden features
            out_features: Number of output features
            act_layer: Activation layer class
            drop: Dropout probability
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Initialize attention module.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for QK dot product
            attn_drop: Attention dropout probability
            proj_drop: Projection dropout probability
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through attention module.

        Args:
            x: Input tensor with shape [batch, seq_len, dim]
            mask: Optional attention mask

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn += mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """Transformer block with multi-head attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        """Initialize transformer block.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for QK dot product
            drop: Dropout probability
            attn_drop: Attention dropout probability
            drop_path: Drop path probability
            act_layer: Activation layer class
            norm_layer: Normalization layer class
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through transformer block.

        Args:
            x: Input tensor
            length: Optional length tensor for attention masking
            return_attention: Whether to return attention weights

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Output tensor or tuple of (output, attention) if return_attention=True
        """
        if length is not None:
            mask_att = get_attention_mask(x, length)
        else:
            mask_att = None

        y, attn = self.attn(self.norm1(x), mask_att)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x


def get_attention_mask(x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """Create attention mask for padded sequences.

    Args:
        x: Input tensor with shape [batch, seq_len, dim]
        length: Length tensor with shape [batch]

    Returns:
        torch.Tensor: Attention mask
    """
    batch_size, max_len, _ = x.shape
    # create mask for padded elements and zero-out them
    mask = torch.arange(max_len, device=length.device).expand(batch_size, max_len) >= length[:, None]
    # extend the mask to attention shape and set weight
    mask = -10000.0 * mask[:, None, None, :]
    mask = mask.expand(batch_size, 1, max_len, max_len).to(x.device)
    return mask


def _no_grad_trunc_normal_(tensor: torch.Tensor, mean: float, std: float, a: float, b: float) -> torch.Tensor:
    """Fill tensor with truncated normal distribution (no gradient tracking).

    Cut & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Args:
        tensor: Tensor to fill
        mean: Mean of the distribution
        std: Standard deviation of the distribution
        a: Lower bound
        b: Upper bound

    Returns:
        torch.Tensor: Filled tensor
    """

    def norm_cdf(x: float) -> float:
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, u], then translate to
        # [2lower-1, 2u-1].
        tensor.uniform_(2 * lower - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def get_num_patches(
    height: int = 64,
    width: int = 1001,
    patch_height: int = 16,
    patch_width: int = 16,
) -> int:
    """Calculate number of patches for given dimensions.

    Args:
        height: Input height
        width: Input width
        patch_height: Patch height
        patch_width: Patch width

    Returns:
        int: Number of patches
    """
    return (height // patch_height) * (width // patch_width)


class PatchEmbed(nn.Module):
    """Patch embedding using convolutional layer."""

    def __init__(
        self,
        patch_height: int = 64,
        patch_width: int = 4,
        embed_dim: int = 768,
        input_dim: int = 1,
    ) -> None:
        """Initialize patch embedding layer.

        Args:
            patch_height: Height of each patch
            patch_width: Width of each patch
            embed_dim: Embedding dimension
            input_dim: Input channel dimension
        """
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.proj = nn.Conv2d(
            input_dim,
            embed_dim,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
        )

    def forward(
        self, melspec: torch.Tensor, length: Optional[torch.Tensor] = None
    ) -> Tuple[None, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through patch embedding.

        Args:
            melspec: Input mel-spectrogram tensor
            length: Optional length tensor

        Returns:
            Tuple[None, torch.Tensor, Optional[torch.Tensor]]:
                (None, patch embeddings, patch lengths)
        """
        height = melspec.shape[2] - melspec.shape[2] % self.patch_height
        patch_embed = self.proj(melspec).squeeze(2).permute(0, 2, 1)

        if length is not None:
            patch_length = (height // self.patch_height) * ((length - length % self.patch_width) // self.patch_width)
        else:
            patch_length = None

        return None, patch_embed, patch_length


class PatchEmbed_v2(nn.Module):
    """Patch embedding using rearrangement and linear projection."""

    def __init__(
        self,
        patch_height: int = 64,
        patch_width: int = 4,
        embed_dim: int = 768,
        input_dim: int = 1,
    ) -> None:
        """Initialize patch embedding layer v2.

        Args:
            patch_height: Height of each patch
            patch_width: Width of each patch
            embed_dim: Embedding dimension
            input_dim: Input channel dimension
        """
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_maker = Rearrange(
            "b c (h p1) (w p2) -> b (w h) (p1 p2 c)",
            p1=patch_height,
            p2=patch_width,
        )
        self.patch_embed = nn.Linear(patch_height * patch_width * input_dim, embed_dim)

    def forward(
        self, melspec: torch.Tensor, length: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through patch embedding v2.

        Args:
            melspec: Input mel-spectrogram tensor
            length: Optional length tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                (patches, patch embeddings, patch lengths)
        """
        height = melspec.shape[2] - melspec.shape[2] % self.patch_height
        width = melspec.shape[3] - melspec.shape[3] % self.patch_width
        patch = self.patch_maker(melspec[:, :, :height, :width])
        patch_embed = self.patch_embed(patch)

        if length is not None:
            patch_length = (height // self.patch_height) * ((length - length % self.patch_width) // self.patch_width)
        else:
            patch_length = None

        return patch, patch_embed, patch_length


class FrameAST(nn.Module):
    """Vision Transformer for audio spectrogram processing."""

    def __init__(
        self,
        nprompt: int = 0,
        spec_h: int = 64,
        spec_w: int = 1001,
        patch_w: int = 16,
        patch_h: int = 16,
        pos_type: str = "cut",
        avg_blocks: int = 0,
        in_chans: int = 1,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_embed: str = "Linear",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize FrameAST model.

        Args:
            nprompt: Number of prompt tokens
            spec_h: Height of input spectrogram
            spec_w: Width of input spectrogram
            patch_w: Patch width
            patch_h: Patch height
            pos_type: Position encoding type
            avg_blocks: Number of blocks to average
            in_chans: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for QK dot product
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Drop path rate
            norm_layer: Normalization layer class
            patch_embed: Type of patch embedding
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.spec_w = spec_w
        self.spec_h = spec_h
        self.embed_dim = embed_dim
        self.patch_w = patch_w
        self.patch_h = patch_h

        self.pos_type = pos_type
        self.avg_blocks = avg_blocks

        if patch_embed == "Linear":
            self.patch_embed = PatchEmbed_v2(patch_h, patch_w, embed_dim)
        elif patch_embed == "CNN":
            self.patch_embed = PatchEmbed(patch_h, patch_w, embed_dim)
        else:
            raise NotImplementedError("patch_embed={} not implemented".format(patch_embed))

        self.mask_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # hack
        self.nprompt = nprompt
        if self.nprompt > 0:
            self.prompt_embed = nn.Parameter(torch.zeros(1, self.nprompt, self.embed_dim))
            trunc_normal_(self.prompt_embed, std=0.02)

        num_patches = get_num_patches(spec_h, spec_w, patch_h, patch_w)
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm_frame = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.mask_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for the model.

        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(
        self,
        x: torch.Tensor,
        mask_index: Optional[torch.Tensor],
        length: Optional[torch.Tensor],
        mask: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Prepare tokens for transformer processing.

        Args:
            x: Input tensor
            mask_index: Mask indices
            length: Sequence lengths
            mask: Whether to apply masking

        Returns:
            Tuple containing processed tokens and related tensors
        """
        B, nc, h, w = x.shape
        mel_patches, x, patch_length = self.patch_embed(x, length)  # patch linear embedding
        B, T, C = x.shape

        if (mask_index is not None) and mask:
            mask_index_expand = mask_index.unsqueeze(2).expand(B, T, self.embed_dim).float()
            x = (1 - mask_index_expand) * x + mask_index_expand * self.mask_embed.expand(B, T, C)

        # add positional encoding to each token
        if self.pos_type == "cut":
            pos = self.pos_embed[:, 1 : T + 1, :].expand(B, -1, -1)
            x = x + pos
        else:
            pos = self.interpolate_pos_encoding(x, h, w)
            x = x + pos[:, 1:]

        # pos = self.pos_embed[:,1:T+1,:].expand(B,-1,-1)
        # x = x + pos

        return self.pos_drop(x), pos, mel_patches, h, w, patch_length

    def freeze(self) -> None:
        """Freeze model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        mask_index: Optional[torch.Tensor] = None,
        mask_input: bool = True,
        length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through FrameAST.

        Args:
            x: Input tensor
            mask_index: Optional mask indices
            mask_input: Whether to mask input
            length: Optional sequence lengths

        Returns:
            torch.Tensor: Output features
        """
        x, pos, mel_patches, h, w, patch_length = self.prepare_tokens(x, mask_index, length, mask_input)

        length_mask = torch.arange(x.shape[1]).to(x.device) < patch_length.unsqueeze(1)
        length_mask = length_mask.to(x.device)
        mask_index = mask_index & length_mask

        if self.nprompt > 0:
            x = torch.cat([self.prompt_embed.expand(x.shape[0], -1, -1), x], dim=1)

        avg_x = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, patch_length + self.nprompt)
            if self.avg_blocks > 0:
                if i >= len(self.blocks) - self.avg_blocks:
                    avg_x.append(F.instance_norm(x.transpose(1, 2)).transpose(1, 2))

        if self.avg_blocks > 0:
            avg_x = torch.mean(torch.stack(avg_x), dim=0)
            frame_repr = avg_x
        else:
            frame_repr = self.norm_frame(x)

        return frame_repr[:, self.nprompt :][mask_index]

    def get_cls(self, x: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get CLS token representation.

        Args:
            x: Input tensor
            length: Optional sequence lengths

        Returns:
            torch.Tensor: CLS token features
        """
        x, pos, mel_patches, h, w, patch_length = self.prepare_tokens(x, None, length, False)

        if self.nprompt > 0:
            x = torch.cat([self.prompt_embed.expand(x.shape[0], -1, -1), x], dim=1)

        for _i, blk in enumerate(self.blocks):
            x = blk(x, patch_length + self.nprompt)

        frame_repr = self.norm_frame(x)

        return torch.mean(frame_repr[:, : self.nprompt], dim=1)

    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate positional encoding for different input sizes.

        Args:
            x: Input tensor
            h: Target height
            w: Target width

        Returns:
            torch.Tensor: Interpolated positional encoding
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == self.spec_w and h == self.spec_h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_width
        h0 = h // self.patch_embed.patch_height
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1,
                self.spec_h // self.patch_h,
                self.spec_w // self.patch_w,
                dim,
            ).permute(0, 3, 1, 2),
            scale_factor=(
                h0 / (self.spec_h // self.patch_h),
                w0 / (self.spec_w // self.patch_w),
            ),
            mode="bicubic",
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """Get self-attention from the last layer.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Self-attention weights
        """
        x, _, _, _, _, _ = self.prepare_tokens(x, mask_index=None, length=None, mask=False)
        atts = []
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x, att = blk(x, return_attention=True)
                atts.append(att)
            else:
                x, att = blk(x, return_attention=True)
                atts.append(att)
                return atts
                # return attention of the last block

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        length: torch.Tensor,
        n: int = 1,
        scene: bool = True,
    ) -> List[torch.Tensor]:
        """Get intermediate layer representations.

        Args:
            x: Input tensor
            length: Sequence lengths
            n: Number of layers to return
            scene: Whether to return scene-level features

        Returns:
            List[torch.Tensor]: Intermediate representations
        """
        x, _, _, _, _, patch_length = self.prepare_tokens(x, mask_index=None, length=length, mask=False)
        # we return the output tokens from the `n` last blocks
        output = []
        if self.nprompt > 0:
            x = torch.cat([self.prompt_embed.expand(x.shape[0], -1, -1), x], dim=1)
        for i, blk in enumerate(self.blocks):
            x = blk(x, patch_length + self.nprompt)

            if len(self.blocks) - i <= n:
                norm_x = self.norm_frame(x)
                if scene:
                    length_mask = torch.arange(x.shape[1] - self.nprompt).to(x.device) < patch_length.unsqueeze(1)
                    avg = torch.sum(
                        norm_x[:, self.nprompt :] * length_mask.unsqueeze(-1),
                        dim=1,
                    ) / (patch_length.unsqueeze(-1) + 1e-6)
                    output.append(avg)
                    if self.nprompt > 0:
                        output.append(torch.mean(x[:, : self.nprompt], dim=1))
                else:
                    output.append(norm_x[:, self.nprompt :])

        return torch.cat(output, dim=-1)


def build_mlp(
    num_layers: int,
    input_dim: int,
    mlp_dim: int,
    output_dim: int,
    last_bn: bool = True,
) -> nn.Sequential:
    """Build multi-layer perceptron.

    Args:
        num_layers: Number of layers
        input_dim: Input dimension
        mlp_dim: Hidden dimension
        output_dim: Output dimension
        last_bn: Whether to use batch norm in last layer

    Returns:
        nn.Sequential: MLP module
    """
    mlp = []
    for layer_idx in range(num_layers):
        dim1 = input_dim if layer_idx == 0 else mlp_dim
        dim2 = output_dim if layer_idx == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if layer_idx < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)


def byol_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True) -> torch.Tensor:
    """Compute BYOL loss.

    Args:
        p: Predicted features
        z: Target features
        simplified: Whether to use simplified version

    Returns:
        torch.Tensor: BYOL loss
    """
    if simplified:
        return 2 - 2 * F.cosine_similarity(p, z, dim=-1).mean()

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    return 2 - 2 * (p * z).sum(dim=1).mean()


def compute_var(y: torch.Tensor) -> torch.Tensor:
    """Compute variance.

    Args:
        y: Input tensor

    Returns:
        torch.Tensor: Variance
    """
    y = y.view(-1, y.size(-1))
    zc = torch.tensor(y.size(0)).cuda()
    zs = y.sum(dim=0)
    zss = (y**2).sum(dim=0)

    torch.distributed.all_reduce(zc)
    torch.distributed.all_reduce(zs)
    torch.distributed.all_reduce(zss)

    var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
    return torch.sqrt(var + 1e-6)


class ByolLoss(nn.Module):
    """BYOL loss module."""

    def __init__(self, symmetric: bool) -> None:
        """Initialize BYOL loss.

        Args:
            symmetric: Whether to use symmetric loss
        """
        super().__init__()
        self.symmetric = symmetric

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Forward pass through BYOL loss.

        Args:
            student: Student features
            teacher: Teacher features

        Returns:
            torch.Tensor: BYOL loss
        """
        stu_frm = student
        tea_frm = teacher

        std_frm_stu = compute_var(F.normalize(stu_frm, dim=-1)).mean()
        std_frm_tea = compute_var(F.normalize(tea_frm, dim=-1)).mean()

        if self.symmetric:
            stu_frm = stu_frm.chunk(2)
            tea_frm = tea_frm.chunk(2)
            total_loss_frm = 0
            n_loss_terms = 0
            for iq, q in enumerate(tea_frm):
                for iv, v in enumerate(stu_frm):
                    if iq == iv:
                        continue
                    loss = byol_loss_func(q, v, simplified=False)
                    n_loss_terms += 1
                    total_loss_frm += loss
            total_loss_frm /= n_loss_terms

        else:
            total_loss_frm = byol_loss_func(tea_frm, stu_frm)
        return total_loss_frm, std_frm_stu, std_frm_tea


class MultiCropWrapper(nn.Module):
    """Multi-crop wrapper for self-supervised learning.

    Performs the forward pass to compute the logits and the loss
    of the view assignment task.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        projector: str = "mlp",
        predictor: bool = True,
    ) -> None:
        """Initialize MultiCropWrapper.

        Args:
            encoder: Backbone encoder
            embed_dim: Embedding dimension
            projector: Type of projector
            predictor: Whether to use predictor
        """
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification

        self.encoder = encoder
        if projector == "mlp":
            self.projector = build_mlp(2, embed_dim, 4096, 256, last_bn=False)
        elif projector == "linear":
            self.projector = nn.Linear(embed_dim, embed_dim)
        else:
            self.projector = nn.Identity()

        if predictor:
            self.predictor = build_mlp(2, 256, 4096, 256, last_bn=False)
        else:
            self.predictor = nn.Identity()

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        length: torch.Tensor,
        mask: torch.Tensor,
        mask_input: bool,
    ) -> torch.Tensor:
        """Forward pass through multi-crop wrapper.

        Args:
            x: Input tensor or list of tensors
            length: Sequence lengths
            mask: Attention mask
            mask_input: Whether to mask input

        Returns:
            torch.Tensor: Output features
        """
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx, output_frame, _output_cls = (
            0,
            torch.empty(0).to(x[0].device),
            torch.empty(0).to(x[0].device),
        )

        for end_idx in idx_crops:
            _out_frame = self.encoder(
                torch.cat(x[start_idx:end_idx]),
                length=torch.cat(length[start_idx:end_idx]),
                mask_index=torch.cat(mask[start_idx:end_idx]),
                mask_input=mask_input,
            )
            # accumulate outputs
            output_frame = torch.cat((output_frame, _out_frame))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.predictor(self.projector(output_frame))


class FrameATST(nn.Module):
    """Frame-level Audio Self-supervised Transformer."""

    def __init__(
        self,
        arch: str = "small",
        symmetric: bool = True,
        pos_type: str = "cut",
        avg_blocks: int = 0,
        patch_embed: str = "Linear",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize FrameATST model.

        Args:
            arch: Model architecture size
            symmetric: Whether to use symmetric loss
            pos_type: Position encoding type
            avg_blocks: Number of blocks to average
            patch_embed: Type of patch embedding
            **kwargs: Additional keyword arguments

        Raises:
            RuntimeError: If architecture is not implemented
        """
        super().__init__()
        if arch == "small":
            encoder_fn = FrameAST_small
            embed_dim = 384
        elif arch == "base":
            encoder_fn = FrameAST_base
            embed_dim = 768
        else:
            raise RuntimeError("arch {} is not implemented".format(arch))
        self.symmetric = symmetric
        if avg_blocks == 0:  # atst-frame
            self.student = MultiCropWrapper(
                encoder_fn(pos_type=pos_type, patch_embed=patch_embed, **kwargs),
                embed_dim,
                predictor=True,
            )
            self.teacher = MultiCropWrapper(
                encoder_fn(pos_type=pos_type, patch_embed=patch_embed, **kwargs),
                embed_dim,
                predictor=False,
            )
        else:  # data2vec, no projector, predictor is linear
            self.student = MultiCropWrapper(
                encoder_fn(pos_type=pos_type, patch_embed=patch_embed, **kwargs),
                embed_dim,
                projector="linear",
                predictor=False,
            )
            self.teacher = MultiCropWrapper(
                encoder_fn(
                    pos_type=pos_type,
                    patch_embed=patch_embed,
                    avg_blocks=8,
                    **kwargs,
                ),
                embed_dim,
                projector=None,
                predictor=False,
            )
        for p in self.teacher.parameters():
            p.requires_grad = False

        if avg_blocks == 0:  # atst-frame
            self.teacher.load_state_dict({k: v for k, v in self.student.state_dict().items() if "predictor" not in k})
        else:  # data2vec
            self.teacher.load_state_dict({k: v for k, v in self.student.state_dict().items() if "projector" not in k})

        self.loss_fn = ByolLoss(symmetric=symmetric)

    def forward(self, x: torch.Tensor, length: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through FrameATST.

        Args:
            x: Input tensor
            length: Sequence lengths
            mask: Attention mask

        Returns:
            torch.Tensor: Loss value
        """
        if self.symmetric:
            tea = self.teacher(x, length, mask, False)
            stu = self.student(x, length, mask, True)
            return self.loss_fn(stu, tea)
        else:
            tea = self.teacher(x[:1], length[:1], mask[:1], False)
            stu = self.student(x[1:], length[1:], mask[1:], True)
            return self.loss_fn(stu, tea)

    def update_teacher(self, m: float) -> None:
        """Update teacher network with exponential moving average.

        Args:
            m: Momentum parameter
        """
        with torch.no_grad():
            for param_q, param_k in zip(
                self.student.encoder.parameters(),
                self.teacher.encoder.parameters(),
                strict=False,
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(
                self.student.projector.parameters(),
                self.teacher.projector.parameters(),
                strict=False,
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def _init_teacher(self) -> None:
        """Initialize teacher network with student weights."""
        self.teacher.load_state_dict({k: v for k, v in self.student.state_dict().items() if "predictor" not in k})


def cosine_scheduler_step(
    base_value: float,
    final_value: float,
    max_steps: int,
    warmup_steps: int = 0,
    start_warmup_value: float = 0,
) -> np.ndarray:
    """Create cosine learning rate schedule.

    Args:
        base_value: Base learning rate value
        final_value: Final learning rate value
        max_steps: Maximum number of steps
        warmup_steps: Number of warmup steps
        start_warmup_value: Starting warmup value

    Returns:
        np.ndarray: Learning rate schedule
    """
    warmup_schedule = np.array([])
    if warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_steps)

    iters = np.arange(max_steps - warmup_steps)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == max_steps
    return schedule


def get_params_groups(model: nn.Module) -> List[dict]:
    """Get parameter groups for optimizer.

    Args:
        model: Model to get parameters from

    Returns:
        List[dict]: Parameter groups
    """
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [
        {"params": regularized},
        {"params": not_regularized, "weight_decay": 0.0},
    ]


def bool_flag(s: Union[str, bool]) -> bool:
    """Parse boolean arguments from the command line.

    Args:
        s: String or boolean value to parse

    Returns:
        bool: Parsed boolean value

    Raises:
        argparse.ArgumentTypeError: If string cannot be parsed as boolean
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


class FrameATSTLightningModule(LightningModule):
    """PyTorch Lightning module for FrameATST."""

    def __init__(
        self,
        arch: str = "small",
        learning_rate: float = 5e-4,
        warmup_steps: int = 1300,
        max_steps: int = 39000,
        ema: float = 0.99,
        symmetric: bool = True,
        pos_type: str = "cut",
        avg_blocks: int = 0,
        patch_embed: str = "Linear",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize FrameATST Lightning module.

        Args:
            arch: Model architecture size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps
            ema: Exponential moving average coefficient
            symmetric: Whether to use symmetric loss
            pos_type: Position encoding type
            avg_blocks: Number of blocks to average
            patch_embed: Type of patch embedding
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.model = FrameATST(
            arch=arch,
            symmetric=symmetric,
            pos_type=pos_type,
            avg_blocks=avg_blocks,
            patch_embed=patch_embed,
            **kwargs,
        )
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.symmetric = symmetric
        self.ema_scheduler = cosine_scheduler_step(ema, 1, max_steps, 0)
        self.wd_scheduler = cosine_scheduler_step(0.04, 0.4, max_steps, 0)
        self.mylr_scheduler = cosine_scheduler_step(learning_rate, 1e-6, max_steps, warmup_steps)
        self.save_hyperparameters()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # noqa: ANN401
        """Training step.

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            torch.Tensor: Loss value
        """
        self.schedule()
        (melspecs, lengths, masks), _ = batch
        total_loss_frm, std_frm_stu, std_frm_tea = self.model(melspecs, lengths, masks)
        loss = total_loss_frm
        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("loss_frm", total_loss_frm, prog_bar=True, logger=True)
        self.log("std_frm_tea", std_frm_tea, prog_bar=True, logger=True)
        self.log("std_frm_stu", std_frm_stu, prog_bar=True, logger=True)
        self.log(
            "ema",
            self.ema_scheduler[self.global_step],
            prog_bar=True,
            logger=True,
        )
        self.log("step", self.global_step, prog_bar=True, logger=True)

        return loss

    def freeze(self) -> None:
        return super().freeze()

    def unfreeze(self) -> None:
        return super().unfreeze()

    def schedule(self) -> None:
        """Update learning rate schedule."""
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            param_group["lr"] = self.mylr_scheduler[self.global_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_scheduler[self.global_step]

        self.log(
            "wd",
            self.wd_scheduler[self.global_step],
            prog_bar=True,
            logger=True,
        )
        self.log("lr", param_group["lr"], prog_bar=True, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers.

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        optimizer = AdamW(
            get_params_groups(self.model.student),
            lr=self.learning_rate,
            weight_decay=0.0,
        )
        return [optimizer]

    def on_train_batch_end(
        self,
        outputs: Any,  # noqa: ANN401
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Called at the end of training batch.

        Args:
            outputs: Training step outputs
            batch: Training batch
            batch_idx: Batch index
            unused: Unused parameter
        """
        m = self.ema_scheduler[self.global_step]
        self.model.update_teacher(m)

    @staticmethod
    def add_model_specific_args(parent_parser: Any) -> Any:  # noqa: ANN401
        """Add model specific arguments to parser.

        Args:
            parent_parser: Parent argument parser

        Returns:
            Any: Parser with added arguments
        """
        parser = parent_parser.add_argument_group("FrameATSTModel")
        parser.add_argument("--arch", type=str, default="small")
        parser.add_argument(
            "--symmetric",
            type=bool_flag,
            default=True,
            help="whether to use symemtric loss",
        )
        parser.add_argument(
            "--nprompt",
            type=int,
            default=0,
            help="number of prompts, not used, always 0",
        )
        parser.add_argument(
            "--learning_rate",
            default=0.0005,
            type=float,
            help="""Learning rate at the end of linear warmup (highest LR used
            during training). The learning rate is linearly scaled with the batch
            size, and specified here for a reference batch size of 256.""",
        )
        parser.add_argument(
            "--ema",
            default=0.99,
            type=float,
            help="""Base EMA parameter for teacher update. The value is increased
            to 1 during training with cosine schedule.""",
        )
        parser.add_argument("--warmup_steps", default=1300, type=int)
        parser.add_argument("--max_steps", default=39010, type=int)
        parser.add_argument(
            "--pos_type",
            default="cut",
            type=str,
            help='"cut" denotes absolute positional embedding, "interpolate" '
            "denotes 2D positional embedding used in SSAST",
        )
        parser.add_argument(
            "--avg_blocks",
            default=0,
            type=int,
            help="0 means atst-frame, a positive int value means data2vec style loss",
        )
        parser.add_argument(
            "--patch_embed",
            default="Linear",
            type=str,
            help="Linear or CNN patch embedding",
        )
        return parent_parser


def FrameAST_small(patch_h: int = 64, patch_w: int = 4, **kwargs: Dict[str, Any]) -> FrameAST:
    """Create small FrameAST model.

    Args:
        patch_h: Patch height
        patch_w: Patch width
        **kwargs: Additional keyword arguments

    Returns:
        FrameAST: Small FrameAST model
    """
    return FrameAST(
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def FrameAST_base(patch_h: int = 64, patch_w: int = 4, **kwargs: Dict[str, Any]) -> FrameAST:
    """Create base FrameAST model.

    Args:
        patch_h: Patch height
        patch_w: Patch width
        **kwargs: Additional keyword arguments

    Returns:
        FrameAST: Base FrameAST model
    """
    return FrameAST(
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def FrameAST_large(patch_h: int, patch_w: int, **kwargs: Dict[str, Any]) -> FrameAST:
    """Create large FrameAST model.

    Args:
        patch_h: Patch height
        patch_w: Patch width
        **kwargs: Additional keyword arguments

    Returns:
        FrameAST: Large FrameAST model
    """
    return FrameAST(
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def load_model(model_path: str, device: str, ssl_model: bool = False) -> nn.Module:
    """Load pretrained model.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        ssl_model: Whether this is an SSL model

    Returns:
        Any: Loaded model
    """
    melspec_t = torchaudio.transforms.MelSpectrogram(
        16000,
        f_min=60,
        f_max=7800,
        hop_length=160,
        win_length=1024,
        n_fft=1024,
        n_mels=64,
    ).to(device)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)
    normalize = MinMax(min=-79.6482, max=50.6842)
    s = universal_torch_load(model_path)

    if ssl_model:
        pretrained_model = FrameATSTLightningModule.load_from_checkpoint(model_path)
        pretrained_encoder = pretrained_model.model.teacher.encoder
    else:
        pretrained_model = FrameATSTLightningModule()
        pretrained_encoder = pretrained_model.model.teacher.encoder
        pretrained_encoder.load_state_dict(s, strict=False)

    if "hyper_parameters" in s:
        pretrained_encoder.hyper_param = s["hyper_parameters"]

    pretrained_encoder.sample_rate = 16000
    pretrained_encoder.scene_embedding_size = pretrained_encoder.embed_dim * 2 * N_BLOCKS
    pretrained_encoder.timestamp_embedding_size = pretrained_encoder.embed_dim * N_BLOCKS

    pretrained_encoder.train()
    pretrained_encoder.transform = transforms.Compose([melspec_t, to_db, normalize])

    return pretrained_encoder


def get_scene_embedding(audio: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Extract scene (clip-level) embedding from an audio clip.

    Args:
        audio: Audio tensor in the shape of [1,N] or [B,1,N]
        model: The pretrained encoder returned by load_model

    Returns:
        torch.Tensor: Scene embedding in the shape of [1,N_BLOCKS*emb_size] or
            [B,N_BLOCKS*emb_size]
    """
    if len(audio.shape) == 2:
        audio = audio.unsqueeze(1)
    else:
        assert len(audio.shape) == 3

    model.to(audio.device)
    model.transform.transforms[0].to(audio.device)
    mel = model.transform(audio)
    chunk_len = 1001  # 10 secnods, consistent with the length of positional embedding
    total_len = mel.shape[-1]
    num_chunks = total_len // chunk_len + 1
    output = []
    for i in range(num_chunks):
        start = i * chunk_len
        end = (i + 1) * chunk_len
        if end > total_len:
            end = total_len
        if end > start:  # and (length +chunk_len//2  > end):
            mel_chunk = mel[:, :, :, start:end]
            len_chunk = mel_chunk.shape[-1]  # if length>end+chunk_len else (length - end)
            len_chunk = torch.tensor([len_chunk]).expand(mel.shape[0]).to(audio.device)
            output_chunk = model.get_intermediate_layers(mel_chunk, len_chunk, n=12)

            output.append(output_chunk)
    output = torch.stack(output, dim=0)
    output = torch.mean(output, dim=0)

    return output


def get_timestamp_embedding(
    audio: torch.Tensor,
    model: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract frame-level embeddings from an audio clip.

    Args:
        audio: Audio tensor in the shape of [1,N] or [B,1,N]
        model: The pretrained encoder returned by load_model

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of (embeddings, timestamps) where
            embeddings are in shape [1,T,N_BLOCKS*emb_size] or [B,T,N_BLOCKS*emb_size],
            timestamps are in milliseconds
    """
    if len(audio.shape) == 2:
        audio = audio.unsqueeze(1)
    else:
        assert len(audio.shape) == 3

    model.to(audio.device)
    model.transform.transforms[0].to(audio.device)
    mel = model.transform(audio)
    output = []

    chunk_len = 1001  # 10 secnods, consistent with the length of positional embedding

    total_len = mel.shape[-1]
    num_chunks = total_len // chunk_len + 1
    for i in range(num_chunks):
        start = i * chunk_len
        end = (i + 1) * chunk_len
        if end > total_len:
            end = total_len
        if end > start:
            mel_chunk = mel[:, :, :, start:end]
            len_chunk = torch.tensor([mel_chunk.shape[-1]]).expand(mel.shape[0]).to(audio.device)

            output_chunk = model.get_intermediate_layers(mel_chunk, len_chunk, n=N_BLOCKS, scene=False)

            output.append(output_chunk)
    output = torch.cat(output, dim=1)
    return output.permute(0, 2, 1)
