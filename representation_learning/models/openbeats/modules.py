"""OpenBEATs neural network modules and components.

This module provides the core neural network components for OpenBEATs including
transformer layers, attention mechanisms with flash attention support, and utility functions.

Based on:
- Paper: https://arxiv.org/abs/2507.14129 (OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder)
- Original BEATs: https://arxiv.org/abs/2212.09058
- Copyright (c) 2022 Microsoft, Licensed under The MIT License
"""

import logging
import math
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from torch.nn import LayerNorm, Parameter

logger = logging.getLogger(__name__)

# Check PyTorch version for flash attention support
is_torch_v25_to_v26 = V(torch.__version__) >= V("2.5.0") and V(torch.__version__) <= V("2.6.0")


class GradMultiply(torch.autograd.Function):
    """Gradient scaling function for layer-wise gradient decay."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Forward pass that passes input unchanged while storing scale factor."""
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """Backward pass that scales gradients by stored factor."""
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
        """Apply padding removal to maintain same size."""
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
        """Apply Swish activation."""
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
            glu_type: Type of gating function
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
        """Apply GLU transformation."""
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = x[:, :, 0 : self.output_dim] * x[:, :, self.output_dim : self.output_dim * 2]
        else:
            x = x[:, :, 0 : self.output_dim] * self.glu_act(x[:, :, self.output_dim : self.output_dim * 2])

        return x


def gelu_accurate(x: torch.Tensor) -> torch.Tensor:
    """Accurate GELU activation function implementation."""
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Standard GELU activation function."""
    return torch.nn.functional.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get activation function by name.

    Args:
        activation: Name of the activation function

    Returns:
        The activation function

    Raises:
        RuntimeError: If activation function is not supported
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        warnings.warn(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate",
            stacklevel=2,
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
        raise RuntimeError(f"--activation-fn {activation} not supported")


def quant_noise(
    module: Union[nn.Linear, nn.Embedding, nn.Conv2d],
    p: float,
    block_size: int,
) -> Union[nn.Linear, nn.Embedding, nn.Conv2d]:
    """Apply quantization noise to module weights.

    Args:
        module: PyTorch module (Linear, Embedding, or Conv2d)
        p: Amount of quantization noise (probability)
        block_size: Size of the blocks for subsequent quantization

    Returns:
        The module with quantization noise applied
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
        assert module.weight.size(1) % block_size == 0, "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert module.in_channels % block_size == 0, "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod: nn.Module, input: Tuple[torch.Tensor, ...]) -> None:
        """Pre-forward hook to apply quantization noise during training."""
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features,
                    device=weight.device,
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
                        in_channels // block_size * out_channels,
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(out_channels, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0) * weight.size(1) * weight.size(2) // block_size,
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(weight.size(0), weight.size(1), weight.size(2))
            # scale weights and apply mask
            mask = mask.to(torch.bool)
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


def init_bert_params(module: nn.Module) -> None:
    """Initialize the weights specific to the BERT Model.

    This overrides the default initializations depending on the specified arguments.
    """

    def normal_(data: torch.Tensor) -> None:
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class MultiheadAttention(nn.Module):
    """Multi-headed attention with optional flash attention support.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        gru_rel_pos: bool = False,
        rescale_init: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        """Initialize MultiheadAttention.

        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            kdim: Dimension of keys (defaults to embed_dim)
            vdim: Dimension of values (defaults to embed_dim)
            dropout: Dropout probability
            bias: Whether to include bias in projections
            add_bias_kv: Whether to add bias to key and value
            add_zero_attn: Whether to add zero attention
            self_attention: Whether this is self-attention
            encoder_decoder_attention: Whether this is encoder-decoder attention
            q_noise: Quantization noise
            qn_block_size: Block size for quantization noise
            has_relative_attention_bias: Whether to use relative position bias
            num_buckets: Number of buckets for relative position embedding
            max_distance: Maximum distance for relative position embedding
            gru_rel_pos: Whether to use gated relative position
            rescale_init: Whether to use rescaled initialization
            use_flash_attn: Whether to use flash attention
        """
        super().__init__()
        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        k_bias = True
        if rescale_init:
            k_bias = False

        k_embed_dim = embed_dim
        q_embed_dim = embed_dim

        self.k_proj = quant_noise(nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size)

        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.initialize()

    def initialize(self) -> None:
        """Initiate parameters in the transformer model."""
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(
        self,
        relative_positions: torch.Tensor,
        bidirectional: bool = True,
    ) -> torch.Tensor:
        """Compute relative position buckets."""
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets = relative_buckets + (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.to(torch.get_default_dtype()) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )

        relative_buckets = relative_buckets + torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        """Compute relative position bias."""
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for multi-head attention.

        Input shape: Time x Batch x Channel

        Args:
            query: Query tensor
            key: Key tensor (optional)
            value: Value tensor (optional)
            key_padding_mask: Mask for keys that are pads
            incremental_state: For incremental decoding
            need_weights: Whether to return attention weights
            static_kv: Whether key/value are static
            attn_mask: Attention mask
            before_softmax: Return raw attention weights before softmax
            need_head_weights: Return per-head attention weights
            position_bias: Pre-computed position bias

        Returns:
            Tuple of (attention output, attention weights, position bias)
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = (
                position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).contiguous().view(bsz * self.num_heads, tgt_len, src_len)
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        alpha = 32
        if not self.use_flash_attn:
            q *= self.scaling
            q *= 1 / alpha

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.q_head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.k_head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.contiguous().view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.contiguous().view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[torch.Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.contiguous().view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.contiguous().view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None
        assert k.size(1) == src_len

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len = src_len + 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)],
                    dim=1,
                )

        # Flash attention path
        if self.use_flash_attn:
            assert not before_softmax, "Flash attention does not support before_softmax"
            assert not need_weights, "Flash attention does not support returning attention weights"

            if is_torch_v25_to_v26:
                q = q.contiguous().view(bsz, self.num_heads, tgt_len, self.q_head_dim)
                k = k.contiguous().view(bsz, self.num_heads, src_len, self.k_head_dim)
                v = v.contiguous().view(bsz, self.num_heads, src_len, self.head_dim)

            if key_padding_mask is not None:
                assert attn_mask is None, "key_padding_mask not supported with attn_mask"
                attn_mask = key_padding_mask == 0
                attn_mask = attn_mask.unsqueeze(1).expand(-1, tgt_len, -1)

            if attn_mask is not None:
                if is_torch_v25_to_v26:
                    attn_mask = attn_mask.unsqueeze(1)
                else:
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                    attn_mask = attn_mask.contiguous().view(bsz * self.num_heads, tgt_len, src_len)

            if position_bias is not None:
                attn_mask_rel_pos = position_bias
                if self.gru_rel_pos == 1:
                    query_layer = q.contiguous().view(bsz, self.num_heads, tgt_len, self.q_head_dim)
                    _B, _H, _L, __ = query_layer.size()
                    gate_a, gate_b = torch.sigmoid(
                        self.grep_linear(query_layer).contiguous().view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)
                    ).chunk(2, dim=-1)
                    gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                    attn_mask_rel_pos = gate_a_1.contiguous().view(bsz * self.num_heads, tgt_len, 1) * position_bias
                attn_mask_rel_pos = (
                    attn_mask_rel_pos.contiguous().view(bsz * self.num_heads, tgt_len, src_len).contiguous()
                )

                if is_torch_v25_to_v26:
                    attn_mask_rel_pos = attn_mask_rel_pos.unsqueeze(1)
                    attn_mask_rel_pos = attn_mask_rel_pos.view(bsz, self.num_heads, tgt_len, src_len).contiguous()

                if attn_mask is not None:
                    attn_mask_ = torch.zeros_like(attn_mask) + attn_mask_rel_pos
                    attn_mask_ = attn_mask_.masked_fill(attn_mask.logical_not(), float("-inf"))
                    attn_mask = attn_mask_
                else:
                    attn_mask = attn_mask_rel_pos

            attn = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_module.p if self.training else 0.0,
                is_causal=False,
            )
            if is_torch_v25_to_v26:
                attn = attn.permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)
            else:
                attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            attn = self.out_proj(attn)
            return attn, None, None

        # Original BEATs implementation
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]) * alpha
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.contiguous().view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.contiguous().view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos == 1:
                query_layer = (
                    q.contiguous().view(bsz, self.num_heads, tgt_len, self.q_head_dim) * alpha / self.scaling
                )
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(
                    self.grep_linear(query_layer).contiguous().view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.contiguous().view(bsz * self.num_heads, tgt_len, 1) * position_bias

            attn_mask_rel_pos = attn_mask_rel_pos.contiguous().view(attn_weights.size())
            attn_weights = attn_weights + attn_mask_rel_pos

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights_out: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights_out = (
                attn_weights_float.contiguous().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            )
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[torch.Tensor],
        prev_key_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[torch.Tensor]:
        """Append previous key padding mask for incremental decoding."""
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Get input buffer for incremental decoding."""
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[torch.Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        buffer: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        """Set input buffer for incremental decoding."""
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(
        self,
        attn_weights: torch.Tensor,
        tgt_len: int,
        src_len: int,
        bsz: int,
    ) -> torch.Tensor:
        """Apply sparse mask to attention weights (no-op by default)."""
        return attn_weights

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """Get incremental state."""
        if incremental_state is None:
            return None
        return incremental_state.get(key)

    def set_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        key: str,
        value: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        """Set incremental state."""
        incremental_state[key] = value
        return incremental_state
