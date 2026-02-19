"""BEATs transformer backbone implementation.

This module provides the transformer backbone architecture for BEATs including
the encoder, attention mechanisms, and positional encoding.

Based on:
- Paper: https://arxiv.org/abs/2212.09058
- Original implementation: https://github.com/microsoft/unilm/tree/master/beats
- Copyright (c) 2022 Microsoft, Licensed under The MIT License
"""

# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import LayerNorm

from .modules import (
    GLU_Linear,
    GradMultiply,
    SamePad,
    get_activation_fn,
)


class TransformerEncoder(nn.Module):
    """Transformer encoder for BEATs model."""

    def __init__(self, args: Any) -> None:  # noqa: ANN401
        """Initialize TransformerEncoder.

        Args:
            args: Configuration object containing encoder parameters
        """
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        # Use parametrizations.weight_norm instead of deprecated utils.weight_norm
        from torch.nn.utils import parametrizations

        self.pos_conv = parametrizations.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList(
            [
                _TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    deep_norm=args.deep_norm,
                    has_relative_attention_bias=self.relative_position_embedding,
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=args.gru_rel_pos,
                    encoder_layers=args.encoder_layers,
                )
                for i in range(args.encoder_layers)
            ]
        )
        if self.relative_position_embedding:
            for i in range(1, args.encoder_layers):
                del self.layers[i].self_attn.relative_attention_bias
                self.layers[i].self_attn.relative_attention_bias = self.layers[0].self_attn.relative_attention_bias

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(_init_bert_params)

        if args.deep_norm:
            deep_norm_beta = math.pow(8 * args.encoder_layers, -1 / 4)
            for i in range(args.encoder_layers):
                nn.init.xavier_normal_(self.layers[i].self_attn.k_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.v_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].self_attn.q_proj.weight, gain=1)
                nn.init.xavier_normal_(
                    self.layers[i].self_attn.out_proj.weight,
                    gain=deep_norm_beta,
                )
                nn.init.xavier_normal_(self.layers[i].fc1.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc2.weight, gain=deep_norm_beta)

        self.layer_wise_gradient_decay_ratio = getattr(args, "layer_wise_gradient_decay_ratio", 1)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        layer: Optional[int] = None,
        disable_layerdrop: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass through the transformer encoder.

        Args:
            x: Input tensor
            padding_mask: Optional padding mask
            layer: Optional target layer index
            disable_layerdrop: Whether to disable layerdrop during forward pass

        Returns:
            Tuple[torch.Tensor, list]: Encoded features and layer results
        """
        x, layer_results = self.extract_features(x, padding_mask, layer, disable_layerdrop)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        tgt_layer: Optional[int] = None,
        disable_layerdrop: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """Extract features from input through transformer layers.

        Args:
            x: Input tensor
            padding_mask: Optional padding mask
            tgt_layer: Optional target layer to stop at
            disable_layerdrop: Whether to disable layerdrop during forward pass

        Returns:
            Tuple[torch.Tensor, list]: Extracted features and layer results
        """
        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            if self.layer_wise_gradient_decay_ratio != 1.0:
                x = GradMultiply.apply(x, self.layer_wise_gradient_decay_ratio)
            if disable_layerdrop:
                should_execute = True
            else:
                dropout_probability = np.random.random()
                should_execute = not self.training or (dropout_probability > self.layerdrop)

            if should_execute:
                x, z, pos_bias = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    pos_bias=pos_bias,
                )
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class _TransformerSentenceEncoderLayer(nn.Module):
    """Transformer sentence encoder layer for BEATs model."""

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        deep_norm: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        gru_rel_pos: bool = False,
        encoder_layers: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = _MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.deep_norm = deep_norm
        if self.deep_norm:
            self.deep_norm_alpha = math.pow(2 * encoder_layers, 1 / 4)
        else:
            self.deep_norm_alpha = 1

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        pos_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None, Optional[torch.Tensor]]:
        """Forward pass through transformer sentence encoder layer.

        Args:
            x: Input tensor of shape ``(T, B, E)``
            self_attn_padding_mask: Optional padding mask of shape ``(B, T)``
            pos_bias: Optional precomputed position bias of shape ``(B, H, T, T)``

        Returns:
            Tuple of (output, None, position_bias)
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, _, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, _, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual * self.deep_norm_alpha + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual * self.deep_norm_alpha + x
            x = self.final_layer_norm(x)

        return x, None, pos_bias


class _MultiheadAttention(nn.Module):
    """Multi-headed attention using PyTorch scaled_dot_product_attention (SDPA).

    Replaces the original fairseq-era manual attention with the fused SDPA kernel
    which dispatches to Flash Attention 2 / memory-efficient attention automatically.
    The GRU-gated relative position bias is precomputed and passed as an additive mask.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        self_attention: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        gru_rel_pos: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.dropout_p = dropout

        self.self_attention = self_attention

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset model parameters with Xavier initialization."""
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
        """Convert relative positions to bucket indices.

        Args:
            relative_positions: Relative position tensor
            bidirectional: Whether to use bidirectional buckets

        Returns:
            torch.Tensor: Bucket indices
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        """Compute relative position bias.

        Args:
            query_length: Length of query sequence
            key_length: Length of key sequence

        Returns:
            torch.Tensor: Position bias tensor of shape ``[num_heads, Q, K]``
        """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, None, Optional[Tensor]]:
        """Multi-head attention forward pass using SDPA.

        Input shape: Time x Batch x Channel

        Args:
            query: Query tensor of shape ``(T, B, E)``
            key: Key tensor (unused for self-attention, kept for API compat)
            value: Value tensor (unused for self-attention, kept for API compat)
            key_padding_mask: Boolean mask of shape ``(B, T)`` where True = padded
            need_weights: Ignored (SDPA does not materialise attention weights)
            attn_mask: Ignored (use position_bias for additive biases)
            position_bias: Precomputed relative position bias of shape
                ``[B, H, T, T]``, or ``None`` to compute from scratch on layer 0

        Returns:
            Tuple of (output, None, position_bias) where output has shape ``(T, B, E)``
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim

        # Compute relative position bias on first layer; subsequent layers reuse it
        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).expand(bsz, -1, -1, -1)  # [B, H, T, T]

        # Project Q, K, V from the same input (self-attention)
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        # Reshape [T, B, E] -> [B, H, T, D] for SDPA
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Build additive attention mask combining position bias and padding
        sdpa_mask: Optional[Tensor] = None

        if position_bias is not None:
            if self.gru_rel_pos:
                # GRU-gated relative position bias: gate is a function of raw Q
                _B, _H, _L, _D = q.size()
                gate_a, gate_b = torch.sigmoid(self.grep_linear(q).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(
                    2, dim=-1
                )
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                sdpa_mask = gate_a_1 * position_bias  # [B,H,T,1] * [B,H,T,T]
            else:
                sdpa_mask = position_bias

        if key_padding_mask is not None:
            pad_mask = torch.zeros(bsz, 1, 1, src_len, device=q.device, dtype=q.dtype)
            pad_mask.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))
            sdpa_mask = sdpa_mask + pad_mask if sdpa_mask is not None else pad_mask

        dropout_p = self.dropout_p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            scale=self.scaling,
        )

        # [B, H, T, D] -> [T, B, E]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, None, position_bias


def _init_bert_params(module: nn.Module) -> None:
    """Initialize the weights specific to the BERT Model.

    Args:
        module: PyTorch module to initialize
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
    if isinstance(module, _MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)
