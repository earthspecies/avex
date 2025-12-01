"""OpenBEATs transformer backbone implementation.

This module provides the transformer backbone architecture for OpenBEATs including
the encoder, attention mechanisms, and positional encoding.

Based on:
- Paper: https://arxiv.org/abs/2507.14129 (OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder)
- Original BEATs: https://arxiv.org/abs/2212.09058
- Copyright (c) 2022 Microsoft, Licensed under The MIT License
"""

import logging
import math
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from .modules import (
    GLU_Linear,
    GradMultiply,
    MultiheadAttention,
    SamePad,
    get_activation_fn,
    init_bert_params,
)

logger = logging.getLogger(__name__)


class TransformerEncoder(nn.Module):
    """Transformer encoder for OpenBEATs model."""

    def __init__(self, args: Any) -> None:
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

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        # Get flash attention setting
        use_flash_attn = getattr(args, "use_flash_attn", False)

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
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
                    use_flash_attn=use_flash_attn,
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

        self.apply(init_bert_params)

        if args.deep_norm:
            logger.info("Deep Norm is applied.")
            deep_norm_beta = math.pow(8 * args.encoder_layers, -1 / 4)
            for i in range(args.encoder_layers):
                nn.init.xavier_normal_(self.layers[i].self_attn.k_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.v_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].self_attn.q_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.out_proj.weight, gain=deep_norm_beta)
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
                    need_weights=False,
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


class TransformerSentenceEncoderLayer(nn.Module):
    """Transformer sentence encoder layer for OpenBEATs model."""

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
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
        encoder_layers: int = 0,
        use_flash_attn: bool = False,
    ) -> None:
        """Initialize TransformerSentenceEncoderLayer.

        Args:
            embedding_dim: Embedding dimension
            ffn_embedding_dim: FFN embedding dimension
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            activation_dropout: Activation dropout probability
            activation_fn: Activation function name
            layer_norm_first: Whether to apply layer norm first
            deep_norm: Whether to use deep norm
            has_relative_attention_bias: Whether to use relative attention bias
            num_buckets: Number of buckets for relative position
            max_distance: Maximum distance for relative position
            rescale_init: Whether to use rescaled initialization
            gru_rel_pos: Whether to use GRU relative position
            encoder_layers: Number of encoder layers
            use_flash_attn: Whether to use flash attention
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
            use_flash_attn=use_flash_attn,
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
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        pos_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass through transformer sentence encoder layer.

        Args:
            x: Input tensor
            self_attn_mask: Optional self-attention mask
            self_attn_padding_mask: Optional self-attention padding mask
            need_weights: Whether to return attention weights
            pos_bias: Optional position bias

        Returns:
            Tuple of (output, attention weights, position bias)
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
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
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
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

        return x, attn, pos_bias
