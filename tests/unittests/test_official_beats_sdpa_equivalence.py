"""Test numerical equivalence between legacy manual attention and SDPA-based attention.

This file is BEATs-specific. It verifies that the SDPA migration preserves
numerical behavior at both:
- isolated attention module level
- full BEATs end-to-end feature extraction
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Iterable
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avex.models.beats.backbone import MultiheadAttention as _MultiheadAttention
from avex.models.beats.beats import BEATs, BEATsConfig

logger = logging.getLogger(__name__)


BEATS_SSL_CONFIG = BEATsConfig(
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=3072,
    encoder_attention_heads=12,
    activation_fn="gelu",
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.0,
    encoder_layerdrop=0.05,
    dropout_input=0.1,
    layer_norm_first=False,
    deep_norm=True,
    conv_bias=False,
    conv_pos=128,
    conv_pos_groups=16,
    relative_position_embedding=True,
    num_buckets=320,
    max_distance=800,
    gru_rel_pos=True,
    input_patch_size=16,
    embed_dim=512,
    finetuned_model=False,
)

BEATS_FINETUNED_CONFIG = BEATsConfig(
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=3072,
    encoder_attention_heads=12,
    activation_fn="gelu",
    dropout=0.0,
    attention_dropout=0.0,
    activation_dropout=0.0,
    encoder_layerdrop=0.05,
    dropout_input=0.0,
    layer_norm_first=False,
    deep_norm=True,
    conv_bias=False,
    conv_pos=128,
    conv_pos_groups=16,
    relative_position_embedding=True,
    num_buckets=320,
    max_distance=800,
    gru_rel_pos=True,
    input_patch_size=16,
    embed_dim=512,
    layer_wise_gradient_decay_ratio=0.6,
    finetuned_model=True,
    predictor_dropout=0.0,
    predictor_class=527,
)


class _LegacyMultiheadAttention(nn.Module):
    """Original fairseq-era manual attention implementation for reference testing."""

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
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        self.scaling = self.head_dim**-0.5
        self.self_attention = self_attention
        self.dropout_module = nn.Dropout(dropout)

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

    def _relative_positions_bucket(self, relative_positions: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
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
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)

        assert self.self_attention
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling
        alpha = 32
        q *= 1 / alpha

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.q_head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.k_head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]) * alpha

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos == 1:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim) * alpha / self.scaling
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(
                    self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, tgt_len, 1) * position_bias

            attn_mask_rel_pos = attn_mask_rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + attn_mask_rel_pos

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        return attn, None, position_bias


def _make_attention_pair(
    embed_dim: int = 768,
    num_heads: int = 12,
    has_relative_attention_bias: bool = True,
    gru_rel_pos: bool = True,
) -> Tuple[_LegacyMultiheadAttention, _MultiheadAttention]:
    kwargs = dict(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        self_attention=True,
        has_relative_attention_bias=has_relative_attention_bias,
        num_buckets=320,
        max_distance=800,
        gru_rel_pos=gru_rel_pos,
    )
    legacy = _LegacyMultiheadAttention(**kwargs)
    new = _MultiheadAttention(**kwargs)
    new.load_state_dict(legacy.state_dict())
    legacy.eval()
    new.eval()
    return legacy, new


def _swap_to_legacy_attention(model: BEATs) -> BEATs:
    encoder = model.encoder
    for layer in encoder.layers:
        old_attn = layer.self_attn
        legacy_attn = _LegacyMultiheadAttention(
            embed_dim=old_attn.embed_dim,
            num_heads=old_attn.num_heads,
            dropout=getattr(old_attn, "dropout_p", old_attn.dropout_module.p),
            self_attention=True,
            has_relative_attention_bias=old_attn.has_relative_attention_bias,
            num_buckets=old_attn.num_buckets,
            max_distance=old_attn.max_distance,
            gru_rel_pos=old_attn.gru_rel_pos,
        )
        legacy_attn.load_state_dict(old_attn.state_dict())
        layer.self_attn = legacy_attn

    if encoder.relative_position_embedding:
        for i in range(1, len(encoder.layers)):
            del encoder.layers[i].self_attn.relative_attention_bias
            encoder.layers[i].self_attn.relative_attention_bias = encoder.layers[0].self_attn.relative_attention_bias
    return model


ATOL = 1e-5
RTOL = 1e-4


class TestAttentionLayerEquivalence:
    def test_basic_no_mask_no_pos_bias(self) -> None:
        legacy, new = _make_attention_pair(has_relative_attention_bias=False, gru_rel_pos=False)
        x = torch.randn(50, 2, 768)
        with torch.no_grad():
            out_legacy, _, _ = legacy(query=x, key=x, value=x)
            out_new, _, _ = new(query=x, key=x, value=x)
        torch.testing.assert_close(out_new, out_legacy, atol=ATOL, rtol=RTOL)


class TestPretrainedCheckpointEquivalence:
    E2E_ATOL = 5e-4

    def test_pretrained_ssl_equivalence(self) -> None:
        from avex.models.beats_model import _get_beats_checkpoint_path
        from avex.utils import universal_torch_load

        path = _get_beats_checkpoint_path(use_naturelm=False, fine_tuned=False)
        try:
            ckpt = universal_torch_load(path, cache_mode="use", map_location="cpu")
        except Exception as e:
            # CI runners may not have credentials to access private GCS buckets.
            msg = str(e)
            skip_tokens: Iterable[str] = (
                "Anonymous caller",
                "storage.objects.get",
                "Permission",
                "401",
                "403",
            )
            if any(tok in msg for tok in skip_tokens):
                pytest.skip(f"Skipping: cannot access checkpoint at {path!r} ({msg})")
            raise
        cfg = BEATsConfig(**ckpt["cfg"])
        weights = ckpt["model"]

        sdpa_model = BEATs(cfg)
        sdpa_model.load_state_dict(weights, strict=False)
        sdpa_model.eval()

        legacy_model = copy.deepcopy(sdpa_model)
        _swap_to_legacy_attention(legacy_model)
        legacy_model.eval()

        audio = torch.randn(2, 32_000)
        with torch.no_grad():
            out_sdpa, _ = sdpa_model.extract_features(audio, feature_only=True, disable_layerdrop=True)
            out_legacy, _ = legacy_model.extract_features(audio, feature_only=True, disable_layerdrop=True)

        max_diff = (out_sdpa - out_legacy).abs().max().item()
        assert max_diff < self.E2E_ATOL
