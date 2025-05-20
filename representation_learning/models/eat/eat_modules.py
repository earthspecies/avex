import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    MultiheadAttention,  # Needed for init_bert_params matching ref
)

# -----------------------------------------------------------------------------
# Lightweight replacements for a few fairseq utility layers
# -----------------------------------------------------------------------------


class LayerNorm(nn.LayerNorm):
    """Direct alias so existing calls keep the same name."""


class SamePad(nn.Module):
    """Identity pad layer – PyTorch convs with explicit `padding=kernel//2` already
    give the SAME output length for odd kernels.  For even kernels, you could
    trim/extend here, but EAT only uses odd sizes (5), so this is a no-op."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.k = kernel_size

    def forward(self, x):
        return x


class SamePad2d(SamePad):
    pass


class TransposeLast(nn.Module):
    """Transpose a given dimension with the last dimension (default swaps −2 & −1)."""

    def __init__(self, tranpose_dim: int = -2):
        super().__init__()
        self.d = tranpose_dim

    def forward(self, x):
        return x.transpose(self.d, -1)


# -----------------------------------------------------------------------------
# original fairseq init_bert_params
# -----------------------------------------------------------------------------


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
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
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


# -----------------------------------------------------------------------------
# Original EAT decoder & attention blocks – only fairseq deps removed
# -----------------------------------------------------------------------------


@dataclass
class D2vDecoderConfig:
    decoder_dim: int = 384
    decoder_groups: int = 16
    decoder_kernel: int = 5
    decoder_layers: int = 5
    input_dropout: float = 0.1

    add_positions_masked: bool = False
    add_positions_all: bool = False

    decoder_residual: bool = True
    projection_layers: int = 1
    projection_ratio: float = 2.0


# -----------------------------------------------------------------------------
# Helper base class -----------------------------------------------------------
# -----------------------------------------------------------------------------


class DecoderBase(nn.Module):
    decoder_cfg: D2vDecoderConfig

    def __init__(self, cfg: D2vDecoderConfig):
        super().__init__()
        self.decoder_cfg = cfg

    def reset_parameters(self):
        for mod in self.proj.modules():
            if isinstance(mod, nn.Linear):
                init_bert_params(mod)

    def add_residual(self, x, residual, i, mask_info):
        if (
            residual is None
            or not self.decoder_cfg.decoder_residual
            or residual.size(1) != x.size(1)
        ):
            return x
        return x + residual


# -----------------------------------------------------------------------------
# 1-D CNN decoder -------------------------------------------------------------
# -----------------------------------------------------------------------------


class Decoder1d(DecoderBase):
    def __init__(self, cfg: D2vDecoderConfig, input_dim: int):
        super().__init__(cfg)

        def make_block(in_dim):
            return nn.Sequential(
                nn.Conv1d(
                    in_dim,
                    cfg.decoder_dim,
                    kernel_size=cfg.decoder_kernel,
                    padding=cfg.decoder_kernel // 2,
                    groups=cfg.decoder_groups,
                ),
                SamePad(cfg.decoder_kernel),
                TransposeLast(),  # (B, C, T) -> (B, T, C)
                LayerNorm(cfg.decoder_dim, elementwise_affine=False),
                TransposeLast(),  # back to (B, C, T)
                nn.GELU(),
            )

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else cfg.decoder_dim)
                for i in range(cfg.decoder_layers)
            ]
        )

        # projection head(s)
        projs = []
        curr_dim = cfg.decoder_dim
        for i in range(cfg.projection_layers - 1):
            next_dim = int(curr_dim * cfg.projection_ratio) if i == 0 else curr_dim
            projs.extend([nn.Linear(curr_dim, next_dim), nn.GELU()])
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, input_dim))
        self.proj = projs[0] if len(projs) == 1 else nn.Sequential(*projs)

    def forward(self, x, mask_info=None):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        residual = x
        for blk in self.blocks:
            x = blk(x)
            x = self.add_residual(x, residual, None, mask_info)
            residual = x
        x = x.transpose(1, 2)  # back to (B, T, C)
        return self.proj(x)


# -----------------------------------------------------------------------------
# 2-D CNN decoder -------------------------------------------------------------
# -----------------------------------------------------------------------------


class Decoder2d(DecoderBase):
    def __init__(self, cfg: D2vDecoderConfig, input_dim: int, h_size: int, w_size: int):
        super().__init__(cfg)
        self.h_size, self.w_size = h_size, w_size

        def make_block(in_dim):
            return nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    cfg.decoder_dim,
                    kernel_size=cfg.decoder_kernel,
                    padding=cfg.decoder_kernel // 2,
                    groups=cfg.decoder_groups,
                ),
                SamePad2d(cfg.decoder_kernel),
                TransposeLast(tranpose_dim=-3),  # swap C & H for LayerNorm
                LayerNorm(cfg.decoder_dim, elementwise_affine=False),
                TransposeLast(tranpose_dim=-3),
                nn.GELU(),
            )

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else cfg.decoder_dim)
                for i in range(cfg.decoder_layers)
            ]
        )
        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def forward(self, x, mask_info=None):
        # TODO: Changed – pad sequence when masked tokens removed so reshape is possible
        B, T, C = x.shape  # expected T == H*W for grid reshape
        expected_tokens = self.h_size * self.w_size
        if T < expected_tokens:
            pad_tokens = expected_tokens - T
            padding = x.new_zeros(B, pad_tokens, C)
            x = torch.cat([x, padding], dim=1)
            T = expected_tokens
        elif T > expected_tokens:
            # In rare cases extra tokens – truncate for now
            x = x[:, :expected_tokens, :]
            T = expected_tokens

        x = x.transpose(1, 2).reshape(B, C, self.h_size, self.w_size)
        residual = x
        for blk in self.blocks:
            x = blk(x)
            x = self.add_residual(x, residual, None, mask_info)
            residual = x
        x = x.reshape(B, -1, T).transpose(1, 2)
        return self.proj(x)


# -----------------------------------------------------------------------------
# Transformer-based decoders --------------------------------------------------
# -----------------------------------------------------------------------------


class TransformerDecoder(nn.Module):
    decoder_cfg: D2vDecoderConfig

    def __init__(self, cfg: D2vDecoderConfig, input_dim: int, encoder: nn.Module):
        super().__init__()
        self.decoder_cfg = cfg
        self.input_proj = nn.Linear(input_dim, cfg.decoder_dim)
        self.encoder = encoder  # expects (B, T, D)
        self.proj = nn.Linear(cfg.decoder_dim, input_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_bert_params)

    def forward(self, x, mask_info=None):
        x = self.input_proj(x)
        x = self.encoder(x, None, None, 1)  # reuse shared encoder
        return self.proj(x)


# -----------------------------------------------------------------------------
#   Alternate attention / block utilities (identical to original code) --------
# -----------------------------------------------------------------------------


class AltAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        cosine_attention: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cosine_attention = cosine_attention
        if cosine_attention:
            self.logit_scale = nn.Parameter(
                torch.log(torch.ones((num_heads, 1, 1)) * 10)
            )

    def forward(self, x, padding_mask=None, alibi_bias=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.cosine_attention:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, : alibi_bias.size(1)] += alibi_bias
        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf")
            )
        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class AltBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        post_mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_norm_first: bool = True,
        ffn_targets: bool = False,
        cosine_attention: bool = False,
    ):
        super().__init__()
        from timm.models.vision_transformer import DropPath, Mlp

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets
        self.norm1 = norm_layer(dim)
        self.attn = AltAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop)

    def forward(self, x, padding_mask=None, alibi_bias=None):
        if self.layer_norm_first:
            x = x + self.drop_path(self.attn(self.norm1(x), padding_mask, alibi_bias))
            residual = x
            trg = x = self.mlp(self.norm2(x))
            x = residual + self.drop_path(self.post_mlp_dropout(x))
            if not self.ffn_targets:
                trg = x
        else:
            x = x + self.drop_path(self.attn(x, padding_mask, alibi_bias))
            residual = self.norm1(x)
            trg = x = self.mlp(residual)
            x = self.norm2(residual + self.drop_path(self.post_mlp_dropout(x)))
            if not self.ffn_targets:
                trg = x
        return x, trg


# -----------------------------------------------------------------------------
# Enc-Dec attention blocks (unchanged except fairseq removal) ------------------
# -----------------------------------------------------------------------------


class EncDecAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        cosine_attention: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = q_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q_proj = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * q_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(q_dim, q_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cosine_attention = cosine_attention
        if cosine_attention:
            self.logit_scale = nn.Parameter(
                torch.log(torch.ones((num_heads, 1, 1)) * 10)
            )

    def forward(self, q, kv, padding_mask=None, alibi_bias=None):
        B, N, C = q.shape
        q = (
            self.q_proj(q)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv_proj(kv)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        if self.cosine_attention:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, : alibi_bias.size(1)] += alibi_bias
        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf")
            )
        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class EncDecBlock(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        post_mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_norm_first: bool = True,
        cosine_attention: bool = False,
        first_residual: bool = True,
    ):
        super().__init__()
        from timm.models.vision_transformer import DropPath, Mlp

        self.layer_norm_first = layer_norm_first
        self.first_residual = first_residual
        self.norm1 = norm_layer(q_dim)
        self.attn = EncDecAttention(
            q_dim,
            kv_dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            drop,
            cosine_attention,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(q_dim)
        hidden_dim = int(q_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=q_dim,
            hidden_features=hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop)

    def forward(self, q, kv, padding_mask=None, alibi_bias=None):
        res = q if self.first_residual else 0
        if self.layer_norm_first:
            q = res + self.drop_path(
                self.attn(self.norm1(q), kv, padding_mask, alibi_bias)
            )
            res = q = self.mlp(self.norm2(q))
            q = res + self.drop_path(self.post_mlp_dropout(q))
        else:
            q = res + self.drop_path(self.attn(q, kv, padding_mask, alibi_bias))
            res = self.norm1(q)
            q = self.mlp(res)
            q = self.norm2(res + self.drop_path(self.post_mlp_dropout(q)))
        return q


class EncDecTransformerDecoder(nn.Module):
    def __init__(self, cfg: D2vDecoderConfig, input_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, cfg.decoder_dim)
        self.blocks = nn.Sequential(
            *[
                EncDecBlock(
                    q_dim=cfg.decoder_dim,
                    kv_dim=input_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    layer_norm_first=False,
                    first_residual=i > 0,
                )
                for i in range(cfg.decoder_layers)
            ]
        )
        self.proj = nn.Linear(cfg.decoder_dim, input_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_bert_params)

    def forward(self, x, kv):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x, kv)
        return self.proj(x)
