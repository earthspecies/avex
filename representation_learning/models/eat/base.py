# base.py  – Fairseq-free re-implementation
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT license.

import logging
import math
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING, II
from torch.cuda.amp import autocast

from .modules import D2vDecoderConfig  # our stripped-down decoders

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Minimal substitutes for three Fairseq helpers                              #
# --------------------------------------------------------------------------- #


class GradMultiply(torch.autograd.Function):
    """Forward identity – back-prop multiplies gradients by *scale*."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.scale, None


def index_put(t: torch.Tensor, mask: torch.Tensor, values: torch.Tensor):
    """Like `t[mask] = values` but keeps shape-broadcasting behaviour used
    in the original Fairseq code."""
    t = t.clone()
    t[mask] = (
        values.view(-1, t.size(-1)) if values.dim() == t.dim() else values
    )
    return t


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    min_masks: int = 1,
    require_same_masks: bool = True,  # ignored (only needed for MT)
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: int = 0,
    indices: Optional[torch.Tensor] = None,
):
    """
    Return a (B, T) numpy bool array where *True* marks masked positions.
    This simplified version covers the usage patterns in EAT: non-overlapping
    spans of length *mask_length* until ~mask_prob of tokens are masked.
    """
    B, T = shape
    rng = np.random.RandomState(seed + epoch if seed is not None else None)
    out = np.zeros((B, T), dtype=np.bool_)

    for b in range(B):
        valid = (
            (~padding_mask[b]).sum().item() if padding_mask is not None else T
        )
        n_span = max(min_masks, int(round(mask_prob * valid / mask_length)))
        n_span = min(n_span, valid // mask_length)

        starts = rng.permutation(valid - mask_length + 1)[:n_span]
        for s in starts:
            out[b, s : s + mask_length] = True

        if padding_mask is not None:
            out[b] &= ~padding_mask[b].cpu().numpy()

        if mask_dropout > 0:
            drop = rng.rand(T) < mask_dropout
            out[b] &= ~drop

    return out


# --------------------------------------------------------------------------- #
#  Public enums & dataclasses                                                 #
# --------------------------------------------------------------------------- #

class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


@dataclass
class D2vModalityConfig:
    # architecture -------------------------------------------------------
    type: Modality = MISSING
    prenet_depth: int = 4
    prenet_layerdrop: float = 0.0
    prenet_dropout: float = 0.0
    start_drop_path_rate: float = 0.0
    end_drop_path_rate: float = 0.0

    # extra tokens -------------------------------------------------------
    num_extra_tokens: int = 0
    init_extra_token_zero: bool = True

    # time-masking -------------------------------------------------------
    mask_noise_std: float = 0.01
    mask_prob_min: Optional[float] = None
    mask_prob: float = 0.7
    inverse_mask: bool = False
    mask_prob_adjust: float = 0.0
    keep_masked_pct: float = 0.0
    mask_length: int = 5
    add_masks: bool = False
    remove_masks: bool = False
    mask_dropout: float = 0.0
    encoder_zero_mask: bool = True

    # channel-masking ----------------------------------------------------
    mask_channel_prob: float = 0.0
    mask_channel_length: int = 64

    # local vs. context encoder -----------------------------------------
    ema_local_encoder: bool = False
    local_grad_mult: float = 1.0

    # alibi settings -----------------------------------------------------
    use_alibi_encoder: bool = False
    alibi_scale: float = 1.0
    learned_alibi: bool = False
    alibi_max_pos: Optional[int] = None
    learned_alibi_scale: bool = False
    learned_alibi_scale_per_head: bool = False
    learned_alibi_scale_per_layer: bool = False
    num_alibi_heads: int = II("model.num_heads")
    model_depth: int = II("model.depth")

    # optional decoder ---------------------------------------------------
    decoder: Optional[D2vDecoderConfig] = field(default_factory=D2vDecoderConfig)


#  Mask bookkeeping structs ---------------------------------------------------

MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])
MaskInfo = namedtuple(
    "MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"]
)

# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #


def get_annealed_rate(start: float, end: float, step: int, total: int):
    """Linear decay from *start* → *end* over *total* steps."""
    if step >= total:
        return end
    return end - (end - start) * (1 - step / total)


def random_masking(
    x: torch.Tensor, mask_ratio: float, seed: Optional[MaskSeed]
) -> MaskInfo:
    """
    MAE-style random masking when `mask_length == 1` in the original code.
    """
    B, L, D = x.shape
    keep = int(L * (1 - mask_ratio))

    g = None
    if seed is not None:
        g = torch.Generator(device=x.device)
        g.manual_seed(
            int(hash((seed.seed, seed.update, seed.ids.sum().item())) % 1_000_000)
        )

    noise = torch.rand(B, L, generator=g, device=x.device)
    idx_shuffle = noise.argsort(dim=1)
    idx_restore = idx_shuffle.argsort(dim=1)

    idx_keep = idx_shuffle[:, :keep]
    idx_keep_exp = idx_keep.unsqueeze(-1).expand(-1, -1, D)
    x_keep = torch.gather(x, 1, idx_keep_exp)

    mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
    mask[:, :keep] = False
    mask = torch.gather(mask, 1, idx_restore)

    idx_restore = idx_restore.unsqueeze(-1).expand(-1, -1, D)
    return MaskInfo(x_keep, mask, idx_restore, idx_keep_exp)


def gather_unmasked(x: torch.Tensor, info: MaskInfo) -> torch.Tensor:
    return torch.gather(x, 1, info.ids_keep)


def gather_unmasked_mask(x: torch.Tensor, info: MaskInfo) -> torch.Tensor:
    return torch.gather(x, 1, info.ids_keep[..., 0])


# Alibi helpers ---------------------------------------------------------------

def _slopes(n):
    if math.log2(n).is_integer():
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]
    half = 2 ** math.floor(math.log2(n))
    return _slopes(half) + _slopes(2 * half)[0::2][: n - half]


def get_alibi(max_pos: int, heads: int) -> torch.Tensor:
    slopes = torch.tensor(_slopes(heads))
    pos = (
        torch.arange(max_pos)
        .unsqueeze(0)
        .sub(torch.arange(max_pos).unsqueeze(1))
        .abs()
        * -1
    )
    return slopes[:, None, None] * pos  # (H, T, T)


def get_alibi_bias(
    cache: dict,
    B: int,
    T: int,
    H: int,
    dtype,
    device,
):
    key = (H, )
    buf = cache.get(key)
    need = H * B
    if buf is None or buf.size(0) < need or buf.size(1) < T \
       or buf.dtype != dtype or buf.device != device:
        big_T = max(T, buf.size(1) if buf is not None else 0)
        big_BH = max(need, buf.size(0) if buf is not None else 0) // H
        buf = (
            get_alibi(big_T, H).to(dtype=dtype, device=device)
            .repeat(big_BH, 1, 1)
        )
        cache[key] = buf
    out = buf[:need, :T, :T].view(B, H, T, T)
    return out


def _learned_alibi_bias(
    alibi_bias: torch.Tensor,
    B: int,
    T: int,
    H: int,
    scale: torch.Tensor,
    dtype,
    device,
):
    if alibi_bias.size(-1) < T:
        pad = math.ceil((T - alibi_bias.size(-1)) / 2)
        alibi_bias = F.pad(alibi_bias, (pad, pad, pad, pad), mode="replicate")
    return (alibi_bias * scale).expand(B, -1, -1, -1)[..., :T, :T]


def masked_alibi(ab: torch.Tensor, info: MaskInfo) -> torch.Tensor:
    H = ab.size(1)
    idx = info.ids_keep[..., 0].unsqueeze(-1)
    tmp = torch.gather(
        ab, -2, idx.expand(-1, H, -1, ab.size(-1))
    )
    return torch.gather(
        tmp, -1, idx.transpose(-1, -2).expand(-1, H, tmp.size(-2), -1)
    )


# --------------------------------------------------------------------------- #
#  Core class: ModalitySpecificEncoder                                        #
# --------------------------------------------------------------------------- #

class ModalitySpecificEncoder(nn.Module):
    """
    Combines a local feature encoder (e.g. CNN/patchify) with a Transformer
    context encoder, masking logic, and (optionally) a decoder head.
    """

    def __init__(
        self,
        cfg: D2vModalityConfig,
        embed_dim: int,
        local_encoder: nn.Module,
        project_features: nn.Module,
        fixed_positional_encoder: Optional[nn.Module],
        relative_positional_encoder: Optional[nn.Module],
        context_encoder: nn.Module,
        decoder: Optional[nn.Module],
        get_alibi_bias: Optional[Callable[..., torch.Tensor]],
    ):
        super().__init__()
        self.cfg = cfg
        self.local_encoder = local_encoder
        self.project_features = project_features
        self.fixed_positional_encoder = fixed_positional_encoder
        self.relative_positional_encoder = relative_positional_encoder
        self.context_encoder = context_encoder
        self.decoder = decoder
        self.get_alibi_bias = (
            get_alibi_bias if cfg.use_alibi_encoder else None
        )
        self.local_grad_mult = cfg.local_grad_mult

        # extra tokens ----------------------------------------------------
        if cfg.num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(
                torch.zeros(1, cfg.num_extra_tokens, embed_dim)
            )
            if not cfg.init_extra_token_zero:
                nn.init.normal_(self.extra_tokens)
            elif self.extra_tokens.size(1) > 1:
                nn.init.normal_(self.extra_tokens[:, 1:])
        else:
            self.extra_tokens = None

        # learned alibi scale --------------------------------------------
        self.alibi_scale = None
        if self.get_alibi_bias is not None:
            layers = (
                cfg.prenet_depth + cfg.model_depth
                if cfg.learned_alibi_scale_per_layer
                else 1
            )
            heads = cfg.num_alibi_heads if cfg.learned_alibi_scale_per_head else 1
            shape = (layers, 1, heads, 1, 1)
            self.alibi_scale = nn.Parameter(
                torch.full(shape, cfg.alibi_scale), requires_grad=cfg.learned_alibi_scale
            )

        # learned alibi bias ---------------------------------------------
        if cfg.learned_alibi and self.get_alibi_bias is not None:
            assert cfg.alibi_max_pos is not None
            base = self.get_alibi_bias(
                {},
                1,
                cfg.alibi_max_pos,
                cfg.num_alibi_heads,
                torch.float32,
                "cpu",
            )
            self.alibi_bias = nn.Parameter(base)
            self.get_alibi_bias = partial(
                _learned_alibi_bias, alibi_bias=self.alibi_bias
            )

    # ------------------------------------------------------------------ #
    #  Local + context encoders                                          #
    # ------------------------------------------------------------------ #

    def local_features(self, feats: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=False):
            w_dtype = next(self.local_encoder.parameters()).dtype
            feats_cast = feats.to(dtype=w_dtype)
            if self.local_grad_mult == 1.0:
                x = self.local_encoder(feats_cast)
            elif self.local_grad_mult > 0:
                x = GradMultiply.apply(self.local_encoder(feats_cast), self.local_grad_mult)
            else:
                with torch.no_grad():
                    x = self.local_encoder(feats_cast)
        return self.project_features(x)

    def convert_padding_mask(self, x, pm):
        return pm

    # ------------------------------------------------------------------ #
    #  Mask-making helpers                                               #
    # ------------------------------------------------------------------ #

    def make_maskinfo(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        shape: Optional[Tuple[int, int, int]] = None,
    ) -> MaskInfo:
        if shape is None:
            B, T, D = x.shape
        else:
            B, T, D = shape

        mask = mask.bool()
        # CUDA kernels do not support argsort on bool tensors.  Cast to int64
        # so False→0, True→1 which preserves the ordering semantics.
        idx_shuffle = mask.long().argsort(dim=1)
        idx_restore = idx_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)

        keep = T - mask[0].sum()
        if self.cfg.keep_masked_pct > 0:
            keep += round((T - int(keep)) * self.cfg.keep_masked_pct)
        idx_keep = idx_shuffle[:, :keep]

        if shape is None:
            idx_keep_exp = idx_keep.unsqueeze(-1).expand(-1, -1, D)
            x_unmasked = torch.gather(x, 1, idx_keep_exp)
        else:
            x_unmasked = None
            idx_keep_exp = idx_keep

        return MaskInfo(x_unmasked, mask, idx_restore, idx_keep_exp)

    def apply_mask(self, x: torch.Tensor, info: Optional[MaskInfo]) -> torch.Tensor:
        cfg = self.cfg
        if info is not None:
            m = info.mask
            if cfg.encoder_zero_mask:
                x = x * (~m).unsqueeze(-1)
            else:
                n = m.sum().item()
                noise = x.new_empty(n, x.size(-1)).normal_(0, cfg.mask_noise_std)
                x = index_put(x, m, noise)

        if cfg.mask_channel_prob > 0:
            B, T, C = x.shape
            c_mask_np = compute_mask_indices(
                (B, C), None, cfg.mask_channel_prob, cfg.mask_channel_length
            )
            c_mask = (
                torch.from_numpy(c_mask_np)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x = index_put(x, c_mask, 0)
        return x

    def compute_mask(
        self,
        x: torch.Tensor,
        pm: Optional[torch.Tensor],
        seed: Optional[MaskSeed],
        apply: bool,
        precomputed_mask,
    ):
        if precomputed_mask is not None:
            mask = precomputed_mask
            info = self.make_maskinfo(x, mask)
        else:
            B, T, _ = x.shape
            cfg = self.cfg
            mprob = cfg.mask_prob
            if cfg.mask_prob_min is not None and 0 <= cfg.mask_prob_min < mprob:
                mprob = np.random.uniform(cfg.mask_prob_min, mprob)
            if cfg.inverse_mask:
                mprob = 1 - mprob

            if mprob > 0:
                if cfg.mask_length == 1:
                    info = random_masking(x, mprob, seed)
                else:
                    mask_np = compute_mask_indices(
                        (B, T),
                        pm,
                        mprob,
                        cfg.mask_length,
                        min_masks=1,
                        seed=seed.seed if seed else None,
                        epoch=seed.update if seed else 0,
                        indices=seed.ids if seed else None,
                        require_same_masks=True,
                        mask_dropout=cfg.mask_dropout,
                        add_masks=cfg.add_masks,
                    )
                    m_t = torch.from_numpy(mask_np).to(x.device)
                    if cfg.inverse_mask:
                        m_t = ~m_t
                    info = self.make_maskinfo(x, m_t)
            else:
                info = None

        if apply:
            x = self.apply_mask(x, info)

        return x, info

    # ------------------------------------------------------------------ #
    #  Contextualised forward                                            #
    # ------------------------------------------------------------------ #

    def contextualised_features(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        do_mask: bool,
        remove_masked: bool,
        clone: int = 1,
        seed: Optional[MaskSeed] = None,
        precomputed_mask=None,
    ):
        if padding_mask is not None:
            padding_mask = self.convert_padding_mask(x, padding_mask)

        local_feats = x.clone() if do_mask and clone == 1 else x

        # absolute pos-enc
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, padding_mask)[:, : x.size(1)]

        # masking --------------------------------------------------------
        if do_mask:
            if clone > 1:
                x = x.repeat_interleave(clone, 0)
                if seed is not None:
                    # de-correlate per-clone RNG seeds
                    offsets = torch.tensor(
                        [int(hash((seed.seed, i)) % 1e10) for i in range(clone)],
                        device=x.device,
                    )
                    ids = seed.ids.repeat_interleave(clone, 0).view(-1, clone)
                    ids += offsets.view(1, -1)
                    seed = MaskSeed(seed.seed, seed.update, ids.view(-1))
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat_interleave(clone, 0)

            x, info = self.compute_mask(
                x, padding_mask, seed, apply=True, precomputed_mask=precomputed_mask
            )
        else:
            info = None

        # relative pos-enc
        if self.relative_positional_encoder is not None:
            x = x + self.relative_positional_encoder(x)

        # remove masked tokens ------------------------------------------
        if do_mask and remove_masked:
            x = info.x_unmasked
            if padding_mask is not None:
                padding_mask = gather_unmasked_mask(padding_mask, info)
                if not padding_mask.any():
                    padding_mask = None

        # alibi ----------------------------------------------------------
        alibi_bias = None
        alibi_scale = self.alibi_scale
        if self.get_alibi_bias is not None:
            alibi_bias = self.get_alibi_bias(
                {},
                x.size(0) // clone,
                x.size(1),
                self.cfg.num_alibi_heads,
                torch.float32,
                x.device,
            )
            if alibi_scale is not None:
                alibi_bias = alibi_bias * alibi_scale.clamp_min(0).type_as(alibi_bias)
            if clone > 1:
                alibi_bias = alibi_bias.repeat_interleave(clone, 0)
            if do_mask and remove_masked:
                alibi_bias = masked_alibi(alibi_bias, info)

        # prepend extra tokens ------------------------------------------
        if self.extra_tokens is not None:
            n_ex = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], 1)
            if padding_mask is not None:
                padding_mask = F.pad(padding_mask, (n_ex, 0))
            if alibi_bias is not None:
                alibi_bias = F.pad(alibi_bias, (n_ex, 0, n_ex, 0))

        # transformer context encoder -----------------------------------
        x = self.context_encoder(x, padding_mask, alibi_bias, None)

        return {
            "x": x,
            "local_features": local_feats,
            "padding_mask": padding_mask,
            "alibi_bias": alibi_bias,
            "alibi_scale": alibi_scale,
            "encoder_mask": info,
        }

    # ------------------------------------------------------------------ #
    #  External API                                                      #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[MaskSeed] = None,
        precomputed_mask=None,
    ):
        x = self.local_features(features)
        return self.contextualised_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )

    def remove_pretraining_modules(self, keep_decoder: bool = False):
        if not keep_decoder:
            self.decoder = None

    # ------------------------------------------------------------------ #
    #  API alias – US vs UK spelling                                     #
    # ------------------------------------------------------------------ #

    def contextualized_features(self, *args, **kwargs):  # noqa: N802 – API alias
        """Alias for :meth:`contextualised_features` (US spelling).

        Converts the keyword ``mask`` → ``do_mask`` expected by the original
        implementation so legacy call-sites continue to work.
        """
        if "mask" in kwargs and "do_mask" not in kwargs:
            kwargs["do_mask"] = kwargs.pop("mask")
        return self.contextualised_features(*args, **kwargs)
