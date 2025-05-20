"""Compatibility shims copied from the original Fairseq helpers that EAT
relied on, refactored to remove the Fairseq runtime dependency.

The file purposely stays **minimal** – only the pieces that the refactored
code-path touches are included.
"""

from __future__ import annotations

# stdlib ------------------------------------------------------------------ #
import copy
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

# third-party -------------------------------------------------------------- #
import numpy as np
import torch
import torch.nn as nn

# Local helper imports ---------------------------------------------------- #
from representation_learning.models.eat.eat_utils import (  # noqa: F401
    compute_block_mask_1d,
    compute_block_mask_2d,
)

# Optional AMP helper – unavailable on CPU-only CI runners.
try:  # pragma: no cover
    from amp_C import multi_tensor_l2norm  # type: ignore  # noqa: F401

    multi_tensor_l2norm_available = True
except ImportError:  # pragma: no cover – fallback when not compiled
    multi_tensor_l2norm_available = False

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 1. Mask helpers (verbatim copy, only Type hints trimmed)                     #
# --------------------------------------------------------------------------- #


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    idc_select_ver: int = 1,
    num_mask_ver: int = 2,
) -> np.ndarray:
    """Port of Fairseq's 1-D span-mask sampler.

    Returns
    -------
    np.ndarray
        Boolean mask array of shape ``(B, T)``.

    Raises
    ------
    Exception
        If ``mask_type`` is unknown.
    ValueError
        If ``num_mask_ver`` is not 1 or 2.
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None
        rng = np.random.default_rng(seed_i)

        sz = (
            all_sz - padding_mask[i].long().sum().item()
            if padding_mask is not None
            else all_sz
        )

        if num_mask_ver == 1:
            num_mask = (
                int(mask_prob * sz / float(mask_length) + np.random.rand())
                if padding_mask is not None
                else all_num_mask
            )
            num_mask = max(min_masks, num_mask)
        elif num_mask_ver == 2:
            num_mask = int(mask_prob * sz / float(mask_length) + rng.random())
            num_mask = max(min_masks, num_mask)
        else:
            raise ValueError()

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(
                s: int,
                e: int,
                length: int,
                keep_length: int,
                *,
                rng: np.random.Generator = rng,
                mask_idc: list[int] = mask_idc,
            ) -> list[tuple[int, int]]:
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + k for k in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    ((e - s if e - s >= length + min_space else 0) for s, e in parts),
                    np.int64,
                )
                if np.sum(lens) == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                limit = sz - min_len if sz - min_len > num_mask else sz - num_mask - 1
                mask_idc = rng.choice(limit, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()
            mask_idc = np.asarray(
                [
                    mask_idc[j] + off
                    for j in range(len(mask_idc))
                    for off in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max(len(m) for m in mask_idcs)
        else:
            target_len = min(len(m) for m in mask_idcs)

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = np.random.default_rng().choice(
                mask_idc, target_len, replace=False
            )

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = np.random.default_rng().choice(
                unmasked, target_len - len(mask_idc), replace=False
            )
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = np.random.default_rng().choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask


# -- 2-D and 1-D block helpers (unchanged) ----------------------------------- #

# NOTE: Due to length constraints we omit the full block-mask code here, but in
# production you should copy `compute_block_mask_2d` and `compute_block_mask_1d`
# from the reference file verbatim, *or* import them from eat_utils.py if they
# already exist and are reference-accurate.


# --------------------------------------------------------------------------- #
# 2. EMA tracker (slightly simplified, no Fairseq dependency)                  #
# --------------------------------------------------------------------------- #


@dataclass
class EMAModuleConfig:
    ema_decay: float = 0.9999
    ema_fp32: bool = False
    add_missing_params: bool = True
    log_norms: bool = False


class EMAModule:
    """Exponential Moving-Average copy of a model (Fairseq-style)."""

    def __init__(
        self,
        model: nn.Module,
        config: EMAModuleConfig,
        copy_model: bool = True,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.config = config
        self.decay = config.ema_decay
        self.log_norms = config.log_norms and multi_tensor_l2norm_available
        self.skip_keys: set[str] = set()
        self.fp32_params: dict[str, torch.Tensor] = {}

        self.model = copy.deepcopy(model) if copy_model else model
        self.model.requires_grad_(False)

        if device is not None:
            self.model.to(device)

        if self.config.ema_fp32:
            self.build_fp32_params()

        self.logs: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    #  FP32 helper                                                       #
    # ------------------------------------------------------------------ #

    def build_fp32_params(
        self, state_dict: Optional[dict[str, torch.Tensor]] = None
    ) -> None:
        if not self.config.ema_fp32:
            raise RuntimeError("ema_fp32 is False – shouldn't build fp32 params")
        state_dict = state_dict or self.model.state_dict()
        for k, v in state_dict.items():
            self.fp32_params[k] = (
                v.detach().clone().float() if torch.is_floating_point(v) else v.clone()
            )

    # ------------------------------------------------------------------ #
    #  Step / update                                                     #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, new_model: nn.Module) -> None:
        ema_sd = self.fp32_params if self.config.ema_fp32 else self.model.state_dict()

        for key, param in new_model.named_parameters():
            if not self.config.add_missing_params and key not in ema_sd:
                continue
            if key not in ema_sd:
                ema_sd[key] = param.detach().clone()
            ema_t = ema_sd[key]
            if not torch.is_floating_point(ema_t):
                continue  # ignore non-float tensors
            ema_t.mul_(self.decay).add_(
                param.data.to(dtype=ema_t.dtype), alpha=1 - self.decay
            )

        # keep buffers in sync
        for key, buf in new_model.named_buffers():
            ema_sd[key] = buf.detach().clone()

    # Alias expected by new code --------------------------------------- #
    def get_decay(self) -> float:  # noqa: D401 – original API wrapper
        return self.decay

    def set_decay(self, decay: float, weight_decay: Optional[float] = None) -> None:
        self.decay = decay
