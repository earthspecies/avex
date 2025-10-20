"""EAT (Data2Vec) utility functions.

This module provides utility functions for EAT (Data2Vec) models including
masking, positional encoding, and other helper functions.

Copyright (c) Facebook, Inc. and its affiliates.
Licensed under the MIT license.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
    idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
    num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
) -> np.ndarray:
    """Return a boolean mask where *True* marks masked positions (1-D variant).

    The algorithm is a faithful port of Fairseq's span-mask sampler.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(B, T)`` batch-size and sequence length.
    padding_mask : torch.Tensor | None
        Boolean mask indicating *padded* (invalid) positions to **skip** in the
        span selection.
    mask_prob : float
        Fraction of tokens to mask overall (before overlaps / clipping).
    mask_length : int
        Length of each contiguous span.
    mask_type : str, default "static"
        Strategy for drawing ``lengths`` – one of ``static``, ``uniform``,
        ``normal``, ``poisson``.
    min_masks : int, default 0
        Lower bound on the *number* of spans per sample.
    no_overlap : bool, default False
        If *True* spans are prevented from overlapping.
    require_same_masks : bool, default True
        Enforce that all samples end up with *exactly* the same number of
        masked positions.
    mask_dropout : float, default 0.0
        Probability of *dropping* an already selected mask position.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(B, T)`` where ``True`` indicates a masked
        position.

    Raises
    ------
    Exception
        If an unknown ``mask_type`` is supplied.
    ValueError
        If ``num_mask_ver`` is invalid.
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * all_sz / float(mask_length) + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length) + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length) + rng.random()
            )
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
            if mask_type == "static":
                raise ValueError("this should never happens")
            else:
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
                mask_idc.extend(span_start + i for i in range(length))

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
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask


def compute_block_mask_2d(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False,
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
    # When the spectrogram is non-square (d[0] != d[1]) pass the explicit
    # shape via *img_shape* so the block mask computes distances correctly.
    img_shape: tuple | None = None,
    flexible_mask: bool = False,
) -> torch.Tensor:
    assert mask_length > 1

    B, L = shape

    d = (int(L**0.5), int(L**0.5))

    if img_shape:
        d = (img_shape[0], img_shape[1])

    if flexible_mask:
        index = np.random.randint(0, 3)
        block_size_options = np.array([(6, 4), (5, 5), (8, 3)])
        block_size = block_size_options[index]

    if inverse_mask:
        mask_prob = 1 - mask_prob

    if flexible_mask:
        mask = torch.zeros((B, d[0], d[1]))
        mask_inds = torch.randint(
            0,
            L,
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / (block_size[0] * block_size[1]))
                    * (1 + mask_dropout)
                ),
            ),
        )
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [], [])

        offset = mask_length // 2
        for i in range(block_size[0]):
            for j in range(block_size[1]):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d[0] - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d[1] - 1)

        mask[(i0, i1, i2)] = 1

    elif non_overlapping:
        sz = math.ceil(d[0] / mask_length)
        inp_len = sz * sz

        inp = torch.zeros((B, 1, sz, sz))
        w = torch.ones((1, 1, mask_length, mask_length))

        mask_inds = torch.multinomial(
            1 - inp.view(B, -1),
            int(inp_len * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
            replacement=False,
        )
        inp.view(B, -1).scatter_(1, mask_inds, 1)

        mask = torch.nn.functional.conv_transpose2d(inp, w, stride=mask_length).squeeze(
            1
        )
        if mask.size(-1) > d[0]:
            mask = mask[..., :d, :d]
    else:
        mask = torch.zeros((B, d[0], d[1]))
        mask_inds = torch.randint(
            0,
            L,
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / mask_length**2)
                    * (1 + mask_dropout)
                ),
            ),
        )
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [], [])

        offset = mask_length // 2
        for i in range(mask_length):
            for j in range(mask_length):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d[0] - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d[1] - 1)

        mask[(i0, i1, i2)] = 1

    def get_nbs(b: int, m: torch.Tensor, w: torch.Tensor) -> torch.Tensor:  # noqa: ANN001 – local helper inside compute_block_mask_2d
        """Return 2-D neighbourhood mask (used during *expand_adjacent*).

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(B, H·W)``.
        """

        all_nbs = torch.nn.functional.conv2d(m.unsqueeze(1), w, padding="same")
        return all_nbs.clamp_max_(1).view(b, -1)

    if require_same_masks and expand_adjcent:
        w = torch.zeros((1, 1, 3, 3))
        w[..., 0, 1] = 1
        w[..., 2, 1] = 1
        w[..., 1, 0] = 1
        w[..., 1, 2] = 1

        all_nbs = get_nbs(B, mask, w)

    mask = mask.reshape(B, -1)

    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * (mask_prob))
        target_len = int(final_target_len * (1 + mask_dropout))

        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.view(1, d[0], d[1]), w).flatten()

                cands = (1 - m + nbs) > 1
                cand_sz = int(cands.sum().item())

                assert cand_sz > 0, f"{nbs} {cand_sz}"

                to_mask = torch.multinomial(
                    cands.float(),
                    min(cand_sz, int(target_len - n)),
                    replacement=False,
                )
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1

            if n > final_target_len:
                to_unmask = torch.multinomial(
                    m, int(n - final_target_len), replacement=False
                )
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(
                    (1 - m), int(final_target_len - n), replacement=False
                )
                m[to_mask] = 1

    if inverse_mask:
        mask = 1 - mask

    return mask


def compute_block_mask_1d(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False,
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
) -> torch.Tensor:
    B, L = shape

    if inverse_mask:
        mask_prob = 1 - mask_prob

    if non_overlapping:
        sz = math.ceil(L / mask_length)

        inp = torch.zeros((B, 1, sz))
        w = torch.ones((1, 1, mask_length))

        mask_inds = torch.multinomial(
            1 - inp.view(B, -1),
            int(sz * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
            replacement=False,
        )
        inp.view(B, -1).scatter_(1, mask_inds, 1)

        mask = torch.nn.functional.conv_transpose1d(inp, w, stride=mask_length).squeeze(
            1
        )
        if mask.size(-1) > L:
            mask = mask[..., :L]

    else:
        mask = torch.zeros((B, L))
        mask_inds = torch.randint(
            0,
            L,
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / mask_length)
                    * (1 + mask_dropout)
                ),
            ),
        )

        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [])

        offset = mask_length // 2
        for i in range(mask_length):
            k1 = i - offset
            inds[0].append(centers[0])
            inds[1].append(centers[1] + k1)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=L - 1)

        mask[(i0, i1)] = 1

    def get_nbs(b: int, m: torch.Tensor, w: torch.Tensor) -> torch.Tensor:  # noqa: ANN001 – private helper
        """Return 1-D neighbourhood mask (used during *expand_adjacent*).

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(B, L)``.
        """

        all_nbs = torch.nn.functional.conv1d(m.unsqueeze(1), w, padding="same")
        return all_nbs.clamp_max_(1).view(b, -1)

    if require_same_masks and expand_adjcent:
        w = torch.ones((1, 1, 3))
        w[..., 1] = 0
        all_nbs = get_nbs(B, mask, w)

    mask = mask.view(B, -1)

    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * (mask_prob))
        target_len = int(final_target_len * (1 + mask_dropout))

        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.unsqueeze(0), w).squeeze(0)

                cands = (1 - m + nbs) > 1
                cand_sz = int(cands.sum().item())

                assert cand_sz > 0, f"{nbs} {cand_sz}"

                to_mask = torch.multinomial(
                    cands.float(),
                    min(cand_sz, int(target_len - n)),
                    replacement=False,
                )
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1

            if n > final_target_len:
                to_unmask = torch.multinomial(
                    m, int(n - final_target_len), replacement=False
                )
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(
                    (1 - m), int(final_target_len - n), replacement=False
                )
                m[to_mask] = 1

    if inverse_mask:
        mask = 1 - mask

    return mask


def get_buckets(sizes: np.ndarray | list[int], num_buckets: int) -> np.ndarray:
    """Return *num_buckets* equally spaced percentile boundaries.

    Parameters
    ----------
    sizes : np.ndarray | list[int]
        1-D collection of lengths.
    num_buckets : int
        Desired number of buckets.

    Returns
    -------
    np.ndarray
        Sorted array of bucket edges.
    """

    buckets: np.ndarray = np.unique(
        np.percentile(
            sizes,
            np.linspace(0, 100, num_buckets + 1)[1:-1],
            interpolation="nearest",
        )
    )
    return buckets


def get_bucketed_sizes(orig_sizes: np.ndarray, buckets: np.ndarray) -> np.ndarray:
    """Map each *orig_sizes* entry to its nearest bucket upper-bound.

    Returns
    -------
    np.ndarray
        Array of the same shape as *orig_sizes* with values snapped to bucket
        edges.
    """

    sizes = np.copy(orig_sizes)
    assert np.min(sizes) >= 0
    start_val = -1
    for end_val in buckets:
        mask = (sizes > start_val) & (sizes <= end_val)
        sizes[mask] = end_val
        start_val = end_val
    return sizes
