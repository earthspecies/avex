from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .eat_modules import *  # noqa: F401,F403


class BlockEncoder(nn.Module):
    """Lightweight wrapper that applies a *list* of transformer blocks.

    The original EAT code used a custom implementation with optional layer-drop
    and final LayerNorm.  For inference / fine-tuning we only need the forward
    path, so this simplified version is sufficient.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        norm: Optional[nn.Module],
        layer_norm_first: bool,
        layerdrop: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.norm = norm
        self.layerdrop = layerdrop
        self.post_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm_first = layer_norm_first

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        alibi_bias: Optional[torch.Tensor],
        _idx: Optional[int] = None,
    ) -> torch.Tensor:  # noqa: D401
        for blk in self.blocks:
            if (
                self.training
                and self.layerdrop > 0
                and torch.rand(1).item() < self.layerdrop
            ):
                continue
            # Blocks in eat_modules.AltBlock return tuple (x, trg). We take x.
            if hasattr(blk, "forward"):
                out = blk(x, padding_mask=padding_mask, alibi_bias=alibi_bias)
            else:
                out = blk(x)
            x = out[0] if isinstance(out, tuple) else out
            x = self.post_dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def reset_parameters(self) -> None:  # noqa: D401
        for blk in self.blocks:
            if hasattr(blk, "reset_parameters"):
                blk.reset_parameters()


# --------------------------------------------------------------------------- #
#  Positional-encoding helper                                                 #
# --------------------------------------------------------------------------- #


class FixedPositionalEncoder(nn.Module):
    """Adds a **pre-computed** (non-learnable) positional embedding tensor."""

    def __init__(self, pos_embed: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(
        self, x: torch.Tensor, _padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # noqa: D401
        # Simply return the (broadcasted) positional embedding cropped to match
        # the current sequence length.
        if self.pos_embed.size(1) < x.size(1):
            raise ValueError("pos_embed length shorter than input sequence")
        return self.pos_embed[:, : x.size(1)]
