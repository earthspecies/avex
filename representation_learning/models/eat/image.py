# images.py – Fairseq-free EAT image-modality encoder
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

from .base import (
    D2vModalityConfig,
    MaskInfo,
    MaskSeed,
    ModalitySpecificEncoder,
    get_alibi_bias,
)
from .mae import PatchEmbed, PatchEmbed_new, get_2d_sincos_pos_embed_flexible
from .modules import (
    BlockEncoder,
    Decoder2d,
    EncDecTransformerDecoder,
    FixedPositionalEncoder,
    TransformerDecoder,
)

# ---------------------------------------------------------------------------- #
#  Image-specific config                                                       #
# ---------------------------------------------------------------------------- #


class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


@dataclass
class D2vImageConfig(D2vModalityConfig):
    type: Modality = Modality.IMAGE

    # patch embedding ----------------------------------------------------
    input_size: int = 224
    in_chans: int = 3
    patch_size: int = 16
    embed_dim: int = 768

    # alibi --------------------------------------------------------------
    alibi_dims: int = 2
    alibi_distance: str = "manhattan"

    # pos-enc ------------------------------------------------------------
    fixed_positions: bool = True

    # decoder choices ----------------------------------------------------
    transformer_decoder: bool = False
    enc_dec_transformer: bool = False  # if True use Enc-Dec transformer
    target_length: int = 1024  # audio-like variable length
    max_length: int = 768  # longest sequence allowed


# ---------------------------------------------------------------------------- #
#  ImageEncoder
# ---------------------------------------------------------------------------- #


class ImageEncoder(ModalitySpecificEncoder):
    """Image (and single-channel spectrogram) modality encoder used by EAT."""

    modality_cfg: D2vImageConfig

    def __init__(
        self,
        modality_cfg: D2vImageConfig,
        embed_dim: int,
        make_block: Callable[[float, Optional[int], Optional[int]], nn.Module],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task: Optional[object] = None,  # kept for API parity – not used here
    ) -> None:
        # ---------------------------------------------------------------- #
        #  1. Patch embedding / local encoder                               #
        # ---------------------------------------------------------------- #
        if modality_cfg.in_chans == 1:  # spectrogram
            img_size = (modality_cfg.target_length, 128)
        else:
            img_size = to_2tuple(modality_cfg.input_size)

        patch_size = to_2tuple(modality_cfg.patch_size)
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.hw = (self.H, self.W)

        local_encoder: nn.Module
        if modality_cfg.in_chans == 3:
            local_encoder = PatchEmbed(
                img_size,
                patch_size,
                modality_cfg.in_chans,
                modality_cfg.embed_dim,
            )
        else:
            # custom patchify variant for 1-channel inputs
            local_encoder = PatchEmbed_new(
                img_size,
                modality_cfg.patch_size,
                modality_cfg.in_chans,
                modality_cfg.embed_dim,
            )

        # Xavier init for conv proj
        w = local_encoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view(w.size(0), -1))

        if modality_cfg.embed_dim != embed_dim:
            local_encoder = nn.Sequential(
                local_encoder, nn.Linear(modality_cfg.embed_dim, embed_dim)
            )

        project_features = nn.Identity()

        # ---------------------------------------------------------------- #
        #  2. Absolute positional embeddings                               #
        # ---------------------------------------------------------------- #
        max_len = modality_cfg.max_length  # in 'time' dimension
        pos_embed = nn.Parameter(
            torch.zeros(1, max_len * self.W, embed_dim), requires_grad=False
        )
        emb = get_2d_sincos_pos_embed_flexible(
            embed_dim, (max_len, self.W), cls_token=False
        )
        pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))
        fixed_pos_enc = (
            FixedPositionalEncoder(pos_embed) if modality_cfg.fixed_positions else None
        )

        # ---------------------------------------------------------------- #
        #  3. Transformer prenet (context encoder)                          #
        # ---------------------------------------------------------------- #
        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )
        blocks = [make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)]
        context_encoder = BlockEncoder(
            nn.ModuleList(blocks),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        # ---------------------------------------------------------------- #
        #  4. Decoder selection                                            #
        # ---------------------------------------------------------------- #
        if modality_cfg.transformer_decoder:
            if modality_cfg.enc_dec_transformer:
                decoder = EncDecTransformerDecoder(modality_cfg.decoder, embed_dim)
            else:
                dec_blocks = [
                    make_block(0, modality_cfg.decoder.decoder_dim, 8)
                    for _ in range(modality_cfg.decoder.decoder_layers)
                ]
                dec_enc = BlockEncoder(
                    nn.ModuleList(dec_blocks),
                    None,
                    layer_norm_first,
                    0.0,
                    0.0,
                )
                decoder = TransformerDecoder(modality_cfg.decoder, embed_dim, dec_enc)
        else:
            decoder = (
                Decoder2d(modality_cfg.decoder, embed_dim, self.H, self.W)
                if modality_cfg.decoder is not None
                else None
            )

        # ---------------------------------------------------------------- #
        #  5. Alibi helper                                                 #
        # ---------------------------------------------------------------- #
        alibi_fn = partial(
            get_alibi_bias,
            alibi_biases=alibi_biases,
            dims=modality_cfg.alibi_dims,
            distance=modality_cfg.alibi_distance,
            heads=modality_cfg.num_alibi_heads,
        )

        # ---------------------------------------------------------------- #
        #  6. Initialise base class                                        #
        # ---------------------------------------------------------------- #
        super().__init__(
            modality_cfg,
            embed_dim,
            local_encoder,
            project_features,
            fixed_pos_enc,
            relative_positional_encoder=None,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_fn,
        )

        # Expose modality config under a stable attribute name so helper
        # methods (e.g. compute_mask) can access it without referring to the
        # base-class generic ``self.cfg``.
        self.modality_cfg = modality_cfg

    # --------------------------------------------------------------------- #
    #  Utility functions (patchify / unpatchify, mask override)             #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images or spectrograms into non-overlapping patches.

        Parameters
        ----------
        imgs : torch.Tensor
            Input tensor of shape ``(B, C, H, W)`` with one (mono) or three
            channels. For spectrograms ``C`` is usually ``1``.

        Returns
        -------
        torch.Tensor
            The flattened patches with shape ``(B, L, P²·C)`` where ``L`` is
            the number of patches and ``P`` is the patch size.
        """
        p = self.modality_cfg.patch_size
        if self.modality_cfg.in_chans == 1:
            h, w = imgs.shape[2] // p, imgs.shape[3] // p
            x = imgs.reshape(imgs.size(0), 1, h, p, w, p)
            x = torch.einsum("nchpwq -> nhwpqc", x).reshape(imgs.size(0), h * w, p * p)
        else:
            h = w = imgs.shape[2] // p
            x = imgs.reshape(imgs.size(0), 3, h, p, w, p)
            x = torch.einsum("nchpwq -> nhwpqc", x).reshape(
                imgs.size(0), h * w, p * p * 3
            )
        return x

    @torch.no_grad()
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.modality_cfg.patch_size
        h = w = int(x.size(1) ** 0.5)
        assert h * w == x.size(1)
        if self.modality_cfg.in_chans == 1:
            x = x.reshape(x.size(0), h, w, p, p, 1)
            x = torch.einsum("nhwpqc -> nchwqp", x).reshape(x.size(0), 1, h * p, w * p)
        else:
            x = x.reshape(x.size(0), h, w, p, p, 3)
            x = torch.einsum("nhwpqc -> nchpwq", x).reshape(x.size(0), 3, h * p, w * p)
        return x

    # Override to support 2-D block masks for images --------------------------

    def compute_mask(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        mask_seed: Optional[MaskSeed],
        apply: bool,
        shape: Optional[Tuple[int, int, int]] = None,
        precomputed_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, MaskInfo]:
        if self.modality_cfg.mask_length <= 1:
            return super().compute_mask(
                x, padding_mask, mask_seed, apply, precomputed_mask
            )

        if precomputed_mask is not None:
            mask = precomputed_mask
        else:
            from .eat_utils import compute_block_mask_2d  # local helper

            if shape is not None:
                B, L, _ = shape
            else:
                B, L, _ = x.shape

            mask = compute_block_mask_2d(
                shape=(B, L),
                mask_prob=self.modality_cfg.mask_prob,
                mask_length=self.modality_cfg.mask_length,
                mask_prob_adjust=self.modality_cfg.mask_prob_adjust,
                inverse_mask=self.modality_cfg.inverse_mask,
                require_same_masks=True,
                mask_dropout=self.modality_cfg.mask_dropout,
                img_shape=self.hw,
            ).to(x.device)

        info = self.make_maskinfo(x, mask, shape)
        if apply:
            x = self.apply_mask(x, info)
        return x, info

    # Enc-Dec transformer needs (query, key-value) rather than patched seq -----

    def decoder_input(
        self, x: torch.Tensor, mask_info: MaskInfo
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # When using a Transformer decoder with separate query/key-value inputs
        # (enc-dec style), return the specialised tuple. Otherwise simply pass
        # through *x* – no need to call the parent as the base class does not
        # implement this hook.

        cfg = self.modality_cfg
        if not (cfg.transformer_decoder and cfg.enc_dec_transformer):
            return x  # Direct pass-through for standard 2-D CNN decoder or no decoder

        inp_drop = cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, self.training)

        kv = x[:, cfg.num_extra_tokens :]
        pos = self.fixed_positional_encoder(x, None)

        mask = mask_info.mask.bool()
        if cfg.decoder.add_positions_all:
            kv = kv + pos[~mask].view(kv.shape)

        q = pos[mask].view(x.size(0), -1, x.size(-1))
        return q, kv

    # reset -------------------------------------------------------------------

    def reset_parameters(self) -> None:
        # Local encoder conv already initialised above; decoder may define its
        # own reset_parameters().  No super call because the parent class does
        # not implement this method.
        if self.decoder is not None and hasattr(self.decoder, "reset_parameters"):
            self.decoder.reset_parameters()
