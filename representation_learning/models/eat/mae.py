# mae_vanilla.py – Minimal MAE / BEiT-style Vision Transformer
# (Fairseq-free rewrite of Facebook's original implementation)

import logging
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

try:
    from apex.normalization import FusedLayerNorm
except ImportError:  # fallback to PyTorch LN
    FusedLayerNorm = nn.LayerNorm

# -----------------------------------------------------------------------------#
#  Config dataclass                                                            #
# -----------------------------------------------------------------------------#


@dataclass
class MaeConfig:
    # patch embedding
    input_size: int = 224
    in_chans: int = 3
    patch_size: int = 16
    embed_dim: int = 768
    # encoder
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    # decoder
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    # misc
    norm_eps: float = 1e-6
    drop_path_rate: float = 0.0
    mask_ratio: float = 0.75
    norm_pix_loss: bool = True

    # experimental toggles (kept for compatibility)
    w2v_block: bool = False  # NOT re-implemented – must stay False
    alt_block: bool = False
    alt_block2: bool = False
    alt_attention: bool = False
    block_dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    layer_norm_first: bool = False
    fused_ln: bool = True
    end_of_block_targets: bool = True

    no_decoder_embed: bool = False
    no_decoder_pos_embed: bool = False
    mask_noise_std: float = 0.0

    single_qkv: bool = False
    use_rel_pos_bias: bool = False
    no_cls: bool = False


# -----------------------------------------------------------------------------#
#  (Optional) alternative Attention / Block variants                            #
# -----------------------------------------------------------------------------#
# We reuse AltAttention / AltBlock we already ported in modules.py earlier.
# If you imported this file standalone, comment these imports out or provide
# your own versions.
try:
    from .modules import AltAttention as AltAttention2
    from .modules import AltBlock as AltBlock2
except Exception:
    AltAttention2 = AltBlock2 = None  # fall back to timm Block if absent


# -----------------------------------------------------------------------------#
#  Logging                                                                      #
# -----------------------------------------------------------------------------#
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
#  Helper: fixed 2-D sin-cos positional embeddings                              #
# -----------------------------------------------------------------------------#
def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w, h
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb = _get_2d_sincos_pos_from_grid(embed_dim, grid)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


def _get_2d_sincos_pos_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = 1.0 / 10000 ** (np.arange(embed_dim, dtype=np.float32) / embed_dim)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# -----------------------------------------------------------------------------#
#  Variant used by EAT's ImageEncoder (supports rectangular grids)             #
# -----------------------------------------------------------------------------#


def get_2d_sincos_pos_embed_flexible(
    embed_dim: int,
    grid_hw: tuple[int, int],
    cls_token: bool = False,
) -> np.ndarray:
    """Generate 2-D sine-cosine positional embeddings for *rectangular* grids.

    This is a drop-in replacement for the original helper that assumed
    square images.  The implementation simply constructs separate ``H`` and
    ``W`` grids and concatenates their respective 1-D embeddings.

    Returns
    -------
    np.ndarray
        Positional embedding matrix with shape ``(H×W + int(cls_token),
        embed_dim)``.
    """

    h, w = grid_hw
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w, h
    grid = np.stack(grid, axis=0).reshape([2, 1, h, w])
    emb = _get_2d_sincos_pos_from_grid(embed_dim // 2, grid)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


# -----------------------------------------------------------------------------#
#  MAE Model                                                                    #
# -----------------------------------------------------------------------------#
class MaeModel(nn.Module):
    def __init__(self, cfg: MaeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_ratio = cfg.mask_ratio

        # 1. Patch embedding ----------------------------------------------------
        self.patch_embed = PatchEmbed(
            cfg.input_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 2. Positional embeddings & CLS token ----------------------------------
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, cfg.embed_dim)) if not cfg.no_cls else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + (0 if cfg.no_cls else 1), cfg.embed_dim),
            requires_grad=False,
        )

        norm_layer = partial(nn.LayerNorm, eps=cfg.norm_eps)
        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()

        # choose block type ------------------------------------------------------
        def make_block(drop_path: float) -> nn.Module:
            if cfg.w2v_block:
                raise NotImplementedError(
                    "cfg.w2v_block=True requires fairseq's "
                    "TransformerSentenceEncoderLayer; set it to False or "
                    "add your own layer."
                )

            if cfg.alt_block and AltBlock2 is not None:
                _window = (
                    cfg.input_size // self.patch_embed.patch_size[0],
                    cfg.input_size // self.patch_embed.patch_size[1],
                )
                return AltBlock2(
                    cfg.embed_dim,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                    layer_norm_first=cfg.layer_norm_first,
                    ffn_targets=not cfg.end_of_block_targets,
                )
            else:
                # timm's vanilla block
                return Block(
                    cfg.embed_dim,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                )

        self.blocks = nn.ModuleList(make_block(d) for d in dpr)
        self.norm = norm_layer(cfg.embed_dim)

        # 3. Decoder -------------------------------------------------------------
        dec_dim = cfg.embed_dim if cfg.no_decoder_embed else cfg.decoder_embed_dim
        if cfg.no_decoder_embed:
            self.decoder_embed = nn.Identity()
        else:
            self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_embed_dim)

        self.mask_token = (
            None if cfg.mask_noise_std > 0 else nn.Parameter(torch.zeros(1, 1, dec_dim))
        )

        self.decoder_pos_embed = (
            None
            if cfg.no_decoder_pos_embed
            else nn.Parameter(
                torch.zeros(1, num_patches + 1, dec_dim), requires_grad=False
            )
        )

        self.decoder_blocks = nn.ModuleList(
            Block(
                dec_dim,
                cfg.decoder_num_heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for _ in range(cfg.decoder_depth)
        )
        self.decoder_norm = norm_layer(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, cfg.patch_size**2 * cfg.in_chans)

        # Init weights -----------------------------------------------------------
        self._initialize_weights()

    # ------------------------------------------------------------------------- #
    #  Weight initialisation                                                    #
    # ------------------------------------------------------------------------- #
    def _initialize_weights(self) -> None:
        # fixed sincos pos-enc
        pos = get_2d_sincos_pos_embed(
            self.pos_embed.size(-1),
            int(self.patch_embed.num_patches**0.5),
            cls_token=not self.cfg.no_cls,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        if self.decoder_pos_embed is not None:
            dpos = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.size(-1),
                int(self.patch_embed.num_patches**0.5),
                cls_token=True,
            )
            self.decoder_pos_embed.data.copy_(
                torch.from_numpy(dpos).float().unsqueeze(0)
            )

        # patch-embed conv like linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, FusedLayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------------- #
    #  Helpers: patchify / unpatchify                                           #
    # ------------------------------------------------------------------------- #
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_embed.patch_size[0]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.size(0), self.cfg.in_chans, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x).reshape(
            imgs.size(0), h * w, p * p * self.cfg.in_chans
        )
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_embed.patch_size[0]
        h = w = int(x.size(1) ** 0.5)
        x = x.reshape(x.size(0), h, w, p, p, self.cfg.in_chans)
        x = torch.einsum("nhwpqc->nchpwq", x).reshape(
            x.size(0), self.cfg.in_chans, h * p, w * p
        )
        return x

    # ------------------------------------------------------------------------- #
    #  Random masking (per-sample)                                              #
    # ------------------------------------------------------------------------- #
    @staticmethod
    def _random_masking(
        x: torch.Tensor, ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, L, D = x.shape
        keep = int(L * (1 - ratio))
        noise = torch.rand(N, L, device=x.device)
        idx_shuffle = noise.argsort(1)
        idx_restore = idx_shuffle.argsort(1)
        idx_keep = idx_shuffle[:, :keep]
        x_keep = torch.gather(x, 1, idx_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(N, L, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, idx_restore)
        return x_keep, mask, idx_restore

    # ------------------------------------------------------------------------- #
    #  Encoder                                                                  #
    # ------------------------------------------------------------------------- #
    def _forward_encoder(
        self, imgs: torch.Tensor, ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x = self.patch_embed(imgs)

        # +pos
        x = x + self.pos_embed[:, (0 if self.cfg.no_cls else 1) :, :]

        mask = ids_restore = None
        if ratio > 0:
            x, mask, ids_restore = self._random_masking(x, ratio)

        if self.cls_token is not None:
            cls_tok = self.cls_token + self.pos_embed[:, :1]
            x = torch.cat([cls_tok.expand(x.size(0), -1, -1), x], 1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    # ------------------------------------------------------------------------- #
    #  Decoder                                                                  #
    # ------------------------------------------------------------------------- #
    def _forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        x = self.decoder_embed(x)

        # append mask tokens
        if self.mask_token is None:
            raise RuntimeError("mask_noise_std > 0 path not yet implemented here.")
        mask_tokens = self.mask_token.repeat(
            x.size(0), ids_restore.size(1) + 1 - x.size(1), 1
        )
        if self.cls_token is not None:
            x_ = torch.cat([x[:, 1:], mask_tokens], 1)
        else:
            x_ = torch.cat([x, mask_tokens], 1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.size(2)))
        if self.cls_token is not None:
            x = torch.cat([x[:, :1], x_], 1)
        else:
            x = x_

        if self.decoder_pos_embed is not None:
            x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        if self.cls_token is not None:
            x = x[:, 1:]
        return self.decoder_pred(x)

    # ------------------------------------------------------------------------- #
    #  Loss & forward                                                           #
    # ------------------------------------------------------------------------- #
    def _loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.cfg.norm_pix_loss:
            mean = target.mean(-1, keepdim=True)
            var = target.var(-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(-1)  # per patch
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(
        self, imgs: torch.Tensor, predictions_only: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        lat, mask, ids_restore = self._forward_encoder(
            imgs, 0 if predictions_only else self.mask_ratio
        )
        if predictions_only:
            return lat  # feature extractor mode

        pred = self._forward_decoder(lat, ids_restore)
        loss = self._loss(imgs, pred, mask)
        return {"loss": loss, "sample_size": mask.sum()}

    # ------------------------------------------------------------------------- #
    #  Public helpers                                                           #
    # ------------------------------------------------------------------------- #
    def remove_pretraining_modules(self) -> None:
        self.decoder_embed = self.decoder_blocks = self.decoder_norm = None
        self.decoder_pos_embed = self.decoder_pred = self.mask_token = None
        if self.cfg.layer_norm_first:
            self.norm = None


# -----------------------------------------------------------------------------#
# Compatibility shim – older EAT code referenced a custom ``PatchEmbed_new``
# variant that only differed in padding logic.  For our purposes a direct
# alias to timm's PatchEmbed is sufficient.
# -----------------------------------------------------------------------------#


class PatchEmbed_new(PatchEmbed):
    """Alias for :class:`timm.models.vision_transformer.PatchEmbed`."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
