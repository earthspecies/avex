import copy
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    D2vModalityConfig,
    MaskSeed,
    ModalitySpecificEncoder,
    get_annealed_rate,
)
from .images import D2vImageConfig, ImageEncoder
from .losses import d2v_loss as _d2v_loss_fn
from .losses import dino_loss as _dino_loss_fn
from .modules import AltBlock, D2vDecoderConfig, Decoder1d

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. ENUMS & CONFIGS
# -----------------------------------------------------------------------------


class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


@dataclass
class D2vModalitiesConfig:
    image: D2vImageConfig = field(default_factory=D2vImageConfig)


@dataclass
class Data2VecMultiConfig:
    # loss & scaling
    loss_beta: float = 0.0
    loss_scale: Optional[float] = None

    # transformer backbone
    depth: int = 12
    start_drop_path_rate: float = 0.0
    end_drop_path_rate: float = 0.0
    num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4
    layer_norm_first: bool = False

    average_top_k_layers: int = 12
    end_of_block_targets: bool = False
    clone_batch: int = 16

    # normalisation switches
    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    # EMA settings
    ema_decay: float = 0.999
    ema_end_decay: float = 0.9999
    ema_anneal_end_step: int = 200_000
    ema_encoder_only: bool = True
    ema_same_dtype: bool = False

    # misc flags
    max_update: int = 200_000
    modalities: D2vModalitiesConfig = field(default_factory=D2vModalitiesConfig)
    shared_decoder: Optional[D2vDecoderConfig] = None
    min_target_var: float = 0.1
    min_pred_var: float = 0.01
    supported_modality: Optional[Modality] = None
    mae_init: bool = False
    seed: int = 1337
    skip_ema: bool = False

    # loss weights
    cls_loss: float = 0.0
    recon_loss: float = 0.0
    d2v_loss: float = 1.0
    decoder_group: bool = False

    # DINO experiment flags
    utterance_level: bool = False
    init_center_token_zero: bool = False
    center_exp: float = 0.9
    softmax_temperature_student: float = 0.1
    softmax_temperature_teacher: float = 0.05


# -----------------------------------------------------------------------------
# 2.  EMA adapter – wraps reference Fairseq implementation
# -----------------------------------------------------------------------------

from representation_learning.models.eat.fairseq_compat import (
    EMAModule as _RefEMAModule,
)
from representation_learning.models.eat.fairseq_compat import (
    EMAModuleConfig as _RefEMACfg,
)


class EMAModel(_RefEMAModule):
    """Thin adapter exposing the interface expected by the refactored code.

    It re-uses the original Fairseq `EMAModule` implementation for feature
    parity (buffer tracking, fp32 shadow params, norm logging…) and adds the
    two convenience helpers (`update`, `to`) that the new codebase relies on.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        fp32: bool = True,
        log_norms: bool = False,
    ) -> None:
        cfg = _RefEMACfg(ema_decay=decay, ema_fp32=fp32, log_norms=log_norms)
        super().__init__(model, cfg, copy_model=True)

    # ------------------------------------------------------------------
    #  Backwards-compat helpers                                          #
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, student: nn.Module):  # noqa: D401 – keep legacy name
        """Alias for :py:meth:`step` so existing calls keep working."""
        self.step(student)

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Move both the EMA model and (if used) fp32 shadow params."""
        self.model.to(*args, **kwargs)
        # move fp32 shadow params manually
        if hasattr(self, "fp32_params"):
            for k, v in self.fp32_params.items():
                if isinstance(v, torch.Tensor):
                    self.fp32_params[k] = v.to(*args, **kwargs)
        return self


# -----------------------------------------------------------------------------
# 3.  Main Model
# -----------------------------------------------------------------------------


class Data2VecMultiModel(nn.Module):
    """Fairseq‑free EAT (data2vec‑multi) implementation."""

    def __init__(self, cfg: Data2VecMultiConfig, modalities: List[Modality], task=None):
        super().__init__()
        self.cfg = cfg
        self.modalities = modalities
        self.task = task
        self.num_updates = 0

        # --- layer norm factory & transformer blocks ---------------------
        make_ln = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_ln,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)
        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])
        self.norm = make_ln(cfg.embed_dim) if cfg.layer_norm_first else None
        self.dropout_input = (
            nn.Dropout(cfg.dropout_input) if cfg.dropout_input > 0 else None
        )

        # --- modality‑specific encoders -----------------------------------
        self.modality_encoders = nn.ModuleDict()
        self.alibi_biases: Dict[str, torch.Tensor] = {}
        for mod in modalities:
            mod_cfg: D2vModalityConfig = getattr(cfg.modalities, mod.name.lower())
            if mod == Modality.IMAGE:
                enc_cls = ImageEncoder
            else:
                raise ValueError(f"Unsupported modality: {mod}")
            self.modality_encoders[mod.name] = enc_cls(
                mod_cfg,
                cfg.embed_dim,
                make_block,
                make_ln,
                cfg.layer_norm_first,
                self.alibi_biases,
                task,
            )

        # --- optional shared decoder & proj heads -------------------------
        self.shared_decoder = (
            Decoder1d(cfg.shared_decoder, cfg.embed_dim) if cfg.shared_decoder else None
        )
        self.recon_proj = (
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 3) if cfg.recon_loss > 0 else None
        )
        self.cls_proj = (
            nn.Linear(cfg.embed_dim, cfg.embed_dim) if cfg.utterance_level else None
        )

        # Initialise optional *shared* decoder to match reference behaviour.
        if self.shared_decoder is not None:
            init_fn = (
                self._init_weights_mae if cfg.mae_init else self._init_weights_bert
            )
            self.shared_decoder.apply(init_fn)

        # init weights ------------------------------------------------------
        self.apply(self._init_weights_mae if cfg.mae_init else self._init_weights_bert)
        for enc in self.modality_encoders.values():
            enc.reset_parameters()

        # teacher -----------------------------------------------------------
        self.ema: Optional[EMAModel] = None
        if not cfg.skip_ema:
            logger.info("[DEBUG] Creating teacher model")
            teacher = self._make_teacher(cfg, modalities)
            # Teacher remains on the same device as the student (default behaviour)
            logger.debug("[DEBUG] EMA model before init")
            self.ema = EMAModel(teacher, decay=cfg.ema_decay, fp32=True)
            logger.debug("[DEBUG] EMA model after init")
        # dino center token -------------------------------------------------
        if cfg.utterance_level:
            self.register_buffer("center", torch.zeros(1, 1, cfg.embed_dim))
            if not cfg.init_center_token_zero:
                nn.init.normal_(self.center)

        # Track current device for safe .to() override
        self._current_device = next(self.parameters()).device

        # --------------------------------------------------------------
        # Parameter annotations for weight-decay handling – replicate
        # Fairseq's behaviour so norms/biases are exempt from WD and the
        # optional decoder group can be isolated.
        # --------------------------------------------------------------
        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
            if self.cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"

    # ------------------------------------------------------------------
    # Init helpers ---------------------------------------------------------
    # ------------------------------------------------------------------
    def _init_weights_bert(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            try:
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            except Exception:
                pass  # Bias might be None or not initialised yet
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _init_weights_mae(self, m):
        self._init_weights_bert(m)  # identical for now

    # ------------------------------------------------------------------
    # Teacher helpers ------------------------------------------------------
    # ------------------------------------------------------------------
    def _make_teacher(self, cfg: Data2VecMultiConfig, modalities):
        # TODO: Changed – avoid recursive EMA creation by passing a copy of the
        # config with skip_ema=True so the teacher itself has no EMA teacher.
        cfg_teacher = copy.deepcopy(cfg)
        cfg_teacher.skip_ema = True
        teacher = Data2VecMultiModel(cfg_teacher, modalities, task=self.task)

        # Weight copy – ensure teacher starts identical to student as in reference
        for p_s, p_t in zip(self.parameters(), teacher.parameters(), strict=False):
            p_t.data.copy_(p_s.data)

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        if cfg.ema_encoder_only:
            teacher = teacher.blocks  # keep only transformer encoder
        return teacher

    @torch.no_grad()
    def set_num_updates(self, num_updates: int):
        self.num_updates = num_updates
        if self.training and self.ema is not None:
            # anneal decay
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    self.ema.set_decay(self.cfg.ema_end_decay)
                else:
                    new_decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                    self.ema.set_decay(new_decay)
            self.ema.update(self.blocks if self.cfg.ema_encoder_only else self)

    # ------------------------------------------------------------------
    # Forward pass ---------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(
        self,
        source: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        *,
        sample_id: Optional[torch.Tensor] = None,
        mode: Optional[Union[Modality, str]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        force_remove_masked: bool = False,
        remove_extra_tokens: bool = True,
        precomputed_mask: Optional[MaskSeed] = None,
    ) -> Dict[str, torch.Tensor]:
        if mode is None:
            mode = self.cfg.supported_modality
        if isinstance(mode, Modality):
            mode = mode.name

        feat_enc: ModalitySpecificEncoder = self.modality_encoders[mode]

        mask_seeds = None
        if sample_id is not None:
            mask_seeds = MaskSeed(
                seed=self.cfg.seed, update=self.num_updates, ids=sample_id
            )

        extractor_out = feat_enc(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        x = extractor_out["x"]  # (B*clone, T', D)
        masked_padding_mask = extractor_out["padding_mask"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                (not self.training)
                or (self.cfg.layerdrop == 0)
                or (np.random.rand() > self.cfg.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)
                x, lr = blk(x, padding_mask=masked_padding_mask, alibi_bias=ab)
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        # -------------------------------------------------------------
        # feature extraction only -------------------------------------
        # -------------------------------------------------------------
        if features_only:
            if remove_extra_tokens:
                x = x[:, feat_enc.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feat_enc.modality_cfg.num_extra_tokens :
                    ]
            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

        # -------------------------------------------------------------
        # PRETRAINING LOSS PATH (mirrors original Fairseq version) -----
        # -------------------------------------------------------------
        result: Dict[str, torch.Tensor] = {"losses": {}, "sample_size": None}

        # 1. DECODER(S) --------------------------------------------------
        xs: List[torch.Tensor] = []
        if self.shared_decoder is not None:
            dx = self._forward_decoder(x, feat_enc, self.shared_decoder, encoder_mask)
            xs.append(dx)
        if feat_enc.decoder is not None:
            dx = self._forward_decoder(x, feat_enc, feat_enc.decoder, encoder_mask)
            xs.append(dx)

        # 2. TEACHER FOR TARGETS ---------------------------------------
        # ------------------------------------------------------------------
        #  Align teacher dtype / device exactly like reference --------------
        # ------------------------------------------------------------------
        device, dtype = x.device, x.dtype
        if self.ema is None:
            raise RuntimeError("EMA / teacher model required but missing")

        p_first = next(self.ema.model.parameters())
        ema_device = p_first.device
        ema_dtype = p_first.dtype

        # honour ema_same_dtype flag
        if not self.cfg.ema_same_dtype:
            dtype = ema_dtype

        if ema_device != device or ema_dtype != dtype:
            logger.info("[EAT] Adjusting EMA dtype/device to %s / %s", dtype, device)
            # move the model
            self.ema.model = self.ema.model.to(device=device, dtype=dtype)

            # move fp32 shadow params dictionary recursively
            def _move_fp32(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        _move_fp32(v)
                    elif isinstance(v, torch.Tensor):
                        d[k] = v.to(device=device)

            if hasattr(self.ema, "fp32_params"):
                _move_fp32(self.ema.fp32_params)

        # Re-fetch pointer after potential move
        ema_model = self.ema.model if self.ema is not None else None
        if ema_model is None:
            raise RuntimeError(
                "EMA / teacher model is required for pretraining but skip_ema=True"
            )

        with torch.no_grad():
            ema_model.eval()
            if self.cfg.ema_encoder_only:
                ema_blocks = ema_model
                ema_input = extractor_out["local_features"]
                ema_input = feat_enc.contextualized_features(
                    ema_input.to(dtype=dtype),
                    padding_mask,
                    mask=False,
                    remove_masked=False,
                )
            else:
                ema_blocks = ema_model.blocks  # type: ignore
                if feat_enc.modality_cfg.ema_local_encoder:
                    inp = (target if target is not None else source).to(dtype=dtype)
                    ema_input = ema_model.modality_encoders[mode](  # type: ignore
                        inp, padding_mask, mask=False, remove_masked=False
                    )
                else:
                    ema_input = extractor_out["local_features"]
                    ema_input = ema_model.modality_encoders[
                        mode
                    ].contextualized_features(  # type: ignore
                        ema_input.to(dtype=dtype),
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )

            ema_padding_mask = ema_input["padding_mask"]
            ema_alibi_bias = ema_input.get("alibi_bias", None)
            ema_alibi_scale = ema_input.get("alibi_scale", None)
            ema_input = ema_input["x"]

            y_layers = []
            extra_tokens = feat_enc.modality_cfg.num_extra_tokens
            for i, blk in enumerate(ema_blocks):
                ab = ema_alibi_bias
                if ab is not None and ema_alibi_scale is not None:
                    scale = (
                        ema_alibi_scale[i]
                        if ema_alibi_scale.size(0) > 1
                        else ema_alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)
                ema_input, lr = blk(
                    ema_input, padding_mask=ema_padding_mask, alibi_bias=ab
                )
                y_layers.append(lr[:, extra_tokens:])

        y = self._make_targets(y_layers, self.cfg.average_top_k_layers)  # (B, T, D)
        orig_targets = y

        # broadcast to clones ------------------------------------------------
        if self.cfg.clone_batch > 1:
            y = y.repeat_interleave(self.cfg.clone_batch, 0)

        # mask handling ------------------------------------------------------
        masked_b = encoder_mask.mask.bool()
        y = y[masked_b]
        xs_masked = [
            x_[masked_b]
            if x_.size(1) == masked_b.size(1)
            else x_.reshape(-1, x_.size(-1))
            for x_ in xs
        ]
        sample_size = encoder_mask.mask.sum().long()
        result["sample_size"] = sample_size

        # 3. LOSS COMPUTATION ---------------------------------------------
        if self.cfg.d2v_loss > 0:
            for i, x_ in enumerate(xs_masked):
                # NOTE: keep per-element loss tensor – reduction happens later in
                # the training loop so that we can apply a shared normalisation
                # by `sample_size` across *all* components.
                reg_elem = _d2v_loss_fn(
                    x_,
                    y,
                    beta=self.cfg.loss_beta,
                    scale=self.cfg.loss_scale,
                )
                key = (
                    f"{mode}_regression_{i}"
                    if len(xs_masked) > 1
                    else f"{mode}_regression"
                )
                result["losses"][key] = reg_elem * self.cfg.d2v_loss

        if self.cfg.recon_loss > 0:
            with torch.no_grad():
                target_patch = feat_enc.patchify(source)
                mean = target_patch.mean(dim=-1, keepdim=True)
                var = target_patch.var(dim=-1, keepdim=True)
                target_patch = (target_patch - mean) / (var + 1e-6) ** 0.5
                if self.cfg.clone_batch > 1:
                    target_patch = target_patch.repeat_interleave(
                        self.cfg.clone_batch, 0
                    )
                target_patch = target_patch[masked_b]
            recon = xs_masked[0]
            if self.recon_proj is not None:
                recon = self.recon_proj(recon)
            result["losses"]["recon"] = (
                _d2v_loss_fn(recon, target_patch) * self.cfg.recon_loss
            )

        if self.cfg.cls_loss > 0:
            assert extra_tokens > 0, "CLS loss requires extra tokens"
            cls_target = orig_targets.mean(dim=1)
            if self.cfg.clone_batch > 1:
                cls_target = cls_target.repeat_interleave(self.cfg.clone_batch, 0)
            cls_pred = x[:, extra_tokens - 1]
            if self.cfg.utterance_level:
                cls_target = cls_target - self.center
                loss_cls = _dino_loss_fn(
                    cls_pred,
                    cls_target,
                    self.cfg.softmax_temperature_student,
                    self.cfg.softmax_temperature_teacher,
                )
                self.center.mul_(self.cfg.center_exp).add_(
                    cls_target.mean(dim=0, keepdim=True), alpha=1 - self.cfg.center_exp
                )
            else:
                loss_cls = _d2v_loss_fn(cls_pred, cls_target)
            # Keep token-level loss; weight and sample_size applied for parity
            # with the reference implementation.
            result["losses"]["cls"] = loss_cls * self.cfg.cls_loss * sample_size

        # 4. MONITORING METRICS -------------------------------------------
        with torch.no_grad():
            if encoder_mask is not None:
                result["masked_pct"] = 1 - (
                    encoder_mask.ids_keep.size(1) / encoder_mask.ids_restore.size(1)
                )
            for i, x_ in enumerate(xs_masked):
                key = f"pred_var_{i}"
                result[key] = self.compute_var(x_.float())
            result["target_var"] = self.compute_var(y.float())

            # early‑stop on degenerate variance
            if self.num_updates > 5000:
                if result["target_var"] < self.cfg.min_target_var:
                    raise RuntimeError(
                        "Target variance below threshold – training diverged."
                    )
                for k, v in result.items():
                    if k.startswith("pred_var") and v < self.cfg.min_pred_var:
                        raise RuntimeError(f"{k} below threshold – training diverged.")

        return result

    # ------------------------------------------------------------------
    # Aux helpers ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _forward_decoder(self, x, feat_enc, decoder, mask_info):
        x_dec_in = feat_enc.decoder_input(x, mask_info)
        # decoder_input may return either a **tensor** or a tuple (q, kv)
        if isinstance(x_dec_in, tuple):
            return decoder(*x_dec_in)
        else:
            return decoder(x_dec_in)

    def _make_targets(self, y_layers: List[torch.Tensor], k: int):
        target_layer_results = y_layers[-k:]
        # optional norm variants ---------------------------------
        permuted = False
        if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
            target_layer_results = [tl.transpose(1, 2) for tl in target_layer_results]
            permuted = True
        if self.cfg.batch_norm_target_layer:
            target_layer_results = [
                F.batch_norm(tl.float(), None, None, training=True)
                for tl in target_layer_results
            ]
        if self.cfg.instance_norm_target_layer:
            target_layer_results = [
                F.instance_norm(tl.float()) for tl in target_layer_results
            ]
        if permuted:
            target_layer_results = [tl.transpose(1, 2) for tl in target_layer_results]
        if self.cfg.layer_norm_target_layer:
            target_layer_results = [
                F.layer_norm(tl.float(), tl.shape[-1:]) for tl in target_layer_results
            ]
        # average --------------------------------------------------
        y = sum(target_layer_results) / len(target_layer_results)
        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])
        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)
        return y

    @staticmethod
    def compute_var(y: torch.Tensor):
        y = y.view(-1, y.size(-1))
        if dist.is_available() and dist.is_initialized():
            zc = torch.tensor(y.size(0), device=y.device)
            zs = y.sum(dim=0)
            zss = (y**2).sum(dim=0)
            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)
            var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    # ------------------------------------------------------------------ #
    #  Device helper – ensure EMA teacher moves with the student          #
    # ------------------------------------------------------------------ #

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Override ``nn.Module.to`` so the EMA teacher follows the student.

        The built-in recursive device transfer skips attributes that are not
        ``nn.Module`` instances (our :class:`EMAModel` wrapper falls into that
        category).  We therefore move it manually to keep weights on the same
        accelerator and avoid dtype/device mismatches at runtime.
        """
        super().to(*args, **kwargs)
        if self.ema is not None:
            self.ema.to(*args, **kwargs)
        # Record for other helpers
        if len(args) > 0 and isinstance(args[0], (str, torch.device)):
            self._current_device = torch.device(args[0])
        elif "device" in kwargs:
            self._current_device = torch.device(kwargs["device"])
        return self
