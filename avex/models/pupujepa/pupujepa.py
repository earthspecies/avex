import torch
import torch.nn as nn
import math
from copy import deepcopy
from typing import Tuple
import random
from timm.models.eva import EvaBlock
from timm.layers import create_rope_embed
from .patch_embed_rope import PatchEmbedRoPE
from torch.utils.checkpoint import checkpoint


class PupuJEPAEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        use_swiglu=False,
        init_values=None,
        num_prefix_tokens=0,
        qk_norm=False,
    ):
        super().__init__()

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                EvaBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    attn_type="rope",
                    num_prefix_tokens=num_prefix_tokens,
                    drop_path=dpr[i],
                    swiglu_mlp=use_swiglu,
                    init_values=init_values,
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x, rope, checkpoint_step=None):
        for i, blk in enumerate(self.blocks):
            if self.training and checkpoint_step != None and i % checkpoint_step == 0:
                x = checkpoint(blk, x, rope=rope, use_reentrant=False)
            else:
                x = blk(x, rope=rope)
        x = self.norm(x)
        return x


class PupuJEPAPredictor(nn.Module):
    def __init__(
        self,
        embed_dim,
        predictor_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        drop_path_rate=0.0,
        use_swiglu=False,
        init_values=None,
        num_prefix_tokens=0,
        qk_norm=False,
    ):
        super().__init__()
        self.proj_in = (
            nn.Linear(embed_dim, predictor_dim)
            if embed_dim != predictor_dim
            else nn.Identity()
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                EvaBlock(
                    dim=predictor_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    attn_type="rope",
                    num_prefix_tokens=num_prefix_tokens,
                    drop_path=dpr[i],
                    swiglu_mlp=use_swiglu,
                    init_values=init_values,
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, embed_dim)

    def forward(self, x_ctx, rope_ctx, rope_tgt, checkpoint_step=None):
        B, N_tgt = rope_tgt.shape[:2]

        x = self.proj_in(x_ctx)
        mask_tokens = self.mask_token.expand(B, N_tgt, -1)
        x_combined = torch.cat([x, mask_tokens], dim=1)
        rope_combined = torch.cat([rope_ctx, rope_tgt], dim=1).unsqueeze(1)

        for i, blk in enumerate(self.blocks):
            if self.training and checkpoint_step != None and i % checkpoint_step == 0:
                x_combined = checkpoint(blk, x_combined, rope=rope_combined, use_reentrant=False)
            else:
                x_combined = blk(x_combined, rope=rope_combined)
        x_combined = self.norm(x_combined)

        N_ctx = x_ctx.shape[1]
        x_pred = x_combined[:, N_ctx:, :]
        return self.proj_out(x_pred)


class MaskGeneratorRandom(nn.Module):
    def __init__(self, mask_ratio=0.8):
        super().__init__()
        self.mask_ratio = mask_ratio

    @torch.no_grad()
    def forward(self, B, device, grid_size: Tuple[int, int]):
        H, W = grid_size
        L = H * W

        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(B, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        masks_enc = ids_shuffle[:, :len_keep]
        masks_pred = ids_shuffle[:, len_keep:]

        return masks_enc, masks_pred


class MaskGeneratorBlock(nn.Module):
    def __init__(
        self,
        mask_ratio=0.8,
        min_block_scale=0.01,
        max_block_scale=0.05,
        prob_vertical=0.5,
        vertical_ar_range=(2.0, 5.0),
        horizontal_ar_range=(0.2, 0.5),
        max_attempts=150,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.min_block_scale = min_block_scale
        self.max_block_scale = max_block_scale
        self.prob_vertical = prob_vertical
        self.vertical_ar_range = vertical_ar_range
        self.horizontal_ar_range = horizontal_ar_range
        self.max_attempts = max_attempts

    @torch.no_grad()
    def forward(self, B, device, grid_size: Tuple[int, int]):
        H, W = grid_size
        L = H * W
        target_mask_count = int(L * self.mask_ratio)

        masks_pred_list = []
        masks_enc_list = []

        for _ in range(B):
            mask_map = torch.zeros(H, W, dtype=torch.bool, device=device)
            current_count = 0
            attempts = 0

            while current_count < target_mask_count and attempts < self.max_attempts:
                attempts += 1

                is_vertical = torch.rand(1).item() < self.prob_vertical
                if is_vertical:
                    ar_min, ar_max = self.vertical_ar_range
                else:
                    ar_min, ar_max = self.horizontal_ar_range
                ar = ar_min + torch.rand(1).item() * (ar_max - ar_min)

                scale = self.min_block_scale + torch.rand(1).item() * (
                    self.max_block_scale - self.min_block_scale
                )

                remaining = target_mask_count - current_count
                block_area = scale * L
                if block_area > remaining:
                    block_area = remaining

                h_w_ratio = math.sqrt(ar)
                h_span = int(math.sqrt(block_area) * h_w_ratio)
                w_span = int(math.sqrt(block_area) / h_w_ratio)

                h_span = max(1, min(H, h_span))
                w_span = max(1, min(W, w_span))

                h_start = torch.randint(0, H - h_span + 1, (1,)).item()
                w_start = torch.randint(0, W - w_span + 1, (1,)).item()

                row_slice = slice(h_start, h_start + h_span)
                col_slice = slice(w_start, w_start + w_span)

                area_mask = mask_map[row_slice, col_slice]
                num_total_in_block = (h_span) * (w_span)
                num_already_masked = area_mask.sum().item()
                num_new = num_total_in_block - num_already_masked

                if num_new == 0:
                    continue

                if current_count + num_new <= target_mask_count:
                    mask_map[row_slice, col_slice] = True
                    current_count += num_new
                else:
                    break

            if current_count < target_mask_count:
                needed = target_mask_count - current_count
                remaining_indices = torch.nonzero(
                    ~mask_map.flatten(), as_tuple=False
                ).squeeze(1)

                if len(remaining_indices) > 0:
                    take = min(needed, len(remaining_indices))
                    perm = torch.randperm(len(remaining_indices), device=device)[:take]
                    mask_map.view(-1)[remaining_indices[perm]] = True

            flat_map = mask_map.flatten()
            masks_pred_list.append(torch.nonzero(flat_map, as_tuple=False).squeeze(1))
            masks_enc_list.append(torch.nonzero(~flat_map, as_tuple=False).squeeze(1))

        masks_pred = torch.stack(masks_pred_list)
        masks_enc = torch.stack(masks_enc_list)

        return masks_enc, masks_pred


class MaskGeneratorTimeFrequency(nn.Module):
    def __init__(
        self,
        mask_ratio=0.8,
        max_attempts=150,
        time_mask_max_width=40,
        freq_mask_max_width=1,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.max_attempts = max_attempts
        self.time_mask_max_width = time_mask_max_width
        self.freq_mask_max_width = freq_mask_max_width

    @torch.no_grad()
    def forward(self, B, device, grid_size: Tuple[int, int]):
        H, W = grid_size
        L = H * W
        target_mask_count = int(L * self.mask_ratio)
        prob_time = H / (H + W)

        masks_pred_list = []
        masks_enc_list = []

        for _ in range(B):
            mask_map = torch.zeros(H, W, dtype=torch.bool, device=device)
            current_count = 0
            attempts = 0

            while current_count < target_mask_count and attempts < self.max_attempts:
                attempts += 1

                is_mask_time = torch.rand(1).item() < prob_time
                if is_mask_time:
                    width = torch.randint(1, self.time_mask_max_width + 1, (1,)).item()
                    remaining = target_mask_count - current_count
                    if width * W > remaining:
                        width = max(1, remaining // W)
                    t_idx = torch.randint(0, H - width + 1, (1,)).item()
                    row_slice = slice(t_idx, t_idx + width)
                    col_slice = slice(None)
                else:
                    width = torch.randint(1, self.freq_mask_max_width + 1, (1,)).item()
                    remaining = target_mask_count - current_count
                    if width * H > remaining:
                        width = max(1, remaining // H)
                    f_idx = torch.randint(0, W - width + 1, (1,)).item()
                    row_slice = slice(None)
                    col_slice = slice(f_idx, f_idx + width)

                area_mask = mask_map[row_slice, col_slice]
                num_already_masked = area_mask.sum().item()
                num_total_in_block = area_mask.shape[0] * area_mask.shape[1]
                num_new = num_total_in_block - num_already_masked

                if num_new == 0:
                    continue

                if current_count + num_new <= target_mask_count:
                    mask_map[row_slice, col_slice] = True
                    current_count += num_new
                else:
                    break

            if current_count < target_mask_count:
                needed = target_mask_count - current_count
                remaining_indices = torch.nonzero(
                    ~mask_map.flatten(), as_tuple=False
                ).squeeze(1)

                if len(remaining_indices) > 0:
                    take = min(needed, len(remaining_indices))
                    perm = torch.randperm(len(remaining_indices), device=device)[:take]
                    mask_map.view(-1)[remaining_indices[perm]] = True

            flat_map = mask_map.flatten()
            masks_pred_list.append(torch.nonzero(flat_map, as_tuple=False).squeeze(1))
            masks_enc_list.append(torch.nonzero(~flat_map, as_tuple=False).squeeze(1))

        masks_pred = torch.stack(masks_pred_list)
        masks_enc = torch.stack(masks_enc_list)

        return masks_enc, masks_pred


class PupuJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = tuple(cfg.model.patch_size)
        self.checkpoint_step = cfg.train.grad_checkpointing_step

        if cfg.model.norm_layer == "layer":
            norm_layer = nn.LayerNorm
        elif cfg.model.norm_layer == "rms":
            norm_layer = nn.RMSNorm
        
        self.patch_embed = PatchEmbedRoPE(
            img_size=tuple(cfg.model.image_size),
            patch_size=self.patch_size,
            in_chans=cfg.model.in_chans,
            embed_dim=cfg.model.embed_dim,
            norm_layer=norm_layer,
            flatten=True,
            frequency_first=cfg.model.frequency_first,
        )

        self.rope_encoder = create_rope_embed(
            rope_type="cat",
            dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            feat_shape=None,
        )

        self.rope_predictor = create_rope_embed(
            rope_type="cat",
            dim=cfg.model.decoder_embed_dim,
            num_heads=cfg.model.decoder_num_heads,
            feat_shape=None,
        )

        self.mask_generator_random = MaskGeneratorRandom(
            mask_ratio=cfg.model.mask_ratio
        )
        self.mask_generator_block = MaskGeneratorBlock(mask_ratio=cfg.model.mask_ratio)
        self.mask_generator_time_frequency = MaskGeneratorTimeFrequency(
            mask_ratio=cfg.model.mask_ratio
        )
        self.mask_probs = cfg.model.mask_probs
        self.mask_warmup_start = cfg.model.mask_warmup_start
        self.mask_warmup_end = cfg.model.mask_warmup_end
        
        enc_drop_path = cfg.model.drop_path_rate
        enc_drop_uniform = cfg.model.drop_path_uniform
        use_swiglu = cfg.model.use_swiglu
        layer_scale_init = cfg.model.layer_scale_init_value
        qk_norm = cfg.model.qk_norm

        self.student = PupuJEPAEncoder(
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            norm_layer=norm_layer,
            drop_path_rate=enc_drop_path,
            drop_path_uniform=enc_drop_uniform,
            use_swiglu=use_swiglu,
            init_values=layer_scale_init,
            qk_norm=qk_norm,
        )

        self.predictor = PupuJEPAPredictor(
            embed_dim=cfg.model.embed_dim,
            predictor_dim=cfg.model.decoder_embed_dim,
            depth=cfg.model.decoder_depth,
            num_heads=cfg.model.decoder_num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            norm_layer=norm_layer,
            drop_path_rate=0.0,
            use_swiglu=use_swiglu,
            init_values=layer_scale_init,
            qk_norm=qk_norm,
        )

        self.initialize_weights()
        self.teacher = deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_dynamic_grid_size(self, imgs):
        _, _, H, W = imgs.shape
        p_h, p_w = self.patch_size
        return (H // p_h, W // p_w)

    def get_current_mask_probs(self, step):
        if step is None or self.mask_warmup_end <= 0:
            return self.mask_probs
        
        if step < self.mask_warmup_start:
            return [1.0, 0.0, 0.0]
        elif step >= self.mask_warmup_end:
            return self.mask_probs
        else:
            alpha = (step - self.mask_warmup_start) / (self.mask_warmup_end - self.mask_warmup_start)
            p1 = alpha * self.mask_probs[1]
            p2 = alpha * self.mask_probs[2]
            p0 = 1.0 - (p1 + p2)
            return [p0, p1, p2]

    def forward(self, imgs, step=None):
        B = imgs.shape[0]
        device = imgs.device

        grid_size = self.get_dynamic_grid_size(imgs)
        x_all = self.patch_embed(imgs)

        current_mask_probs = self.get_current_mask_probs(step)
        masking_strategy = random.choices([0, 1, 2], weights=current_mask_probs, k=1)[0]
        if masking_strategy == 0:
            masks_enc, masks_pred = self.mask_generator_random(B, device, grid_size)
        elif masking_strategy == 1:
            masks_enc, masks_pred = self.mask_generator_block(B, device, grid_size)
        else:
            masks_enc, masks_pred = self.mask_generator_time_frequency(
                B, device, grid_size
            )
            
        rope_full_enc = self.rope_encoder.get_embed(grid_size)
        rope_full_pred = self.rope_predictor.get_embed(grid_size)

        with torch.no_grad():
            idx_tgt_input = masks_pred.unsqueeze(-1).expand(-1, -1, x_all.shape[-1])
            x_target_patches = torch.gather(x_all, 1, idx_tgt_input)
            rope_target_enc = rope_full_enc[masks_pred]
            h_target = self.teacher(x_target_patches, rope=rope_target_enc.unsqueeze(1), checkpoint_step=self.checkpoint_step)

        idx_ctx_input = masks_enc.unsqueeze(-1).expand(-1, -1, x_all.shape[-1])
        x_context_patches = torch.gather(x_all, 1, idx_ctx_input)

        rope_context_enc = rope_full_enc[masks_enc]
        rope_context_pred = rope_full_pred[masks_enc]
        rope_target_pred = rope_full_pred[masks_pred]

        z_context = self.student(x_context_patches, rope=rope_context_enc.unsqueeze(1), checkpoint_step=self.checkpoint_step)
        z_pred = self.predictor(z_context, rope_context_pred, rope_target_pred, checkpoint_step=self.checkpoint_step)

        return z_pred, h_target