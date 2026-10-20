import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from slot_jepa.predictors.components.modules import ACBlock as Block
from slot_jepa.utils.tensors import trunc_normal_


def _build_intrinsics(height, width, hfov_deg, device, dtype):
    hfov = math.radians(hfov_deg)
    fx = 0.5 * width / math.tan(0.5 * hfov)
    fy = fx
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def _build_query_patch_centers(height, width, patch_size, device, dtype):
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x = (xs + 0.5) * patch_size - 0.5
    y = (ys + 0.5) * patch_size - 0.5
    ones = torch.ones_like(x)
    return torch.stack([x, y, ones], dim=-1)


def _euler_xyz_to_matrix_torch(rot_xyz):
    rx, ry, rz = rot_xyz.unbind(dim=-1)
    cx, cy, cz = torch.cos(rx), torch.cos(ry), torch.cos(rz)
    sx, sy, sz = torch.sin(rx), torch.sin(ry), torch.sin(rz)

    ones = torch.ones_like(rx)
    zeros = torch.zeros_like(rx)

    rot_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )
    rot_y = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    rot_z = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return rot_z @ rot_y @ rot_x


def _rotvec_to_matrix_torch(rotvec):
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    small = theta < 1.0e-6
    axis = rotvec / theta.clamp_min(1.0e-6)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    k_mat = torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(rotvec.shape[:-1] + (3, 3))
    sin_theta = torch.sin(theta)[..., None]
    cos_theta = torch.cos(theta)[..., None]
    rot = eye + sin_theta * k_mat + (1.0 - cos_theta) * (k_mat @ k_mat)
    return torch.where(small[..., None], eye, rot)


def _extrinsics_to_pose_torch(extrinsics, rotation_mode):
    translation = extrinsics[..., :3]
    rotation = extrinsics[..., 3:]
    if rotation_mode == "rotvec":
        rot_mat = _rotvec_to_matrix_torch(rotation)
    else:
        rot_mat = _euler_xyz_to_matrix_torch(rotation)
    return rot_mat, translation


def _build_context_query_attention_mask(cache, T_ctx, T_query, H, W, add_tokens=1, device=None):
    cache_key = (
        T_ctx,
        T_query,
        H,
        W,
        add_tokens,
        device.type if device is not None else "cpu",
        device.index if device is not None else None,
    )
    if cache_key in cache:
        return cache[cache_key]

    block_tokens = add_tokens + (H * W)
    total_tokens = (T_ctx + T_query) * block_tokens
    mask = torch.zeros(total_tokens, total_tokens).bool()
    block_mask = torch.ones(block_tokens, block_tokens).bool()

    for hist_idx in range(T_ctx):
        mask[
            hist_idx * block_tokens : (hist_idx + 1) * block_tokens,
            : T_ctx * block_tokens,
        ] = True

    for query_idx in range(T_query):
        row_start = (T_ctx + query_idx) * block_tokens
        row_end = row_start + block_tokens
        mask[row_start:row_end, : T_ctx * block_tokens] = True
        col_start = (T_ctx + query_idx) * block_tokens
        col_end = col_start + block_tokens
        mask[row_start:row_end, col_start:col_end] = block_mask

    if device is not None:
        mask = mask.to(device, non_blocking=True)
    cache[cache_key] = mask
    return mask


def _init_transformer_weights(module, init_std):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=init_std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


def _rescale_blocks(blocks):
    def rescale(param, layer_id):
        param.div_(math.sqrt(2.0 * layer_id))

    for layer_id, layer in enumerate(blocks):
        rescale(layer.attn.proj.weight.data, layer_id + 1)
        rescale(layer.mlp.fc2.weight.data, layer_id + 1)


class LegacyBilinearSpatialResizer(nn.Module):
    """Parameter-free resize used by the original Slot-AC predictor."""

    def __init__(self, input_spatial_size, output_spatial_size):
        super().__init__()
        self.input_spatial_size = int(input_spatial_size)
        self.output_spatial_size = int(output_spatial_size)

    def forward(self, x):
        expected = (self.input_spatial_size, self.input_spatial_size)
        if tuple(x.shape[-3:-1]) != expected:
            raise ValueError(
                f"Expected channel-last spatial input [..., {expected[0]}, "
                f"{expected[1]}, C], got {tuple(x.shape)}"
            )
        if self.input_spatial_size == self.output_spatial_size:
            return x
        leading_shape = x.shape[:-3]
        channels = x.shape[-1]
        resized = x.reshape(-1, *expected, channels).permute(0, 3, 1, 2)
        resized = F.interpolate(
            resized,
            size=(self.output_spatial_size, self.output_spatial_size),
            mode="bilinear",
            align_corners=False,
        )
        resized = resized.permute(0, 2, 3, 1)
        return resized.reshape(
            *leading_shape,
            self.output_spatial_size,
            self.output_spatial_size,
            channels,
        )


class ReconstructedFeaturePredictorBranch(nn.Module):
    """Original global branch that predicts from all-object reconstructed features."""

    def __init__(
        self,
        embed_dim=768,
        feature_dim=196,
        predictor_embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        spatial_size=24,
        num_objects=24,
        state_dim=7,
        use_extrinsics=True,
        extrinsics_dim=6,
        num_obj_tokens=1,
        concat_mask_input=True,
        use_warp_concat=False,
        use_checkpoint=False,
        input_spatial_size=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.spatial_size = int(spatial_size)
        self.input_spatial_size = int(
            self.spatial_size if input_spatial_size is None else input_spatial_size
        )
        self.num_patches = spatial_size * spatial_size
        self.num_objects = num_objects
        self.state_dim = state_dim
        self.use_extrinsics = use_extrinsics
        self.extrinsics_dim = extrinsics_dim
        self.num_obj_tokens = num_obj_tokens
        self.concat_mask_input = concat_mask_input
        self.use_warp_concat = use_warp_concat
        self.use_checkpoint = use_checkpoint
        self.init_std = init_std

        self.obj_latent_reduce = nn.Linear(embed_dim, feature_dim, bias=True)
        predictor_in_dim = feature_dim + (num_objects if concat_mask_input else 0)
        if self.use_warp_concat:
            predictor_in_dim *= 2
        self.entry_downsample = LegacyBilinearSpatialResizer(
            self.input_spatial_size,
            self.spatial_size,
        )
        self.predictor_embed = nn.Linear(predictor_in_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(state_dim, predictor_embed_dim, bias=True)
        if self.use_extrinsics:
            self.extrinsics_encoder = nn.Linear(extrinsics_dim, predictor_embed_dim, bias=True)
        self.obj_encoder = nn.Linear(
            num_objects * feature_dim,
            predictor_embed_dim * num_obj_tokens,
            bias=True,
        )
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, self.num_patches, predictor_in_dim))
        # One transition occupies one temporal RoPE position. Fuse its source
        # and target patches; the target half is learned for query blocks.
        self.pair_fusion = nn.Linear(2 * predictor_embed_dim, predictor_embed_dim, bias=True)

        self.cond_tokens = 1 + self.num_obj_tokens
        if self.use_extrinsics:
            self.cond_tokens += 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=True,
                    grid_size=spatial_size,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.GELU,
                    wide_silu=True,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(predictor_embed_dim)
        self.apply(lambda module: _init_transformer_weights(module, self.init_std))
        trunc_normal_(self.query_tokens, std=self.init_std)
        with torch.no_grad():
            self.pair_fusion.weight.zero_()
            eye = torch.eye(predictor_embed_dim, dtype=self.pair_fusion.weight.dtype)
            self.pair_fusion.weight[:, :predictor_embed_dim].copy_(0.5 * eye)
            self.pair_fusion.weight[:, predictor_embed_dim:].copy_(0.5 * eye)
            self.pair_fusion.bias.zero_()
        _rescale_blocks(self.blocks)
        self._attn_mask_cache = {}

    def _build_input(
        self,
        cur_masks,
        cur_warped_masks,
        obj_latent_low,
        precomputed_base_parts=None,
        return_base_parts=False,
    ):
        if precomputed_base_parts is None:
            recon_low = torch.einsum(
                "btkn,bkd->btnd",
                cur_masks.to(obj_latent_low.dtype),
                obj_latent_low,
            )
            base_parts = [recon_low]
            if self.concat_mask_input:
                base_parts.append(
                    cur_masks.permute(0, 1, 3, 2).to(recon_low.dtype)
                )
        else:
            base_parts = precomputed_base_parts
            recon_low = base_parts[0]
        parts = list(base_parts)
        if cur_warped_masks is not None:
            warped_recon_low = torch.einsum(
                "btkn,bkd->btnd",
                cur_warped_masks.to(obj_latent_low.dtype),
                obj_latent_low,
            )
            parts.append(warped_recon_low)
            if self.concat_mask_input:
                parts.append(
                    cur_warped_masks.permute(0, 1, 3, 2).to(recon_low.dtype)
                )
        inputs = torch.cat(parts, dim=-1)
        inputs = inputs.reshape(
            *inputs.shape[:-2],
            self.input_spatial_size,
            self.input_spatial_size,
            inputs.shape[-1],
        )
        inputs = self.entry_downsample(inputs)
        inputs = inputs.reshape(
            *inputs.shape[:-3], self.num_patches, inputs.shape[-1]
        )
        if return_base_parts:
            return inputs, base_parts
        return inputs

    def forward(
        self,
        obj_latent,
        mask_probs,
        states,
        next_states,
        target_mask_probs=None,
        query_source_mask_probs=None,
        extrinsics=None,
        next_extrinsics=None,
        warped_mask_probs=None,
        target_warped_mask_probs=None,
        query_source_warped_mask_probs=None,
        precomputed_obj_latent_low=None,
        return_conditioning=False,
    ):
        B, T_ctx, K, N = mask_probs.shape
        T_query = next_states.shape[1]

        obj_latent_low = precomputed_obj_latent_low
        if obj_latent_low is None:
            obj_latent_low = self.obj_latent_reduce(obj_latent)
        if target_mask_probs is None:
            target_mask_probs = mask_probs
        if query_source_mask_probs is None:
            query_source_mask_probs = mask_probs[:, -1:].expand(-1, T_query, -1, -1)

        source_input, source_base_parts = self._build_input(
            mask_probs,
            warped_mask_probs,
            obj_latent_low,
            return_base_parts=True,
        )
        source_patch_tokens = self.predictor_embed(source_input)
        reuse_identical_inputs = not self.training and not torch.is_grad_enabled()
        if (
            reuse_identical_inputs
            and target_mask_probs is mask_probs
            and target_warped_mask_probs is warped_mask_probs
        ):
            target_patch_tokens = source_patch_tokens
        else:
            target_patch_tokens = self.predictor_embed(
                self._build_input(
                    target_mask_probs,
                    target_warped_mask_probs,
                    obj_latent_low,
                    precomputed_base_parts=(
                        source_base_parts
                        if reuse_identical_inputs
                        and target_mask_probs is mask_probs
                        else None
                    ),
                )
            )
        hist_patch_tokens = self.pair_fusion(
            torch.cat([source_patch_tokens, target_patch_tokens], dim=-1)
        )
        query_source_tokens = self.predictor_embed(
            self._build_input(
                query_source_mask_probs,
                query_source_warped_mask_probs,
                obj_latent_low,
            )
        )
        learned_target_tokens = self.predictor_embed(
            self.query_tokens.expand(B, T_query, -1, -1)
        )
        query_patch_tokens = self.pair_fusion(
            torch.cat([query_source_tokens, learned_target_tokens], dim=-1)
        )
        D_pred = hist_patch_tokens.size(-1)

        history_state_tokens = self.state_encoder(states).view(B, T_ctx, 1, D_pred)
        query_state_tokens = self.state_encoder(next_states).view(B, T_query, 1, D_pred)
        obj_cond = self.obj_encoder(obj_latent_low.flatten(1)).view(B, 1, self.num_obj_tokens, D_pred)
        history_obj_cond = obj_cond.expand(-1, T_ctx, -1, -1)
        query_obj_cond = obj_cond.expand(-1, T_query, -1, -1)

        if self.use_extrinsics:
            history_extr_tokens = self.extrinsics_encoder(extrinsics).view(B, T_ctx, 1, D_pred)
            query_extr_tokens = self.extrinsics_encoder(next_extrinsics).view(B, T_query, 1, D_pred)
            hist_blocks = torch.cat([history_state_tokens, history_extr_tokens, history_obj_cond, hist_patch_tokens], dim=2)
            query_blocks = torch.cat([query_state_tokens, query_extr_tokens, query_obj_cond, query_patch_tokens], dim=2)
        else:
            hist_blocks = torch.cat([history_state_tokens, history_obj_cond, hist_patch_tokens], dim=2)
            query_blocks = torch.cat([query_state_tokens, query_obj_cond, query_patch_tokens], dim=2)

        total_blocks = T_ctx + T_query
        x = torch.cat([hist_blocks, query_blocks], dim=1).flatten(1, 2)
        attn_mask = _build_context_query_attention_mask(
            self._attn_mask_cache,
            T_ctx,
            T_query,
            self.spatial_size,
            self.spatial_size,
            add_tokens=self.cond_tokens,
            device=x.device,
        )

        for blk in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=total_blocks,
                    H=self.spatial_size,
                    W=self.spatial_size,
                    action_tokens=self.cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=total_blocks,
                    H=self.spatial_size,
                    W=self.spatial_size,
                    action_tokens=self.cond_tokens,
                )

        x = x.view(B, total_blocks, self.cond_tokens + self.num_patches, D_pred)
        query_x = x[:, T_ctx:]
        query_patch_features = self.norm(query_x[:, :, self.cond_tokens :, :])
        if return_conditioning:
            # The first condition token is the query state/action token.  It
            # has passed through the complete predictor transformer and can
            # attend to context and visual patch tokens, making it the global
            # predictor representation for non-spatial outputs such as
            # proprioception.
            query_conditioning = self.norm(query_x[:, :, 0, :])
            return query_patch_features, query_conditioning
        return query_patch_features

    def prepare_context(
        self,
        obj_latent,
        mask_probs,
        states,
        target_mask_probs,
        extrinsics=None,
        warped_mask_probs=None,
        target_warped_mask_probs=None,
        precomputed_obj_latent_low=None,
    ):
        """Precompute all query-independent transformer state for inference."""
        B, T_ctx, _, _ = mask_probs.shape
        obj_latent_low = precomputed_obj_latent_low
        if obj_latent_low is None:
            obj_latent_low = self.obj_latent_reduce(obj_latent)
        source_input, source_base_parts = self._build_input(
            mask_probs,
            warped_mask_probs,
            obj_latent_low,
            return_base_parts=True,
        )
        source_tokens = self.predictor_embed(source_input)
        reuse_identical_inputs = not self.training and not torch.is_grad_enabled()
        if (
            reuse_identical_inputs
            and target_mask_probs is mask_probs
            and target_warped_mask_probs is warped_mask_probs
        ):
            target_tokens = source_tokens
        else:
            target_tokens = self.predictor_embed(
                self._build_input(
                    target_mask_probs,
                    target_warped_mask_probs,
                    obj_latent_low,
                    precomputed_base_parts=(
                        source_base_parts
                        if reuse_identical_inputs
                        and target_mask_probs is mask_probs
                        else None
                    ),
                )
            )
        patch_tokens = self.pair_fusion(
            torch.cat([source_tokens, target_tokens], dim=-1)
        )
        dim = patch_tokens.shape[-1]
        state_tokens = self.state_encoder(states).view(B, T_ctx, 1, dim)
        obj_condition = self.obj_encoder(obj_latent_low.flatten(1)).view(
            B, 1, self.num_obj_tokens, dim
        )
        obj_tokens = obj_condition.expand(-1, T_ctx, -1, -1)
        if self.use_extrinsics:
            extrinsic_tokens = self.extrinsics_encoder(extrinsics).view(
                B, T_ctx, 1, dim
            )
            blocks = torch.cat(
                [state_tokens, extrinsic_tokens, obj_tokens, patch_tokens], dim=2
            )
        else:
            blocks = torch.cat([state_tokens, obj_tokens, patch_tokens], dim=2)
        hidden = blocks.flatten(1, 2)
        layer_kv = []
        for block in self.blocks:
            hidden, context_kv = block.prepare_context(
                hidden,
                T=T_ctx,
                H=self.spatial_size,
                W=self.spatial_size,
                action_tokens=self.cond_tokens,
            )
            layer_kv.append(context_kv)
        return {
            "batch_size": B,
            "num_context": T_ctx,
            "obj_latent_low": obj_latent_low,
            "obj_condition": obj_condition,
            "layer_kv": layer_kv,
        }

    def forward_query_cached(
        self,
        cache,
        query_source_mask_probs,
        next_states,
        next_extrinsics=None,
        query_source_warped_mask_probs=None,
        return_conditioning=False,
    ):
        """Predict one query frame using a prepared context cache."""
        if next_states.shape[1] != 1:
            raise ValueError("Cached predictor path supports one query frame")
        B = cache["batch_size"]
        obj_latent_low = cache["obj_latent_low"]
        query_source_tokens = self.predictor_embed(
            self._build_input(
                query_source_mask_probs,
                query_source_warped_mask_probs,
                obj_latent_low,
            )
        )
        learned_target_tokens = self.predictor_embed(
            self.query_tokens.expand(B, 1, -1, -1)
        )
        patch_tokens = self.pair_fusion(
            torch.cat([query_source_tokens, learned_target_tokens], dim=-1)
        )
        dim = patch_tokens.shape[-1]
        state_tokens = self.state_encoder(next_states).view(B, 1, 1, dim)
        if self.use_extrinsics:
            extrinsic_tokens = self.extrinsics_encoder(next_extrinsics).view(
                B, 1, 1, dim
            )
            query_blocks = torch.cat(
                [
                    state_tokens,
                    extrinsic_tokens,
                    cache["obj_condition"],
                    patch_tokens,
                ],
                dim=2,
            )
        else:
            query_blocks = torch.cat(
                [state_tokens, cache["obj_condition"], patch_tokens], dim=2
            )
        hidden = query_blocks.flatten(1, 2)
        for block, context_kv in zip(self.blocks, cache["layer_kv"]):
            hidden = block.forward_query(
                hidden,
                context_kv=context_kv,
                T_ctx=cache["num_context"],
                H=self.spatial_size,
                W=self.spatial_size,
                action_tokens=self.cond_tokens,
            )
        hidden = hidden.view(
            B, 1, self.cond_tokens + self.num_patches, dim
        )
        patch_features = self.norm(hidden[:, :, self.cond_tokens :])
        if return_conditioning:
            return patch_features, self.norm(hidden[:, :, 0])
        return patch_features

    @staticmethod
    def expand_context_cache(cache, sample_count, case_start=0, case_end=None):
        case_end = cache["batch_size"] if case_end is None else int(case_end)

        def expand(tensor):
            selected = tensor[int(case_start) : case_end]
            return selected[:, None].expand(
                selected.shape[0], sample_count, *selected.shape[1:]
            ).reshape(selected.shape[0] * sample_count, *selected.shape[1:])

        return {
            "batch_size": (case_end - int(case_start)) * sample_count,
            "num_context": cache["num_context"],
            "obj_latent_low": expand(cache["obj_latent_low"]),
            "obj_condition": expand(cache["obj_condition"]),
            "layer_kv": [
                (expand(key), expand(value))
                for key, value in cache["layer_kv"]
            ],
        }


class ObjectScalePredictorBranch(nn.Module):
    """One object-wise context/query transformer at a single spatial scale."""

    def __init__(
        self,
        input_spatial_size,
        spatial_size,
        embed_dim,
        branch_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        state_dim=7,
        use_extrinsics=True,
        extrinsics_dim=6,
        num_obj_tokens=1,
        use_warp_concat=False,
        use_checkpoint=False,
        output_spatial_size=None,
    ):
        super().__init__()
        self.input_spatial_size = int(input_spatial_size)
        self.spatial_size = int(spatial_size)
        self.output_spatial_size = int(
            self.spatial_size if output_spatial_size is None else output_spatial_size
        )
        self.num_patches = spatial_size * spatial_size
        self.branch_dim = branch_dim
        self.depth = depth
        self.num_heads = num_heads
        self.use_extrinsics = use_extrinsics
        self.num_obj_tokens = num_obj_tokens
        self.use_warp_concat = use_warp_concat
        self.use_checkpoint = use_checkpoint
        self.init_std = init_std

        self.obj_latent_reduce = nn.Linear(embed_dim, branch_dim, bias=True)
        input_dim = branch_dim * (2 if self.use_warp_concat else 1)
        self.entry_downsample = LegacyBilinearSpatialResizer(
            self.input_spatial_size,
            self.spatial_size,
        )
        self.input_embed = nn.Linear(input_dim, branch_dim, bias=True)
        self.state_encoder = nn.Linear(state_dim, branch_dim, bias=True)
        if self.use_extrinsics:
            self.extrinsics_encoder = nn.Linear(extrinsics_dim, branch_dim, bias=True)
        self.obj_encoder = nn.Linear(embed_dim, branch_dim * num_obj_tokens, bias=True)
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, self.num_patches, branch_dim))
        self.pair_fusion = nn.Linear(2 * branch_dim, branch_dim, bias=True)

        self.cond_tokens = 1 + self.num_obj_tokens
        if self.use_extrinsics:
            self.cond_tokens += 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=True,
                    grid_size=spatial_size,
                    dim=branch_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.GELU,
                    wide_silu=True,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(branch_dim)
        self.apply(lambda module: _init_transformer_weights(module, self.init_std))
        trunc_normal_(self.query_tokens, std=self.init_std)
        with torch.no_grad():
            self.pair_fusion.weight.zero_()
            eye = torch.eye(branch_dim, dtype=self.pair_fusion.weight.dtype)
            self.pair_fusion.weight[:, :branch_dim].copy_(0.5 * eye)
            self.pair_fusion.weight[:, branch_dim:].copy_(0.5 * eye)
            self.pair_fusion.bias.zero_()
        _rescale_blocks(self.blocks)
        self._attn_mask_cache = {}

    def _resize_object_features(self, object_features):
        B, T_ctx, K, N, D = object_features.shape
        if N != self.input_spatial_size * self.input_spatial_size:
            raise ValueError(
                f"Expected {self.input_spatial_size ** 2} object patches, got {N}"
            )
        x = object_features.view(
            B,
            T_ctx,
            K,
            self.input_spatial_size,
            self.input_spatial_size,
            D,
        )
        x = self.entry_downsample(x)
        x = x.reshape(B, T_ctx, K, self.num_patches, D)
        return x

    def _build_tokens(
        self,
        cur_masks,
        cur_warped_masks,
        obj_latent_low,
        precomputed_base_features=None,
        return_base_features=False,
    ):
        B, _, K, _ = cur_masks.shape
        base_features = precomputed_base_features
        if base_features is None:
            base_features = (
                cur_masks.to(obj_latent_low.dtype).unsqueeze(-1)
                * obj_latent_low[:, None, :, None, :]
            )
        object_features = base_features
        if cur_warped_masks is not None:
            warped_features = (
                cur_warped_masks.to(obj_latent_low.dtype).unsqueeze(-1)
                * obj_latent_low[:, None, :, None, :]
            )
            object_features = torch.cat(
                [object_features, warped_features], dim=-1
            )
        object_features = self._resize_object_features(object_features)
        timesteps = object_features.shape[1]
        object_features = object_features.permute(0, 2, 1, 3, 4).reshape(
            B * K, timesteps, self.num_patches, object_features.size(-1)
        )
        tokens = self.input_embed(object_features)
        if return_base_features:
            return tokens, base_features
        return tokens

    def forward(
        self,
        obj_latent,
        mask_probs,
        states,
        next_states,
        target_mask_probs=None,
        query_source_mask_probs=None,
        extrinsics=None,
        next_extrinsics=None,
        warped_mask_probs=None,
        target_warped_mask_probs=None,
        query_source_warped_mask_probs=None,
    ):
        B, T_ctx, K, _ = mask_probs.shape
        T_query = next_states.shape[1]

        obj_latent_low = self.obj_latent_reduce(obj_latent)
        if target_mask_probs is None:
            target_mask_probs = mask_probs
        if query_source_mask_probs is None:
            query_source_mask_probs = mask_probs[:, -1:].expand(-1, T_query, -1, -1)

        source_patch_tokens, source_base_features = self._build_tokens(
            mask_probs,
            warped_mask_probs,
            obj_latent_low,
            return_base_features=True,
        )
        reuse_identical_inputs = not self.training and not torch.is_grad_enabled()
        if (
            reuse_identical_inputs
            and target_mask_probs is mask_probs
            and target_warped_mask_probs is warped_mask_probs
        ):
            target_patch_tokens = source_patch_tokens
        else:
            target_patch_tokens = self._build_tokens(
                target_mask_probs,
                target_warped_mask_probs,
                obj_latent_low,
                precomputed_base_features=(
                    source_base_features
                    if reuse_identical_inputs
                    and target_mask_probs is mask_probs
                    else None
                ),
            )
        hist_patch_tokens = self.pair_fusion(
            torch.cat([source_patch_tokens, target_patch_tokens], dim=-1)
        )
        query_source_tokens = self._build_tokens(
            query_source_mask_probs,
            query_source_warped_mask_probs,
            obj_latent_low,
        )
        learned_target_tokens = self.query_tokens.expand(B * K, T_query, -1, -1)
        query_patch_tokens = self.pair_fusion(
            torch.cat([query_source_tokens, learned_target_tokens], dim=-1)
        )

        states_obj = states[:, None].expand(B, K, T_ctx, states.size(-1)).reshape(B * K, T_ctx, states.size(-1))
        next_states_obj = next_states[:, None].expand(B, K, T_query, next_states.size(-1)).reshape(
            B * K,
            T_query,
            next_states.size(-1),
        )
        history_state_tokens = self.state_encoder(states_obj).view(B * K, T_ctx, 1, self.branch_dim)
        query_state_tokens = self.state_encoder(next_states_obj).view(B * K, T_query, 1, self.branch_dim)

        obj_cond = self.obj_encoder(obj_latent.reshape(B * K, -1)).view(B * K, 1, self.num_obj_tokens, self.branch_dim)
        history_obj_cond = obj_cond.expand(-1, T_ctx, -1, -1)
        query_obj_cond = obj_cond.expand(-1, T_query, -1, -1)

        if self.use_extrinsics:
            extrinsics_obj = extrinsics[:, None].expand(B, K, T_ctx, extrinsics.size(-1)).reshape(
                B * K,
                T_ctx,
                extrinsics.size(-1),
            )
            next_extrinsics_obj = next_extrinsics[:, None].expand(B, K, T_query, next_extrinsics.size(-1)).reshape(
                B * K,
                T_query,
                next_extrinsics.size(-1),
            )
            history_extr_tokens = self.extrinsics_encoder(extrinsics_obj).view(
                B * K,
                T_ctx,
                1,
                self.branch_dim,
            )
            query_extr_tokens = self.extrinsics_encoder(next_extrinsics_obj).view(
                B * K,
                T_query,
                1,
                self.branch_dim,
            )
            hist_blocks = torch.cat([history_state_tokens, history_extr_tokens, history_obj_cond, hist_patch_tokens], dim=2)
            query_blocks = torch.cat([query_state_tokens, query_extr_tokens, query_obj_cond, query_patch_tokens], dim=2)
        else:
            hist_blocks = torch.cat([history_state_tokens, history_obj_cond, hist_patch_tokens], dim=2)
            query_blocks = torch.cat([query_state_tokens, query_obj_cond, query_patch_tokens], dim=2)

        total_blocks = T_ctx + T_query
        x = torch.cat([hist_blocks, query_blocks], dim=1).flatten(1, 2)
        attn_mask = _build_context_query_attention_mask(
            self._attn_mask_cache,
            T_ctx,
            T_query,
            self.spatial_size,
            self.spatial_size,
            add_tokens=self.cond_tokens,
            device=x.device,
        )

        for blk in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=total_blocks,
                    H=self.spatial_size,
                    W=self.spatial_size,
                    action_tokens=self.cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=total_blocks,
                    H=self.spatial_size,
                    W=self.spatial_size,
                    action_tokens=self.cond_tokens,
                )

        x = x.view(B * K, total_blocks, self.cond_tokens + self.num_patches, self.branch_dim)
        x = self.norm(x[:, T_ctx:, self.cond_tokens :, :])
        x = x.reshape(B, K, T_query, self.num_patches, self.branch_dim).permute(0, 2, 1, 3, 4)

        if self.spatial_size != self.output_spatial_size:
            x = x.reshape(B * T_query * K, self.spatial_size, self.spatial_size, self.branch_dim)
            x = x.permute(0, 3, 1, 2)
            x = F.interpolate(
                x,
                size=(self.output_spatial_size, self.output_spatial_size),
                mode="bilinear",
                align_corners=False,
            )
            x = x.permute(0, 2, 3, 1).reshape(
                B,
                T_query,
                K,
                self.output_spatial_size * self.output_spatial_size,
                self.branch_dim,
            )

        return x

    def prepare_context(
        self,
        obj_latent,
        mask_probs,
        states,
        target_mask_probs,
        extrinsics=None,
        warped_mask_probs=None,
        target_warped_mask_probs=None,
    ):
        """Precompute this scale's query-independent transformer context."""
        B, T_ctx, K, _ = mask_probs.shape
        obj_latent_low = self.obj_latent_reduce(obj_latent)
        source_tokens, source_base_features = self._build_tokens(
            mask_probs,
            warped_mask_probs,
            obj_latent_low,
            return_base_features=True,
        )
        reuse_identical_inputs = not self.training and not torch.is_grad_enabled()
        if (
            reuse_identical_inputs
            and target_mask_probs is mask_probs
            and target_warped_mask_probs is warped_mask_probs
        ):
            target_tokens = source_tokens
        else:
            target_tokens = self._build_tokens(
                target_mask_probs,
                target_warped_mask_probs,
                obj_latent_low,
                precomputed_base_features=(
                    source_base_features
                    if reuse_identical_inputs
                    and target_mask_probs is mask_probs
                    else None
                ),
            )
        patch_tokens = self.pair_fusion(
            torch.cat([source_tokens, target_tokens], dim=-1)
        )
        states_obj = states[:, None].expand(
            B, K, T_ctx, states.shape[-1]
        ).reshape(B * K, T_ctx, states.shape[-1])
        state_tokens = self.state_encoder(states_obj).view(
            B * K, T_ctx, 1, self.branch_dim
        )
        obj_condition = self.obj_encoder(obj_latent.reshape(B * K, -1)).view(
            B * K, 1, self.num_obj_tokens, self.branch_dim
        )
        obj_tokens = obj_condition.expand(-1, T_ctx, -1, -1)
        if self.use_extrinsics:
            extrinsics_obj = extrinsics[:, None].expand(
                B, K, T_ctx, extrinsics.shape[-1]
            ).reshape(B * K, T_ctx, extrinsics.shape[-1])
            extrinsic_tokens = self.extrinsics_encoder(extrinsics_obj).view(
                B * K, T_ctx, 1, self.branch_dim
            )
            blocks = torch.cat(
                [state_tokens, extrinsic_tokens, obj_tokens, patch_tokens], dim=2
            )
        else:
            blocks = torch.cat([state_tokens, obj_tokens, patch_tokens], dim=2)
        hidden = blocks.flatten(1, 2)
        layer_kv = []
        for block in self.blocks:
            hidden, context_kv = block.prepare_context(
                hidden,
                T=T_ctx,
                H=self.spatial_size,
                W=self.spatial_size,
                action_tokens=self.cond_tokens,
            )
            layer_kv.append(context_kv)
        return {
            "batch_size": B,
            "num_objects": K,
            "num_context": T_ctx,
            "obj_latent_low": obj_latent_low,
            "obj_condition": obj_condition,
            "layer_kv": layer_kv,
        }

    def forward_query_cached(
        self,
        cache,
        query_source_mask_probs,
        next_states,
        next_extrinsics=None,
        query_source_warped_mask_probs=None,
    ):
        """Predict one query frame using this scale's context cache."""
        if next_states.shape[1] != 1:
            raise ValueError("Cached predictor path supports one query frame")
        B = cache["batch_size"]
        K = cache["num_objects"]
        query_tokens = self._build_tokens(
            query_source_mask_probs,
            query_source_warped_mask_probs,
            cache["obj_latent_low"],
        )
        learned_tokens = self.query_tokens.expand(B * K, 1, -1, -1)
        patch_tokens = self.pair_fusion(
            torch.cat([query_tokens, learned_tokens], dim=-1)
        )
        states_obj = next_states[:, None].expand(
            B, K, 1, next_states.shape[-1]
        ).reshape(B * K, 1, next_states.shape[-1])
        state_tokens = self.state_encoder(states_obj).view(
            B * K, 1, 1, self.branch_dim
        )
        if self.use_extrinsics:
            extrinsics_obj = next_extrinsics[:, None].expand(
                B, K, 1, next_extrinsics.shape[-1]
            ).reshape(B * K, 1, next_extrinsics.shape[-1])
            extrinsic_tokens = self.extrinsics_encoder(extrinsics_obj).view(
                B * K, 1, 1, self.branch_dim
            )
            blocks = torch.cat(
                [
                    state_tokens,
                    extrinsic_tokens,
                    cache["obj_condition"],
                    patch_tokens,
                ],
                dim=2,
            )
        else:
            blocks = torch.cat(
                [state_tokens, cache["obj_condition"], patch_tokens], dim=2
            )
        hidden = blocks.flatten(1, 2)
        for block, context_kv in zip(self.blocks, cache["layer_kv"]):
            hidden = block.forward_query(
                hidden,
                context_kv=context_kv,
                T_ctx=cache["num_context"],
                H=self.spatial_size,
                W=self.spatial_size,
                action_tokens=self.cond_tokens,
            )
        hidden = hidden.view(
            B * K, 1, self.cond_tokens + self.num_patches, self.branch_dim
        )
        output = self.norm(hidden[:, :, self.cond_tokens :])
        output = output.reshape(
            B, K, 1, self.num_patches, self.branch_dim
        ).permute(0, 2, 1, 3, 4)
        if self.spatial_size != self.output_spatial_size:
            output = output.reshape(
                B * K,
                self.spatial_size,
                self.spatial_size,
                self.branch_dim,
            ).permute(0, 3, 1, 2)
            output = F.interpolate(
                output,
                size=(self.output_spatial_size, self.output_spatial_size),
                mode="bilinear",
                align_corners=False,
            )
            output = output.permute(0, 2, 3, 1).reshape(
                B,
                1,
                K,
                self.output_spatial_size * self.output_spatial_size,
                self.branch_dim,
            )
        return output

    @staticmethod
    def expand_context_cache(cache, sample_count, case_start=0, case_end=None):
        batch_size = cache["batch_size"]
        num_objects = cache["num_objects"]
        case_end = batch_size if case_end is None else int(case_end)
        case_start = int(case_start)

        def expand_case_tensor(tensor):
            selected = tensor[case_start:case_end]
            return selected[:, None].expand(
                selected.shape[0], sample_count, *selected.shape[1:]
            ).reshape(selected.shape[0] * sample_count, *selected.shape[1:])

        def expand_object_tensor(tensor):
            shaped = tensor.reshape(batch_size, num_objects, *tensor.shape[1:])
            selected = shaped[case_start:case_end]
            return selected[:, None].expand(
                selected.shape[0], sample_count, *selected.shape[1:]
            ).reshape(
                selected.shape[0] * sample_count * num_objects,
                *tensor.shape[1:],
            )

        return {
            "batch_size": (case_end - case_start) * sample_count,
            "num_objects": num_objects,
            "num_context": cache["num_context"],
            "obj_latent_low": expand_case_tensor(cache["obj_latent_low"]),
            "obj_condition": expand_object_tensor(cache["obj_condition"]),
            "layer_kv": [
                (expand_object_tensor(key), expand_object_tensor(value))
                for key, value in cache["layer_kv"]
            ],
        }


class ObjectFeaturePyramidPredictorBranch(nn.Module):
    """Object-wise branch that merges multiple spatial scales."""

    def __init__(
        self,
        input_spatial_size=24,
        embed_dim=768,
        scales=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        state_dim=7,
        use_extrinsics=True,
        extrinsics_dim=6,
        num_obj_tokens=1,
        use_warp_concat=False,
        use_checkpoint=False,
        output_spatial_size=None,
    ):
        super().__init__()
        if scales is None:
            scales = [
                {"spatial_size": 16, "dim": 48, "depth": 4, "num_heads": 3},
                {"spatial_size": 12, "dim": 96, "depth": 4, "num_heads": 3},
                {"spatial_size": 6, "dim": 196, "depth": 4, "num_heads": 4},
            ]
        self.scales = scales
        self.output_dim = sum(int(scale["dim"]) for scale in scales)
        self.branches = nn.ModuleList(
            [
                ObjectScalePredictorBranch(
                    input_spatial_size=input_spatial_size,
                    spatial_size=int(scale["spatial_size"]),
                    embed_dim=embed_dim,
                    branch_dim=int(scale["dim"]),
                    depth=int(scale.get("depth", 3)),
                    num_heads=int(scale.get("num_heads", 3)),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,
                    init_std=init_std,
                    state_dim=state_dim,
                    use_extrinsics=use_extrinsics,
                    extrinsics_dim=extrinsics_dim,
                    num_obj_tokens=num_obj_tokens,
                    use_warp_concat=use_warp_concat,
                    use_checkpoint=use_checkpoint,
                    output_spatial_size=output_spatial_size,
                )
                for scale in scales
            ]
        )

    def forward(
        self,
        obj_latent,
        mask_probs,
        states,
        next_states,
        target_mask_probs=None,
        query_source_mask_probs=None,
        extrinsics=None,
        next_extrinsics=None,
        warped_mask_probs=None,
        target_warped_mask_probs=None,
        query_source_warped_mask_probs=None,
    ):
        scale_features = [
            branch(
                obj_latent,
                mask_probs,
                states,
                next_states,
                target_mask_probs=target_mask_probs,
                query_source_mask_probs=query_source_mask_probs,
                extrinsics=extrinsics,
                next_extrinsics=next_extrinsics,
                warped_mask_probs=warped_mask_probs,
                target_warped_mask_probs=target_warped_mask_probs,
                query_source_warped_mask_probs=query_source_warped_mask_probs,
            )
            for branch in self.branches
        ]
        return torch.cat(scale_features, dim=-1)

    def prepare_context(self, *args, **kwargs):
        return [
            branch.prepare_context(*args, **kwargs)
            for branch in self.branches
        ]

    def forward_query_cached(
        self,
        caches,
        query_source_mask_probs,
        next_states,
        next_extrinsics=None,
        query_source_warped_mask_probs=None,
    ):
        if len(caches) != len(self.branches):
            raise ValueError("Object pyramid cache does not match branch count")
        return torch.cat(
            [
                branch.forward_query_cached(
                    cache,
                    query_source_mask_probs=query_source_mask_probs,
                    next_states=next_states,
                    next_extrinsics=next_extrinsics,
                    query_source_warped_mask_probs=(
                        query_source_warped_mask_probs
                    ),
                )
                for branch, cache in zip(self.branches, caches)
            ],
            dim=-1,
        )

    def expand_context_cache(
        self, caches, sample_count, case_start=0, case_end=None
    ):
        return [
            branch.expand_context_cache(
                cache,
                sample_count=sample_count,
                case_start=case_start,
                case_end=case_end,
            )
            for branch, cache in zip(self.branches, caches)
        ]


class ActionConditionalMaskPredictor(nn.Module):
    """
    Predictor wrapper that fuses the original global reconstruction branch with
    a new object-wise multi-scale branch before predicting mask logits.
    """

    def __init__(
        self,
        embed_dim=768,
        feature_dim=196,
        predictor_embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        num_context_frames=7,
        spatial_size=24,
        num_objects=24,
        state_dim=7,
        use_extrinsics=True,
        extrinsics_dim=6,
        num_obj_tokens=1,
        concat_mask_input=True,
        use_warp_concat=False,
        warp_hfov_deg=60.0,
        warp_extrinsics_format="c2w",
        warp_rotation_mode="euler_xyz",
        warp_depth_bins=None,
        patch_size=16,
        use_checkpoint=False,
        object_branch_scales=None,
        proprio_output_dim=None,
        predictor_spatial_size=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.num_context_frames = num_context_frames
        self.input_spatial_size = int(spatial_size)
        self.spatial_size = int(
            self.input_spatial_size
            if predictor_spatial_size is None
            else predictor_spatial_size
        )
        if self.input_spatial_size < self.spatial_size:
            raise ValueError(
                "Input mask resolution must not be smaller than the predictor "
                "resolution, got "
                f"{self.input_spatial_size} and {self.spatial_size}"
            )
        self.output_scale = self.input_spatial_size / self.spatial_size
        self.input_num_patches = self.input_spatial_size * self.input_spatial_size
        self.num_patches = self.spatial_size * self.spatial_size
        self.num_objects = num_objects
        self.state_dim = state_dim
        self.use_extrinsics = use_extrinsics
        self.extrinsics_dim = extrinsics_dim
        self.num_obj_tokens = num_obj_tokens
        self.concat_mask_input = concat_mask_input
        self.use_warp_concat = use_warp_concat
        self.warp_hfov_deg = warp_hfov_deg
        self.warp_extrinsics_format = warp_extrinsics_format
        self.warp_rotation_mode = warp_rotation_mode
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.proprio_output_dim = (
            None if proprio_output_dim is None else int(proprio_output_dim)
        )
        if warp_depth_bins is None:
            warp_depth_bins = [0.15, 0.25, 0.4, 0.7, 1.2, 2.0]
        self.register_buffer(
            "warp_depth_bins",
            torch.tensor(warp_depth_bins, dtype=torch.float32),
            persistent=False,
        )

        self.recon_branch = ReconstructedFeaturePredictorBranch(
            embed_dim=embed_dim,
            feature_dim=feature_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_std=init_std,
            spatial_size=self.spatial_size,
            num_objects=num_objects,
            state_dim=state_dim,
            use_extrinsics=use_extrinsics,
            extrinsics_dim=extrinsics_dim,
            num_obj_tokens=num_obj_tokens,
            concat_mask_input=concat_mask_input,
            use_warp_concat=use_warp_concat,
            use_checkpoint=use_checkpoint,
            input_spatial_size=self.input_spatial_size,
        )
        self.object_branch = ObjectFeaturePyramidPredictorBranch(
            input_spatial_size=self.input_spatial_size,
            embed_dim=embed_dim,
            scales=object_branch_scales,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_std=init_std,
            state_dim=state_dim,
            use_extrinsics=use_extrinsics,
            extrinsics_dim=extrinsics_dim,
            num_obj_tokens=num_obj_tokens,
            use_warp_concat=use_warp_concat,
            use_checkpoint=use_checkpoint,
            output_spatial_size=self.spatial_size,
        )
        fusion_dim = predictor_embed_dim + self.object_branch.output_dim
        self.mask_head = nn.Sequential(
            norm_layer(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            # Match the pretrained head exactly: one logit per internal token.
            # Native-grid recovery below is parameter-free bilinear resize.
            nn.Linear(fusion_dim, 1),
        )
        self.mask_head.apply(lambda module: _init_transformer_weights(module, init_std))
        if self.proprio_output_dim is not None:
            # Global prediction branch from the predictor's query condition
            # token.  This is parallel to mask_head rather than downstream of
            # slot/patch mask features.
            self.proprio_head = nn.Sequential(
                norm_layer(predictor_embed_dim),
                nn.Linear(predictor_embed_dim, predictor_embed_dim),
                nn.GELU(),
                nn.Linear(predictor_embed_dim, self.proprio_output_dim),
            )
            self.proprio_head.apply(
                lambda module: _init_transformer_weights(module, init_std)
            )
        else:
            self.proprio_head = None
        self.debug_timing = False
        self.latest_timing = {}

    @staticmethod
    def _cuda_mem_mb():
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 2)

    def _sync_device(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _tic(self):
        self._sync_device()
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def _toc_ms(self, start_end):
        start, end = start_end
        end.record()
        self._sync_device()
        return start.elapsed_time(end)

    def _upsample_mask_logits(self, coarse_subpixel_logits):
        """Resize internal-grid logits to the native slot grid without parameters."""
        B, T, K, N, channels = coarse_subpixel_logits.shape
        if N != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} coarse patches, got {N}")
        if channels != 1:
            raise ValueError(f"Expected one logit per predictor token, got {channels}")
        if self.input_spatial_size == self.spatial_size:
            # When predictor and slot grids match, return logits directly.
            return coarse_subpixel_logits[..., 0].contiguous()
        logits = coarse_subpixel_logits.reshape(
            B * T * K,
            self.spatial_size,
            self.spatial_size,
            1,
        ).permute(0, 3, 1, 2)
        logits = F.interpolate(
            logits,
            size=(self.input_spatial_size, self.input_spatial_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits.reshape(B, T, K, self.input_num_patches).contiguous()

    def _get_warp_geometry_cache(self, device):
        cache = getattr(self, "_warp_geometry_cache", None)
        if cache is None:
            cache = {}
            self._warp_geometry_cache = cache
        key = (device.type, device.index)
        if key not in cache:
            dtype = torch.float32
            height = self.input_spatial_size
            width = self.input_spatial_size
            query_pixels = _build_query_patch_centers(height, width, self.patch_size, device, dtype).view(-1, 3)
            K = _build_intrinsics(
                self.patch_size * height,
                self.patch_size * width,
                self.warp_hfov_deg,
                device,
                dtype,
            )
            cache[key] = {
                "query_rays": query_pixels @ torch.linalg.inv(K).transpose(-1, -2),
                "K": K,
            }
        return cache[key]

    def _sample_sources_with_depth_maps(self, source_feats, source_extrinsics, target_extrinsics, depth_maps):
        device = source_feats.device
        source_dtype = source_feats.dtype
        dtype = torch.float32
        source_feats = source_feats.to(dtype)
        source_extrinsics = source_extrinsics.to(device=device, dtype=dtype)
        target_extrinsics = target_extrinsics.to(device=device, dtype=dtype)
        depth_maps = depth_maps.to(device=device, dtype=dtype)

        batch_size, num_sources, height, width, channels = source_feats.shape
        if target_extrinsics.ndim == 2:
            target_extrinsics = target_extrinsics.unsqueeze(1)
        if depth_maps.ndim == 3:
            depth_maps = depth_maps.unsqueeze(1)
        num_queries = target_extrinsics.size(1)
        num_depth = depth_maps.size(2)
        num_patches = height * width
        geom = self._get_warp_geometry_cache(device)
        query_rays = geom["query_rays"]
        K = geom["K"]

        query_points_tgt = query_rays.view(1, 1, 1, num_patches, 3) * depth_maps[..., None]

        tgt_rot, tgt_t = _extrinsics_to_pose_torch(target_extrinsics, self.warp_rotation_mode)
        src_rot, src_t = _extrinsics_to_pose_torch(source_extrinsics, self.warp_rotation_mode)
        if self.warp_extrinsics_format == "c2w":
            points_world = query_points_tgt @ tgt_rot[:, :, None].transpose(-1, -2) + tgt_t[:, :, None, None, :]
            points_src = torch.einsum(
                "bqsdnc,bsck->bqsdnk",
                points_world[:, :, None] - src_t[:, None, :, None, None, :],
                src_rot,
            )
        else:
            points_world = (query_points_tgt - tgt_t[:, :, None, None, :]) @ tgt_rot[:, :, None]
            points_src = torch.einsum(
                "bqdnc,bskc->bqsdnk",
                points_world,
                src_rot.transpose(-1, -2),
            ) + src_t[:, None, :, None, None, :]

        z = points_src[..., 2]
        z_valid = z > 1.0e-6
        safe_z = torch.where(z_valid, z, torch.ones_like(z))
        src_x = K[0, 0] * (points_src[..., 0] / safe_z) + K[0, 2]
        src_y = K[1, 1] * (points_src[..., 1] / safe_z) + K[1, 2]
        src_feat_x = (src_x + 0.5) / self.patch_size - 0.5
        src_feat_y = (src_y + 0.5) / self.patch_size - 0.5
        grid_x = (src_feat_x / max(width - 1, 1)) * 2.0 - 1.0
        grid_y = (src_feat_y / max(height - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(
            batch_size * num_queries * num_sources * num_depth,
            height,
            width,
            2,
        )

        feat_tensor = (
            source_feats.permute(0, 1, 4, 2, 3)
            .unsqueeze(1)
            .unsqueeze(3)
            .expand(-1, num_queries, -1, num_depth, -1, -1, -1)
            .reshape(batch_size * num_queries * num_sources * num_depth, channels, height, width)
        )
        sampled = F.grid_sample(feat_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.view(batch_size, num_queries, num_sources, num_depth, channels, height, width).permute(
            0, 1, 3, 2, 5, 6, 4
        )

        in_bounds = (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
        valid = (z_valid & in_bounds).view(batch_size, num_queries, num_sources, num_depth, height, width).permute(
            0, 1, 3, 2, 4, 5
        )
        return sampled.to(source_dtype), valid

    def _solve_warped_context_masks(self, obj_latent_low, mask_probs, extrinsics, target_extrinsics):
        B, T_ctx, K, N = mask_probs.shape
        height = self.input_spatial_size
        width = self.input_spatial_size
        depth_bins = self.warp_depth_bins.to(device=mask_probs.device, dtype=obj_latent_low.dtype)
        num_queries = target_extrinsics.size(1)
        recon_low = torch.einsum("btkn,bkd->btnd", mask_probs.to(obj_latent_low.dtype), obj_latent_low)
        num_depth = depth_bins.numel()
        constant_depth_maps = depth_bins.view(1, 1, num_depth, 1).expand(B, num_queries, -1, N)

        feat_stack, valid_stack = self._sample_sources_with_depth_maps(
            recon_low.view(B, T_ctx, height, width, -1),
            extrinsics,
            target_extrinsics,
            constant_depth_maps,
        )
        valid_float = valid_stack.to(feat_stack.dtype)
        support = valid_float.sum(dim=3)
        safe_support = support.clamp_min(1.0)
        mean_feat = (feat_stack * valid_float.unsqueeze(-1)).sum(dim=3) / safe_support.unsqueeze(-1)
        variance = ((feat_stack - mean_feat.unsqueeze(3)).pow(2).mean(dim=-1) * valid_float).sum(dim=3) / safe_support
        variance = torch.where(support >= 2, variance, torch.full_like(variance, float("inf")))
        variance = variance.view(B, num_queries, num_depth, N)
        any_valid = torch.isfinite(variance).any(dim=2)
        default_idx = torch.full((B, num_queries, N), num_depth // 2, device=mask_probs.device, dtype=torch.long)
        best_idx = torch.argmin(variance, dim=2)
        best_idx = torch.where(any_valid, best_idx, default_idx)
        best_depth_maps = depth_bins[best_idx].unsqueeze(2)

        source_masks = mask_probs.permute(0, 1, 3, 2).view(B, T_ctx, height, width, K)
        sampled_masks, valid_masks = self._sample_sources_with_depth_maps(
            source_masks,
            extrinsics,
            target_extrinsics,
            best_depth_maps,
        )
        sampled_masks = torch.where(valid_masks.unsqueeze(-1), sampled_masks, torch.zeros_like(sampled_masks))
        sampled_masks = sampled_masks[:, :, 0]
        warped_mask_probs = sampled_masks.view(B, num_queries, T_ctx, N, K).permute(0, 1, 2, 4, 3)
        warped_mask_probs = warped_mask_probs / warped_mask_probs.sum(dim=3, keepdim=True).clamp_min(1.0e-6)
        return warped_mask_probs

    def prepare_context_cache(
        self,
        obj_latent,
        context_masks,
        states,
        context_target_masks=None,
        extrinsics=None,
        next_extrinsics=None,
        external_warped_mask_probs=None,
    ):
        """Build an inference-only cache for context transformer keys/values."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Predictor context caching is inference-only")
        B, T_ctx, K, N = context_masks.shape
        if K != self.num_objects or N != self.input_num_patches:
            raise ValueError("Context cache input shape does not match predictor")
        if context_target_masks is None:
            context_target_masks = context_masks
        mask_probs = torch.softmax(context_masks, dim=2)
        target_mask_probs = torch.softmax(context_target_masks, dim=2)
        obj_latent_low = self.recon_branch.obj_latent_reduce(obj_latent)
        warped_mask_probs = external_warped_mask_probs
        if self.use_warp_concat:
            if warped_mask_probs is None:
                if not self.use_extrinsics or next_extrinsics is None:
                    raise ValueError("Warp concat cache requires target extrinsics")
                warped_mask_probs = self._solve_warped_context_masks(
                    obj_latent_low,
                    mask_probs,
                    extrinsics,
                    next_extrinsics[:, :1],
                )
            if warped_mask_probs.ndim == 5:
                if warped_mask_probs.shape[1] != 1:
                    raise ValueError(
                        "Context cache requires one query-independent warped context"
                    )
                warped_mask_probs = warped_mask_probs[:, 0]
        recon_cache = self.recon_branch.prepare_context(
            obj_latent,
            mask_probs,
            states,
            target_mask_probs=target_mask_probs,
            extrinsics=extrinsics,
            warped_mask_probs=warped_mask_probs,
            target_warped_mask_probs=(
                target_mask_probs if self.use_warp_concat else None
            ),
            precomputed_obj_latent_low=obj_latent_low,
        )
        object_cache = self.object_branch.prepare_context(
            obj_latent,
            mask_probs,
            states,
            target_mask_probs=target_mask_probs,
            extrinsics=extrinsics,
            warped_mask_probs=warped_mask_probs,
            target_warped_mask_probs=(
                target_mask_probs if self.use_warp_concat else None
            ),
        )
        return {
            "batch_size": B,
            "num_context": T_ctx,
            "recon": recon_cache,
            "object": object_cache,
        }

    def forward_query_cached(
        self,
        cache,
        query_source_masks,
        next_states,
        next_extrinsics=None,
        return_features=False,
        return_proprio=False,
    ):
        """Predict a single query frame against a prepared context cache."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Predictor context caching is inference-only")
        if next_states.shape[1] != 1 or query_source_masks.shape[1] != 1:
            raise ValueError("Cached predictor path supports one query frame")
        query_mask_probs = torch.softmax(query_source_masks, dim=2)
        query_warped = query_mask_probs if self.use_warp_concat else None
        recon_output = self.recon_branch.forward_query_cached(
            cache["recon"],
            query_source_mask_probs=query_mask_probs,
            next_states=next_states,
            next_extrinsics=next_extrinsics,
            query_source_warped_mask_probs=query_warped,
            return_conditioning=return_proprio,
        )
        if return_proprio:
            recon_features, query_conditioning = recon_output
        else:
            recon_features = recon_output
        object_features = self.object_branch.forward_query_cached(
            cache["object"],
            query_source_mask_probs=query_mask_probs,
            next_states=next_states,
            next_extrinsics=next_extrinsics,
            query_source_warped_mask_probs=query_warped,
        )
        B = cache["batch_size"]
        recon_features = recon_features[:, :, None].expand(
            B,
            1,
            self.num_objects,
            self.num_patches,
            self.predictor_embed_dim,
        )
        fused_features = torch.cat([recon_features, object_features], dim=-1)
        logits = self._upsample_mask_logits(self.mask_head(fused_features))
        if return_proprio:
            if self.proprio_head is None:
                raise RuntimeError("return_proprio requires a proprio head")
            pred_proprio = self.proprio_head(query_conditioning.float())
            if return_features:
                return logits, pred_proprio, fused_features.contiguous()
            return logits, pred_proprio
        if return_features:
            return logits, fused_features.contiguous()
        return logits

    def expand_context_cache(
        self, cache, sample_count, case_start=0, case_end=None
    ):
        """Match a per-case cache to CEM's case-major candidate ordering."""
        case_end = cache["batch_size"] if case_end is None else int(case_end)
        return {
            "batch_size": (case_end - int(case_start)) * int(sample_count),
            "num_context": cache["num_context"],
            "recon": self.recon_branch.expand_context_cache(
                cache["recon"],
                sample_count=int(sample_count),
                case_start=case_start,
                case_end=case_end,
            ),
            "object": self.object_branch.expand_context_cache(
                cache["object"],
                sample_count=int(sample_count),
                case_start=case_start,
                case_end=case_end,
            ),
        }

    def forward(
        self,
        obj_latent,
        context_masks,
        states,
        next_states,
        context_target_masks=None,
        query_source_masks=None,
        extrinsics=None,
        next_extrinsics=None,
        external_warped_mask_probs=None,
        external_warp_matches_context=False,
        precomputed_mask_probs=None,
        precomputed_obj_latent_low=None,
        return_features=False,
        return_proprio=False,
    ):
        """
        Args:
            obj_latent: [B, K, D_obj]
            context_masks: [B, T, K, N]
              Source-frame mask logits for observed transitions.
            context_target_masks: [B, T, K, N]
              Target-frame mask logits paired with each context source.
            query_source_masks: [B, T_query, K, N]
              Current-frame mask logits paired with learned query patches.
            states: [B, T, S], source-state/action transition tokens.
            next_states: [B, T_query, S], query source-state/action tokens.
            extrinsics: [B, T, E] or None
            next_extrinsics: [B, T_query, E] or None

        Returns:
            next_logits: [B, T_query, K, N]
        """
        timing = {}
        start_all = None
        if self.debug_timing and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_all = self._tic()
            start_all[0].record()
        if context_masks.ndim != 4:
            raise ValueError(f"Expected context_masks [B, T, K, N], got {tuple(context_masks.shape)}")
        B, T_ctx, K, N = context_masks.shape
        if T_ctx < 1 or T_ctx > self.num_context_frames:
            raise ValueError(f"Expected context length in [1, {self.num_context_frames}], got {T_ctx}")
        if K != self.num_objects:
            raise ValueError(f"Expected {self.num_objects} objects, got {K}")
        if N != self.input_num_patches:
            raise ValueError(f"Expected {self.input_num_patches} patches, got {N}")
        if obj_latent.shape[:2] != (B, K):
            raise ValueError(f"Expected obj_latent [B, K, D], got {tuple(obj_latent.shape)}")
        context_masks_are_shared = context_target_masks is None
        if context_target_masks is None:
            context_target_masks = context_masks
        if context_target_masks.shape != context_masks.shape:
            raise ValueError(
                "Context transition source/target masks must match, got "
                f"{tuple(context_masks.shape)} and {tuple(context_target_masks.shape)}"
            )
        if states.shape[:2] != (B, T_ctx):
            raise ValueError(f"Expected states [B, {T_ctx}, S], got {tuple(states.shape)}")
        if next_states.ndim != 3 or next_states.shape[0] != B:
            raise ValueError(f"Expected next_states [B, T_query, S], got {tuple(next_states.shape)}")
        T_query = next_states.shape[1]
        query_masks_are_shared = query_source_masks is None
        if query_source_masks is None:
            query_source_masks = context_target_masks[:, -1:].expand(
                -1, T_query, -1, -1
            )
        if query_source_masks.shape != (B, T_query, K, N):
            raise ValueError(
                f"Expected query_source_masks {(B, T_query, K, N)}, got "
                f"{tuple(query_source_masks.shape)}"
            )
        if self.use_extrinsics:
            if extrinsics is None or next_extrinsics is None:
                raise ValueError("Expected extrinsics when use_extrinsics=True")
            if extrinsics.shape[:2] != (B, T_ctx):
                raise ValueError(f"Expected extrinsics [B, {T_ctx}, E], got {tuple(extrinsics.shape)}")
            if next_extrinsics.shape[:2] != (B, T_query):
                raise ValueError(f"Expected next_extrinsics [B, {T_query}, E], got {tuple(next_extrinsics.shape)}")

        mask_probs = precomputed_mask_probs
        if mask_probs is None:
            mask_probs = torch.softmax(context_masks, dim=2)
        elif mask_probs.shape != context_masks.shape:
            raise ValueError(
                "precomputed_mask_probs must match context_masks, got "
                f"{tuple(mask_probs.shape)} and {tuple(context_masks.shape)}"
            )
        reuse_identical_inputs = not self.training and not torch.is_grad_enabled()
        target_mask_probs = (
            mask_probs
            if reuse_identical_inputs and context_masks_are_shared
            else torch.softmax(context_target_masks, dim=2)
        )
        query_source_mask_probs = (
            target_mask_probs[:, -1:].expand(-1, T_query, -1, -1)
            if reuse_identical_inputs and query_masks_are_shared
            else torch.softmax(query_source_masks, dim=2)
        )
        obj_latent_low = precomputed_obj_latent_low
        if obj_latent_low is None:
            obj_latent_low = self.recon_branch.obj_latent_reduce(obj_latent)
        elif obj_latent_low.shape[:2] != (B, K):
            raise ValueError(
                "precomputed_obj_latent_low must start with [B, K], got "
                f"{tuple(obj_latent_low.shape)}"
            )
        warped_mask_probs = external_warped_mask_probs
        if self.use_warp_concat:
            if warped_mask_probs is None:
                if not self.use_extrinsics:
                    raise ValueError("Warp concat requires extrinsics")
                with torch.no_grad():
                    warped_mask_probs = self._solve_warped_context_masks(
                        obj_latent_low.detach(),
                        mask_probs.detach(),
                        extrinsics.detach(),
                        next_extrinsics.detach(),
                    )
        if not self.use_warp_concat:
            recon_output = self.recon_branch(
                obj_latent,
                mask_probs,
                states,
                next_states,
                target_mask_probs=target_mask_probs,
                query_source_mask_probs=query_source_mask_probs,
                extrinsics=extrinsics,
                next_extrinsics=next_extrinsics,
                warped_mask_probs=None,
                precomputed_obj_latent_low=obj_latent_low,
                return_conditioning=return_proprio,
            )
            if return_proprio:
                recon_features, query_conditioning = recon_output
            else:
                recon_features = recon_output
            object_features = self.object_branch(
                obj_latent,
                mask_probs,
                states,
                next_states,
                target_mask_probs=target_mask_probs,
                query_source_mask_probs=query_source_mask_probs,
                extrinsics=extrinsics,
                next_extrinsics=next_extrinsics,
                warped_mask_probs=None,
            )

            recon_features = recon_features[:, :, None].expand(B, T_query, K, self.num_patches, self.predictor_embed_dim)
            fused_features = torch.cat([recon_features, object_features], dim=-1)
            logits = self._upsample_mask_logits(self.mask_head(fused_features))
            if return_proprio:
                if self.proprio_head is None:
                    raise RuntimeError(
                        "return_proprio=True requires proprio_output_dim at construction"
                    )
                pred_proprio = self.proprio_head(query_conditioning.float())
                if return_features:
                    return logits, pred_proprio, fused_features.contiguous()
                return logits, pred_proprio
            if return_features:
                return logits, fused_features.contiguous()
            return logits

        expanded_batch = B * T_query
        obj_latent_exp = obj_latent[:, None].expand(B, T_query, K, obj_latent.size(-1)).reshape(expanded_batch, K, obj_latent.size(-1))
        obj_latent_low_exp = obj_latent_low[:, None].expand(B, T_query, K, obj_latent_low.size(-1)).reshape(
            expanded_batch,
            K,
            obj_latent_low.size(-1),
        )
        mask_probs_exp = mask_probs[:, None].expand(B, T_query, T_ctx, K, N).reshape(expanded_batch, T_ctx, K, N)
        target_mask_probs_exp = (
            mask_probs_exp
            if target_mask_probs is mask_probs
            else target_mask_probs[:, None].expand(
                B, T_query, T_ctx, K, N
            ).reshape(expanded_batch, T_ctx, K, N)
        )
        query_source_mask_probs_exp = query_source_mask_probs.reshape(
            expanded_batch, 1, K, N
        )
        warped_mask_probs_exp = warped_mask_probs.reshape(expanded_batch, T_ctx, K, N)
        target_warped_mask_probs_exp = (
            warped_mask_probs_exp
            if external_warp_matches_context
            and target_mask_probs_exp is mask_probs_exp
            else target_mask_probs_exp
        )
        states_exp = states[:, None].expand(B, T_query, T_ctx, states.size(-1)).reshape(expanded_batch, T_ctx, states.size(-1))
        next_states_exp = next_states.reshape(expanded_batch, 1, next_states.size(-1))
        extrinsics_exp = None
        next_extrinsics_exp = None
        if extrinsics is not None:
            extrinsics_exp = extrinsics[:, None].expand(B, T_query, T_ctx, extrinsics.size(-1)).reshape(
                expanded_batch,
                T_ctx,
                extrinsics.size(-1),
            )
        if next_extrinsics is not None:
            next_extrinsics_exp = next_extrinsics.reshape(expanded_batch, 1, next_extrinsics.size(-1))

        t = None
        if self.debug_timing and torch.cuda.is_available():
            t = self._tic()
            t[0].record()
        recon_output = self.recon_branch(
            obj_latent_exp,
            mask_probs_exp,
            states_exp,
            next_states_exp,
            target_mask_probs=target_mask_probs_exp,
            query_source_mask_probs=query_source_mask_probs_exp,
            extrinsics=extrinsics_exp,
            next_extrinsics=next_extrinsics_exp,
            warped_mask_probs=warped_mask_probs_exp,
            target_warped_mask_probs=target_warped_mask_probs_exp,
            query_source_warped_mask_probs=query_source_mask_probs_exp,
            precomputed_obj_latent_low=obj_latent_low_exp,
            return_conditioning=return_proprio,
        )
        if return_proprio:
            recon_features, query_conditioning = recon_output
        else:
            recon_features = recon_output
        if self.debug_timing and torch.cuda.is_available():
            timing["predictor_time_recon_branch_ms"] = self._toc_ms(t)
            timing["predictor_mem_after_recon_branch_mb"] = self._cuda_mem_mb()

        t = None
        if self.debug_timing and torch.cuda.is_available():
            t = self._tic()
            t[0].record()
        object_features = self.object_branch(
            obj_latent_exp,
            mask_probs_exp,
            states_exp,
            next_states_exp,
            target_mask_probs=target_mask_probs_exp,
            query_source_mask_probs=query_source_mask_probs_exp,
            extrinsics=extrinsics_exp,
            next_extrinsics=next_extrinsics_exp,
            warped_mask_probs=warped_mask_probs_exp,
            target_warped_mask_probs=target_warped_mask_probs_exp,
            query_source_warped_mask_probs=query_source_mask_probs_exp,
        )
        if self.debug_timing and torch.cuda.is_available():
            timing["predictor_time_object_branch_ms"] = self._toc_ms(t)
            timing["predictor_mem_after_object_branch_mb"] = self._cuda_mem_mb()

        recon_features = recon_features[:, :, None].expand(
            expanded_batch,
            1,
            K,
            self.num_patches,
            self.predictor_embed_dim,
        )
        fused_features = torch.cat([recon_features, object_features], dim=-1)
        t = None
        if self.debug_timing and torch.cuda.is_available():
            t = self._tic()
            t[0].record()
        logits = self._upsample_mask_logits(self.mask_head(fused_features))
        if self.debug_timing and torch.cuda.is_available():
            timing["predictor_time_mask_head_ms"] = self._toc_ms(t)
            timing["predictor_mem_after_mask_head_mb"] = self._cuda_mem_mb()
            timing["predictor_mem_peak_mb"] = self._cuda_mem_mb()
            timing["predictor_mem_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            timing["predictor_time_total_ms"] = self._toc_ms(start_all)
            self.latest_timing = timing
        logits = logits.view(B, T_query, K, N).contiguous()
        if return_proprio:
            if self.proprio_head is None:
                raise RuntimeError(
                    "return_proprio=True requires proprio_output_dim at construction"
                )
            query_conditioning = query_conditioning.view(
                B, T_query, self.predictor_embed_dim
            )
            pred_proprio = self.proprio_head(query_conditioning.float())
            if return_features:
                fused_features = fused_features.view(
                    B, T_query, K, self.num_patches, -1
                ).contiguous()
                return logits, pred_proprio, fused_features
            return logits, pred_proprio
        if return_features:
            fused_features = fused_features.view(
                B, T_query, K, self.num_patches, -1
            ).contiguous()
            return logits, fused_features
        return logits
