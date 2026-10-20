# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import drop_path


def build_action_block_causal_attention_mask(T, H, W, add_tokens=1):
    N_T = add_tokens + (H * W)
    N = T * N_T
    mask = torch.zeros(N, N).bool()
    mask_block = torch.ones(N_T, N_T).bool()
    local_window_time = T

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block

    return mask


def _build_rope_factors(dim, dtype, device, pos):
    omega = torch.arange(dim // 2, dtype=dtype, device=device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freq.sin()
    emb_cos = freq.cos()
    # Preserve the pretrained model's historical pair-frequency layout.
    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
    return emb_sin, emb_cos


def rotate_queries_or_keys(x, pos, rope_factors=None):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position
    if rope_factors is None:
        rope_factors = _build_rope_factors(D, x.dtype, x.device, pos)
    emb_sin, emb_cos = rope_factors

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class ACRoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        self._position_cache = {}
        self._rope_factor_cache = {}

    @staticmethod
    def _device_key(device):
        return device.type, device.index

    def _cached_positions(self, T, H, W, device, time_offset=0):
        key = (
            int(T),
            int(H),
            int(W),
            int(time_offset),
            float(self.grid_size),
            *self._device_key(device),
        )
        cached = self._position_cache.get(key)
        if cached is not None:
            return key, cached
        ids = torch.arange(int(T * H * W), device=device)
        d_pos, h_pos, w_pos = self.separate_positions(ids, H, W)
        if time_offset:
            d_pos = d_pos + int(time_offset)
        h_pos = h_pos * (self.grid_size / H)
        w_pos = w_pos * (self.grid_size / W)
        frame_positions = torch.arange(
            int(time_offset), int(time_offset) + int(T), device=device
        )
        cached = (d_pos, h_pos, w_pos, frame_positions)
        self._position_cache[key] = cached
        return key, cached

    def _cached_rotate(self, x, pos, position_key, component):
        factor_key = (
            position_key,
            component,
            x.shape[-1],
            x.dtype,
            *self._device_key(x.device),
        )
        factors = self._rope_factor_cache.get(factor_key)
        if factors is None:
            factors = _build_rope_factors(
                x.shape[-1], x.dtype, x.device, pos
            )
            self._rope_factor_cache[factor_key] = factors
        return rotate_queries_or_keys(x, pos, rope_factors=factors)

    def _get_frame_pos(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches, W_patches):
        # Remove frame component from ids
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        B, N, C = x.size()

        # -- compute position of each frame token
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)
            h_mask *= self.grid_size / H
            w_mask *= self.grid_size / W
            position_key = None
            frame_positions = torch.arange(T, device=x.device)
        else:
            position_key, positions = self._cached_positions(
                T, H, W, x.device
            )
            d_mask, h_mask, w_mask, frame_positions = positions

        # -- split out action tokens from sequence
        if action_tokens > 0:
            x = x.view(B, -1, action_tokens + H * W, C)  # [B, T, 1+H*W, D]

            action_q, action_k, action_v = [], [], []
            for i in range(action_tokens):
                a = x[:, :, i : i + 1, :].flatten(1, 2)
                # Note action tokens do not work with masking
                # -- compute qkv for action tokens and rotate
                qkv = self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
                # --
                if position_key is None:
                    qd = rotate_queries_or_keys(
                        q[..., : self.d_dim], pos=frame_positions
                    )
                    kd = rotate_queries_or_keys(
                        k[..., : self.d_dim], pos=frame_positions
                    )
                else:
                    qd = self._cached_rotate(
                        q[..., : self.d_dim],
                        frame_positions,
                        position_key,
                        "action_depth",
                    )
                    kd = self._cached_rotate(
                        k[..., : self.d_dim],
                        frame_positions,
                        position_key,
                        "action_depth",
                    )
                qr = q[..., self.d_dim :]
                kr = k[..., self.d_dim :]
                action_q += [torch.cat([qd, qr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_k += [torch.cat([kd, kr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_v += [v.view(B, self.num_heads, T, 1, -1)]

            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)

        # -- compute qkv for frame tokens and rotate
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        s = 0
        # Rotate depth
        rotate = (
            lambda tensor, pos, component: rotate_queries_or_keys(tensor, pos)
            if position_key is None
            else self._cached_rotate(
                tensor, pos, position_key, component
            )
        )
        qd = rotate(q[..., s : s + self.d_dim], d_mask, "patch_depth")
        kd = rotate(k[..., s : s + self.d_dim], d_mask, "patch_depth")
        s += self.d_dim
        # Rotate height dim
        qh = rotate(q[..., s : s + self.h_dim], h_mask, "patch_height")
        kh = rotate(k[..., s : s + self.h_dim], h_mask, "patch_height")
        s += self.h_dim
        # Rotate width dim
        qw = rotate(q[..., s : s + self.w_dim], w_mask, "patch_width")
        kw = rotate(k[..., s : s + self.w_dim], w_mask, "patch_width")
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens > 0:

            def merge_(tx, ta):
                """tx, tx in [B, num_heads, N, D]"""
                tx = tx.view(B, self.num_heads, T, H * W, -1)  # [B, T, H*W, D]
                ta = ta.view(B, self.num_heads, T, action_tokens, -1)  # [B, T, A, D]
                return torch.cat([ta, tx], dim=3).flatten(2, 3)

            q = merge_(q, action_q)
            k = merge_(k, action_k)
            v = merge_(v, action_v)

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _project_qkv_at_time(self, x, T, H, W, action_tokens, time_offset):
        """Project tokens exactly as ``forward``, with an explicit RoPE time offset."""
        B, _, C = x.shape
        position_key, positions = self._cached_positions(
            T, H, W, x.device, time_offset=time_offset
        )
        d_pos, h_pos, w_pos, frame_positions = positions
        x = x.view(B, T, action_tokens + H * W, C)

        action_q, action_k, action_v = [], [], []
        for index in range(action_tokens):
            action = x[:, :, index : index + 1].flatten(1, 2)
            qkv = self.qkv(action).unflatten(
                -1, (3, self.num_heads, -1)
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            qd = self._cached_rotate(
                q[..., : self.d_dim],
                frame_positions,
                position_key,
                "action_depth",
            )
            kd = self._cached_rotate(
                k[..., : self.d_dim],
                frame_positions,
                position_key,
                "action_depth",
            )
            action_q.append(
                torch.cat([qd, q[..., self.d_dim :]], dim=-1).view(
                    B, self.num_heads, T, 1, -1
                )
            )
            action_k.append(
                torch.cat([kd, k[..., self.d_dim :]], dim=-1).view(
                    B, self.num_heads, T, 1, -1
                )
            )
            action_v.append(v.view(B, self.num_heads, T, 1, -1))

        patches = x[:, :, action_tokens:].flatten(1, 2)
        qkv = self.qkv(patches).unflatten(
            -1, (3, self.num_heads, -1)
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        split = 0
        qd = self._cached_rotate(
            q[..., split : split + self.d_dim],
            d_pos,
            position_key,
            "patch_depth",
        )
        kd = self._cached_rotate(
            k[..., split : split + self.d_dim],
            d_pos,
            position_key,
            "patch_depth",
        )
        split += self.d_dim
        qh = self._cached_rotate(
            q[..., split : split + self.h_dim],
            h_pos,
            position_key,
            "patch_height",
        )
        kh = self._cached_rotate(
            k[..., split : split + self.h_dim],
            h_pos,
            position_key,
            "patch_height",
        )
        split += self.h_dim
        qw = self._cached_rotate(
            q[..., split : split + self.w_dim],
            w_pos,
            position_key,
            "patch_width",
        )
        kw = self._cached_rotate(
            k[..., split : split + self.w_dim],
            w_pos,
            position_key,
            "patch_width",
        )
        split += self.w_dim
        if split < self.head_dim:
            q = torch.cat([qd, qh, qw, q[..., split:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., split:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens:
            action_q = torch.cat(action_q, dim=3)
            action_k = torch.cat(action_k, dim=3)
            action_v = torch.cat(action_v, dim=3)

            def merge(patch_tokens, condition_tokens):
                patch_tokens = patch_tokens.view(
                    B, self.num_heads, T, H * W, -1
                )
                return torch.cat([condition_tokens, patch_tokens], dim=3).flatten(2, 3)

            q = merge(q, action_q)
            k = merge(k, action_k)
            v = merge(v, action_v)
        return q, k, v

    def prepare_context_kv(self, x, T, H, W, action_tokens=0):
        """Return the context keys/values needed by later query-only passes."""
        _, key, value = self._project_qkv_at_time(
            x, T=T, H=H, W=W, action_tokens=action_tokens, time_offset=0
        )
        return key, value

    def forward_query(self, x, context_kv, T_ctx, H, W, action_tokens=0):
        """Evaluate one query block against precomputed context keys/values."""
        B, N, C = x.shape
        query, query_key, query_value = self._project_qkv_at_time(
            x,
            T=1,
            H=H,
            W=W,
            action_tokens=action_tokens,
            time_offset=T_ctx,
        )
        context_key, context_value = context_kv
        key = torch.cat([context_key, query_key], dim=2)
        value = torch.cat([context_value, query_value], dim=2)
        with torch.backends.cuda.sdp_kernel():
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.proj_drop_prob,
                is_causal=False,
            )
        output = output.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(output))


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        # Remove frame component from ids
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ACBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = ACRoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        y = self.norm1(x)
        if isinstance(self.attn, ACRoPEAttention):
            y = self.attn(y, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=action_tokens)
        else:
            y = self.attn(y, mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        y = self.norm2(x)
        x = x + self.drop_path(self.mlp(y))
        return x

    def prepare_context(self, x, T, H, W, action_tokens=0):
        """Advance context tokens and retain this layer's normalized KV cache."""
        if not isinstance(self.attn, ACRoPEAttention):
            raise TypeError("Context caching requires ACRoPEAttention")
        normalized = self.norm1(x)
        context_kv = self.attn.prepare_context_kv(
            normalized, T=T, H=H, W=W, action_tokens=action_tokens
        )
        attended = self.attn(
            normalized, T=T, H=H, W=W, action_tokens=action_tokens
        )
        x = x + self.drop_path(attended)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, context_kv

    def forward_query(self, x, context_kv, T_ctx, H, W, action_tokens=0):
        """Advance a single query block against cached context KV."""
        if not isinstance(self.attn, ACRoPEAttention):
            raise TypeError("Context caching requires ACRoPEAttention")
        attended = self.attn.forward_query(
            self.norm1(x),
            context_kv=context_kv,
            T_ctx=T_ctx,
            H=H,
            W=W,
            action_tokens=action_tokens,
        )
        x = x + self.drop_path(attended)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        # self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        return q


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q
