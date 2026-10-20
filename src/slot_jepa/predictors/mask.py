import torch
import torch.nn as nn
import torch.utils.checkpoint
from slot_jepa.predictors.components.modules import ACBlock as Block
from slot_jepa.predictors.components.modules import build_action_block_causal_attention_mask
from slot_jepa.utils.tensors import trunc_normal_


class ObjectiveMaskPredictor(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        predictor_embed_dim=256,
        bottleneck_dim=256,
        depth=8,
        num_heads=4,
        mlp_ratio=4.0,
        drop_rate=0.0,
        num_context_frames=7,
        spatial_size=24,
        num_objects=16,
        num_inverse_tokens=4,
        use_checkpoint=False,
        inverse_depth=8,
        inverse_heads=4,
    ):
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_context_frames = num_context_frames
        self.spatial_size = spatial_size
        self.num_patches = spatial_size * spatial_size
        self.num_objects = num_objects
        self.num_inverse_tokens = num_inverse_tokens
        self.use_checkpoint = use_checkpoint
        self.num_action_tokens = self.num_objects + self.num_inverse_tokens

        self.obj_latent_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.context_mask_embed = nn.Linear(self.num_objects, predictor_embed_dim)

        self.inverse_obj_proj = nn.Linear(embed_dim, predictor_embed_dim)
        self.inverse_mask_proj = nn.Linear(2 * self.num_objects, predictor_embed_dim)
        self.inverse_spatial_cls = nn.Parameter(torch.zeros(1, self.num_inverse_tokens, predictor_embed_dim))
        self.inverse_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=predictor_embed_dim,
                nhead=inverse_heads,
                dim_feedforward=int(mlp_ratio * predictor_embed_dim),
                dropout=drop_rate,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(inverse_depth)
        ])
        self.inverse_norm = nn.LayerNorm(predictor_embed_dim)
        self.inverse_mlp = nn.Sequential(
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, bottleneck_dim),
        )
        self.bottleneck_up_proj = nn.Linear(bottleneck_dim, predictor_embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, predictor_embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                use_rope=True,
                grid_size=spatial_size,
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=0.0,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.pred_proj = nn.Linear(predictor_embed_dim, self.num_objects)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode_inverse(self, obj_latent, prev_masks, next_masks):
        # obj_latent: [B, K, D]
        # prev_masks/next_masks: [B, T_ctx, K, N]
        B, T, K, N = prev_masks.shape
        if K != self.num_objects:
            raise ValueError(f"Expected {self.num_objects} objects, got {K}")
        pair_masks = torch.cat([prev_masks, next_masks], dim=2).permute(0, 1, 3, 2)  # [B, T, N, 2K]
        mask_tokens = self.inverse_mask_proj(pair_masks)
        mask_tokens = mask_tokens + self.spatial_pos_embed[:, :N].to(mask_tokens.dtype).unsqueeze(1)
        obj_token = self.inverse_obj_proj(obj_latent).mean(dim=1, keepdim=True).to(mask_tokens.dtype)
        obj_token = obj_token.unsqueeze(1).expand(-1, T, -1, -1)
        spatial_cls = self.inverse_spatial_cls.to(mask_tokens.dtype).expand(B, T, -1, -1)
        mask_tokens = torch.cat([spatial_cls, obj_token, mask_tokens], dim=2).reshape(
            B * T,
            N + 1 + self.num_inverse_tokens,
            -1,
        )

        for blk in self.inverse_blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                mask_tokens = torch.utils.checkpoint.checkpoint(
                    blk, mask_tokens, use_reentrant=False
                )
            else:
                mask_tokens = blk(mask_tokens)

        mask_tokens = self.inverse_norm(mask_tokens)
        inverse_tokens = self.inverse_mlp(mask_tokens[:, :self.num_inverse_tokens]).view(B, T, self.num_inverse_tokens, -1)
        return inverse_tokens

    def forward(self, obj_latent, context_masks, inverse_tokens):
        # obj_latent: [B, K, D]
        # context_masks: [B, T_ctx, K, N]
        B, T_ctx, K, N = context_masks.shape
        if T_ctx < 1 or T_ctx > self.num_context_frames:
            raise ValueError(f"Expected context length in [1, {self.num_context_frames}], got {T_ctx}")
        if K != self.num_objects:
            raise ValueError(f"Expected {self.num_objects} objects, got {K}")
        if inverse_tokens.shape[:3] != (B, T_ctx, self.num_inverse_tokens):
            raise ValueError(
                f"Expected inverse tokens with shape [B, {T_ctx}, {self.num_inverse_tokens}, C], got {tuple(inverse_tokens.shape)}"
            )
        inverse_tokens = self.bottleneck_up_proj(inverse_tokens)

        obj_tokens = self.obj_latent_embed(obj_latent)
        context_embed = self.context_mask_embed(context_masks.permute(0, 1, 3, 2))
        spatial_pos = self.spatial_pos_embed.to(context_embed.dtype).unsqueeze(1)
        context_embed = context_embed + spatial_pos

        ctx_action = torch.cat(
            [
                obj_tokens.unsqueeze(1).expand(-1, T_ctx, -1, -1),
                inverse_tokens,
            ],
            dim=2,
        )
        ctx_frames = torch.cat([ctx_action, context_embed], dim=2).flatten(1, 2)
        attn_mask = build_action_block_causal_attention_mask(
            T_ctx, self.spatial_size, self.spatial_size, add_tokens=self.num_action_tokens
        ).to(ctx_frames.device, non_blocking=True)

        x = ctx_frames

        for blk in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                def block_forward(x_in):
                    return blk(
                        x_in,
                        attn_mask=attn_mask,
                        action_tokens=self.num_action_tokens,
                        T=T_ctx,
                        H=self.spatial_size,
                        W=self.spatial_size,
                    )
                x = torch.utils.checkpoint.checkpoint(block_forward, x, use_reentrant=False)
            else:
                x = blk(
                    x,
                    attn_mask=attn_mask,
                    action_tokens=self.num_action_tokens,
                    T=T_ctx,
                    H=self.spatial_size,
                    W=self.spatial_size,
                )

        x = x.view(B, T_ctx, self.num_action_tokens + N, -1)
        target_tokens = x[:, :, self.num_action_tokens:, :]
        target_tokens = self.norm(target_tokens)
        next_logits = self.pred_proj(target_tokens).permute(0, 1, 3, 2)
        return next_logits, inverse_tokens
