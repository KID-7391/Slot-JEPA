import torch
import torch.nn as nn
import torch.utils.checkpoint

from slot_jepa.utils.tensors import trunc_normal_


class InverseConditionModule(nn.Module):
    """
    Infer pseudo conditions from slot reconstructions for video-only pretraining.

    These pseudo conditions are only used during video-only pretraining so that
    the downstream SLOT-AC predictor sees the same input interface as the
    action-conditioned finetuning stage.
    """

    def __init__(
        self,
        embed_dim=768,
        feature_dim=196,
        predictor_embed_dim=192,
        num_objects=24,
        state_dim=7,
        extrinsics_dim=6,
        num_condition_tokens=3,
        depth=8,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.0,
        spatial_size=24,
        concat_mask_input=True,
        use_checkpoint=False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_objects = num_objects
        self.state_dim = state_dim
        self.extrinsics_dim = extrinsics_dim
        self.num_condition_tokens = num_condition_tokens
        self.predictor_embed_dim = predictor_embed_dim
        self.spatial_size = spatial_size
        self.num_patches = spatial_size * spatial_size
        self.concat_mask_input = concat_mask_input
        self.use_checkpoint = use_checkpoint

        self.obj_latent_reduce = nn.Linear(embed_dim, feature_dim)
        inverse_in_dim = 2 * feature_dim + (2 * num_objects if concat_mask_input else 0)
        self.pair_embed = nn.Linear(inverse_in_dim, predictor_embed_dim)
        self.obj_proj = nn.Linear(num_objects * feature_dim, predictor_embed_dim)
        frame_in_dim = feature_dim + (num_objects if concat_mask_input else 0)
        self.frame_proj = nn.Linear(frame_in_dim, predictor_embed_dim)
        self.condition_cls = nn.Parameter(torch.zeros(1, num_condition_tokens, predictor_embed_dim))
        self.frame_condition_cls = nn.Parameter(torch.zeros(1, 2, predictor_embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, predictor_embed_dim))

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=predictor_embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(mlp_ratio * predictor_embed_dim),
                    dropout=drop_rate,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.frame_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=predictor_embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(mlp_ratio * predictor_embed_dim),
                    dropout=drop_rate,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.frame_norm = nn.LayerNorm(predictor_embed_dim)
        self.state_head = nn.Linear(predictor_embed_dim, state_dim)
        self.extrinsics_head = nn.Linear(predictor_embed_dim, extrinsics_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, obj_latent, prev_masks, next_masks, all_masks=None):
        """
        Args:
            obj_latent: [B, K, D]
            prev_masks: [B, T, K, N]
            next_masks: [B, T, K, N]

        Returns:
            pseudo_states: [B, T_all, state_dim]
            pseudo_extrinsics: [B, T_all, extrinsics_dim]
        """
        if prev_masks.shape != next_masks.shape:
            raise ValueError(
                f"Expected prev_masks and next_masks to match, got {tuple(prev_masks.shape)} vs {tuple(next_masks.shape)}"
            )
        B, T_ctx, K, N = prev_masks.shape
        if K != self.num_objects:
            raise ValueError(f"Expected {self.num_objects} objects, got {K}")
        if N != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {N}")
        if all_masks is None:
            all_masks = torch.cat([prev_masks, next_masks[:, -1:]], dim=1)
        if all_masks.ndim != 4 or all_masks.shape[0] != B or all_masks.shape[2:] != (K, N):
            raise ValueError(
                f"Expected all_masks [B, T_all, {K}, {N}], got {tuple(all_masks.shape)}"
            )

        obj_latent_low = self.obj_latent_reduce(obj_latent)
        prev_probs = torch.softmax(prev_masks, dim=2)
        next_probs = torch.softmax(next_masks, dim=2)
        all_probs = torch.softmax(all_masks, dim=2)
        prev_recon = torch.einsum("btkn,bkd->btnd", prev_probs.to(obj_latent_low.dtype), obj_latent_low)
        next_recon = torch.einsum("btkn,bkd->btnd", next_probs.to(obj_latent_low.dtype), obj_latent_low)
        all_recon = torch.einsum("btkn,bkd->btnd", all_probs.to(obj_latent_low.dtype), obj_latent_low)

        if self.concat_mask_input:
            prev_mask_input = prev_probs.permute(0, 1, 3, 2).to(prev_recon.dtype)
            next_mask_input = next_probs.permute(0, 1, 3, 2).to(prev_recon.dtype)
            pair_input = torch.cat([prev_recon, next_recon, prev_mask_input, next_mask_input], dim=-1)
        else:
            pair_input = torch.cat([prev_recon, next_recon], dim=-1)

        pair_tokens = self.pair_embed(pair_input)
        pair_tokens = pair_tokens + self.spatial_pos_embed[:, :N].to(pair_tokens.dtype).unsqueeze(1)

        obj_token = self.obj_proj(obj_latent_low.flatten(1)).view(B, 1, -1).to(pair_tokens.dtype)
        obj_token = obj_token.unsqueeze(1).expand(-1, T_ctx, -1, -1)
        condition_cls = self.condition_cls.to(pair_tokens.dtype).expand(B, T_ctx, -1, -1)

        x = torch.cat([condition_cls, obj_token, pair_tokens], dim=2).reshape(
            B * T_ctx,
            self.num_condition_tokens + 1 + N,
            self.predictor_embed_dim,
        )

        for blk in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm(x).view(B, T_ctx, self.num_condition_tokens + 1 + N, self.predictor_embed_dim)
        condition_tokens = x[:, :, : self.num_condition_tokens]
        total_frames = all_recon.size(1)
        if self.concat_mask_input:
            all_mask_input = all_probs.permute(0, 1, 3, 2).to(all_recon.dtype)
            frame_input = torch.cat([all_recon, all_mask_input], dim=-1)
        else:
            frame_input = all_recon
        frame_tokens = self.frame_proj(frame_input)
        frame_tokens = frame_tokens + self.spatial_pos_embed[:, :N].to(frame_tokens.dtype).unsqueeze(1)
        frame_obj_token = self.obj_proj(obj_latent_low.flatten(1)).view(B, 1, 1, -1).to(frame_tokens.dtype)
        frame_obj_token = frame_obj_token.expand(-1, total_frames, -1, -1)
        frame_condition_cls = self.frame_condition_cls.to(frame_tokens.dtype).expand(B, total_frames, -1, -1)
        frame_tokens = torch.cat([frame_condition_cls, frame_obj_token, frame_tokens], dim=2).reshape(
            B * total_frames,
            N + 3,
            self.predictor_embed_dim,
        )
        for blk in self.frame_blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                frame_tokens = torch.utils.checkpoint.checkpoint(blk, frame_tokens, use_reentrant=False)
            else:
                frame_tokens = blk(frame_tokens)
        frame_tokens = self.frame_norm(frame_tokens).view(B, total_frames, N + 3, self.predictor_embed_dim)
        pseudo_states = self.state_head(frame_tokens[:, :, 0])
        pseudo_extrinsics = self.extrinsics_head(frame_tokens[:, :, 1])
        return pseudo_states, pseudo_extrinsics
