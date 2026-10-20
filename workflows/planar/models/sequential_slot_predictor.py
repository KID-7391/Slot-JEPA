"""V-JEPA-style causal sequential predictor for Slot-AC ablation A1.

This keeps the global reconstructed-feature branch, the multi-scale object
branch, and the output heads of ``ActionConditionalMaskPredictor`` unchanged.
Unlike TMP, it never appends a learned target patch token or an observed target
transition.  A token block at time t contains only frame-t masks and the
frame-t state/action condition; its output is decoded as the prediction for
frame t+1.
"""

import torch
import torch.nn.functional as F

from slot_jepa.predictors import (
    ActionConditionalMaskPredictor,
)


def _causal_block_mask(cache, timesteps, block_tokens, device):
    key = (
        int(timesteps),
        int(block_tokens),
        device.type,
        device.index,
    )
    if key not in cache:
        temporal = torch.ones(
            timesteps, timesteps, dtype=torch.bool, device=device
        ).tril()
        cache[key] = temporal.repeat_interleave(
            block_tokens, dim=0
        ).repeat_interleave(block_tokens, dim=1)
    return cache[key]


def _run_causal_blocks(branch, blocks, timesteps):
    batch_size, _, block_tokens, dim = blocks.shape
    hidden = blocks.flatten(1, 2)
    attention_mask = _causal_block_mask(
        branch._sequential_attn_mask_cache,
        timesteps,
        block_tokens,
        hidden.device,
    )
    for block in branch.blocks:
        if branch.use_checkpoint and branch.training and torch.is_grad_enabled():
            hidden = torch.utils.checkpoint.checkpoint(
                block,
                hidden,
                mask=None,
                attn_mask=attention_mask,
                T=timesteps,
                H=branch.spatial_size,
                W=branch.spatial_size,
                action_tokens=branch.cond_tokens,
                use_reentrant=False,
            )
        else:
            hidden = block(
                hidden,
                mask=None,
                attn_mask=attention_mask,
                T=timesteps,
                H=branch.spatial_size,
                W=branch.spatial_size,
                action_tokens=branch.cond_tokens,
            )
    return hidden.view(batch_size, timesteps, block_tokens, dim)


class SequentialActionConditionalMaskPredictor(ActionConditionalMaskPredictor):
    """Causal t-to-t+1 variant with the same branch dimensions as TMP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # These TMP-only components construct learned target/query halves of a
        # transition pair. The sequential architecture consumes observed
        # source blocks directly, so keeping them would add dead parameters
        # and break ordinary DDP gradient reduction.
        self.recon_branch.query_tokens = None
        self.recon_branch.pair_fusion = None
        self.recon_branch._sequential_attn_mask_cache = {}
        for branch in self.object_branch.branches:
            branch.query_tokens = None
            branch.pair_fusion = None
            branch._sequential_attn_mask_cache = {}

    def _reconstruction_sequence(
        self, obj_latent, mask_probs, state_tokens, warped_mask_probs
    ):
        branch = self.recon_branch
        batch_size, timesteps = mask_probs.shape[:2]
        obj_latent_low = branch.obj_latent_reduce(obj_latent)
        patch_tokens = branch.predictor_embed(
            branch._build_input(mask_probs, warped_mask_probs, obj_latent_low)
        )
        dim = patch_tokens.shape[-1]
        condition = branch.state_encoder(state_tokens).view(
            batch_size, timesteps, 1, dim
        )
        obj_condition = branch.obj_encoder(obj_latent_low.flatten(1)).view(
            batch_size, 1, branch.num_obj_tokens, dim
        ).expand(-1, timesteps, -1, -1)
        parts = [condition]
        if branch.use_extrinsics:
            # Preserve the TMP token count and dimensions without exposing
            # state_{t+1}: the second condition is another projection of the
            # observed frame-t state/action token.
            parts.append(
                branch.extrinsics_encoder(state_tokens).view(
                    batch_size, timesteps, 1, dim
                )
            )
        parts.extend([obj_condition, patch_tokens])
        hidden = _run_causal_blocks(branch, torch.cat(parts, dim=2), timesteps)
        patch_features = branch.norm(hidden[:, :, branch.cond_tokens :])
        conditioning = branch.norm(hidden[:, :, 0])
        return patch_features, conditioning

    @staticmethod
    def _object_scale_sequence(
        branch, obj_latent, mask_probs, state_tokens, warped_mask_probs
    ):
        batch_size, timesteps, num_objects = mask_probs.shape[:3]
        obj_latent_low = branch.obj_latent_reduce(obj_latent)
        patch_tokens = branch._build_tokens(
            mask_probs, warped_mask_probs, obj_latent_low
        )
        state_per_object = state_tokens[:, None].expand(
            batch_size,
            num_objects,
            timesteps,
            state_tokens.shape[-1],
        ).reshape(
            batch_size * num_objects, timesteps, state_tokens.shape[-1]
        )
        condition = branch.state_encoder(state_per_object).view(
            batch_size * num_objects, timesteps, 1, branch.branch_dim
        )
        obj_condition = branch.obj_encoder(
            obj_latent.reshape(batch_size * num_objects, -1)
        ).view(
            batch_size * num_objects,
            1,
            branch.num_obj_tokens,
            branch.branch_dim,
        ).expand(-1, timesteps, -1, -1)
        parts = [condition]
        if branch.use_extrinsics:
            parts.append(
                branch.extrinsics_encoder(state_per_object).view(
                    batch_size * num_objects,
                    timesteps,
                    1,
                    branch.branch_dim,
                )
            )
        parts.extend([obj_condition, patch_tokens])
        hidden = _run_causal_blocks(branch, torch.cat(parts, dim=2), timesteps)
        output = branch.norm(hidden[:, :, branch.cond_tokens :])
        output = output.reshape(
            batch_size,
            num_objects,
            timesteps,
            branch.num_patches,
            branch.branch_dim,
        ).permute(0, 2, 1, 3, 4)
        if branch.spatial_size != branch.output_spatial_size:
            output = output.reshape(
                batch_size * timesteps * num_objects,
                branch.spatial_size,
                branch.spatial_size,
                branch.branch_dim,
            ).permute(0, 3, 1, 2)
            output = F.interpolate(
                output,
                size=(branch.output_spatial_size, branch.output_spatial_size),
                mode="bilinear",
                align_corners=False,
            )
            output = output.permute(0, 2, 3, 1).reshape(
                batch_size,
                timesteps,
                num_objects,
                branch.output_spatial_size**2,
                branch.branch_dim,
            )
        return output

    def forward_sequence(
        self,
        obj_latent,
        source_masks,
        state_tokens,
        return_features=False,
        return_proprio=False,
    ):
        """Predict frame t+1 at every input position t.

        Args:
            obj_latent: ``[B, K, D_obj]`` fixed object latents.
            source_masks: ``[B, T, K, N]`` frame-t mask logits.
            state_tokens: ``[B, T, D_state]`` frame-t state/action tokens.
        """
        if source_masks.ndim != 4:
            raise ValueError(
                f"Expected source_masks [B, T, K, N], got {tuple(source_masks.shape)}"
            )
        batch_size, timesteps, num_objects, num_patches = source_masks.shape
        if not 1 <= timesteps <= self.num_context_frames:
            raise ValueError(
                f"Expected sequence length in [1, {self.num_context_frames}], "
                f"got {timesteps}"
            )
        if num_objects != self.num_objects or num_patches != self.input_num_patches:
            raise ValueError("Sequential mask shape does not match predictor")
        if state_tokens.shape[:2] != (batch_size, timesteps):
            raise ValueError("Sequential state/action tokens do not align with masks")

        mask_probs = torch.softmax(source_masks, dim=2)
        # The original branches reserve the warp-concat channel. A1 keeps its
        # width but fills it only with observed frame-t masks, never t+1 data.
        warped_mask_probs = mask_probs if self.use_warp_concat else None
        recon_features, conditioning = self._reconstruction_sequence(
            obj_latent, mask_probs, state_tokens, warped_mask_probs
        )
        object_features = torch.cat(
            [
                self._object_scale_sequence(
                    branch,
                    obj_latent,
                    mask_probs,
                    state_tokens,
                    warped_mask_probs,
                )
                for branch in self.object_branch.branches
            ],
            dim=-1,
        )
        recon_features = recon_features[:, :, None].expand(
            batch_size,
            timesteps,
            num_objects,
            self.num_patches,
            self.predictor_embed_dim,
        )
        fused = torch.cat([recon_features, object_features], dim=-1)
        logits = self._upsample_mask_logits(self.mask_head(fused))
        if return_proprio:
            if self.proprio_head is None:
                raise RuntimeError("return_proprio requires a proprio head")
            proprio = self.proprio_head(conditioning.float())
            if return_features:
                return logits, proprio, fused.contiguous()
            return logits, proprio
        if return_features:
            return logits, fused.contiguous()
        return logits
