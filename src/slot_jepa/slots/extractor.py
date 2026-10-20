import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math


def generate_3d_pe(T, H, W, D, device):
    t, y, x = torch.meshgrid(
        torch.arange(T, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
    pe_y = torch.zeros(T, H, W, D, device=device)
    pe_x = torch.zeros(T, H, W, D, device=device)
    
    pe_y[..., 0::2] = torch.sin(y.unsqueeze(-1) * div_term)
    pe_y[..., 1::2] = torch.cos(y.unsqueeze(-1) * div_term)
    
    pe_x[..., 0::2] = torch.sin(x.unsqueeze(-1) * div_term)
    pe_x[..., 1::2] = torch.cos(x.unsqueeze(-1) * div_term)
    
    # Only 2D spatial positional encoding replicated across T frames
    pe_3d = pe_y + pe_x
    
    return pe_3d


class TwoStageSlotAttentionMerge(nn.Module):
    def __init__(self, window_size=[2, 6], slots_per_window=4, global_k=16, D_in: int = 768, iters=3, mask_temp: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.slots_per_window = slots_per_window
        self.global_k = global_k
        self.iters = iters
        self.D_in = D_in
        self.mask_temp = mask_temp
        
        # Stage 1 (Local) Parameters
        self.norm_slots_1 = nn.LayerNorm(D_in)
        self.norm_inputs_1_local = nn.LayerNorm(D_in)
        self.q_proj_1 = nn.Linear(D_in, D_in)
        self.k_proj_1 = nn.Linear(D_in, D_in)
        self.v_proj_1 = nn.Linear(D_in, D_in)
        self.gru_1 = nn.GRUCell(D_in, D_in)
        self.mlp_1 = nn.Sequential(
            nn.Linear(D_in, D_in * 2),
            nn.ReLU(),
            nn.Linear(D_in * 2, D_in)
        )
        self.norm_pre_mlp_1 = nn.LayerNorm(D_in)
        
        # Stage 2 (Global) Parameters
        self.norm_slots_2 = nn.LayerNorm(D_in)
        self.norm_inputs_2_global = nn.LayerNorm(D_in)
        self.q_proj_2 = nn.Linear(D_in, D_in)
        self.k_proj_2 = nn.Linear(D_in, D_in)
        self.v_proj_2 = nn.Linear(D_in, D_in)
        self.gru_2 = nn.GRUCell(D_in, D_in)
        self.mlp_2 = nn.Sequential(
            nn.Linear(D_in, D_in * 2),
            nn.ReLU(),
            nn.Linear(D_in * 2, D_in)
        )
        self.norm_pre_mlp_2 = nn.LayerNorm(D_in)

    def _slot_attention(self, inputs, slots, q_proj, k_proj, v_proj, gru, mlp, norm_inputs, norm_slots, norm_pre_mlp, input_mask=None):
        B_eff, N, D_in = inputs.shape
        num_slots = slots.shape[1]
        
        inputs_norm = norm_inputs(inputs)
        k = k_proj(inputs_norm)
        v = v_proj(inputs_norm)
        
        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = norm_slots(slots)
            q = q_proj(slots_norm)
            
            attn_logits = torch.bmm(k, q.transpose(1, 2)) * (self.D_in ** -0.5)
            attn = F.softmax(attn_logits / self.mask_temp, dim=-1)
            if input_mask is not None:
                attn = attn * input_mask.unsqueeze(-1).to(attn.dtype)
            attn_norm = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
            
            updates = torch.bmm(attn_norm.transpose(1, 2), v)
            
            slots = gru(updates.reshape(-1, self.D_in), slots_prev.reshape(-1, self.D_in)).reshape(B_eff, num_slots, self.D_in)
            slots = slots + mlp(norm_pre_mlp(slots))
        return slots, attn

    def _init_slots_from_valid_tokens(self, tokens, valid_mask, num_slots):
        B_eff, _, D = tokens.shape
        init_slots = torch.zeros(B_eff, num_slots, D, device=tokens.device, dtype=tokens.dtype)
        for b in range(B_eff):
            valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            if valid_idx.numel() >= num_slots:
                select = torch.linspace(
                    0, valid_idx.numel() - 1, steps=num_slots, device=tokens.device
                ).long()
                picked = valid_idx[select]
            else:
                repeat = torch.arange(num_slots, device=tokens.device) % valid_idx.numel()
                picked = valid_idx[repeat]
            init_slots[b] = tokens[b, picked]
        return init_slots

    def farthest_point_sampling(self, points, K, valid_mask=None):
        B, N, D = points.shape
        device = points.device
        centroids = torch.zeros(B, K, D, device=device)

        if valid_mask is None:
            valid_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        for b in range(B):
            valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            farthest = valid_idx[torch.randint(0, valid_idx.numel(), (1,), device=device)].item()
            dist_b = torch.ones(N, device=device) * 1e10
            for i in range(K):
                centroid = points[b, farthest]
                centroids[b, i] = centroid
                dist = torch.sum((points[b] - centroid.unsqueeze(0)) ** 2, -1)
                dist_b = torch.minimum(dist_b, dist)
                dist_b = torch.where(valid_mask[b], dist_b, torch.full_like(dist_b, -1.0))
                farthest = torch.argmax(dist_b).item()
            
        return centroids

    def forward(self, Z0_5d: torch.Tensor, valid_mask: torch.Tensor = None):
        B, T, H, W, D = Z0_5d.shape
        if valid_mask is None:
            valid_mask = torch.ones(B, T, H, W, dtype=torch.bool, device=Z0_5d.device)
        
        T_w, S_w = self.window_size
        T_pad = (T_w - (T % T_w)) % T_w if T % T_w != 0 else 0
        H_pad = (S_w - (H % S_w)) % S_w if H % S_w != 0 else 0
        W_pad = (S_w - (W % S_w)) % S_w if W % S_w != 0 else 0
        
        if T_pad > 0 or H_pad > 0 or W_pad > 0:
            Z0_5d = F.pad(Z0_5d, (0, 0, 0, W_pad, 0, H_pad, 0, T_pad))
            valid_mask = F.pad(valid_mask.float(), (0, W_pad, 0, H_pad, 0, T_pad)).bool()
            
        _, T_p, H_p, W_p, _ = Z0_5d.shape
        
        num_windows_t = T_p // T_w
        num_windows_h = H_p // S_w
        num_windows_w = W_p // S_w
        
        Z0_win = Z0_5d.view(B, num_windows_t, T_w, num_windows_h, S_w, num_windows_w, S_w, D)
        windows = Z0_win.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(B * num_windows_t * num_windows_h * num_windows_w, T_w * S_w * S_w, D)
        valid_win = valid_mask.view(B, num_windows_t, T_w, num_windows_h, S_w, num_windows_w, S_w)
        valid_windows = valid_win.permute(0, 1, 3, 5, 2, 4, 6).reshape(B * num_windows_t * num_windows_h * num_windows_w, T_w * S_w * S_w)
        
        initial_slots_1 = self._init_slots_from_valid_tokens(windows, valid_windows, self.slots_per_window)

        local_slots, _ = self._slot_attention(
            windows.contiguous(), initial_slots_1,
            self.q_proj_1, self.k_proj_1, self.v_proj_1, self.gru_1, self.mlp_1,
            self.norm_inputs_1_local, self.norm_slots_1, self.norm_pre_mlp_1,
            input_mask=valid_windows
        )
        
        num_windows_total = num_windows_t * num_windows_h * num_windows_w
        global_inputs = local_slots.view(B, num_windows_total * self.slots_per_window, D)
        window_valid = valid_windows.view(B, num_windows_total, -1).any(dim=-1)
        global_valid = window_valid.unsqueeze(-1).expand(B, num_windows_total, self.slots_per_window).reshape(B, num_windows_total * self.slots_per_window)
        
        global_initial_slots = self.farthest_point_sampling(global_inputs, self.global_k, valid_mask=global_valid)
        
        final_slots, _ = self._slot_attention(
            global_inputs.contiguous(), global_initial_slots,
            self.q_proj_2, self.k_proj_2, self.v_proj_2, self.gru_2, self.mlp_2,
            self.norm_inputs_2_global, self.norm_slots_2, self.norm_pre_mlp_2,
            input_mask=global_valid
        )
        
        return final_slots, None

def extract_instance_latents(mask_out, Z0_5d, L_max=24):
    """
    基于 m_{k,t} 提取各个 k 对应的最大激活帧，然后划分连通域，重新提特征得到实例 latents s_k
    mask_out: [B, T, K, N]
    Z0_5d: [B, T, H, W, D]
    返回: s_latents [B, K, L_max, D] 或打平表示等, 并在前向传递
    """
    B, T, K, N = mask_out.shape
    H, W = Z0_5d.shape[2], Z0_5d.shape[3]
    D = Z0_5d.shape[-1]
    device = mask_out.device
    mask_out = mask_out.detach()

    s_latents = torch.zeros(B, L_max, D, device=device)
    s_masks = torch.zeros(B, L_max, H, W, device=device)
    s_valid = torch.zeros(B, L_max, dtype=torch.bool, device=device)
    s_frame_idx = torch.zeros(B, L_max, dtype=torch.long, device=device)

    for b in range(B):
        all_comp_scores = []
        for k in range(K):
            # 找到面积最大的时刻 t_max
            # mask_out: [B, T, K, N]
            mask_btk = mask_out[b, :, k, :] # [T, N]
            
            mask_area_t = mask_btk.sum(dim=-1) # [T]
            t_max = mask_area_t.argmax().item()
            
            m_t_max = mask_btk[t_max].view(H, W) # [H, W]
            
            # 1. 唯一性初始化 (Initialization)
            bin_mask = (m_t_max > 0.3).float() # 二值化前景 Mask
            if bin_mask.sum() == 0:
                continue
                
            ids = torch.arange(1, H * W + 1, device=device).view(1, 1, H, W).float()
            curr_ids = ids * bin_mask.view(1, 1, H, W)
            
            # 迭代收敛 (Iteration)
            while True:
                # 2. 局部通信 (Max-Pooling)
                next_ids = F.max_pool2d(curr_ids, kernel_size=3, stride=1, padding=1)
                # 3. 约束边界 (Masking)
                next_ids = next_ids * bin_mask.view(1, 1, H, W)
                
                if torch.all(next_ids == curr_ids):
                    break
                curr_ids = next_ids
                
            curr_ids = curr_ids.squeeze() # [H, W]
            unique_ids = torch.unique(curr_ids)
            unique_ids = unique_ids[unique_ids > 0] # 去除背景
            
            if len(unique_ids) > 0:
                for uid in unique_ids:
                    comp_mask = (curr_ids == uid).float()
                    # compute internal area score
                    comp_area = comp_mask.sum().item()
                    all_comp_scores.append((comp_area, k, t_max, comp_mask, m_t_max))

        all_comp_scores.sort(key=lambda x: x[0], reverse=True)
        top_p = min(L_max, len(all_comp_scores))
        
        for l in range(top_p):
            score, k, t_max, comp_mask, m_t_max = all_comp_scores[l]
            instance_mask = m_t_max * comp_mask
            
            if instance_mask.sum() > 1e-3:
                instance_mask_norm = instance_mask / instance_mask.sum()
                
                feat_t = Z0_5d[b, t_max] # [H, W, D]
                s_latent = (feat_t * instance_mask_norm.unsqueeze(-1)).sum(dim=(0, 1)) # [D]
                
                s_latents[b, l] = s_latent
                s_masks[b, l] = instance_mask
                s_valid[b, l] = True
                s_frame_idx[b, l] = t_max

    return s_latents, s_masks, s_valid, s_frame_idx

class TwoStageExtractor(nn.Module):
    def __init__(self, window_size=[2, 6], slots_per_window=4, global_k=16, D_in=768, mask_temp=1.0, L_max=32, use_decoder_checkpoint=False):
        super().__init__()
        self.merger = TwoStageSlotAttentionMerge(
            window_size=window_size,
            slots_per_window=slots_per_window,
            global_k=global_k,
            D_in=D_in,
            mask_temp=mask_temp
        )
        self.decoder = SIMONeDecoder(D_in=D_in, use_checkpoint=use_decoder_checkpoint)
        self.L_max = L_max

    def forward(self, Z0, spatial_dims=None, T_dim=1, valid_mask=None):
        B, T, H, W, C = Z0.shape
        
        obj_latents, _ = self.merger(Z0, valid_mask=valid_mask)
        
        Z_decode_flat = Z0.view(B, T, H*W, C)
        decoded_state = self.decoder(obj_latents, Z_decode_flat, H, W, valid_mask=valid_mask)
        
        state = GaussianState(T_dim=T_dim)
        state.add_layer(decoded_state)
        
        return state, obj_latents

    def decode_with_latents(self, obj_latents, Z0, spatial_dims=None, T_dim=1, valid_mask=None):
        B, T, H, W, C = Z0.shape
        Z_decode_flat = Z0.view(B, T, H*W, C)
        decoded_state = self.decoder(obj_latents, Z_decode_flat, H, W, valid_mask=valid_mask)
        
        state = GaussianState(T_dim=T_dim)
        state.add_layer(decoded_state)
        return state

    def extract_global_reconstruction(self, Z0_5d):
        B, T, H, W, D = Z0_5d.shape
        num_samples = min(T, 8)
        sampled_indices = torch.linspace(0, T - 1, num_samples).long().to(Z0_5d.device)
        Z_sampled = Z0_5d[:, sampled_indices]
        
        obj_latents, _ = self.merger(Z_sampled)
        
        Z_decode_flat_global = Z0_5d.view(B, T, H*W, D)
        out_recon = []
        for t_start in range(0, T, 4):
            t_end = min(t_start + 4, T)
            Z_batch_global = Z_decode_flat_global[:, t_start:t_end]
            decoded_state_g = self.decoder(obj_latents, Z_batch_global, H, W)
            mask_chunk = decoded_state_g['mask']
            recon_chunk = torch.einsum('btkn,bkd->btnd', mask_chunk, obj_latents)
            out_recon.append(recon_chunk)
            
        recon_all = torch.cat(out_recon, dim=1) # [B, T, N, D]
        return recon_all.view(B, T, H, W, D)

    def inference_with_masks(self, Z0_5d, t_idx, first_seg_flat):
        """
        Z0_5d: [B, T, H, W, D]
        t_idx: frame index corresponding to the GT masks (usually 0)
        first_seg_flat: [B, C, H, W] - 0-th frame GT mask
        """
        B, T, H, W, D = Z0_5d.shape
        C = first_seg_flat.shape[1]
        
        # 1. 均匀采样最多8帧提取全局 obj_latent
        num_samples = min(T, 8)
        sampled_indices = torch.linspace(0, T - 1, num_samples).long().to(Z0_5d.device)
        Z_sampled = Z0_5d[:, sampled_indices] # [B, num_samples, H, W, D]
        
        obj_latents, _ = self.merger(Z_sampled)
        # obj_latents: [B, K_global, D]
        K_global = obj_latents.shape[1]
        
        # 2. 对每个 obj_latent 解码出所有帧的 mask
        Z_decode_flat_global = Z0_5d.view(B, T, H*W, D)
        out_global_masks = []
        out_global_logits = []
        for t_start in range(0, T, 4):
            t_end = min(t_start + 4, T)
            Z_batch_global = Z_decode_flat_global[:, t_start:t_end]
            decoded_state_g = self.decoder(obj_latents, Z_batch_global, H, W)
            out_global_masks.append(decoded_state_g['mask'])
            out_global_logits.append(decoded_state_g['mask_logits'])
        
        all_global_masks = torch.cat(out_global_masks, dim=1) # [B, T, K_global, N]
        all_global_logits = torch.cat(out_global_logits, dim=1) # [B, T, K_global, N]
        
        global_masks_t = all_global_masks[:, t_idx].view(B, K_global, H, W)
        
        # 3. 和第0帧的GT mask 求交集作为 instance mask, 从而提取 instance slot (s_latents)
        s_masks_list = []
        slot_to_class = []
        s_latents_list = []
        
        for b in range(B):
            for k in range(K_global):
                m_k = global_masks_t[b, k] # [H, W]
                # Binarize or keep soft? We binarize the extracted mask to find overlap
                # m_k_bin = (m_k > 0.3).float()
                # if m_k_bin.sum() < 1:
                #     continue
                
                # Check overlap with each class in GT
                for c in range(1, C): # exclude background class 0
                    gt_c = first_seg_flat[b, c] # [H, W]
                    intersection = m_k * gt_c
                    if intersection.sum() > 0:
                        # use intersection as the new instance mask
                        mask_norm = intersection / intersection.sum()
                        feat_t = Z0_5d[b, t_idx] # [H, W, D]
                        s_latent = (feat_t * mask_norm.unsqueeze(-1)).sum(dim=(0, 1))
                        
                        s_latents_list.append(s_latent)
                        s_masks_list.append(intersection)
                        slot_to_class.append(c)
        
        if len(s_latents_list) == 0:
            # Fallback
            s_latents = torch.zeros(1, 1, D, device=Z0_5d.device)
            s_masks_stack = torch.zeros(1, 1, H, W, device=Z0_5d.device)
            slot_to_class = [0]
        else:
            s_latents = torch.stack(s_latents_list).unsqueeze(0) # [1, K, D], assuming B=1 for VOS
            s_masks_stack = torch.stack(s_masks_list).unsqueeze(0) # [1, K, H, W]
            
        # 4. 根据提取的 instance slot 解码所有帧
        # 为了和训练时行为一致 (训练时将 L_s 放入 Batch 维度, 使得 K=1 进行独立解码)，我们这里也将 K 放到 Batch 维度
        K_inst = s_latents.shape[1]
        s_latents_batch = s_latents.view(B * K_inst, 1, D)  # [B*K, 1, D]
        
        Z_decode_flat = Z0_5d.view(B, T, H*W, D)
        Z_decode_batch = Z_decode_flat.expand(B * K_inst, T, H*W, D) # 扩展 Z 以匹配 Batch
        
        decode_batch_size = 4  # 这里 batch_size 的含义变成了时间维度T的分块
        out_masks = []
        for t_start in range(0, T, decode_batch_size):
            t_end = min(t_start + decode_batch_size, T)
            Z_batch = Z_decode_batch[:, t_start:t_end] # [B*K, T_chunk, H*W, D]
            
            # 由于可能 B*K 很大，时间块也比较大，显存可能吃紧，我们也可以在空间上减少或者切块
            # 但这里我们保持不变，直接解码
            decoded_state = self.decoder(s_latents_batch, Z_batch, H, W)
            # mask_logits 从 [B*K, T_chunk, 1, N] 重新 reshape 为 [B, T_chunk, K, N]
            m_logits = decoded_state['mask_logits'].view(B, K_inst, -1, H*W).transpose(1, 2)
            out_masks.append(m_logits)
            
        return torch.cat(out_masks, dim=1), slot_to_class, s_masks_stack, all_global_logits

# ==========================================
# [新增] 分层几何解码器 (Hierarchical Geometric Decoder)
# ==========================================
class GaussianState:
    def __init__(self, T_dim=1):
        self.layer_states = [] # list of dicts with mu, scale, rot, depth, feat. 每个张量通常含 T 维: [B, T, K, D]
        self.T_dim = T_dim

    def add_layer(self, state_dict):
        self.layer_states.append(state_dict)

    @property
    def num_layers(self):
        return len(self.layer_states)

    def get_layer(self, layer_idx, t=None):
        state = self.layer_states[layer_idx]
        
        if t is not None:
            out_state = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.dim() == 4 and v.shape[1] == self.T_dim:
                    out_state[k] = v[:, t] 
                else:
                    out_state[k] = v
            return out_state
        return state
            
    def reconstruct_features(self, layer_idx, t=None):
        state = self.get_layer(layer_idx, t=t)
        mask = state['mask']
        
        # Use dense x_hat if available, otherwise broadcast the global Gaussian feat
        feat = state.get('x_hat', state['feat'])
        
        # When instance_state is evaluated, we have a batch dimension of K. Let's trace it carefully.
        # Check shapes:
        # mask is usually [B, K, N] or [B, K_dim, N] or [B, K1, K2, N] when nested.
        # Given we repacked into state2:
        # mask shape: [B, K_outer, K_inner, N]
        if mask.dim() == 4: # [B, T, K, N] or [B, K_outer, K_inner, N]
            if feat.dim() == 4: # [B, T, K, D] or [B, K_outer, K_inner, D]
                feat = feat.unsqueeze(3) # [B, T, K, 1, D]
            recon_Z = (mask.unsqueeze(-1) * feat).sum(dim=2) # [B, T, N, D] or [B, K_outer, N, D]
        elif mask.dim() == 5: # [B, T, K_outer, K_inner, N]
            if feat.dim() == 5:
                feat = feat.unsqueeze(4) # [B, T, K_outer, K_inner, 1, D]
            recon_Z = (mask.unsqueeze(-1) * feat).sum(dim=3) # [B, T, K_outer, N, D]
        elif mask.dim() == 3: # [B, K, N]
            if feat.dim() == 3: # [B, K, D]
                feat = feat.unsqueeze(2) # [B, K, 1, D]
            recon_Z = (mask.unsqueeze(-1) * feat).sum(dim=1) # [B, N, D]
        else:
            # Fallback for whatever dimension
            if feat.dim() == mask.dim():
                feat = feat.unsqueeze(-2)
            recon_Z = (mask.unsqueeze(-1) * feat).sum(dim=-3)
        return recon_Z


class SIMONeDecoder(nn.Module):
    def __init__(self, D_in=768, n_heads=8, num_layers=2, use_checkpoint=False):
        super().__init__()
        self.D_in = D_in
        self.use_checkpoint = use_checkpoint
        
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(D_in, n_heads, batch_first=True) for _ in range(num_layers)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(D_in) for _ in range(num_layers)
        ])
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=D_in, nhead=n_heads, dim_feedforward=4*D_in, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Mask 相关模块 (先各自降维一半，再 concat 输入 MLP)
        D_mask = 128
        self.mask_p_proj = nn.Linear(D_in, D_mask)
        self.mask_k_proj = nn.Linear(D_in, D_mask)
        self.mask_mlp = nn.Sequential(
            nn.Linear(2*D_mask, D_mask),
            nn.ReLU(True),
            nn.Linear(D_mask, D_mask),
            nn.ReLU(True),
            nn.Linear(D_mask, 1)
        )

    def forward(self, obj_latents, Z0_frames, H_new, W_new, valid_mask=None):
        """
        obj_latents: [B, K, D]
        Z0_frames: [B, T, N, D]
        """
        B, K, D = obj_latents.shape
        T = Z0_frames.shape[1]
        N = H_new * W_new

        # [B*T, N, D]
        patches = Z0_frames.reshape(B * T, N, D)
        patch_key_padding_mask = None
        if valid_mask is not None:
            valid_mask_flat = valid_mask.reshape(B * T, N)
            patches = patches.masked_fill(~valid_mask_flat.unsqueeze(-1), 0.0)
            patch_key_padding_mask = ~valid_mask_flat
        
        # 拼接作为 Query: [B*T, N, D]
        seq = patches
        
        # KV 为 obj_latents: [B*T, K, D]
        obj_kv = obj_latents.unsqueeze(1).expand(B, T, K, D).reshape(B * T, K, D)
        
        # Cross Attention + Transformer Layers交替
        x = seq
        for c_attn, c_norm, s_layer in zip(self.cross_attns, self.cross_norms, self.transformer_layers):
            def layer_forward(x_in):
                x_norm = c_norm(x_in)
                attn_out, _ = c_attn(query=x_norm, key=obj_kv, value=obj_kv)
                x_out = x_in + attn_out
                return s_layer(x_out, src_key_padding_mask=patch_key_padding_mask)

            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(layer_forward, x, use_reentrant=False)
            else:
                x = layer_forward(x)
            
        patch_out = x.reshape(B, T, N, D) # [B, T, N, D]
        
        def compute_mask_chunk(p, k_chunk):
            # p: [B, T, N, 1, D/2], k_chunk: [B, 1, 1, K_chunk, D/2]
            p_expand = p.expand(-1, -1, -1, k_chunk.shape[3], -1)
            k_expand = k_chunk.expand(-1, p.shape[1], p.shape[2], -1, -1)
            # # 通过 Concat 操作避免纯标量相乘导致的信息稀疏
            concat_feat = torch.cat([p_expand, k_expand], dim=-1).contiguous() # [B, T, N, K_chunk, D]
            # concat_feat = p_expand * k_expand
            return self.mask_mlp(concat_feat).squeeze(-1)

        p_proj = self.mask_p_proj(patch_out).unsqueeze(3) # [B, T, N, 1, D_mask]
        k_proj = self.mask_k_proj(obj_latents).unsqueeze(1).unsqueeze(2) # [B, 1, 1, K, D_mask]

        mask_logits_list = []
        chunk_size = 256
        
        for i in range(0, B, chunk_size):
            end = min(i + chunk_size, B)
            p_chunk = p_proj[i:end] # [B_chunk, 1, 1, K, D_mask]
            k_chunk = k_proj[i:end] # [B_chunk, 1, 1, K, D_mask]

            # if self.training and torch.is_grad_enabled():
            #     # 利用梯度检查点重计算中间激活张量
            #     logits_chunk = torch.utils.checkpoint.checkpoint(
            #         compute_mask_chunk, p_chunk, k_chunk, use_reentrant=False
            #     )
            # else:
            logits_chunk = compute_mask_chunk(p_chunk, k_chunk)
                
            mask_logits_list.append(logits_chunk)
            
        mask_logits = torch.cat(mask_logits_list, dim=0) # [B, T, N, K]
        mask_probs = F.softmax(mask_logits, dim=-1) # [B, T, N, K]
        
        # 将 obj_latents 扩展用于后续特征相似度计算: [B, T, K, D]
        o_feat = obj_latents.unsqueeze(1).expand(B, T, K, D)
        
        mask_out = mask_probs.permute(0, 1, 3, 2) # [B, T, K, N]
        mask_logits_out = mask_logits.permute(0, 1, 3, 2) # [B, T, K, N]
        
        return {
            'feat': o_feat, 
            'obj_feat': obj_latents,
            'mask': mask_out,
            'mask_logits': mask_logits_out
        }
