import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

_ssim_windows = {}

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    device = img1.device
    key = (window_size, channel, device)
    
    if key not in _ssim_windows:
        _ssim_windows[key] = create_window(window_size, channel).to(device)
    window = _ssim_windows[key]
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def feature_recon_loss(Z_hat, Z0, loss_type='kl', temp_s=0.15, temp_t=0.15):
    """
    Z_hat: B x N x D
    Z0: B x N x D  (N = H*W)
    """
    B, N, D = Z_hat.shape

    if loss_type == 'l1':
        return F.l1_loss(Z_hat, Z0)
    elif loss_type == 'kl':
        log_probs = F.log_softmax(Z_hat / temp_s, dim=-1)
        targets = F.softmax(Z0 / temp_t, dim=-1)
        return -torch.sum(targets * log_probs, dim=-1).mean()
    elif loss_type == 'mix':
        H = int(math.sqrt(N))
        W = N // H
        Z_hat_img = Z_hat.view(B, H, W, D).permute(0, 3, 1, 2)
        Z0_img = Z0.view(B, H, W, D).permute(0, 3, 1, 2)
        
        loss_l1 = F.l1_loss(Z_hat_img, Z0_img)
        loss_ssim = 1.0 - ssim(Z_hat_img, Z0_img)
        
        lambda_dssim = 0.2
        return (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * loss_ssim
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def slot_orthogonality_loss(slots):
    """
    Computes the orthogonality constraint loss for slots.
    slots: [B, K, D]
    """
    B, K, D = slots.shape
    norm_slots = F.normalize(slots, p=2, dim=-1, eps=1e-5)
    dot_products = torch.bmm(norm_slots, norm_slots.transpose(1, 2))
    off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=slots.device).unsqueeze(0).expand(B, -1, -1)
    off_diag_sims = dot_products[off_diag_mask]
    return (off_diag_sims ** 2).mean()

def volume_regularization(scale, weight=1e-3):
    """
    惩罚最大的4个高斯的面积，防止单个高斯膨胀覆盖全局。
    scale: B x K x 2
    """
    vol = scale.abs().prod(dim=-1)  # B x K
    # Get top 4 largest areas per batch
    topk_vol, _ = torch.topk(vol, k=4, dim=-1)
    return weight * topk_vol.mean()

def tracking_reg_loss(S_t, S_prev, detach_prev=True):
    """
    Computes regularization loss (e.g. MSE of mu) between matched IDs in S_t and S_prev.
    Handles dynamic birth/death of entities.
    """
    if 'track_ids' not in S_t or 'track_ids' not in S_prev:
        mu_prev = S_prev['mu'].detach() if detach_prev else S_prev['mu']
        return F.mse_loss(S_t['mu'], mu_prev)

    ids_t = S_t['track_ids'] # [B, K_t]
    ids_prev = S_prev['track_ids'] # [B, K_prev]
    
    mu_t = S_t['mu']
    mu_prev = S_prev['mu'].detach() if detach_prev else S_prev['mu']
    
    B = mu_t.shape[0]
    total_loss = 0.0
    valid_count = 0
    
    for b in range(B):
        id_t_b = ids_t[b]
        id_prev_b = ids_prev[b]
        
        match_mask = (id_t_b.unsqueeze(1) == id_prev_b.unsqueeze(0)) # [K_t, K_prev]
        idx_t, idx_prev = torch.where(match_mask)
        
        if len(idx_t) > 0:
            diff = mu_t[b, idx_t] - mu_prev[b, idx_prev]
            total_loss = total_loss + (diff ** 2).sum()
            valid_count += len(idx_t)
            
    if valid_count > 0:
        return total_loss / valid_count
    return torch.tensor(0.0, device=mu_t.device)

if __name__ == '__main__':
    Zh = torch.randn(2, 24 * 24, 768)
    Z0 = torch.randn(2, 24 * 24, 768)
    print(feature_recon_loss(Zh, Z0))

def dynamics_loss(pred_states, true_states):
    """
    pred_states: dict with 'mu', 'scale', 'rot', 'feat'
    true_states: dict with 'mu', 'scale', 'rot', 'feat'
    """
    pred_mu = pred_states['mu']
    true_mu = true_states['mu']
    pred_scale = pred_states['scale']
    true_scale = true_states['scale']
    pred_rot = pred_states['rot']
    true_rot = true_states['rot']
    pred_feat = pred_states.get('slot_feat', pred_states.get('feat'))
    true_feat = true_states.get('slot_feat', true_states.get('feat'))
    
    loss_mu = F.mse_loss(pred_mu, true_mu)
    loss_scale = F.mse_loss(pred_scale, true_scale)
    loss_rot = F.mse_loss(pred_rot, true_rot)
    
    loss_feat = 1.0 - F.cosine_similarity(pred_feat, true_feat, dim=-1).mean()
    
    return loss_mu + loss_scale + loss_rot + loss_feat


def unmasked_loss(student_refine_states, teacher_refine_states):
    """
    Both should be [B, T, N_unmasked, D] or [B, N_unmasked, D]
    """
    return F.l1_loss(student_refine_states, teacher_refine_states)

def masked_loss(predicted_masked_states, teacher_refine_states_masked):
    """
    predicted: [B, N_masked, D]
    target: [B, N_masked, D]
    """
    return F.l1_loss(predicted_masked_states, teacher_refine_states_masked)

def recon_loss(rendered_features, ground_truth_features):
    """
    Already implemented as feature_recon_loss
    """
    return feature_recon_loss(rendered_features, ground_truth_features, loss_type='l1')

def identity_consistency_loss(S_t, S_prev, Z_t, Z_prev, spatial_dims, spawn_mask=None):
    """
    Computes soft pooling identity consistency loss between frames.
    S_t, S_prev: Dict of states with 'mu', 'scale', 'rot' [B, K, ...]
    Z_t, Z_prev: V-JEPA features [B, N, D] or [B, D, H, W]
    spatial_dims: (H, W)
    spawn_mask: boolean tensor [B, K], True if the entity was just spawned (should be ignored)
    """
    H, W = spatial_dims
    B, K = S_t['mu'].shape[:2]
    device = S_t['mu'].device
    
    if Z_t.dim() == 3:
        D = Z_t.shape[-1]
        Z_t = Z_t.transpose(1, 2).reshape(B, D, H, W)
        Z_prev = Z_prev.transpose(1, 2).reshape(B, D, H, W)
        
    xs = torch.linspace(-1.0, 1.0, steps=W, device=device)
    ys = torch.linspace(-1.0, 1.0, steps=H, device=device)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1) # [H, W, 2]
    grid = grid.view(1, 1, H*W, 2) # [1, 1, P, 2]
    
    def get_mask(S):
        mu = S['mu'].unsqueeze(2) # [B, K, 1, 2]
        scale = S['scale'].unsqueeze(2) # [B, K, 1, 2]
        scale_sq = scale ** 2 + 1e-5
        
        # 增加旋转矩阵防止因为旋转错位导致 identity loss 计算位置偏离
        cos = S['rot'][..., 0]  # [B, K]
        sin = S['rot'][..., 1]  # [B, K]
        
        # 构建逆旋转矩阵 (相当于点相对高斯的坐标系旋转)
        R_inv = torch.stack([
            torch.stack([cos, sin], dim=-1),
            torch.stack([-sin, cos], dim=-1)
        ], dim=-2) # [B, K, 2, 2]
        
        delta = grid - mu # [B, K, P, 2]
        delta_rot = torch.einsum('bkij,bkpj->bkpi', R_inv, delta)

        m = (delta_rot ** 2 / scale_sq).sum(dim=-1) # [B, K, P]
        
        G = torch.exp(-0.5 * m) # [B, K, P]
        # 添加归一化，防止由于 Gaussian 缩放 (面积膨胀/缩小) 带来的 feature 绝对值偏移导致拟合崩溃
        G = G / (G.sum(dim=-1, keepdim=True) + 1e-5)
        
        return G.view(B, K, H, W)
    
    mask_t = get_mask(S_t) # [B, K, H, W]
    mask_prev = get_mask(S_prev) # [B, K, H, W]
    
    V_t = torch.einsum('bkhw,bdhw->bkd', mask_t, Z_t)
    V_prev = torch.einsum('bkhw,bdhw->bkd', mask_prev, Z_prev.detach())

    diff_sq = (V_t - V_prev) ** 2 # [B, K, D]
    loss_per_slot = diff_sq.mean(dim=-1) # [B, K]
    
    if spawn_mask is not None:
        valid_mask = (~spawn_mask).float()
        return (loss_per_slot * valid_mask).sum() / (valid_mask.sum() + 1e-5)
    return loss_per_slot.mean()
