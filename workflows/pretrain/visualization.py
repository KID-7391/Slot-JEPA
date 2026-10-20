import math
import torch
from typing import Optional
from torchvision.utils import make_grid
import torch.nn.functional as F

def generate_pca_feature_vis(features, patch_size: int = 16, spatial_shape: tuple = (24, 24), image: Optional[torch.Tensor] = None):
    """
    Generate PCA-based feature visualization.
    Computes PCA ONLY on the original feature map to establish a baseline color space,
    then projects both original and reconstructed features into this same space.
    """
    is_list = isinstance(features, list)
    if is_list:
        feat_orig = features[0]  # Ground truth features
    else:
        feat_orig = features

    # 1. Compute PCA using ONLY the original features
    feat_orig_mean = feat_orig.mean(dim=0, keepdim=True)
    feat_orig_centered = feat_orig - feat_orig_mean
    U, S, V = torch.pca_lowrank(feat_orig_centered, q=3)
    
    # 2. Get Min/Max for normalization from the original features
    pca_orig = torch.matmul(feat_orig_centered, V)
    pca_min = pca_orig.amin(dim=0, keepdim=True)
    pca_max = pca_orig.amax(dim=0, keepdim=True)
    
    # Helper function to project and reshape any feature tensor
    def process_to_img(f_tensor):
        # Center with original mean, project with original V
        f_centered = f_tensor - feat_orig_mean
        f_pca = torch.matmul(f_centered, V)
        
        # Normalize using original min/max bounds, clamp to valid image range
        f_norm = (f_pca - pca_min) / (pca_max - pca_min + 1e-8)
        f_norm = f_norm.clamp(0.0, 1.0)
        
        # Reshape to spatial grid
        H_f, W_f = spatial_shape
        H_img, W_img = H_f * patch_size, W_f * patch_size
        
        pca_grid = f_norm.view(H_f, W_f, 3).permute(2, 0, 1)  # [3, H_f, W_f]
        pca_img = F.interpolate(pca_grid.unsqueeze(0), size=(H_img, W_img), mode='nearest').squeeze(0)
        return pca_img

    if is_list:
        img_list = [process_to_img(f) for f in features]
    else:
        img_list = [process_to_img(feat_orig)]
        
    if image is not None:
        device = image.device
        C = image.shape[0]
        unnormalize_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(C, 1, 1)
        unnormalize_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(C, 1, 1)
        image_unnorm = image * unnormalize_std + unnormalize_mean
        image_unnorm = image_unnorm.clamp(0.0, 1.0).cpu()
        
        H_img, W_img = spatial_shape[0] * patch_size, spatial_shape[1] * patch_size
        if image_unnorm.shape[1] != H_img or image_unnorm.shape[2] != W_img:
            image_unnorm = F.interpolate(image_unnorm.unsqueeze(0), size=(H_img, W_img), mode='bilinear', align_corners=False).squeeze(0)
        img_list.insert(0, image_unnorm)

    if len(img_list) > 1:
        # side-by-side
        return make_grid(img_list, nrow=len(img_list), padding=10, pad_value=1.0)
    else:
        return img_list[0]

def make_pca_grid_with_points(gt_features, recon_features, original_imgs, patch_size=16, return_list=True, sampled_indices_dict=None, frame_indices=None):
    """
    Helper function to draw PCA grids and optionally draw circles on them based on sampled_indices_dict.
    sampled_indices_dict: Dict indicating which indices should be circled for each frame index.
                          e.g., {t_k: torch.Tensor([...])}
    """
    # Fit PCA ONLY on ground truth features to ensure consistent color mapping
    if isinstance(gt_features, list) and len(gt_features) > 0 and gt_features[0].dim() == 3:
        gt_all = torch.cat(gt_features, dim=1).squeeze(0)
    else:
        gt_all = torch.cat([f.reshape(-1, f.shape[-1]) for f in gt_features], dim=0)

    mean = gt_all.mean(dim=0, keepdim=True)
    centered_gt = gt_all - mean
    
    U, S, V = torch.pca_lowrank(centered_gt, q=3)
    gt_pca = torch.matmul(centered_gt, V)
    
    p_min = gt_pca.amin(dim=0, keepdim=True)
    p_max = gt_pca.amax(dim=0, keepdim=True)
    
    def to_img(f, t_idx=None, is_pred=False):
        f_c = f.reshape(-1, f.shape[-1]) - mean
        f_p = torch.matmul(f_c, V)
        
        f_norm = (f_p - p_min) / (p_max - p_min + 1e-8)
        f_norm = f_norm.clamp(0, 1)
        
        h_spatial = int(math.sqrt(f_norm.shape[0]))
        f_norm_img = f_norm.view(h_spatial, h_spatial, 3).permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(f_norm_img, scale_factor=patch_size, mode='nearest').squeeze(0)
        
        # Draw points
        if not is_pred and sampled_indices_dict is not None and frame_indices is not None and t_idx is not None:
            t = frame_indices[t_idx]
            sampled_indices = sampled_indices_dict.get(t, None)
            if sampled_indices is not None:
                import cv2
                import numpy as np
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
                
                H_f, W_f = h_spatial, h_spatial
                for idx in sampled_indices:
                    if idx.item() == -1:
                        continue
                    y_f = idx.item() // W_f
                    x_f = idx.item() % W_f
                    center_y = int(y_f * patch_size + patch_size / 2)
                    center_x = int(x_f * patch_size + patch_size / 2)
                    radius = patch_size // 2 - 1
                    cv2.circle(img_np, (center_x, center_y), radius, (255, 255, 255), 2)
                    cv2.circle(img_np, (center_x, center_y), radius+1, (0, 0, 0), 1)
                    
                img = torch.from_numpy(img_np).permute(2, 0, 1).float().to(f.device) / 255.0
                
        if is_pred:
            pad = 4
            border_color = torch.tensor([1., 0., 0.], device=img.device).view(3, 1, 1)
            img = img.clone()
            img[:, :pad, :] = border_color
            img[:, -pad:, :] = border_color
            img[:, :, :pad] = border_color
            img[:, :, -pad:] = border_color
            
        return img

    gt_imgs = [to_img(f, i, False) for i, f in enumerate(gt_features)]
    
    recon_imgs = []
    if len(recon_features) > 0:
        recon_imgs = [to_img(f, None, True) for f in recon_features]

    all_imgs = []
    all_imgs.extend(original_imgs)
    all_imgs.extend(gt_imgs)
    all_imgs.extend(recon_imgs)
    
    if return_list:
        return all_imgs
    return make_grid(all_imgs, nrow=max(1, len(gt_features)))


def make_pca_grid(gt_features, recon_features, original_imgs, patch_size=16, error_maps_vis=None, cand_mus_vis=None, return_list=False):
    # Fit PCA ONLY on ground truth features to ensure consistent color mapping
    gt_all = torch.cat(gt_features, dim=1).squeeze(0)  # Shape depends on features
    mean = gt_all.mean(dim=0, keepdim=True)
    centered_gt = gt_all - mean
    
    # Perform PCA on centered GT features
    U, S, V = torch.pca_lowrank(centered_gt, q=3)
    gt_pca = torch.matmul(centered_gt, V)
    
    # Compute min and max bounds for normalization from GT features
    p_min = gt_pca.amin(dim=0, keepdim=True)
    p_max = gt_pca.amax(dim=0, keepdim=True)
    
    def to_img(f, is_pred=False):
        # Apply the SAME PCA transform fitted early to everything
        f_c = f.squeeze(0) - mean
        f_p = torch.matmul(f_c, V)
        
        # Normalize using the GT bounds
        f_norm = (f_p - p_min) / (p_max - p_min + 1e-8)
        
        # Clip to [0, 1] and reshape to image
        h_spatial = int(math.sqrt(f_norm.shape[0]))
        f_norm = f_norm.clamp(0, 1).view(h_spatial, h_spatial, 3).permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(f_norm, scale_factor=patch_size, mode='nearest').squeeze(0)
        
        if is_pred:
            pad = 4
            border_color = torch.tensor([1., 0., 0.], device=img.device).view(3, 1, 1)
            img[:, :pad, :] = border_color
            img[:, -pad:, :] = border_color
            img[:, :, :pad] = border_color
            img[:, :, -pad:] = border_color
        return img

    gt_imgs = [to_img(f, False) for f in gt_features]
    
    recon_imgs = []
    if len(recon_features) > 0:
        recon_imgs = [to_img(recon_features[0], False)] + [to_img(f, True) for f in recon_features[1:]]
    
    all_imgs = []
    all_imgs.extend(original_imgs)
    all_imgs.extend(gt_imgs)
    all_imgs.extend(recon_imgs)

    if return_list:
        return all_imgs
    return make_grid(all_imgs, nrow=len(gt_features))


def generate_gaussian_mask_grid(image: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor, rot: torch.Tensor,
                                unnormalize_mean: Optional[torch.Tensor] = None,
                                unnormalize_std: Optional[torch.Tensor] = None,
                                eps: float = 1e-8,
                                blend: bool = True,
                                overlay_alpha: float = 0.6,
                                alphas: Optional[torch.Tensor] = None,
                                depths: Optional[torch.Tensor] = None,
                                annotate_alphas: bool = True,
                                tau: float = 4.0):
    """
    Generate a montage grid of K images where each image is the input `image` masked by a 2D Gaussian.


    Args:
        image: Tensor [C, H, W], assumed normalized (e.g., ImageNet normalized). Will be un-normalized to [0,1].
        means: Tensor [K, 2] -- Gaussian centers in normalized image coordinates [-1, 1] (x, y)
        covariances: Tensor either [K, 2, 2] (full covariance matrices) or [K, 2] (diagonal sigma values).
        unnormalize_mean: optional Tensor [3] to un-normalize (default ImageNet mean)
        unnormalize_std: optional Tensor [3] to un-normalize (default ImageNet std)
        eps: small epsilon for numeric stability

    Returns:
        grid_img: Tensor [3, H_grid, W_grid] (float, in [0,1]) suitable for writer.add_image or torchvision.save_image

    Implementation notes:
      - Fully vectorized: no Python loops over pixels. Broadcasting used across K and HW.
      - Means are expected in normalized coordinates [-1,1] where (-1,-1) is top-left and (1,1) bottom-right.
      - If covariances is provided as [K,2] it's interpreted as standard deviations (sigma_x, sigma_y).
    """
    assert image.ndim == 3, "image must be [C,H,W]"
    C, H, W = image.shape
    device = image.device

    # # Un-normalize to [0,1]
    # if unnormalize_mean is None:
    #     unnormalize_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(C, 1, 1)
    # else:
    #     unnormalize_mean = unnormalize_mean.to(device).view(C, 1, 1)
    # if unnormalize_std is None:
    #     unnormalize_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(C, 1, 1)
    # else:
    #     unnormalize_std = unnormalize_std.to(device).view(C, 1, 1)

    # image_unnorm = image * unnormalize_std + unnormalize_mean
    # image_unnorm = image_unnorm.clamp(0.0, 1.0)
    image_unnorm = image

    # Prepare grid of (x,y) coordinates in normalized [-1,1] matching renderer convention
    # 渲染是在特征图维度做的 (降采样了patch_size倍)，所以这里先在低分辨率计算，避免出现高分辨率离散采样导致的面积偏差
    patch_size = 16
    H_f, W_f = H // patch_size, W // patch_size
    xs = torch.linspace(-1.0, 1.0, W_f, device=device)
    ys = torch.linspace(-1.0, 1.0, H_f, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # H_f x W_f
    grid = torch.stack([xx, yy], dim=-1)  # H_f x W_f x 2
    M = H_f * W_f
    grid_flat = grid.view(M, 2)  # (H_f*W_f) x 2

    K = mu.shape[0]
    mu = mu.to(device)
    scale = scale.to(device)
    rot = rot.to(device)

    # handle depths/alphas sorting: if depths provided, sort kernels front-to-back
    sort_idx = None
    sorted_alphas = None
    sorted_depths = None
    if depths is not None:
        d = depths.to(device).view(-1)
        if d.shape[0] != K:
            raise ValueError('depths length must match K')
        sort_idx = torch.argsort(d, descending=True)
        mu = mu[sort_idx]
        scale = scale[sort_idx]
        rot = rot[sort_idx]
        sorted_depths = d[sort_idx].cpu().numpy().tolist()

    if alphas is not None:
        a = alphas.to(device).view(-1)
        if a.shape[0] != K:
            raise ValueError('alphas length must match K')
        if sort_idx is not None:
            a = a[sort_idx]
        sorted_alphas = a.cpu().numpy().tolist()

    grid_e = grid_flat.unsqueeze(0).unsqueeze(0)  # 1 x 1 x P x 2
    offsets = grid_e - mu.view(1, K, 1, 2)
    cos_r, sin_r = rot[..., 0], rot[..., 1]
    rotation = torch.stack(
        [cos_r, -sin_r, sin_r, cos_r], dim=-1
    ).view(K, 2, 2)
    local_offsets = torch.einsum("kij,bkpj->bkpi", rotation.transpose(-1, -2), offsets)
    inv_scale = scale.clamp_min(eps).reciprocal().view(1, K, 1, 2)
    masks = torch.exp(-0.5 * ((local_offsets * inv_scale) ** 2).sum(dim=-1) / tau).squeeze(0)

    masks = masks.view(K, H_f, W_f)  # [K,H_f,W_f]
    # 上采样回到原图分辨率，保持与重建特征空间一致的块状效应
    masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

    # Apply masks: produce either masked images (multiplicative) or blended overlays
    img = image_unnorm.unsqueeze(0).expand(K, C, H, W)
    masks_unsq = masks.unsqueeze(1)  # [K,1,H,W]

    if blend:
        # generate distinct colors per kernel via hue-roation-like mapping
        hs = torch.linspace(0.0, 1.0, K, device=device)
        # simple HSV->RGB approximation using phase-shifted sinusoids for visual variety
        colors = torch.stack([
            (0.5 * (torch.sin(2 * math.pi * (hs + 0.0)) + 1.0)),
            (0.5 * (torch.sin(2 * math.pi * (hs + 1.0/3.0)) + 1.0)),
            (0.5 * (torch.sin(2 * math.pi * (hs + 2.0/3.0)) + 1.0)),
        ], dim=1).to(dtype=image_unnorm.dtype)  # [K,3]

        colors = colors.view(K, C, 1, 1)  # [K,3,1,1]
        colored_masks = masks_unsq * colors  # [K,3,H,W]

        # alpha blend: result = img*(1 - alpha*mask) + colored_mask*(alpha*mask)
        a = overlay_alpha
        blended = img * (1.0 - masks_unsq * a) + colored_masks * (masks_unsq * a)
        out = blended
    else:
        out = img * masks_unsq

    # Make montage grid
    nrow = int(math.ceil(math.sqrt(K)))
    grid_img = make_grid(out, nrow=nrow, pad_value=0)
    # grid_img is [3, H_grid, W_grid], in [0,1]
    # annotate alpha values using PIL if requested
    if annotate_alphas and (sorted_alphas is not None or sorted_depths is not None):
        try:
            from torchvision.transforms.functional import to_pil_image, to_tensor
            from PIL import ImageDraw, ImageFont

            pil = to_pil_image(grid_img.cpu())
            draw = ImageDraw.Draw(pil)
            font = None
            try:
                import glob
                font_paths = glob.glob('/usr/share/fonts/**/*.ttf', recursive=True)
                if font_paths:
                    font = ImageFont.truetype(font_paths[0], 24)
            except Exception:
                pass

            # tile sizes
            tile_w = W
            tile_h = H
            cols = nrow
            for idx in range(K):
                row = idx // cols
                col = idx % cols
                x = col * tile_w + 4 + (col * 2)  # pad offset from make_grid
                y = row * tile_h + 4 + (row * 2)
                
                text_parts = []
                if sorted_alphas is not None:
                    text_parts.append(f'A: {sorted_alphas[idx]:.3f}')
                if sorted_depths is not None:
                    text_parts.append(f'D: {sorted_depths[idx]:.3f}')
                text = '\n'.join(text_parts)
                
                kwargs = {}
                if font is not None:
                    kwargs['font'] = font
                
                # draw outline for readability
                draw.text((x - 2, y - 2), text, fill='black', **kwargs)
                draw.text((x + 2, y - 2), text, fill='black', **kwargs)
                draw.text((x - 2, y + 2), text, fill='black', **kwargs)
                draw.text((x + 2, y + 2), text, fill='black', **kwargs)
                draw.text((x, y), text, fill='white', **kwargs)

            grid_img = to_tensor(pil)
        except Exception as e:
            print("PIL annotation error:", e)
            # if PIL not available for some reason, skip annotations
            pass

    return grid_img


# Integration snippet (example) -- put this into your DDP training loop on rank 0:
#
# Example: call generate_gaussian_mask_grid with decoded Gaussian parameters.
# if rank == 0 and writer is not None:
#     # `image` should be a CPU tensor [C,H,W] (normalized). `mu` and `cov` are tensors on CPU.
#     grid_img = generate_gaussian_mask_grid(image.cpu(), mu.cpu(), cov.cpu())
#     writer.add_image('Train_Vis/Gaussian_Masks', grid_img, global_step)

def generate_composite_gaussian_overlay(image, mu, covs, alphas, depths, cmap_name='hsv'):
    """
    Generate a single composite overlay of all Gaussians onto the image canvas.
    Args:
        image: Tensor [3, H, W] in [0, 1]
        mu: Tensor [K, 2] in [-1, 1]
        covs: Tensor [K, 2, 2]
        alphas: Tensor [K] in [0, 1]
        depths: Tensor [K]
    Returns:
        canvas: Tensor [3, H, W] in [0, 1]
    """
    import matplotlib.pyplot as plt

    device = image.device
    C, H, W = image.shape
    K = mu.shape[0]

    # Generate K distinct colors
    cmap = plt.get_cmap(cmap_name)
    colors = []
    for i in range(K):
        # vary color appropriately
        c = cmap(i / max(1, K - 1))[:3]
        colors.append(torch.tensor(c, dtype=image.dtype, device=device).view(3, 1, 1))
    
    # Grid of (X, Y)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # H x W
    grid = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (H*W) x 2

    # Canvas initialization
    canvas = image.clone() * 0.4  # dampen the background

    # Sort indices by depth in descending order (back-to-front rendering)
    sort_idx = torch.argsort(depths, descending=False)

    for idx in sort_idx:
        center = mu[idx].view(1, 2)
        cov = covs[idx] + torch.eye(2, device=device) * 1e-6
        inv_cov = torch.linalg.inv(cov)
        
        diff = grid - center  # (H*W) x 2
        # Mahalanobis distance: (diff @ inv_cov * diff).sum(dim=-1)
        mahal = (torch.matmul(diff, inv_cov) * diff).sum(dim=-1)
        
        G = torch.where(mahal < 2.25, torch.exp(-0.5 * mahal), torch.zeros_like(mahal)).view(1, H, W)
        mask = G * alphas[idx].view(1, 1, 1)
        
        # Alpha blending
        color = colors[idx]
        canvas = mask * color + (1.0 - mask) * canvas

    return canvas.clamp(0.0, 1.0)
