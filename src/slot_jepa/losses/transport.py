import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _generalized_kl(x, y, eps):
    x_safe = x.clamp_min(eps)
    y_safe = y.clamp_min(eps)
    return (x_safe * (torch.log(x_safe) - torch.log(y_safe)) - x_safe + y_safe).sum(dim=-1)


def _slot_change_weights(
    reference_mask_probs,
    target_mask_probs,
    min_weight=0.05,
    max_weight=5.0,
):
    """Weight slots by observed context-to-goal change, never by predictions."""
    reference = reference_mask_probs.detach().float()
    target = target_mask_probs.detach().float()
    if reference.shape != target.shape:
        raise ValueError(
            "Expected reference and target masks to align, got "
            f"{tuple(reference.shape)} vs {tuple(target.shape)}"
        )
    weights = torch.mean(torch.abs(target - reference), dim=-1)
    mean = weights.mean(dim=-1, keepdim=True)
    weights = torch.where(
        mean > 1.0e-8,
        weights / mean.clamp_min(1.0e-8),
        torch.ones_like(weights),
    )
    return weights.clamp(float(min_weight), float(max_weight)).float()


def _slot_area_normalizer(context_mask_probs, target_mask_probs, eps=1.0e-3):
    """Return stable per-case/per-slot max areas over context and goal frames."""
    if context_mask_probs.ndim != 4 or target_mask_probs.ndim != 4:
        raise ValueError(
            "Expected context/target masks [B,T,K,N], got "
            f"{tuple(context_mask_probs.shape)} and {tuple(target_mask_probs.shape)}"
        )
    if context_mask_probs.shape[0] != target_mask_probs.shape[0]:
        raise ValueError("Context and target batch sizes must match")
    if context_mask_probs.shape[2:] != target_mask_probs.shape[2:]:
        raise ValueError("Context and target slot/patch dimensions must match")
    masks = torch.cat(
        [context_mask_probs.detach().float(), target_mask_probs.detach().float()],
        dim=1,
    )
    return masks.sum(dim=-1).amax(dim=1).clamp_min(float(eps))


def _broadcast_slot_values(values, target):
    values = values.to(target.device, dtype=target.dtype)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError(f"Expected per-slot values [K] or [B,K], got {tuple(values.shape)}")
    if values.shape[0] == 1 and target.shape[0] != 1:
        values = values.expand(target.shape[0], -1)
    if values.shape[0] != target.shape[0] or values.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"Per-slot values {tuple(values.shape)} cannot broadcast to {tuple(target.shape)}"
        )
    return values.view(values.shape[0], *([1] * (target.ndim - 2)), values.shape[-1])


def _uot_mask_loss(
    pred_mask_probs,
    target_mask_probs,
    sinkhorn_iters=5,
    entropy_reg=0.05,
    unbalanced_ot_rho=0.2,
    eps=1.0e-8,
    batch_size=128,
    transport_cache=None,
    slot_weights=None,
    area_normalizer=None,
    area_normalize=False,
    area_normalize_eps=1.0e-3,
    reduce_time=True,
    return_per_slot=False,
    transport_cost_addition=None,
):
    if pred_mask_probs.shape != target_mask_probs.shape:
        raise ValueError(
            f"Expected equal shapes for UOT objective: {tuple(pred_mask_probs.shape)} vs {tuple(target_mask_probs.shape)}"
        )
    num_patches = pred_mask_probs.shape[-1]
    spatial_size = int(np.sqrt(num_patches))
    if spatial_size * spatial_size != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} patches")
    cache_key = (
        pred_mask_probs.device.type,
        pred_mask_probs.device.index,
        num_patches,
        float(entropy_reg),
        float(eps),
    )
    cached = None if transport_cache is None else transport_cache.get(cache_key)
    if cached is None:
        ys, xs = torch.meshgrid(
            torch.linspace(0.0, 1.0, spatial_size, device=pred_mask_probs.device),
            torch.linspace(0.0, 1.0, spatial_size, device=pred_mask_probs.device),
            indexing="ij",
        )
        coords = torch.stack([ys, xs], dim=-1).reshape(-1, 2)
        cost = torch.cdist(coords, coords, p=2).pow(2).float()
        kernel = torch.exp(-cost / entropy_reg).clamp_min(eps)
        if transport_cache is not None:
            transport_cache[cache_key] = (cost, kernel)
    else:
        cost, kernel = cached
    tau = unbalanced_ot_rho / (unbalanced_ot_rho + entropy_reg)

    custom_cost = None
    if transport_cost_addition is not None:
        addition = transport_cost_addition.to(
            device=pred_mask_probs.device, dtype=torch.float32
        )
        expected_without_time = (
            pred_mask_probs.shape[0],
            pred_mask_probs.shape[-2],
            num_patches,
            num_patches,
        )
        expected_with_time = (
            *pred_mask_probs.shape[:-1], num_patches, num_patches
        )
        if tuple(addition.shape) == expected_without_time:
            addition = addition[:, None].expand(
                pred_mask_probs.shape[0],
                pred_mask_probs.shape[1],
                pred_mask_probs.shape[-2],
                num_patches,
                num_patches,
            )
        elif tuple(addition.shape) != expected_with_time:
            raise ValueError(
                "transport_cost_addition must have shape [B,K,N,N] or "
                f"[B,T,K,N,N], got {tuple(addition.shape)} for masks "
                f"{tuple(pred_mask_probs.shape)}"
            )
        custom_cost = (
            cost.view(*([1] * (addition.ndim - 2)), num_patches, num_patches)
            + addition
        ).reshape(-1, num_patches, num_patches)

    pred_flat = pred_mask_probs.reshape(-1, num_patches).float().clamp_min(eps)
    target_flat = target_mask_probs.reshape(-1, num_patches).float().clamp_min(eps)
    out = pred_flat.new_zeros((pred_flat.shape[0],))
    batch_size = max(1, int(batch_size))
    kernel_t = kernel.transpose(0, 1)
    for start in range(0, pred_flat.shape[0], batch_size):
        end = min(start + batch_size, pred_flat.shape[0])
        a = pred_flat[start:end]
        b = target_flat[start:end]
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        if custom_cost is None:
            chunk_cost = cost.unsqueeze(0)
            chunk_kernel = kernel.unsqueeze(0)
            for _ in range(sinkhorn_iters):
                kv = torch.matmul(v, kernel_t).clamp_min(eps)
                u = (a / kv).pow(tau)
                ktu = torch.matmul(u, kernel).clamp_min(eps)
                v = (b / ktu).pow(tau)
        else:
            chunk_cost = custom_cost[start:end]
            chunk_kernel = torch.exp(-chunk_cost / entropy_reg).clamp_min(eps)
            for _ in range(sinkhorn_iters):
                kv = torch.bmm(chunk_kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
                u = (a / kv).pow(tau)
                ktu = torch.bmm(
                    chunk_kernel.transpose(1, 2), u.unsqueeze(-1)
                ).squeeze(-1).clamp_min(eps)
                v = (b / ktu).pow(tau)
        plan = u.unsqueeze(-1) * chunk_kernel * v.unsqueeze(-2)
        transport_cost = (plan * chunk_cost).sum(dim=(1, 2))
        transported_pred = plan.sum(dim=2)
        transported_target = plan.sum(dim=1)
        mass_penalty = unbalanced_ot_rho * (
            _generalized_kl(transported_pred, a, eps)
            + _generalized_kl(transported_target, b, eps)
        )
        out[start:end] = transport_cost + mass_penalty
    loss_per_slot = out.view(*pred_mask_probs.shape[:-1])
    if area_normalize:
        if area_normalizer is None:
            # Safe fallback for callers without context frames. Prefer passing
            # a context+goal max-area normalizer from the planner.
            area_normalizer = target_mask_probs.detach().float().sum(dim=-1).amax(dim=1)
        denom = _broadcast_slot_values(area_normalizer, loss_per_slot)
        loss_per_slot = (
            loss_per_slot / denom.clamp_min(float(area_normalize_eps))
        ) * float(pred_mask_probs.shape[-2])

    if slot_weights is None:
        loss_per_time = loss_per_slot.mean(dim=-1)
    else:
        weights = _broadcast_slot_values(slot_weights, loss_per_slot).clamp_min(0.0)
        weighted = (loss_per_slot * weights).sum(dim=-1)
        denom = weights.expand_as(loss_per_slot).sum(dim=-1).clamp_min(1.0e-6)
        loss_per_time = weighted / denom
    loss = (
        loss_per_time.mean(dim=tuple(range(1, loss_per_time.ndim)))
        if reduce_time and loss_per_time.ndim > 1
        else loss_per_time
    )
    if return_per_slot:
        return loss, loss_per_slot
    return loss


def _mask_l1_loss(
    pred_mask_probs,
    target_mask_probs,
    slot_weights=None,
    area_normalizer=None,
    area_normalize=False,
    area_normalize_eps=1.0e-3,
    reduce_time=True,
    return_per_slot=False,
):
    """Mask L1 with the same per-slot area/weight reduction as UOT."""
    if pred_mask_probs.shape != target_mask_probs.shape:
        raise ValueError(
            "Expected equal shapes for mask L1 objective: "
            f"{tuple(pred_mask_probs.shape)} vs {tuple(target_mask_probs.shape)}"
        )
    loss_per_slot = (
        pred_mask_probs.float() - target_mask_probs.float()
    ).abs().mean(dim=-1)
    if area_normalize:
        if area_normalizer is None:
            area_normalizer = target_mask_probs.detach().float().sum(dim=-1).amax(dim=1)
        denom = _broadcast_slot_values(area_normalizer, loss_per_slot)
        loss_per_slot = (
            loss_per_slot / denom.clamp_min(float(area_normalize_eps))
        ) * float(pred_mask_probs.shape[-2])

    if slot_weights is None:
        loss_per_time = loss_per_slot.mean(dim=-1)
    else:
        weights = _broadcast_slot_values(slot_weights, loss_per_slot).clamp_min(0.0)
        weighted = (loss_per_slot * weights).sum(dim=-1)
        denominator = weights.expand_as(loss_per_slot).sum(dim=-1).clamp_min(1.0e-6)
        loss_per_time = weighted / denominator
    loss = (
        loss_per_time.mean(dim=tuple(range(1, loss_per_time.ndim)))
        if reduce_time and loss_per_time.ndim > 1
        else loss_per_time
    )
    if return_per_slot:
        return loss, loss_per_slot
    return loss


def _mask_kl_loss(
    pred_mask_logits,
    target_mask_probs,
    slot_weights=None,
    area_normalizer=None,
    area_normalize=False,
    area_normalize_eps=1.0e-3,
    reduce_time=True,
    return_per_slot=False,
):
    """Per-slot Bernoulli KL (soft-target BCE minus target entropy).

    Each slot is compared as an independent foreground/background mask.  Its
    divergence is non-negative before slot weighting, so area/change weights
    can be applied directly without breaking the categorical KL cancellation.
    Predictions still come from the shared softmax and therefore retain the
    mutually-exclusive slot partition used by the model.
    """
    if pred_mask_logits.shape != target_mask_probs.shape:
        raise ValueError(
            "Expected equal shapes for mask KL objective: "
            f"{tuple(pred_mask_logits.shape)} vs {tuple(target_mask_probs.shape)}"
        )
    pred = F.softmax(pred_mask_logits.float(), dim=-2).clamp(
        min=1.0e-6, max=1.0 - 1.0e-6
    )
    target = target_mask_probs.detach().float()
    target_safe = target.clamp(min=1.0e-6, max=1.0 - 1.0e-6)
    bernoulli_kl = (
        target * (target_safe.log() - pred.log())
        + (1.0 - target)
        * ((1.0 - target_safe).log() - (1.0 - pred).log())
    ).clamp_min(0.0)
    loss_per_slot = bernoulli_kl.mean(dim=-1)
    if area_normalize:
        if area_normalizer is None:
            area_normalizer = target.sum(dim=-1).amax(dim=1)
        denom = _broadcast_slot_values(area_normalizer, loss_per_slot)
        loss_per_slot = (
            loss_per_slot / denom.clamp_min(float(area_normalize_eps))
        ) * float(pred_mask_logits.shape[-2])

    if slot_weights is None:
        loss_per_time = loss_per_slot.mean(dim=-1)
    else:
        weights = _broadcast_slot_values(slot_weights, loss_per_slot).clamp_min(0.0)
        weighted = (loss_per_slot * weights).sum(dim=-1)
        denominator = weights.expand_as(loss_per_slot).sum(dim=-1).clamp_min(1.0e-6)
        loss_per_time = weighted / denominator
    loss = (
        loss_per_time.mean(dim=tuple(range(1, loss_per_time.ndim)))
        if reduce_time and loss_per_time.ndim > 1
        else loss_per_time
    )
    if return_per_slot:
        return loss, loss_per_slot
    return loss


def _mask_sobel_loss(
    pred_mask_probs,
    target_mask_probs,
    slot_weights=None,
    area_normalizer=None,
    area_normalize=False,
    area_normalize_eps=1.0e-3,
    reduce_time=True,
    return_per_slot=False,
):
    """Per-slot L1 loss between signed x/y Sobel responses."""
    if pred_mask_probs.shape != target_mask_probs.shape:
        raise ValueError(
            "Expected equal shapes for mask Sobel objective: "
            f"{tuple(pred_mask_probs.shape)} vs {tuple(target_mask_probs.shape)}"
        )
    num_patches = pred_mask_probs.shape[-1]
    spatial_size = int(np.sqrt(num_patches))
    if spatial_size * spatial_size != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} patches")

    # Compare signed x/y derivatives instead of gradient magnitudes so the loss
    # also supervises the local contour normal. Replicate padding avoids
    # introducing an artificial edge at the image boundary.
    kernels = pred_mask_probs.new_tensor(
        [
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        ],
        dtype=torch.float32,
    ).unsqueeze(1) / 8.0
    leading_shape = pred_mask_probs.shape[:-2]
    num_slots = pred_mask_probs.shape[-2]
    pred_flat = pred_mask_probs.float().reshape(-1, 1, spatial_size, spatial_size)
    target_flat = target_mask_probs.detach().float().reshape(
        -1, 1, spatial_size, spatial_size
    )
    pred_grad = F.conv2d(
        F.pad(pred_flat, (1, 1, 1, 1), mode="replicate"), kernels
    ).reshape(*leading_shape, num_slots, 2, spatial_size, spatial_size)
    target_grad = F.conv2d(
        F.pad(target_flat, (1, 1, 1, 1), mode="replicate"), kernels
    ).reshape(*leading_shape, num_slots, 2, spatial_size, spatial_size)

    loss_per_slot = (pred_grad - target_grad).abs().mean(dim=(-3, -2, -1))
    if area_normalize:
        if area_normalizer is None:
            area_normalizer = (
                target_mask_probs.detach().float().sum(dim=-1).amax(dim=1)
            )
        denom = _broadcast_slot_values(area_normalizer, loss_per_slot)
        loss_per_slot = (
            loss_per_slot / denom.clamp_min(float(area_normalize_eps))
        ) * float(num_slots)

    if slot_weights is None:
        loss_per_time = loss_per_slot.mean(dim=-1)
    else:
        weights = _broadcast_slot_values(slot_weights, loss_per_slot).clamp_min(0.0)
        weighted = (loss_per_slot * weights).sum(dim=-1)
        denominator = weights.expand_as(loss_per_slot).sum(dim=-1).clamp_min(1.0e-6)
        loss_per_time = weighted / denominator
    loss = (
        loss_per_time.mean(dim=tuple(range(1, loss_per_time.ndim)))
        if reduce_time and loss_per_time.ndim > 1
        else loss_per_time
    )
    if return_per_slot:
        return loss, loss_per_slot
    return loss


def _straight_line_obstacle_cost(obstacle_mask, line_samples=21):
    """Return max obstacle occupancy along every patch-to-patch line."""
    if obstacle_mask.ndim != 2:
        raise ValueError(
            f"Expected obstacle masks [B,N], got {tuple(obstacle_mask.shape)}"
        )
    num_patches = obstacle_mask.shape[-1]
    spatial_size = int(np.sqrt(num_patches))
    if spatial_size * spatial_size != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} patches")
    line_samples = int(line_samples)
    if line_samples < 2:
        raise ValueError("line_samples must be at least 2")

    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, spatial_size, device=obstacle_mask.device),
        torch.linspace(-1.0, 1.0, spatial_size, device=obstacle_mask.device),
        indexing="ij",
    )
    coords = torch.stack([xs, ys], dim=-1).reshape(num_patches, 2)
    interpolation = torch.linspace(
        0.0, 1.0, line_samples, device=obstacle_mask.device
    )
    lines = (
        coords[:, None, None, :] * (1.0 - interpolation[None, None, :, None])
        + coords[None, :, None, :] * interpolation[None, None, :, None]
    )
    grid = lines.reshape(1, num_patches * num_patches, line_samples, 2).expand(
        obstacle_mask.shape[0], -1, -1, -1
    )
    image = obstacle_mask.reshape(
        obstacle_mask.shape[0], 1, spatial_size, spatial_size
    )
    sampled = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, 0].amax(dim=-1).reshape(
        obstacle_mask.shape[0], num_patches, num_patches
    )


def _geodesic_obstacle_cost(
    obstacle_mask,
    obstacle_dilation=0,
    penalty_weight=10.0,
):
    """Return soft obstacle-weighted all-pairs patch-grid path costs.

    Patches are connected with an 8-neighbour graph in the same normalized
    coordinates used by the UOT ground cost. Obstacle occupancy continuously
    increases each edge length instead of making any patch impassable. Returned
    values are squared path lengths so they match the Euclidean UOT cost units.
    """
    if obstacle_mask.ndim != 2:
        raise ValueError(
            f"Expected obstacle masks [B,N], got {tuple(obstacle_mask.shape)}"
        )
    num_patches = obstacle_mask.shape[-1]
    spatial_size = int(np.sqrt(num_patches))
    if spatial_size * spatial_size != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} patches")

    obstacle_dilation = int(obstacle_dilation)
    if obstacle_dilation < 0:
        raise ValueError("obstacle_dilation must be non-negative")
    occupancy = obstacle_mask.reshape(-1, 1, spatial_size, spatial_size)
    if obstacle_dilation:
        kernel_size = 2 * obstacle_dilation + 1
        occupancy = F.max_pool2d(
            occupancy,
            kernel_size=kernel_size,
            stride=1,
            padding=obstacle_dilation,
        )
    occupancy = occupancy[:, 0].clamp(0.0, 1.0)

    batch_size = obstacle_mask.shape[0]
    dtype = torch.float32
    device = obstacle_mask.device
    distances = torch.full(
        (batch_size, num_patches, num_patches),
        float("inf"),
        device=device,
        dtype=dtype,
    )
    diagonal = torch.arange(num_patches, device=device)
    distances[:, diagonal, diagonal] = 0.0
    grid_step = 1.0 / float(max(1, spatial_size - 1))

    # Add each undirected edge once. For diagonal edges, include both cells at
    # the crossed corner so a path cannot exploit a low-valued diagonal gap.
    # Scaling the penalty by edge length approximates a resolution-independent
    # line integral of obstacle occupancy along the path.
    neighbour_offsets = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(spatial_size):
        for col in range(spatial_size):
            source = row * spatial_size + col
            for delta_row, delta_col in neighbour_offsets:
                next_row = row + delta_row
                next_col = col + delta_col
                if not (
                    0 <= next_row < spatial_size
                    and 0 <= next_col < spatial_size
                ):
                    continue
                target = next_row * spatial_size + next_col
                edge_occupancy = torch.maximum(
                    occupancy[:, row, col], occupancy[:, next_row, next_col]
                )
                if delta_row and delta_col:
                    edge_occupancy = torch.maximum(
                        edge_occupancy,
                        torch.maximum(
                            occupancy[:, row, next_col],
                            occupancy[:, next_row, col],
                        ),
                    )
                    edge_length = np.sqrt(2.0) * grid_step
                else:
                    edge_length = grid_step
                edge = float(edge_length) * (
                    1.0 + float(penalty_weight) * edge_occupancy
                )
                distances[:, source, target] = edge
                distances[:, target, source] = edge

    # Batched Floyd-Warshall is computed only once per planning context, not
    # per candidate. Callers chunk native-resolution grids to bound memory.
    for intermediate in range(num_patches):
        distances = torch.minimum(
            distances,
            distances[:, :, intermediate : intermediate + 1]
            + distances[:, intermediate : intermediate + 1, :],
        )
    return distances.square()


def _build_obstacle_transport_metadata(
    context_mask_probs,
    probe_mask_probs,
    static_change_threshold,
    moving_change_threshold,
    local_change_fraction,
    hardening_temperature,
    line_samples,
    penalty_weight,
    cost_mode="line_penalty",
    obstacle_dilation=0,
    moving_topk=0,
    static_bottomk=0,
    moving_requires_both_scores=False,
    all_non_moving_static=False,
    motion_score_mode="max",
    motion_topk_patches=10,
    fixed_slot_change_scores=None,
    static_slot_indices=None,
):
    """Identify static obstacles and construct a moving-slot ground cost."""
    if context_mask_probs.ndim != 4:
        raise ValueError(
            f"Expected context [B,T,K,N], got {tuple(context_mask_probs.shape)}"
        )
    use_fixed_scores = fixed_slot_change_scores is not None
    if not use_fixed_scores and (
        probe_mask_probs is None or probe_mask_probs.ndim != 4
    ):
        probe_shape = (
            None if probe_mask_probs is None else tuple(probe_mask_probs.shape)
        )
        raise ValueError(
            "Expected context [B,T,K,N] and probes [B,P,K,N], got "
            f"{tuple(context_mask_probs.shape)} and {probe_shape}"
        )
    initial = context_mask_probs[:, -1].detach().float()
    probes = (
        initial[:, None]
        if use_fixed_scores
        else probe_mask_probs.detach().float()
    )
    if probes.shape[0] != initial.shape[0] or probes.shape[2:] != initial.shape[1:]:
        raise ValueError("Probe masks must align with context cases, slots, and patches")

    # The mass-normalized score works well for compact slots, but dilutes a
    # local change in a large background slot. Retain a second score over the
    # most-changed spatial patches, then leave moderately changing slots in an
    # uncertainty band instead of forcing them into either semantic class.
    initial_mass = initial.sum(dim=-1).clamp_min(1.0e-6)
    absolute_change = (probes - initial[:, None]).abs()
    mass_relative_change = (
        absolute_change.sum(dim=-1)
        / initial_mass[:, None]
    ).mean(dim=1)
    local_patch_count = max(
        1, int(np.ceil(initial.shape[-1] * float(local_change_fraction)))
    )
    local_change = absolute_change.topk(
        local_patch_count, dim=-1
    ).values.mean(dim=-1).mean(dim=1)
    motion_topk_patches = int(motion_topk_patches)
    if not 1 <= motion_topk_patches <= initial.shape[-1]:
        raise ValueError(
            "motion_topk_patches must lie in "
            f"[1, {initial.shape[-1]}], got {motion_topk_patches}"
        )
    initial_topk_values, initial_topk_indices = initial.topk(
        motion_topk_patches, dim=-1
    )
    initial_topk_mask = torch.zeros_like(initial).scatter(
        -1, initial_topk_indices, initial_topk_values
    )
    probe_topk_values, probe_topk_indices = probes.topk(
        motion_topk_patches, dim=-1
    )
    probe_topk_mask = torch.zeros_like(probes).scatter(
        -1, probe_topk_indices, probe_topk_values
    )
    topk_support = (probe_topk_mask > 0) | (initial_topk_mask[:, None] > 0)
    topk_difference = (
        (probe_topk_mask - initial_topk_mask[:, None]).abs().sum(dim=-1)
        / topk_support.sum(dim=-1).clamp_min(1)
    ).mean(dim=1)
    topk_ranked_difference = (
        probe_topk_values - initial_topk_values[:, None]
    ).abs().mean(dim=-1).mean(dim=1)
    per_frame_difference_topk = absolute_change.topk(
        motion_topk_patches, dim=-1
    ).values.mean(dim=-1).mean(dim=1)
    mean_mask = torch.cat([initial[:, None], probes], dim=1).mean(dim=1)
    aligned_topk_indices = mean_mask.topk(motion_topk_patches, dim=-1).indices
    initial_aligned_topk = initial.gather(-1, aligned_topk_indices)
    probe_aligned_topk = probes.gather(
        -1,
        aligned_topk_indices[:, None].expand(
            -1, probes.shape[1], -1, -1
        ),
    )
    mean_topk_aligned_difference = (
        probe_aligned_topk - initial_aligned_topk[:, None]
    ).abs().mean(dim=(-1, 1))
    spatial_size = int(np.sqrt(initial.shape[-1]))
    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, spatial_size, device=initial.device),
        torch.linspace(0.0, 1.0, spatial_size, device=initial.device),
        indexing="ij",
    )
    coords = torch.stack([ys, xs], dim=-1).reshape(1, 1, -1, 2)
    initial_distribution = initial / initial_mass[:, :, None]
    probe_mass = probes.sum(dim=-1).clamp_min(1.0e-6)
    probe_distribution = probes / probe_mass[:, :, :, None]
    initial_centroid = (initial_distribution[..., None] * coords).sum(dim=-2)
    probe_centroid = (
        probe_distribution[..., None] * coords[:, None]
    ).sum(dim=-2)
    centroid_displacement = torch.linalg.vector_norm(
        probe_centroid - initial_centroid[:, None], dim=-1
    ).mean(dim=1)
    centered = coords - initial_centroid[:, :, None, :]
    spatial_radius = torch.sqrt(
        (initial_distribution[..., None] * centered.square()).sum(dim=(-2, -1))
    ).clamp_min(1.0e-4)
    relative_centroid_displacement = centroid_displacement / spatial_radius
    motion_score_mode = str(motion_score_mode)
    if motion_score_mode == "max":
        change_score = torch.maximum(mass_relative_change, local_change)
    elif motion_score_mode == "topk_difference":
        change_score = topk_difference
    elif motion_score_mode == "topk_ranked_difference":
        change_score = topk_ranked_difference
    elif motion_score_mode == "mean_topk_aligned_difference":
        change_score = mean_topk_aligned_difference
    elif motion_score_mode == "per_frame_difference_topk":
        change_score = per_frame_difference_topk
    else:
        raise ValueError(
            "motion_score_mode must be 'max', 'topk_difference', or "
            "'topk_ranked_difference', 'mean_topk_aligned_difference', or "
            "'per_frame_difference_topk', got "
            f"{motion_score_mode!r}"
        )
    if use_fixed_scores:
        fixed_scores = torch.as_tensor(
            fixed_slot_change_scores, device=initial.device, dtype=initial.dtype
        )
        if fixed_scores.ndim != 1 or fixed_scores.numel() != initial.shape[1]:
            raise ValueError(
                "fixed_slot_change_scores must contain exactly one score per "
                f"slot ({initial.shape[1]}), got shape {tuple(fixed_scores.shape)}"
            )
        change_score = fixed_scores[None].expand(initial.shape[0], -1)
    area_fraction = initial_mass / float(initial.shape[-1])
    num_slots = change_score.shape[-1]
    moving_topk = int(moving_topk)
    static_bottomk = int(static_bottomk)
    if not 0 <= moving_topk <= num_slots:
        raise ValueError(
            f"moving_topk must lie in [0, {num_slots}], got {moving_topk}"
        )
    if not 0 <= static_bottomk <= num_slots:
        raise ValueError(
            f"static_bottomk must lie in [0, {num_slots}], got {static_bottomk}"
        )

    # The legacy union mode treats either a distributed mass change or a local
    # patch change as sufficient evidence of motion.  The strict mode requires
    # both signals, which rejects background slots that merely absorb a moving
    # object's probability mass or flicker in a small region.
    if moving_requires_both_scores:
        moving_slots = (
            (mass_relative_change > float(moving_change_threshold))
            & (local_change > float(moving_change_threshold))
        )
    else:
        moving_slots = change_score > float(moving_change_threshold)
    obstacle_slots = change_score <= float(static_change_threshold)
    if moving_topk:
        moving_indices = change_score.topk(moving_topk, dim=-1).indices
        moving_slots.scatter_(dim=-1, index=moving_indices, value=True)
    if static_bottomk:
        static_indices = change_score.topk(
            static_bottomk, dim=-1, largest=False
        ).indices
        obstacle_slots.scatter_(dim=-1, index=static_indices, value=True)
    # In strict obstacle mode, every slot without clear motion contributes to
    # the obstacle field.  Otherwise preserve the legacy uncertainty band.
    if all_non_moving_static:
        obstacle_slots = ~moving_slots
    else:
        # A slot cannot define the obstacle field and be transported through it.
        # In the degenerate overlap case, preserve the moving-slot guarantee.
        obstacle_slots &= ~moving_slots
    if static_slot_indices is not None:
        indices = [int(index) for index in static_slot_indices]
        if len(set(indices)) != len(indices) or any(
            index < 0 or index >= num_slots for index in indices
        ):
            raise ValueError(f"Invalid static_slot_indices: {indices}")
        obstacle_slots = torch.zeros_like(moving_slots)
        obstacle_slots[:, indices] = True
        if bool(torch.any(obstacle_slots & moving_slots)):
            raise ValueError("static_slot_indices overlap moving slots")
    ignored_slots = ~(obstacle_slots | moving_slots)

    # Use one predicted action as requested. Normalize each selected slot by
    # its own spatial peak, then apply a temperature exponent. This removes the
    # diffuse tail of a wall mask without assigning every background patch to
    # one of the selected static slots.
    selected = (initial if use_fixed_scores else probes[:, 0]).clamp_min(1.0e-8)
    spatial_peak = selected.amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    hard_slot_masks = (selected / spatial_peak).pow(
        float(hardening_temperature)
    )
    obstacle_mask = (
        hard_slot_masks
        * obstacle_slots[:, :, None].to(hard_slot_masks.dtype)
    ).amax(dim=1).clamp(0.0, 1.0)
    if cost_mode == "line_penalty":
        crossing_cost = _straight_line_obstacle_cost(
            obstacle_mask, line_samples=line_samples
        )
        addition = (
            float(penalty_weight)
            * moving_slots[:, :, None, None].to(crossing_cost.dtype)
            * crossing_cost[:, None]
        )
    elif cost_mode == "geodesic":
        geodesic_cost = _geodesic_obstacle_cost(
            obstacle_mask,
            obstacle_dilation=obstacle_dilation,
            penalty_weight=penalty_weight,
        )
        spatial_size = int(np.sqrt(initial.shape[-1]))
        ys, xs = torch.meshgrid(
            torch.linspace(0.0, 1.0, spatial_size, device=initial.device),
            torch.linspace(0.0, 1.0, spatial_size, device=initial.device),
            indexing="ij",
        )
        coords = torch.stack([ys, xs], dim=-1).reshape(-1, 2)
        euclidean_cost = torch.cdist(coords, coords, p=2).square().float()
        # Geodesic paths cannot be shorter than straight lines. Clamping also
        # removes tiny negative differences caused by floating-point rounding.
        geodesic_addition = (
            geodesic_cost - euclidean_cost.unsqueeze(0)
        ).clamp_min(0.0)
        addition = (
            moving_slots[:, :, None, None].to(geodesic_addition.dtype)
            * geodesic_addition[:, None]
        )
    else:
        raise ValueError(f"Unsupported obstacle transport cost_mode: {cost_mode}")
    metadata = {
        "transport_cost_addition": addition,
        "obstacle_slot_change": change_score,
        "obstacle_slot_mass_relative_change": mass_relative_change,
        "obstacle_slot_local_change": local_change,
        "obstacle_slot_topk_difference": topk_difference,
        "obstacle_slot_topk_ranked_difference": topk_ranked_difference,
        "obstacle_slot_mean_topk_aligned_difference": mean_topk_aligned_difference,
        "obstacle_slot_per_frame_difference_topk": per_frame_difference_topk,
        "obstacle_slot_centroid_displacement": centroid_displacement,
        "obstacle_slot_spatial_radius": spatial_radius,
        "obstacle_slot_relative_centroid_displacement": relative_centroid_displacement,
        "obstacle_slot_area_fraction": area_fraction,
        "obstacle_slot_flags": obstacle_slots,
        "moving_slot_flags": moving_slots,
        "ignored_slot_flags": ignored_slots,
        "hardened_obstacle_mask": obstacle_mask,
    }
    return metadata


def create_objective_fn(alpha, base, mode="last"):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt, **_objective_kwargs):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt, **_objective_kwargs):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError


def create_uot_objective_fn(
    alpha,
    base=2,
    mode="last",
    uot_loss_weight=1.0,
    mask_l1_loss_weight=10.0,
    sinkhorn_iters=5,
    entropy_reg=0.05,
    unbalanced_ot_rho=0.2,
    uot_batch_size=128,
    mask_transport_area_normalize=False,
    mask_transport_area_normalize_eps=1.0e-3,
    slot_change_weighting=False,
    slot_change_min_weight=0.05,
    slot_change_max_weight=5.0,
    obstacle_transport=None,
):
    uot_loss_weight = float(uot_loss_weight)
    if uot_loss_weight < 0.0:
        raise ValueError("uot_loss_weight must be non-negative")
    mask_l1_loss_weight = float(mask_l1_loss_weight)
    if mask_l1_loss_weight < 0.0:
        raise ValueError("mask_l1_loss_weight must be non-negative")
    obstacle_transport = dict(obstacle_transport or {})
    obstacle_transport_enabled = bool(obstacle_transport.pop("enabled", False))
    obstacle_probe_num_actions = int(obstacle_transport.pop("num_probe_actions", 16))
    obstacle_probe_action_std = float(obstacle_transport.pop("probe_action_std", 1.0))
    obstacle_probe_seed = int(obstacle_transport.pop("probe_seed", 1729))
    obstacle_static_change_threshold = float(
        obstacle_transport.pop("static_change_threshold", 0.15)
    )
    obstacle_moving_change_threshold = float(
        obstacle_transport.pop(
            "moving_change_threshold", obstacle_static_change_threshold
        )
    )
    obstacle_local_change_fraction = float(
        obstacle_transport.pop("local_change_fraction", 0.05)
    )
    obstacle_hardening_temperature = float(
        obstacle_transport.pop("hardening_temperature", 20.0)
    )
    obstacle_cost_mode = str(obstacle_transport.pop("cost_mode", "line_penalty"))
    obstacle_line_samples = int(obstacle_transport.pop("line_samples", 21))
    obstacle_penalty_weight = float(obstacle_transport.pop("penalty_weight", 5.0))
    obstacle_dilation = int(obstacle_transport.pop("obstacle_dilation", 0))
    obstacle_moving_topk = int(obstacle_transport.pop("moving_topk", 0))
    obstacle_static_bottomk = int(obstacle_transport.pop("static_bottomk", 0))
    obstacle_moving_requires_both_scores = bool(
        obstacle_transport.pop("moving_requires_both_scores", False)
    )
    obstacle_all_non_moving_static = bool(
        obstacle_transport.pop("all_non_moving_static", False)
    )
    obstacle_motion_score_mode = str(
        obstacle_transport.pop("motion_score_mode", "max")
    )
    obstacle_motion_topk_patches = int(
        obstacle_transport.pop("motion_topk_patches", 10)
    )
    obstacle_fixed_slot_change_scores = obstacle_transport.pop(
        "fixed_slot_change_scores", None
    )
    obstacle_static_slot_indices = obstacle_transport.pop("static_slot_indices", None)
    if obstacle_fixed_slot_change_scores is not None:
        obstacle_fixed_slot_change_scores = [
            float(value) for value in obstacle_fixed_slot_change_scores
        ]
    if obstacle_transport:
        raise ValueError(
            f"Unknown obstacle_transport options: {sorted(obstacle_transport)}"
        )
    if obstacle_transport_enabled:
        if (
            obstacle_fixed_slot_change_scores is None
            and obstacle_probe_num_actions < 1
        ):
            raise ValueError("num_probe_actions must be positive")
        if obstacle_probe_action_std <= 0.0:
            raise ValueError("probe_action_std must be positive")
        if obstacle_static_change_threshold < 0.0:
            raise ValueError("static_change_threshold must be non-negative")
        if obstacle_moving_change_threshold < obstacle_static_change_threshold:
            raise ValueError(
                "moving_change_threshold must be greater than or equal to "
                "static_change_threshold"
            )
        if not 0.0 < obstacle_local_change_fraction <= 1.0:
            raise ValueError("local_change_fraction must lie in (0, 1]")
        if obstacle_hardening_temperature <= 0.0:
            raise ValueError("hardening_temperature must be positive")
        if obstacle_cost_mode not in {"line_penalty", "geodesic"}:
            raise ValueError(
                "obstacle_transport.cost_mode must be 'line_penalty' or 'geodesic'"
            )
        if obstacle_line_samples < 2:
            raise ValueError("line_samples must be at least 2")
        if obstacle_penalty_weight < 0.0:
            raise ValueError("penalty_weight must be non-negative")
        if obstacle_dilation < 0:
            raise ValueError("obstacle_dilation must be non-negative")
        if obstacle_moving_topk < 0:
            raise ValueError("moving_topk must be non-negative")
        if obstacle_static_bottomk < 0:
            raise ValueError("static_bottomk must be non-negative")
    # The spatial grid is fixed throughout planning. Reuse its cost and
    # entropic kernel instead of rebuilding them for every CEM micro-batch.
    transport_cache = {}

    def build_slot_metadata(context_mask_probs, target_mask_probs):
        current = context_mask_probs[:, -1]
        target = target_mask_probs[:, -1]
        metadata = {}
        if slot_change_weighting:
            metadata["slot_weights"] = _slot_change_weights(
                current,
                target,
                min_weight=slot_change_min_weight,
                max_weight=slot_change_max_weight,
            )
        if mask_transport_area_normalize:
            metadata["area_normalizer"] = _slot_area_normalizer(
                context_mask_probs,
                target_mask_probs,
                eps=mask_transport_area_normalize_eps,
            )
        return metadata

    uot_kwargs = {
        "sinkhorn_iters": sinkhorn_iters,
        "entropy_reg": entropy_reg,
        "unbalanced_ot_rho": unbalanced_ot_rho,
        "batch_size": uot_batch_size,
        "transport_cache": transport_cache,
        "area_normalize": mask_transport_area_normalize,
        "area_normalize_eps": mask_transport_area_normalize_eps,
    }

    def objective_fn_last(
        z_obs_pred,
        z_obs_tgt,
        slot_weights=None,
        area_normalizer=None,
        transport_cost_addition=None,
    ):
        if "mask_probs" not in z_obs_pred or "mask_probs" not in z_obs_tgt:
            raise KeyError("UOT objective requires `mask_probs` in both prediction and target dicts")
        pred_masks = z_obs_pred["mask_probs"][:, -1:]
        target_masks = z_obs_tgt["mask_probs"]
        loss_uot = (
            pred_masks.new_zeros(pred_masks.shape[0], dtype=torch.float32)
            if uot_loss_weight == 0.0
            else _uot_mask_loss(
                pred_masks,
                target_masks,
                slot_weights=slot_weights,
                area_normalizer=area_normalizer,
                transport_cost_addition=transport_cost_addition,
                **uot_kwargs,
            )
        )
        loss_mask_l1 = _mask_l1_loss(
            pred_masks,
            target_masks,
            slot_weights=slot_weights,
            area_normalizer=area_normalizer,
            area_normalize=mask_transport_area_normalize,
            area_normalize_eps=mask_transport_area_normalize_eps,
        )
        loss_visual = uot_loss_weight * loss_uot + mask_l1_loss_weight * loss_mask_l1
        loss_proprio = ((z_obs_pred["proprio"][:, -1:] - z_obs_tgt["proprio"]) ** 2).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        return loss_visual + alpha * loss_proprio

    def loss_components_last(
        z_obs_pred,
        z_obs_tgt,
        slot_weights=None,
        area_normalizer=None,
        transport_cost_addition=None,
    ):
        """Return the exact summed terms plus diagnostics used by CEM."""
        if "mask_probs" not in z_obs_pred or "mask_probs" not in z_obs_tgt:
            raise KeyError(
                "UOT objective requires `mask_probs` in both prediction and target dicts"
            )
        pred_masks = z_obs_pred["mask_probs"][:, -1:]
        target_masks = z_obs_tgt["mask_probs"]
        if uot_loss_weight == 0.0:
            loss_uot = pred_masks.new_zeros(
                pred_masks.shape[0], dtype=torch.float32
            )
            loss_uot_per_slot = pred_masks.new_zeros(
                pred_masks.shape[:-1], dtype=torch.float32
            )
            loss_uot_euclidean = loss_uot.clone()
            loss_uot_euclidean_per_slot = loss_uot_per_slot.clone()
        else:
            loss_uot, loss_uot_per_slot = _uot_mask_loss(
                pred_masks,
                target_masks,
                slot_weights=slot_weights,
                area_normalizer=area_normalizer,
                transport_cost_addition=transport_cost_addition,
                return_per_slot=True,
                **uot_kwargs,
            )
            loss_uot_euclidean, loss_uot_euclidean_per_slot = _uot_mask_loss(
                pred_masks,
                target_masks,
                slot_weights=slot_weights,
                area_normalizer=area_normalizer,
                return_per_slot=True,
                **uot_kwargs,
            )
        loss_mask_l1, loss_mask_l1_per_slot = _mask_l1_loss(
            pred_masks,
            target_masks,
            slot_weights=slot_weights,
            area_normalizer=area_normalizer,
            area_normalize=mask_transport_area_normalize,
            area_normalize_eps=mask_transport_area_normalize_eps,
            return_per_slot=True,
        )
        loss_proprio = (
            (z_obs_pred["proprio"][:, -1:] - z_obs_tgt["proprio"]) ** 2
        ).mean(dim=tuple(range(1, z_obs_pred["proprio"].ndim)))
        weighted_mask_l1 = mask_l1_loss_weight * loss_mask_l1
        weighted_proprio = alpha * loss_proprio
        weighted_uot = uot_loss_weight * loss_uot
        total = weighted_uot + weighted_mask_l1 + weighted_proprio
        return {
            "total": total,
            "uot": loss_uot,
            "weighted_uot": weighted_uot,
            "uot_euclidean": loss_uot_euclidean,
            "obstacle_effect": loss_uot - loss_uot_euclidean,
            "mask_l1": loss_mask_l1,
            "weighted_mask_l1": weighted_mask_l1,
            "proprio_mse": loss_proprio,
            "weighted_proprio": weighted_proprio,
            "uot_per_slot": loss_uot_per_slot,
            "uot_euclidean_per_slot": loss_uot_euclidean_per_slot,
            "mask_l1_per_slot": loss_mask_l1_per_slot,
        }

    def objective_fn_all(
        z_obs_pred,
        z_obs_tgt,
        slot_weights=None,
        area_normalizer=None,
        transport_cost_addition=None,
    ):
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["mask_probs"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["mask_probs"].device)
        target_masks = z_obs_tgt["mask_probs"]
        if target_masks.shape[1] == 1 and z_obs_pred["mask_probs"].shape[1] != 1:
            target_masks = target_masks.expand(
                -1, z_obs_pred["mask_probs"].shape[1], -1, -1
            )
        loss_uot = (
            z_obs_pred["mask_probs"].new_zeros(
                z_obs_pred["mask_probs"].shape[:-2], dtype=torch.float32
            )
            if uot_loss_weight == 0.0
            else _uot_mask_loss(
                z_obs_pred["mask_probs"],
                target_masks,
                slot_weights=slot_weights,
                area_normalizer=area_normalizer,
                reduce_time=False,
                transport_cost_addition=transport_cost_addition,
                **uot_kwargs,
            )
        )
        loss_mask_l1 = _mask_l1_loss(
            z_obs_pred["mask_probs"],
            target_masks,
            slot_weights=slot_weights,
            area_normalizer=area_normalizer,
            area_normalize=mask_transport_area_normalize,
            area_normalize_eps=mask_transport_area_normalize_eps,
            reduce_time=False,
        )
        loss_visual = uot_loss_weight * loss_uot + mask_l1_loss_weight * loss_mask_l1
        loss_proprio = ((z_obs_pred["proprio"] - z_obs_tgt["proprio"]) ** 2).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        return (loss_visual * coeffs).sum(dim=1) + alpha * (loss_proprio * coeffs).sum(dim=1)

    def loss_components_all(
        z_obs_pred,
        z_obs_tgt,
        slot_weights=None,
        area_normalizer=None,
        transport_cost_addition=None,
    ):
        """All-frame counterpart of ``loss_components_last``."""
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["mask_probs"].shape[1])],
            dtype=np.float32,
        )
        coeffs = torch.tensor(
            coeffs / np.sum(coeffs), device=z_obs_pred["mask_probs"].device
        )
        target_masks = z_obs_tgt["mask_probs"]
        if target_masks.shape[1] == 1 and z_obs_pred["mask_probs"].shape[1] != 1:
            target_masks = target_masks.expand(
                -1, z_obs_pred["mask_probs"].shape[1], -1, -1
            )
        if uot_loss_weight == 0.0:
            loss_uot_time = z_obs_pred["mask_probs"].new_zeros(
                z_obs_pred["mask_probs"].shape[:-2], dtype=torch.float32
            )
            loss_uot_per_slot = z_obs_pred["mask_probs"].new_zeros(
                z_obs_pred["mask_probs"].shape[:-1], dtype=torch.float32
            )
            loss_uot_euclidean_time = loss_uot_time.clone()
            loss_uot_euclidean_per_slot = loss_uot_per_slot.clone()
        else:
            loss_uot_time, loss_uot_per_slot = _uot_mask_loss(
                z_obs_pred["mask_probs"],
                target_masks,
                slot_weights=slot_weights,
                area_normalizer=area_normalizer,
                reduce_time=False,
                transport_cost_addition=transport_cost_addition,
                return_per_slot=True,
                **uot_kwargs,
            )
            loss_uot_euclidean_time, loss_uot_euclidean_per_slot = _uot_mask_loss(
                z_obs_pred["mask_probs"],
                target_masks,
                slot_weights=slot_weights,
                area_normalizer=area_normalizer,
                reduce_time=False,
                return_per_slot=True,
                **uot_kwargs,
            )
        loss_mask_l1_time, loss_mask_l1_per_slot = _mask_l1_loss(
            z_obs_pred["mask_probs"],
            target_masks,
            slot_weights=slot_weights,
            area_normalizer=area_normalizer,
            area_normalize=mask_transport_area_normalize,
            area_normalize_eps=mask_transport_area_normalize_eps,
            reduce_time=False,
            return_per_slot=True,
        )
        loss_proprio_time = (
            (z_obs_pred["proprio"] - z_obs_tgt["proprio"]) ** 2
        ).mean(dim=tuple(range(2, z_obs_pred["proprio"].ndim)))

        reduce_time = lambda value: (value * coeffs).sum(dim=1)
        loss_uot = reduce_time(loss_uot_time)
        loss_uot_euclidean = reduce_time(loss_uot_euclidean_time)
        loss_mask_l1 = reduce_time(loss_mask_l1_time)
        loss_proprio = reduce_time(loss_proprio_time)
        weighted_mask_l1 = mask_l1_loss_weight * loss_mask_l1
        weighted_proprio = alpha * loss_proprio
        weighted_uot = uot_loss_weight * loss_uot
        total = weighted_uot + weighted_mask_l1 + weighted_proprio
        return {
            "total": total,
            "uot": loss_uot,
            "weighted_uot": weighted_uot,
            "uot_euclidean": loss_uot_euclidean,
            "obstacle_effect": loss_uot - loss_uot_euclidean,
            "mask_l1": loss_mask_l1,
            "weighted_mask_l1": weighted_mask_l1,
            "proprio_mse": loss_proprio,
            "weighted_proprio": weighted_proprio,
            "uot_per_slot": loss_uot_per_slot,
            "uot_euclidean_per_slot": loss_uot_euclidean_per_slot,
            "mask_l1_per_slot": loss_mask_l1_per_slot,
        }

    if mode == "last":
        objective_fn = objective_fn_last
        objective_fn.loss_components = loss_components_last
    elif mode == "all":
        objective_fn = objective_fn_all
        objective_fn.loss_components = loss_components_all
    else:
        raise NotImplementedError
    objective_fn.build_slot_metadata = build_slot_metadata
    objective_fn.uot_loss_weight = uot_loss_weight
    objective_fn.mask_l1_loss_weight = mask_l1_loss_weight
    objective_fn.obstacle_probe_num_actions = (
        obstacle_probe_num_actions
        if (
            uot_loss_weight > 0.0
            and obstacle_transport_enabled
            and obstacle_fixed_slot_change_scores is None
        )
        else 0
    )
    objective_fn.obstacle_use_fixed_slot_scores = (
        uot_loss_weight > 0.0
        and obstacle_transport_enabled
        and obstacle_fixed_slot_change_scores is not None
    )
    objective_fn.obstacle_probe_action_std = obstacle_probe_action_std
    objective_fn.obstacle_probe_seed = obstacle_probe_seed

    def build_obstacle_metadata(context_mask_probs, probe_mask_probs):
        metadata = _build_obstacle_transport_metadata(
            context_mask_probs=context_mask_probs,
            probe_mask_probs=probe_mask_probs,
            static_change_threshold=obstacle_static_change_threshold,
            moving_change_threshold=obstacle_moving_change_threshold,
            local_change_fraction=obstacle_local_change_fraction,
            hardening_temperature=obstacle_hardening_temperature,
            line_samples=obstacle_line_samples,
            penalty_weight=obstacle_penalty_weight,
            cost_mode=obstacle_cost_mode,
            obstacle_dilation=obstacle_dilation,
            moving_topk=obstacle_moving_topk,
            static_bottomk=obstacle_static_bottomk,
            moving_requires_both_scores=obstacle_moving_requires_both_scores,
            all_non_moving_static=obstacle_all_non_moving_static,
            motion_score_mode=obstacle_motion_score_mode,
            motion_topk_patches=obstacle_motion_topk_patches,
            fixed_slot_change_scores=obstacle_fixed_slot_change_scores,
            static_slot_indices=obstacle_static_slot_indices,
        )
        objective_fn.last_obstacle_probe = {
            key: value.detach() for key, value in metadata.items()
            if key != "transport_cost_addition"
        }
        return {
            "transport_cost_addition": metadata["transport_cost_addition"]
        }

    objective_fn.build_obstacle_metadata = (
        build_obstacle_metadata
        if uot_loss_weight > 0.0 and obstacle_transport_enabled
        else None
    )
    objective_fn.last_obstacle_probe = None
    return objective_fn
