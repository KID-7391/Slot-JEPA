"""VOS evaluation orchestrator."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import queue

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from .config import resolve_protocol_frame_short_side
from .dataset_handlers import get_handler
from .label_propagation import (
    extract_feature,
    extract_video_clip_target_features,
    label_propagation,
    upsample_and_norm_mask,
)
from .utils import (
    DEFAULT_ALIGN,
    imwrite_indexed,
    load_color_palette,
    read_frame,
    read_frame_list,
    read_seg,
)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _stable_video_seed(base_seed: int, video_name: str) -> int:
    """Derive an order-independent RNG seed for one video."""
    digest = hashlib.sha256(str(video_name).encode("utf-8")).digest()
    video_offset = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return (int(base_seed) + video_offset) % ((1 << 63) - 1)


def _mask_center_token_indices(
    first_frame_segmentation,
    feat_h: int,
    feat_w: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return patch indices at first-frame instance-mask centers."""
    if first_frame_segmentation is None:
        return None
    segmentation = np.asarray(first_frame_segmentation)
    if segmentation.ndim == 3:
        segmentation = segmentation[..., 0]
    if segmentation.ndim != 2:
        raise ValueError(
            "Expected first-frame segmentation with shape [H,W], "
            f"got {tuple(segmentation.shape)}"
        )

    seg_h, seg_w = segmentation.shape
    center_indices = []
    object_ids = [int(value) for value in np.unique(segmentation) if int(value) not in {0, 255}]
    for object_id in object_ids:
        coordinates = np.argwhere(segmentation == object_id)
        if coordinates.size == 0:
            continue
        mean_yx = coordinates.mean(axis=0)
        nearest_idx = np.square(coordinates - mean_yx).sum(axis=1).argmin()
        center_y, center_x = coordinates[nearest_idx]
        feat_y = min(int((float(center_y) + 0.5) * feat_h / seg_h), feat_h - 1)
        feat_x = min(int((float(center_x) + 0.5) * feat_w / seg_w), feat_w - 1)
        center_index = (feat_y * feat_w) + feat_x
        if center_index not in center_indices:
            center_indices.append(center_index)

    if not center_indices:
        return None
    return torch.tensor(center_indices, device=device, dtype=torch.long).unsqueeze(0)


def _compute_align(patch_size, explicit_align=None):
    """Compute frame alignment factor from patch_size."""
    if explicit_align is not None:
        return int(explicit_align)
    if not patch_size or patch_size <= 0:
        return DEFAULT_ALIGN
    return int(patch_size)


def _compute_scale_size(frame_short_side):
    return (int(frame_short_side),)


def _expected_output_frames(handler, frame_list):
    outputs = [os.path.basename(frame_path).replace(".jpg", ".png") for frame_path in frame_list]
    if outputs:
        outputs[0] = handler.get_first_frame_output_name(frame_list)
    return outputs


def _video_is_completed(handler, video_folder, frame_list):
    if not os.path.isdir(video_folder):
        return False
    expected = _expected_output_frames(handler, frame_list)
    return all(os.path.exists(os.path.join(video_folder, frame_name)) for frame_name in expected)


def _build_cpu_video_params(vos_params):
    cpu_params = copy.deepcopy(vos_params)
    cpu_params["affinity_dtype"] = "float32"
    cpu_params["feature_dtype"] = "float32"
    cpu_params["label_agg_device"] = "cpu"
    return cpu_params


def _scores_to_label_map(scores, output_hw, align_corners=True):
    scores = upsample_and_norm_mask(scores, size=output_hw, align_corners=align_corners)
    _, seg = torch.max(scores, dim=0)
    return np.array(seg.squeeze().cpu(), dtype=np.uint8)


def _save_visualization(frame_path, pred_mask, output_path, color_palette):
    image = Image.open(frame_path).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)
    if pred_mask.shape != image_np.shape[:2]:
        pred_mask = np.array(
            Image.fromarray(pred_mask.astype(np.uint8)).resize((image_np.shape[1], image_np.shape[0]), 0),
            dtype=np.uint8,
        )
    overlay = image_np.copy()
    labels = np.unique(pred_mask)
    for label in labels:
        if int(label) <= 0:
            continue
        class_mask = pred_mask == label
        if not np.any(class_mask):
            continue
        color = np.array(color_palette[int(label) % len(color_palette)], dtype=np.float32)
        overlay[class_mask] = (
            0.55 * overlay[class_mask].astype(np.float32) + 0.45 * color
        ).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)


def _save_resized_frame_visualization(frame_tensor, output_path):
    frame = frame_tensor.detach().cpu().clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.228, 0.224, 0.225]).view(3, 1, 1)
    frame = (frame * std) + mean
    frame = frame.clamp(0.0, 1.0)
    frame_np = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(frame_np).save(output_path)


def _save_debug_exports(video_folder, frame_stem, frame_path, frame_tar, pred_mask, color_palette):
    debug_prefix = os.path.join(video_folder, f"debug_{frame_stem}")
    _save_resized_frame_visualization(frame_tar, f"{debug_prefix}_input.png")
    imwrite_indexed(f"{debug_prefix}_mask.png", pred_mask.astype(np.uint8), color_palette)
    _save_visualization(
        frame_path,
        pred_mask,
        f"{debug_prefix}_vis.png",
        color_palette,
    )
    debug_meta = {
        "resized_hw": [int(frame_tar.shape[1]), int(frame_tar.shape[2])],
    }
    with open(f"{debug_prefix}_meta.json", "w", encoding="utf-8") as handle:
        json.dump(debug_meta, handle, indent=2)
        handle.write("\n")


def _build_vjepa_clip(frames, frame_idx, video_infer_frames):
    start_idx = max(0, frame_idx - video_infer_frames + 1)
    clip_frames = frames[start_idx : frame_idx + 1]
    # Ensure the current frame falls into the final tubelet.
    if len(clip_frames) % 2 == 1:
        prefix_frame = clip_frames[0] if len(clip_frames) == 1 else clip_frames[-2]
        clip_frames = [prefix_frame] + clip_frames
    return torch.stack(clip_frames, dim=0)


def _estimate_patch_token_count(frame_hw, model):
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    patch_size = int(getattr(model, "patch_size", 0) or 0)
    patch_stride = int(getattr(model, "patch_stride", patch_size) or patch_size)
    patch_padding = int(getattr(model, "patch_padding", 0) or 0)
    if patch_size <= 0 or patch_stride <= 0:
        return None
    token_h = int((frame_h + (2 * patch_padding) - patch_size) // patch_stride + 1)
    token_w = int((frame_w + (2 * patch_padding) - patch_size) // patch_stride + 1)
    if token_h <= 0 or token_w <= 0:
        return None
    return token_h * token_w


def _extract_all_video_features(
    vos_params,
    model,
    frames,
    device,
    patch_size,
    video_name=None,
    first_frame_segmentation=None,
):
    feature_dtype = vos_params.get("feature_dtype", "float32")
    shift_and_stitch_factor = int(vos_params.get("shift_and_stitch_factor", 1))
    if not bool(vos_params.get("shift_and_stitch_enabled", False)):
        shift_and_stitch_factor = 1
    video_infer_frames = int(vos_params.get("video_infer_frames", 1))
    feature_chunk_size = int(vos_params.get("feature_chunk_size", 1))
    use_video_clip_features = bool(vos_params.get("use_video_clip_features", False)) and video_infer_frames > 1

    if not frames:
        raise ValueError("Expected at least one frame when precomputing video features")

    estimated_tokens = _estimate_patch_token_count((frames[0].shape[1], frames[0].shape[2]), model)
    patch_stride = int(getattr(model, "patch_stride", patch_size) or patch_size or 0)
    if (
        estimated_tokens is not None
        and patch_size
        and patch_stride > 0
        and patch_stride < patch_size
        and shift_and_stitch_factor > 1
    ):
        raise RuntimeError(
            "Unsupported VOS feature-extraction configuration: "
            f"patch_size={patch_size}, patch_stride={patch_stride}, "
            f"shift_and_stitch_factor={shift_and_stitch_factor}, "
            f"estimated_tokens_per_frame={estimated_tokens}. "
            "For VJEPA2 dense patch grids (patch_stride < patch_size), shift-and-stitch "
            "falls back to single-frame extraction and becomes prohibitively expensive. "
            "Set vos.shift_and_stitch_enabled=False, or use patch_stride == patch_size."
        )

    all_features = [None] * len(frames)
    feat_h = feat_w = None
    progress_desc = "feature-precompute" if video_name is None else f"{video_name} features"

    if hasattr(model, "extract_obj_latents") and hasattr(model, "mix_with_obj_latents"):
        raw_tokens = []
        for _, frame in tqdm(
            list(enumerate(frames)),
            total=len(frames),
            ascii=True,
            desc=progress_desc,
            leave=False,
        ):
            frame_feat, h_i, w_i = extract_feature(
                model,
                frame,
                device,
                return_h_w=True,
                patch_size=patch_size,
                feature_dtype=feature_dtype,
                shift_and_stitch_factor=1,
            )
            if feat_h is None:
                feat_h, feat_w = h_i, w_i
            elif feat_h != h_i or feat_w != w_i:
                raise ValueError(
                    "Inconsistent single-frame feature shapes across slot video: "
                    f"expected=({feat_h},{feat_w}) got=({h_i},{w_i})"
                )
            raw_tokens.append(frame_feat)

        video_tokens = torch.stack(raw_tokens, dim=0)
        if float(getattr(model, "alpha", 0.0)) > 0.0:
            target_token_maps = model.reshape_video_patch_embeddings(video_tokens, feat_h=feat_h, feat_w=feat_w)
            fps_init_mode = str(vos_params.get("slot_fps_init_mode", "first_frame_mask_center"))
            fps_initial_token_indices = None
            if fps_init_mode == "first_frame_mask_center":
                fps_initial_token_indices = _mask_center_token_indices(
                    first_frame_segmentation,
                    feat_h,
                    feat_w,
                    video_tokens.device,
                )
                if fps_initial_token_indices is None:
                    raise ValueError(
                        "slot_fps_init_mode=first_frame_mask_center requires at least "
                        f"one foreground object in the first-frame mask for video={video_name}"
                    )
            # Default VOS behavior for slot mixing:
            # 1) extract obj latents from uniformly sampled reference frames;
            # 2) reconstruct all frames in the current video;
            # 3) mix raw and reconstructed features with alpha.
            obj_latents, _ = model.extract_obj_latents(
                target_token_maps,
                max_reference_frames=int(vos_params.get("slot_reference_num_frames", 8)),
                fps_seed=_stable_video_seed(
                    int(vos_params.get("experiment_seed", 0)),
                    video_name or "feature-precompute",
                ),
                fps_initial_token_indices=fps_initial_token_indices,
            )
            mixed_token_maps, _ = model.mix_with_obj_latents(target_token_maps, obj_latents=obj_latents)
            mixed_tokens = model.flatten_video_token_maps(mixed_token_maps)
            mixed_tokens = mixed_tokens[0] if mixed_tokens.ndim == 4 else mixed_tokens
        else:
            mixed_tokens = video_tokens
        for frame_idx in range(len(frames)):
            all_features[frame_idx] = mixed_tokens[frame_idx].T.contiguous()
        return all_features, feat_h, feat_w

    if use_video_clip_features:
        chunk_starts = list(range(0, len(frames), feature_chunk_size))
        for chunk_start in tqdm(
            chunk_starts,
            ascii=True,
            desc=progress_desc,
            leave=False,
        ):
            chunk_end = min(chunk_start + feature_chunk_size, len(frames))
            clip_groups = {}
            for frame_idx in range(chunk_start, chunk_end):
                clip = _build_vjepa_clip(frames, frame_idx, video_infer_frames)
                clip_groups.setdefault(int(clip.shape[0]), []).append((frame_idx, clip))

            for _, grouped_clips in sorted(clip_groups.items()):
                frame_indices = [frame_idx for frame_idx, _ in grouped_clips]
                clip_batch = torch.stack([clip for _, clip in grouped_clips], dim=0)
                clip_feats, h_i, w_i = extract_video_clip_target_features(
                    model,
                    clip_batch,
                    device,
                    return_h_w=True,
                    patch_size=patch_size,
                    feature_dtype=feature_dtype,
                    shift_and_stitch_factor=shift_and_stitch_factor,
                    temporal_pool="last",
                )
                if feat_h is None:
                    feat_h, feat_w = h_i, w_i
                elif feat_h != h_i or feat_w != w_i:
                    raise ValueError(
                        "Inconsistent VJEPA feature shapes across chunks: "
                        f"expected=({feat_h},{feat_w}) got=({h_i},{w_i})"
                    )
                for batch_idx, frame_idx in enumerate(frame_indices):
                    all_features[frame_idx] = clip_feats[batch_idx].T
    else:
        for frame_idx, frame in tqdm(
            list(enumerate(frames)),
            total=len(frames),
            ascii=True,
            desc=progress_desc,
            leave=False,
        ):
            frame_feat, h_i, w_i = extract_feature(
                model,
                frame,
                device,
                return_h_w=True,
                patch_size=patch_size,
                feature_dtype=feature_dtype,
                shift_and_stitch_factor=shift_and_stitch_factor,
            )
            if feat_h is None:
                feat_h, feat_w = h_i, w_i
            elif feat_h != h_i or feat_w != w_i:
                raise ValueError(
                    "Inconsistent single-frame feature shapes across video: "
                    f"expected=({feat_h},{feat_w}) got=({h_i},{w_i})"
                )
            all_features[frame_idx] = frame_feat.T

    return all_features, feat_h, feat_w


@torch.no_grad()
def process_single_video(vos_params, model, handler, video_name, color_palette,
                         device, output_dir, patch_size, align):
    """Process a single video: propagate first-frame labels to all subsequent frames."""
    scale_size = _compute_scale_size(vos_params["frame_short_side"])
    align_mode = vos_params.get("frame_align_mode", "floor")
    video_dir = handler.get_video_dir(video_name)
    frame_list = read_frame_list(video_dir)
    if not frame_list:
        print(f"Warning: No frames found in {video_dir}, skipping.")
        return

    seg_path = handler.get_first_seg_path(video_name, frame_list)
    first_frame_out = handler.get_first_frame_output_name(frame_list)
    video_folder = os.path.join(output_dir, video_name)
    os.makedirs(video_folder, exist_ok=True)

    que = queue.Queue(vos_params["n_last_frames"])
    visualize_outputs = bool(vos_params.get("visualize_outputs", False))
    align_corners = bool(vos_params.get("align_corners", True))

    frames = []
    frame_oris = []
    for frame_path in frame_list:
        frame_tensor, ori_h, ori_w = read_frame(
            frame_path,
            scale_size=scale_size,
            align=align,
            align_mode=align_mode,
        )
        frames.append(frame_tensor)
        frame_oris.append((ori_h, ori_w))

    first_frame_segmentation = np.asarray(Image.open(seg_path))

    all_frame_feats, feat_h, feat_w = _extract_all_video_features(
        vos_params,
        model,
        frames,
        device,
        patch_size,
        video_name=video_name,
        first_frame_segmentation=first_frame_segmentation,
    )

    frame1 = frames[0]
    frame1_feat = all_frame_feats[0]
    first_seg, seg_ori = read_seg(
        seg_path, (feat_h, feat_w), frame_hw=(frame1.shape[1], frame1.shape[2]),
        scale_size=scale_size, align=align, align_mode=align_mode,
    )

    out_path = os.path.join(video_folder, first_frame_out)
    imwrite_indexed(out_path, seg_ori, color_palette)
    if visualize_outputs:
        _save_debug_exports(
            video_folder,
            os.path.splitext(first_frame_out)[0],
            frame_list[0],
            frame1,
            np.array(Image.fromarray(seg_ori).resize((frame1.shape[2], frame1.shape[1]), 0), dtype=np.uint8),
            color_palette,
        )

    mask_neighborhood = {}

    progress = tqdm(total=max(len(frame_list) - 1, 0), ascii=True, desc=video_name)
    try:
        for cnt in range(1, len(frame_list)):
            frame_tar = frames[cnt]
            tar_ori_h, tar_ori_w = frame_oris[cnt]

            queue_items = list(que.queue)
            used_frame_feats = [frame1_feat] + [pair[0] for pair in queue_items]
            used_segs = [first_seg] + [pair[1] for pair in queue_items]

            frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(
                vos_params, model, frame_tar, used_frame_feats, used_segs,
                device, mask_neighborhood, patch_size,
                precomputed_feat_tar=all_frame_feats[cnt],
            )

            if que.qsize() == vos_params["n_last_frames"]:
                que.get()
            seg_local = copy.deepcopy(frame_tar_avg)
            que.put([feat_tar, seg_local])

            frame_tar_seg = _scores_to_label_map(
                frame_tar_avg,
                (frame_tar.shape[1], frame_tar.shape[2]),
                align_corners=align_corners,
            )

            frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((tar_ori_w, tar_ori_h), 0))
            frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg", ".png")
            out_path = os.path.join(video_folder, frame_nm)
            imwrite_indexed(out_path, frame_tar_seg, color_palette)
            if visualize_outputs:
                _save_visualization(
                    frame_list[cnt],
                    frame_tar_seg,
                    os.path.join(video_folder, f"vis_{frame_nm}"),
                    color_palette,
                )
                _save_debug_exports(
                    video_folder,
                    os.path.splitext(frame_nm)[0],
                    frame_list[cnt],
                    frame_tar,
                    np.array(Image.fromarray(frame_tar_seg).resize((frame_tar.shape[2], frame_tar.shape[1]), 0), dtype=np.uint8),
                    color_palette,
                )
            progress.update(1)
    finally:
        progress.close()

    if str(device).startswith("cuda"):
        pass


@torch.no_grad()
def evaluate_vos(model, cfg, device, cpu_fallback_model=None, run_official_eval=True):
    """Run VOS evaluation for all videos in the configured dataset."""
    import concurrent.futures

    data_cfg = cfg["data"]
    vos_params = copy.deepcopy(cfg["vos"])
    vos_params["experiment_seed"] = int(cfg["experiment"].get("seed", 0))
    output_dir = cfg["experiment"]["output_dir"]
    patch_size = int(cfg["model"].get("patch_size", 0) or 0)
    vos_params["frame_short_side"] = resolve_protocol_frame_short_side(
        data_cfg,
        cfg["model"],
        vos_params,
    )
    align = _compute_align(patch_size, vos_params.get("frame_align"))
    frame_short_side = vos_params["frame_short_side"]
    frame_align_mode = vos_params.get("frame_align_mode", "floor")

    handler = get_handler(
        data_cfg["dataset"],
        data_cfg["data_path"],
        data_cfg["subset"],
        MODULE_DIR,
        video_list_file=data_cfg.get("video_list_file"),
    )
    video_list = handler.get_video_list()
    color_palette = load_color_palette(os.path.join(MODULE_DIR, "palette.txt"))
    cpu_fallback_videos = set(vos_params.get("cpu_fallback_videos", []))
    skip_completed_videos = bool(vos_params.get("skip_completed_videos", True))

    print(f"Evaluating {len(video_list)} videos on {data_cfg['dataset']} ({data_cfg['subset']})")
    print(f"Output directory: {output_dir}")
    if data_cfg.get("video_list_file"):
        print(f"Video list file: {data_cfg['video_list_file']}")
    print(f"Patch size: {patch_size}, frame alignment: {align}")
    print(f"Frame short side: {frame_short_side}, align mode: {frame_align_mode}")
    print(f"Affinity dtype: {vos_params.get('affinity_dtype', 'auto')}")
    print(f"Feature dtype: {vos_params.get('feature_dtype', 'auto')}")
    print(
        "Shift-and-stitch: "
        f"enabled={bool(vos_params.get('shift_and_stitch_enabled', False))}, "
        f"factor={int(vos_params.get('shift_and_stitch_factor', 1))}, "
        f"visualize_outputs={bool(vos_params.get('visualize_outputs', False))}, "
        f"align_corners={bool(vos_params.get('align_corners', True))}"
    )
    if cpu_fallback_videos:
        print(f"CPU fallback videos: {sorted(cpu_fallback_videos)}")

    test_workers = int(vos_params.get("num_workers", 1))
    print(f"Using multithreading for videos with max_workers={test_workers}")

    def evaluate_video(i, video_name):
        video_name = video_name.strip()
        video_dir = handler.get_video_dir(video_name)
        frame_list = read_frame_list(video_dir)
        video_folder = os.path.join(output_dir, video_name)
        if skip_completed_videos and frame_list and _video_is_completed(handler, video_folder, frame_list):
            print(f"[{i}/{len(video_list)}] Skipping completed video: {video_name}")
            return

        if video_name in cpu_fallback_videos:
            if cpu_fallback_model is None:
                raise RuntimeError(
                    f"Video {video_name} requires CPU fallback, but no cpu_fallback_model was provided"
                )
            run_model = cpu_fallback_model
            run_device = "cpu"
            run_vos_params = _build_cpu_video_params(vos_params)
            print(f"[{i}/{len(video_list)}] [{run_device}] Processing video: {video_name}")
        else:
            run_model = model
            run_device = device
            run_vos_params = vos_params
            print(f"[{i}/{len(video_list)}] [{run_device}] Processing video: {video_name}")

        process_single_video(
            run_vos_params,
            run_model,
            handler,
            video_name,
            color_palette,
            run_device,
            output_dir,
            patch_size,
            align,
        )

    if test_workers <= 1:
        for i, video_name in enumerate(video_list, 1):
            evaluate_video(i, video_name)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=test_workers) as executor:
            futures = [executor.submit(evaluate_video, i, video_name) for i, video_name in enumerate(video_list, 1)]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    if run_official_eval:
        print("Running official VOS evaluation...")
        handler.run_evaluation(output_dir)
