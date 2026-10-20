#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml


MODEL_PRESETS = {
    "vitb_384": {
        "vjepa_checkpoint": "workflows/pretrain/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt",
        "student_encoder_impl": "slot_jepa",
        "student_encoder_checkpoint_key": "ema_encoder",
        "student_encoder_patch_size": 16,
        "student_encoder_num_register_tokens": 0,
        "student_model_name": "vit_base",
        "student_crop_size": 384,
        "student_train_last_n_layers": 6,
        "teacher_encoder_impl": "dinov2_vitb14_reg",
        "teacher_encoder_checkpoint_key": None,
        "teacher_encoder_patch_size": 14,
        "teacher_encoder_num_register_tokens": 4,
        "teacher_model_name": "dinov2_vitb14_reg",
        "teacher_crop_size": 336,
        "pred_depth": 12,
        "pred_embed_dim": 192,
        "pred_num_heads": 3,
        "dtype": "bfloat16",
        "global_k": 16,
        "L_max": 24,
        "mask_temp": 0.05,
        "window_size": [2, 4],
        "slots_per_window": 4,
        "slot_feature_dim": 192,
        "inverse_depth": 8,
        "inverse_heads": 4,
        "inverse_token_dim": 32,
    },
}

DINO_ENCODER_IMPLS = {"dinov2", "dinov2_vitb14", "dinov2_vitb14_reg"}
VJEPA_ENCODER_IMPLS = {"slot_jepa", "vjepa", "vjepa2"}
ENCODER_IMPL_CHOICES = sorted(DINO_ENCODER_IMPLS | VJEPA_ENCODER_IMPLS)


def _encoder_defaults(encoder_impl, preset, role):
    if encoder_impl in DINO_ENCODER_IMPLS:
        model_name = "dinov2_vitb14_reg" if encoder_impl == "dinov2_vitb14_reg" else "dinov2_vitb14"
        return {
            "model_name": model_name,
            "checkpoint_key": None,
            "patch_size": 14,
            "num_register_tokens": 4 if model_name.endswith("_reg") else 0,
            "crop_size": 336,
        }
    return {
        "model_name": "vit_base",
        "checkpoint_key": "ema_encoder",
        "patch_size": 16,
        "num_register_tokens": 0,
        "crop_size": 384,
    }


def _resolve_optional_checkpoint(path_value, repo_root):
    if path_value is None:
        return None
    path_arg = Path(path_value).expanduser()
    return path_arg.resolve() if path_arg.is_absolute() else (repo_root / path_arg).resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and launch the shared pretrain + DROID downstream camera-base pipeline."
    )
    parser.add_argument("--config", default=None, help="Existing pipeline_config.yaml. If omitted, a new one is generated from CLI args.")
    parser.add_argument(
        "--stage",
        choices=["prepare", "pretrain", "downstream", "all"],
        default=None,
        help="prepare: only write configs; pretrain/downstream: launch one stage; all: run both sequentially.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Backward-compatible alias for --stage prepare.")
    parser.add_argument("--launch", action="store_true", help="Backward-compatible alias for --stage all.")
    parser.add_argument("--skip-data-check", action="store_true", help="Allow prepare to skip downstream dataset scanning. CSV will be materialized when downstream training starts.")
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS.keys()), default="vitb_384")
    parser.add_argument("--output-dir", default="outputs/slot_ac_camera_base_pipeline_vitb_384")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"])
    parser.add_argument("--debugmode", action="store_true")

    parser.add_argument("--frames-per-clip", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--camera-views", nargs="+", default=["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"])
    parser.add_argument("--multiview-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--encoder-impl", choices=ENCODER_IMPL_CHOICES, default=None, help="Deprecated alias for --teacher-encoder-impl.")
    parser.add_argument("--encoder-model-name", default=None)
    parser.add_argument("--encoder-checkpoint-key", default=None)
    parser.add_argument("--encoder-patch-size", type=int, default=None)
    parser.add_argument("--encoder-num-register-tokens", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--student-encoder-impl", choices=ENCODER_IMPL_CHOICES, default=None)
    parser.add_argument("--teacher-encoder-impl", choices=ENCODER_IMPL_CHOICES, default=None)
    parser.add_argument("--student-model-name", default=None)
    parser.add_argument("--teacher-model-name", default=None)
    parser.add_argument("--student-encoder-checkpoint-key", default=None)
    parser.add_argument("--teacher-encoder-checkpoint-key", default=None)
    parser.add_argument("--student-encoder-patch-size", type=int, default=None)
    parser.add_argument("--teacher-encoder-patch-size", type=int, default=None)
    parser.add_argument("--student-encoder-num-register-tokens", type=int, default=None)
    parser.add_argument("--teacher-encoder-num-register-tokens", type=int, default=None)
    parser.add_argument("--student-checkpoint", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--student-crop-size", type=int, default=None)
    parser.add_argument("--teacher-crop-size", type=int, default=None)
    parser.add_argument("--student-train-last-n-layers", type=int, default=None)

    parser.add_argument("--recon-loss-weight", type=float, default=0.1)
    parser.add_argument("--transport-loss-weight", type=float, default=3.0)
    parser.add_argument("--mask-l1-loss-weight", type=float, default=3.0)
    parser.add_argument("--target-num-frames", type=int, default=3)

    parser.add_argument("--vjepa-checkpoint", default=None)
    parser.add_argument("--slot-pretrain-checkpoint", default=None, help="Optional override for downstream init checkpoint. Defaults to the generated pretrain latest.pt.")

    parser.add_argument("--pretrain-dataset-root", default="data/Kinetics-400")
    parser.add_argument("--pretrain-valid-list", default="valid_video_names.txt")
    parser.add_argument("--pretrain-use-lmdb", type=bool, default=True)
    parser.add_argument("--pretrain-lmdb-path", default="data/Kinetics-400/kinetics_train.lmdb")
    parser.add_argument("--pretrain-batch-size", type=int, default=2, help="Number of videos per pretraining iteration.")
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--pretrain-prefetch-factor", type=int, default=2)

    parser.add_argument("--downstream-dataset-root", default="data/droid_frame_camera_base_multiview")
    parser.add_argument("--downstream-batch-size", type=int, default=4)
    parser.add_argument("--downstream-epochs", type=int, default=100)
    parser.add_argument("--downstream-ipe", type=int, default=300)
    parser.add_argument(
        "--downstream-num-workers",
        type=int,
        default=None,
        help="Optional downstream dataloader workers per rank. Direct V-JEPA downstream defaults to min(num_workers, 2).",
    )
    parser.add_argument(
        "--downstream-trainer",
        choices=["slot-ac", "vjepa-droid", "vjepa-query-predictor"],
        default="slot-ac",
        help="Choose which downstream trainer writes to downstream.yaml.",
    )
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument(
        "--share-multiview-obj-latents",
        action="store_true",
        help="Share one jointly extracted obj latent set across all downstream camera views.",
    )
    parser.add_argument(
        "--disable-multiview-shared-obj-latents",
        action="store_true",
        help="Deprecated compatibility flag. Obj latents are not shared by default.",
    )
    return parser.parse_args()


def resolve_stage(args):
    if args.stage is not None:
        return args.stage
    if args.prepare_only:
        return "prepare"
    if args.launch:
        return "all"
    return "prepare"


def build_episode_csv(dataset_root: Path, csv_path: Path):
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    episode_dirs = sorted(
        str(path)
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "labels.json").exists()
    )
    if not episode_dirs:
        episode_dirs = sorted({str(path.parent) for path in dataset_root.rglob("labels.json")})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter=" ")
        for episode_dir in episode_dirs:
            writer.writerow([episode_dir])
    return len(episode_dirs)


def build_pipeline_config(args, repo_root: Path):
    preset = MODEL_PRESETS[args.model_preset]
    student_encoder_impl = args.student_encoder_impl or preset["student_encoder_impl"]
    teacher_encoder_impl = args.teacher_encoder_impl or args.encoder_impl or preset["teacher_encoder_impl"]
    student_defaults = _encoder_defaults(student_encoder_impl, preset, "student")
    teacher_defaults = _encoder_defaults(teacher_encoder_impl, preset, "teacher")
    student_encoder_checkpoint_key = (
        args.student_encoder_checkpoint_key
        if args.student_encoder_checkpoint_key is not None
        else student_defaults["checkpoint_key"]
    )
    teacher_encoder_checkpoint_key = (
        args.teacher_encoder_checkpoint_key
        if args.teacher_encoder_checkpoint_key is not None
        else (args.encoder_checkpoint_key if args.encoder_checkpoint_key is not None else teacher_defaults["checkpoint_key"])
    )
    student_encoder_patch_size = (
        args.student_encoder_patch_size
        if args.student_encoder_patch_size is not None
        else student_defaults["patch_size"]
    )
    teacher_encoder_patch_size = (
        args.teacher_encoder_patch_size
        if args.teacher_encoder_patch_size is not None
        else (args.encoder_patch_size if args.encoder_patch_size is not None else teacher_defaults["patch_size"])
    )
    student_model_name = args.student_model_name or student_defaults["model_name"]
    teacher_model_name = args.teacher_model_name or args.encoder_model_name or teacher_defaults["model_name"]
    student_encoder_num_register_tokens = (
        args.student_encoder_num_register_tokens
        if args.student_encoder_num_register_tokens is not None
        else student_defaults["num_register_tokens"]
    )
    teacher_encoder_num_register_tokens = (
        args.teacher_encoder_num_register_tokens
        if args.teacher_encoder_num_register_tokens is not None
        else (
            args.encoder_num_register_tokens
            if args.encoder_num_register_tokens is not None
            else teacher_defaults["num_register_tokens"]
        )
    )
    student_crop_size = (
        args.student_crop_size
        if args.student_crop_size is not None
        else (args.crop_size if args.crop_size is not None else student_defaults["crop_size"])
    )
    student_train_last_n_layers = (
        args.student_train_last_n_layers
        if args.student_train_last_n_layers is not None
        else preset["student_train_last_n_layers"]
    )
    teacher_crop_size = (
        args.teacher_crop_size
        if args.teacher_crop_size is not None
        else teacher_defaults["crop_size"]
    )
    if student_crop_size // student_encoder_patch_size != teacher_crop_size // teacher_encoder_patch_size:
        raise ValueError(
            "Student and teacher patch grids must match: "
            f"{student_crop_size}//{student_encoder_patch_size} != {teacher_crop_size}//{teacher_encoder_patch_size}"
        )
    output_dir_arg = Path(args.output_dir).expanduser()
    output_dir = output_dir_arg.resolve() if output_dir_arg.is_absolute() else (repo_root / output_dir_arg).resolve()
    default_vjepa_checkpoint = str(_resolve_optional_checkpoint(args.vjepa_checkpoint or preset["vjepa_checkpoint"], repo_root))
    student_checkpoint = _resolve_optional_checkpoint(args.student_checkpoint, repo_root)
    teacher_checkpoint = _resolve_optional_checkpoint(args.teacher_checkpoint, repo_root)
    if student_checkpoint is None and student_encoder_impl in VJEPA_ENCODER_IMPLS:
        student_checkpoint = Path(default_vjepa_checkpoint)
    if teacher_checkpoint is None and teacher_encoder_impl in VJEPA_ENCODER_IMPLS:
        teacher_checkpoint = Path(default_vjepa_checkpoint)
    downstream_dataset_root_arg = Path(args.downstream_dataset_root).expanduser()
    downstream_dataset_root = (
        downstream_dataset_root_arg.resolve()
        if downstream_dataset_root_arg.is_absolute()
        else (repo_root / downstream_dataset_root_arg).resolve()
    )
    pretrain_latest_checkpoint = output_dir / "pretrain" / "run" / "latest.pt"
    if args.slot_pretrain_checkpoint:
        downstream_checkpoint_arg = Path(args.slot_pretrain_checkpoint).expanduser()
        downstream_checkpoint = (
            downstream_checkpoint_arg.resolve()
            if downstream_checkpoint_arg.is_absolute()
            else (repo_root / downstream_checkpoint_arg).resolve()
        )
    else:
        downstream_checkpoint = pretrain_latest_checkpoint
    return {
        "pipeline_version": 1,
        "model_preset": args.model_preset,
        "devices": list(args.devices),
        "debugmode": bool(args.debugmode),
        "paths": {
            "repo_root": str(repo_root),
            "output_dir": str(output_dir),
            "vjepa_checkpoint": default_vjepa_checkpoint,
            "student_checkpoint": None if student_checkpoint is None else str(student_checkpoint),
            "teacher_checkpoint": None if teacher_checkpoint is None else str(teacher_checkpoint),
            "pretrain_config": str(output_dir / "pretrain.yaml"),
            "downstream_config": str(output_dir / "downstream.yaml"),
            "pipeline_config": str(output_dir / "pipeline_config.yaml"),
            "downstream_csv": str(output_dir / "droid_frame_camera_base_train.csv"),
            "pretrain_latest_checkpoint": str(pretrain_latest_checkpoint),
            "downstream_slot_pretrain_checkpoint": str(downstream_checkpoint),
        },
        "shared": {
            "frames_per_clip": args.frames_per_clip,
            "num_workers": args.num_workers,
            "camera_views": list(args.camera_views),
            "multiview_training": bool(args.multiview_training),
            "recon_loss_weight": args.recon_loss_weight,
            "transport_loss_weight": args.transport_loss_weight,
            "mask_l1_loss_weight": args.mask_l1_loss_weight,
            "target_num_frames": args.target_num_frames,
            "model": {
                "encoder_impl": student_encoder_impl,
                "model_name": student_model_name,
                "encoder_checkpoint_key": student_encoder_checkpoint_key,
                "encoder_patch_size": student_encoder_patch_size,
                "encoder_num_register_tokens": student_encoder_num_register_tokens,
                "crop_size": student_crop_size,
                "student_encoder_impl": student_encoder_impl,
                "student_model_name": student_model_name,
                "student_encoder_checkpoint_key": student_encoder_checkpoint_key,
                "student_encoder_patch_size": student_encoder_patch_size,
                "student_encoder_num_register_tokens": student_encoder_num_register_tokens,
                "student_crop_size": student_crop_size,
                "student_train_last_n_layers": student_train_last_n_layers,
                "teacher_encoder_impl": teacher_encoder_impl,
                "teacher_model_name": teacher_model_name,
                "teacher_encoder_checkpoint_key": teacher_encoder_checkpoint_key,
                "teacher_encoder_patch_size": teacher_encoder_patch_size,
                "teacher_encoder_num_register_tokens": teacher_encoder_num_register_tokens,
                "teacher_crop_size": teacher_crop_size,
                "pred_depth": preset["pred_depth"],
                "pred_embed_dim": preset["pred_embed_dim"],
                "pred_num_heads": preset["pred_num_heads"],
                "dtype": preset["dtype"],
                "global_k": preset["global_k"],
                "L_max": preset["L_max"],
                "mask_temp": preset["mask_temp"],
                "window_size": list(preset["window_size"]),
                "slots_per_window": preset["slots_per_window"],
                "slot_feature_dim": preset["slot_feature_dim"],
                "inverse_depth": preset["inverse_depth"],
                "inverse_heads": preset["inverse_heads"],
                "inverse_token_dim": preset["inverse_token_dim"],
                "share_obj_latents_across_views": bool(args.share_multiview_obj_latents)
                and not bool(args.disable_multiview_shared_obj_latents),
            },
        },
        "pretrain": {
            "dataset_root": str(Path(args.pretrain_dataset_root).expanduser().resolve()),
            "valid_list": args.pretrain_valid_list,
            "use_lmdb": bool(args.pretrain_use_lmdb),
            "lmdb_path": args.pretrain_lmdb_path,
            "batch_size": args.pretrain_batch_size,
            "epochs": args.pretrain_epochs,
            "lr": args.pretrain_lr,
            "prefetch_factor": args.pretrain_prefetch_factor,
        },
        "downstream": {
            "dataset_root": str(downstream_dataset_root),
            "trainer": args.downstream_trainer,
            "batch_size": args.downstream_batch_size,
            "epochs": args.downstream_epochs,
            "ipe": args.downstream_ipe,
            "num_workers": args.downstream_num_workers,
            "overfit_one_batch": bool(args.overfit_one_batch),
        },
    }


def build_pretrain_config(pipeline_config):
    paths = pipeline_config["paths"]
    shared = pipeline_config["shared"]
    model_shared = shared["model"]
    pretrain = pipeline_config["pretrain"]
    frames_per_clip = int(shared["frames_per_clip"])
    return {
        "app": "slot_jepa",
        "cpus_per_task": max(4, int(shared["num_workers"])),
        "folder": str(Path(paths["output_dir"]) / "pretrain" / "run"),
        "mem_per_gpu": "24G",
        "nodes": 1,
        "tasks_per_node": max(1, len(pipeline_config["devices"])),
        "data": {
            "batch_size": int(pretrain["batch_size"]),
            "crop_size": int(model_shared["crop_size"]),
            "dataset_root": pretrain["dataset_root"],
            "dataset_fpcs": [frames_per_clip],
            "valid_list": pretrain["valid_list"],
            "use_lmdb": bool(pretrain["use_lmdb"]),
            "lmdb_path": pretrain["lmdb_path"],
            "num_workers": int(shared["num_workers"]),
            "prefetch_factor": int(pretrain["prefetch_factor"]),
            "pin_mem": True,
            "persistent_workers": True,
        },
        "data_aug": {
            "auto_augment": False,
            "horizontal_flip": False,
            "motion_shift": False,
            "random_resize_aspect_ratio": [0.75, 1.35],
            "random_resize_scale": [1.777, 1.777],
            "reprob": 0.0,
        },
        "loss": {
            "recon_loss_type": "l1",
            "recon_loss_weight": float(shared["recon_loss_weight"]),
            "transport_loss_weight": float(shared["transport_loss_weight"]),
            "mask_l1_loss_weight": float(shared["mask_l1_loss_weight"]),
            "sinkhorn_iters": 5,
            "sinkhorn_entropy_reg": 0.05,
            "target_num_frames": int(shared["target_num_frames"]),
        },
        "meta": {
            "dtype": model_shared["dtype"],
            "resume_checkpoint": None,
            "pretrain_checkpoint": paths.get("teacher_checkpoint"),
            "student_pretrain_checkpoint": paths.get("student_checkpoint"),
            "teacher_pretrain_checkpoint": paths.get("teacher_checkpoint"),
            "save_every_freq": 5,
        },
        "model": {
            "encoder_impl": model_shared.get("encoder_impl", "slot_jepa"),
            "model_name": model_shared["model_name"],
            "encoder_checkpoint_key": model_shared.get("encoder_checkpoint_key", "ema_encoder"),
            "encoder_patch_size": int(model_shared.get("encoder_patch_size", 16)),
            "encoder_num_register_tokens": int(model_shared.get("encoder_num_register_tokens", 0)),
            "student_encoder_impl": model_shared.get("student_encoder_impl", model_shared.get("encoder_impl", "slot_jepa")),
            "student_model_name": model_shared.get("student_model_name", model_shared["model_name"]),
            "student_encoder_checkpoint_key": model_shared.get(
                "student_encoder_checkpoint_key", model_shared.get("encoder_checkpoint_key", "ema_encoder")
            ),
            "student_encoder_patch_size": int(
                model_shared.get("student_encoder_patch_size", model_shared.get("encoder_patch_size", 16))
            ),
            "student_encoder_num_register_tokens": int(
                model_shared.get("student_encoder_num_register_tokens", model_shared.get("encoder_num_register_tokens", 0))
            ),
            "student_crop_size": int(model_shared.get("student_crop_size", model_shared["crop_size"])),
            "student_train_last_n_layers": int(model_shared.get("student_train_last_n_layers", 6)),
            "teacher_encoder_impl": model_shared.get("teacher_encoder_impl", "dinov2_vitb14_reg"),
            "teacher_model_name": model_shared.get("teacher_model_name", "dinov2_vitb14_reg"),
            "teacher_encoder_checkpoint_key": model_shared.get("teacher_encoder_checkpoint_key"),
            "teacher_encoder_patch_size": int(model_shared.get("teacher_encoder_patch_size", 14)),
            "teacher_encoder_num_register_tokens": int(model_shared.get("teacher_encoder_num_register_tokens", 4)),
            "teacher_crop_size": int(model_shared.get("teacher_crop_size", 336)),
            "slot_feature_dim": int(model_shared["slot_feature_dim"]),
            "pred_depth": int(model_shared["pred_depth"]),
            "pred_embed_dim": int(model_shared["pred_embed_dim"]),
            "pred_num_heads": int(model_shared["pred_num_heads"]),
            "global_k": int(model_shared["global_k"]),
            "L_max": int(model_shared["L_max"]),
            "mask_temp": float(model_shared["mask_temp"]),
            "window_size": list(model_shared["window_size"]),
            "slots_per_window": int(model_shared["slots_per_window"]),
            "num_obj_tokens": 1,
            "concat_mask_input": True,
            "use_extrinsics": True,
            "inverse_depth": int(model_shared["inverse_depth"]),
            "inverse_heads": int(model_shared["inverse_heads"]),
            "inverse_token_dim": int(model_shared["inverse_token_dim"]),
            "use_activation_checkpointing": False,
            "use_decoder_checkpoint": False,
            "correspondence_jitter_std": 0.05,
            "correspondence_radius": 2,
        },
        "optimization": {
            "epochs": int(pretrain["epochs"]),
            "lr": float(pretrain["lr"]),
        },
    }


def build_slot_ac_downstream_config(pipeline_config):
    paths = pipeline_config["paths"]
    shared = pipeline_config["shared"]
    model_shared = shared["model"]
    downstream = pipeline_config["downstream"]
    overfit_one_batch = bool(downstream["overfit_one_batch"])
    batch_size = 4 if overfit_one_batch else int(downstream["batch_size"])
    ipe = 1000 if overfit_one_batch else int(downstream["ipe"])
    downstream_num_workers = downstream.get("num_workers")
    num_workers = int(shared["num_workers"] if downstream_num_workers is None else downstream_num_workers)
    frames_per_clip = int(shared["frames_per_clip"])
    return {
        "app": "slot_ac",
        "cpus_per_task": max(4, int(shared["num_workers"])),
        "folder": str(Path(paths["output_dir"]) / "downstream" / "run"),
        "mem_per_gpu": "24G",
        "nodes": 1,
        "tasks_per_node": max(1, len(pipeline_config["devices"])),
        "data": {
            "batch_size": batch_size,
            "camera_views": list(shared["camera_views"]),
            "camera_frame": True,
            "dataset_format": "exported_frames",
            "crop_size": int(model_shared["crop_size"]),
            "datasets": [paths["downstream_csv"]],
            "dataset_fpcs": [frames_per_clip],
            "fps": 1,
            "multiview_training": bool(shared["multiview_training"]),
            "num_workers": num_workers,
            "patch_size": int(model_shared.get("encoder_patch_size", 16)),
            "pin_mem": True,
            "persistent_workers": True,
            "stereo_view": False,
            "tubelet_size": 2,
        },
        "data_aug": {
            "auto_augment": False,
            "horizontal_flip": False,
            "motion_shift": False,
            "random_resize_aspect_ratio": [0.75, 1.35],
            "random_resize_scale": [1.777, 1.777],
            "reprob": 0.0,
        },
        "loss": {
            "auto_steps": 2,
            "recon_loss_type": "l1",
            "recon_loss_weight": float(shared["recon_loss_weight"]),
            "transport_loss_weight": float(shared["transport_loss_weight"]),
            "mask_l1_loss_weight": float(shared["mask_l1_loss_weight"]),
            "sinkhorn_iters": 5,
            "sinkhorn_entropy_reg": 0.05,
            "sampling_strategies": ["extrapolation"],
            "target_num_frames": int(shared["target_num_frames"]),
        },
        "meta": {
            "dtype": model_shared["dtype"],
            "eval_freq": 100,
            "overfit_one_batch": overfit_one_batch,
            "resume_checkpoint": None,
            "pretrain_checkpoint": paths.get("teacher_checkpoint"),
            "vjepa_checkpoint_path": paths.get("student_checkpoint"),
            "student_pretrain_checkpoint": paths.get("student_checkpoint"),
            "teacher_pretrain_checkpoint": paths.get("teacher_checkpoint"),
            "slot_pretrain_checkpoint": paths["downstream_slot_pretrain_checkpoint"],
            "load_predictor_shared": True,
            "save_every_freq": 25,
            "seed": 239,
            "use_sdpa": True,
            "tb_log_freq": 10,
            "tb_vis_freq": 100,
            "tb_vis_frames": frames_per_clip,
        },
        "model": {
            "encoder_impl": model_shared.get("encoder_impl", "slot_jepa"),
            "model_name": model_shared["model_name"],
            "encoder_checkpoint_key": model_shared.get("encoder_checkpoint_key", "ema_encoder"),
            "encoder_patch_size": int(model_shared.get("encoder_patch_size", 16)),
            "encoder_num_register_tokens": int(model_shared.get("encoder_num_register_tokens", 0)),
            "student_encoder_impl": model_shared.get("student_encoder_impl", model_shared.get("encoder_impl", "slot_jepa")),
            "student_model_name": model_shared.get("student_model_name", model_shared["model_name"]),
            "student_encoder_checkpoint_key": model_shared.get(
                "student_encoder_checkpoint_key", model_shared.get("encoder_checkpoint_key", "ema_encoder")
            ),
            "student_encoder_patch_size": int(
                model_shared.get("student_encoder_patch_size", model_shared.get("encoder_patch_size", 16))
            ),
            "student_encoder_num_register_tokens": int(
                model_shared.get("student_encoder_num_register_tokens", model_shared.get("encoder_num_register_tokens", 0))
            ),
            "student_crop_size": int(model_shared.get("student_crop_size", model_shared["crop_size"])),
            "student_train_last_n_layers": int(model_shared.get("student_train_last_n_layers", 6)),
            "teacher_encoder_impl": model_shared.get("teacher_encoder_impl", "dinov2_vitb14_reg"),
            "teacher_model_name": model_shared.get("teacher_model_name", "dinov2_vitb14_reg"),
            "teacher_encoder_checkpoint_key": model_shared.get("teacher_encoder_checkpoint_key"),
            "teacher_encoder_patch_size": int(model_shared.get("teacher_encoder_patch_size", 14)),
            "teacher_encoder_num_register_tokens": int(model_shared.get("teacher_encoder_num_register_tokens", 4)),
            "teacher_crop_size": int(model_shared.get("teacher_crop_size", 336)),
            "pred_depth": int(model_shared["pred_depth"]),
            "pred_embed_dim": int(model_shared["pred_embed_dim"]),
            "pred_num_heads": int(model_shared["pred_num_heads"]),
            "pred_is_frame_causal": True,
            "uniform_power": True,
            "use_activation_checkpointing": False,
            "use_extrinsics": True,
            "use_rope": True,
            "embed_dim": 768,
            "global_k": int(model_shared["global_k"]),
            "L_max": int(model_shared["L_max"]),
            "mask_temp": float(model_shared["mask_temp"]),
            "window_size": list(model_shared["window_size"]),
            "slots_per_window": int(model_shared["slots_per_window"]),
            "slot_feature_dim": int(model_shared["slot_feature_dim"]),
            "state_dim": 7,
            "extrinsics_dim": 6,
            "num_obj_tokens": 1,
            "concat_mask_input": True,
            "use_wrist_warp_concat": True,
            "warp_hfov_deg": 60.0,
            "warp_extrinsics_format": "c2w",
            "warp_rotation_mode": "euler_xyz",
            "warp_depth_bins": [0.15, 0.25, 0.4, 0.7, 1.2, 2.0],
            "freeze_extractor": True,
            "freeze_student": True,
            "use_decoder_checkpoint": False,
            "inverse_depth": int(model_shared["inverse_depth"]),
            "inverse_heads": int(model_shared["inverse_heads"]),
            "inverse_token_dim": int(model_shared["inverse_token_dim"]),
            "share_obj_latents_across_views": bool(model_shared.get("share_obj_latents_across_views", False)),
        },
        "optimization": {
            "anneal": 15,
            "epochs": 1 if overfit_one_batch else int(downstream["epochs"]),
            "final_lr": 0.0,
            "final_weight_decay": 0.01,
            "ipe": ipe,
            "lr": 1e-3,
            "start_lr": 1e-4,
            "warmup": 15,
            "weight_decay": 0.01,
            "enc_lr_scale": 1.0,
        },
    }


def _build_direct_droid_data_config(pipeline_config, batch_size, frames_per_clip):
    paths = pipeline_config["paths"]
    shared = pipeline_config["shared"]
    model_shared = shared["model"]
    downstream = pipeline_config["downstream"]
    downstream_num_workers = downstream.get("num_workers")
    num_workers = min(int(shared["num_workers"]), 2) if downstream_num_workers is None else int(downstream_num_workers)
    return {
        "batch_size": batch_size,
        "camera_views": list(shared["camera_views"]),
        "camera_frame": True,
        "dataset_format": "exported_frames",
        "crop_size": int(model_shared["crop_size"]),
        "datasets": [paths["downstream_csv"]],
        "dataset_fpcs": [frames_per_clip],
        "fps": 1,
        "multiview_training": bool(shared["multiview_training"]),
        "num_workers": num_workers,
        "patch_size": int(model_shared.get("encoder_patch_size", 16)),
        "pin_mem": True,
        "persistent_workers": False,
        "stereo_view": False,
        "tubelet_size": 2,
    }


def _build_direct_droid_meta_config(pipeline_config, frames_per_clip, overfit_one_batch):
    paths = pipeline_config["paths"]
    model_shared = pipeline_config["shared"]["model"]
    return {
        "dtype": model_shared["dtype"],
        "eval_freq": 100,
        "overfit_one_batch": overfit_one_batch,
        "resume_checkpoint": None,
        "pretrain_checkpoint": paths.get("student_checkpoint") or paths.get("vjepa_checkpoint"),
        "context_encoder_key": model_shared.get("encoder_checkpoint_key", "ema_encoder"),
        "target_encoder_key": model_shared.get("encoder_checkpoint_key", "ema_encoder"),
        "load_encoder": True,
        "load_predictor": False,
        "save_every_freq": 25,
        "seed": 239,
        "use_sdpa": True,
        "tb_log_freq": 10,
        "tb_vis_freq": 100,
        "tb_vis_frames": frames_per_clip,
    }


def _build_single_route_direct_predictor_config(model_shared):
    return {
        "pred_depth": int(model_shared["pred_depth"]),
        "pred_embed_dim": 384,
        "pred_num_heads": int(model_shared["pred_num_heads"]),
        "separate_wrist_predictor": False,
    }


def build_vjepa_droid_config(pipeline_config):
    paths = pipeline_config["paths"]
    shared = pipeline_config["shared"]
    model_shared = shared["model"]
    downstream = pipeline_config["downstream"]
    overfit_one_batch = bool(downstream["overfit_one_batch"])
    batch_size = 4 if overfit_one_batch else int(downstream["batch_size"])
    ipe = 1000 if overfit_one_batch else int(downstream["ipe"])
    frames_per_clip = int(shared["frames_per_clip"])
    return {
        "app": "vjepa_droid",
        "cpus_per_task": max(4, int(shared["num_workers"])),
        "folder": str(Path(paths["output_dir"]) / "vjepa_droid" / "run"),
        "mem_per_gpu": "24G",
        "nodes": 1,
        "tasks_per_node": max(1, len(pipeline_config["devices"])),
        "data": _build_direct_droid_data_config(pipeline_config, batch_size, frames_per_clip),
        "data_aug": {
            "auto_augment": False,
            "horizontal_flip": False,
            "motion_shift": False,
            "random_resize_aspect_ratio": [0.75, 1.35],
            "random_resize_scale": [1.777, 1.777],
            "reprob": 0.0,
        },
        "loss": {
            "auto_steps": 2,
            "loss_exp": 1.0,
            "normalize_reps": False,
        },
        "meta": _build_direct_droid_meta_config(pipeline_config, frames_per_clip, overfit_one_batch),
        "model": {
            "encoder_impl": model_shared.get("encoder_impl", "slot_jepa"),
            "model_name": model_shared["model_name"],
            **_build_single_route_direct_predictor_config(model_shared),
            "pred_is_frame_causal": True,
            "uniform_power": True,
            "use_activation_checkpointing": False,
            "use_extrinsics": True,
            "use_rope": True,
            "use_silu": False,
            "use_pred_silu": False,
            "wide_silu": True,
        },
        "optimization": {
            "anneal": 15,
            "epochs": 1 if overfit_one_batch else int(downstream["epochs"]),
            "final_lr": 0.0,
            "final_weight_decay": 0.01,
            "ipe": ipe,
            "lr": 1e-3,
            "start_lr": 1e-4,
            "warmup": 15,
            "weight_decay": 0.01,
            "enc_lr_scale": 1.0,
        },
    }


def build_vjepa_query_predictor_config(pipeline_config):
    paths = pipeline_config["paths"]
    shared = pipeline_config["shared"]
    model_shared = shared["model"]
    downstream = pipeline_config["downstream"]
    overfit_one_batch = bool(downstream["overfit_one_batch"])
    batch_size = 4 if overfit_one_batch else int(downstream["batch_size"])
    ipe = 1000 if overfit_one_batch else int(downstream["ipe"])
    frames_per_clip = int(shared["frames_per_clip"])
    return {
        "app": "vjepa_query_predictor",
        "cpus_per_task": max(4, int(shared["num_workers"])),
        "folder": str(Path(paths["output_dir"]) / "vjepa-query-predictor" / "run"),
        "mem_per_gpu": "24G",
        "nodes": 1,
        "tasks_per_node": max(1, len(pipeline_config["devices"])),
        "data": _build_direct_droid_data_config(pipeline_config, batch_size, frames_per_clip),
        "data_aug": {
            "auto_augment": False,
            "horizontal_flip": False,
            "motion_shift": False,
            "random_resize_aspect_ratio": [0.75, 1.35],
            "random_resize_scale": [1.777, 1.777],
            "reprob": 0.0,
        },
        "loss": {
            "normalize_reps": False,
            "target_num_frames": int(shared["target_num_frames"]),
        },
        "meta": _build_direct_droid_meta_config(pipeline_config, frames_per_clip, overfit_one_batch),
        "model": {
            "encoder_impl": model_shared.get("encoder_impl", "slot_jepa"),
            "model_name": model_shared["model_name"],
            **_build_single_route_direct_predictor_config(model_shared),
            "uniform_power": True,
            "use_activation_checkpointing": False,
            "use_extrinsics": True,
            "use_rope": True,
            "use_silu": False,
            "wide_silu": True,
            "state_dim": 7,
            "extrinsics_dim": 6,
            "use_warp_concat": True,
            "warp_hfov_deg": 60.0,
            "warp_extrinsics_format": "c2w",
            "warp_rotation_mode": "euler_xyz",
            "warp_depth_bins": [0.15, 0.25, 0.4, 0.7, 1.2, 2.0],
        },
        "optimization": {
            "epochs": 1 if overfit_one_batch else int(downstream["epochs"]),
            "ipe": ipe,
            "lr": 1e-3,
            "weight_decay": 0.01,
        },
    }


def build_downstream_config(pipeline_config):
    trainer = pipeline_config.get("downstream", {}).get("trainer", "slot-ac")
    if trainer == "slot-ac":
        return build_slot_ac_downstream_config(pipeline_config)
    if trainer == "vjepa-droid":
        return build_vjepa_droid_config(pipeline_config)
    if trainer == "vjepa-query-predictor":
        return build_vjepa_query_predictor_config(pipeline_config)
    raise ValueError(f"Unsupported downstream trainer: {trainer}")


def load_pipeline_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as fin:
        return yaml.safe_load(fin)


def materialize_configs(pipeline_config, skip_data_check=False):
    paths = pipeline_config["paths"]
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    downstream_dataset_root = Path(pipeline_config["downstream"]["dataset_root"])
    csv_path = Path(paths["downstream_csv"])
    if skip_data_check:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        episode_count = None
    else:
        episode_count = build_episode_csv(downstream_dataset_root, csv_path)

    pretrain_config = build_pretrain_config(pipeline_config)
    downstream_config = build_downstream_config(pipeline_config)

    for config_path_key, config_obj in (
        ("pretrain_config", pretrain_config),
        ("downstream_config", downstream_config),
    ):
        config_path = Path(paths[config_path_key])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as fout:
            yaml.safe_dump(config_obj, fout, sort_keys=False)

    pipeline_path = Path(paths["pipeline_config"])
    with pipeline_path.open("w", encoding="utf-8") as fout:
        yaml.safe_dump(pipeline_config, fout, sort_keys=False)

    return {
        "episode_count": episode_count,
        "pipeline_path": pipeline_path,
        "pretrain_config_path": Path(paths["pretrain_config"]),
        "downstream_config_path": Path(paths["downstream_config"]),
        "downstream_checkpoint_path": Path(paths["downstream_slot_pretrain_checkpoint"]),
    }


def launch_stage(config_path: Path, devices, debugmode: bool, repo_root: Path):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(device.replace("cuda:", "") for device in devices)
    env.setdefault("TORCH_HOME", str(repo_root / ".cache" / "torch"))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    with config_path.open("r", encoding="utf-8") as stream:
        app = yaml.safe_load(stream).get("app")
    entry = Path("../pretrain/train.py") if app == "slot_jepa" else Path("train.py")
    if debugmode:
        cmd = [sys.executable, str(entry), "--fname", str(config_path)]
    else:
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(len(devices)),
            str(entry),
            "--fname",
            str(config_path),
        ]
    print("[launch]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)


def _to_repo_relative(path: Path, repo_root: Path):
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_pipeline_stage_command(pipeline_config_path: Path, stage: str, repo_root: Path, skip_data_check: bool = False):
    rel_config_path = _to_repo_relative(pipeline_config_path, repo_root)
    cmd = f"python train_slot_ac_camera_base.py --config {rel_config_path} --stage {stage}"
    if skip_data_check:
        cmd += " --skip-data-check"
    return cmd


def ensure_downstream_csv(pipeline_config):
    paths = pipeline_config["paths"]
    downstream_dataset_root = Path(pipeline_config["downstream"]["dataset_root"])
    csv_path = Path(paths["downstream_csv"])
    if csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"[prepare] Reusing existing downstream CSV: {csv_path}", flush=True)
        return None
    episode_count = build_episode_csv(downstream_dataset_root, csv_path)
    print(f"[prepare] Wrote {episode_count} episodes to {csv_path}", flush=True)
    return episode_count


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    stage = resolve_stage(args)

    if args.config:
        pipeline_config = load_pipeline_config(Path(args.config).expanduser().resolve())
    else:
        pipeline_config = build_pipeline_config(args, repo_root)

    skip_data_check = bool(args.skip_data_check) or stage in ("pretrain", "downstream", "all")
    materialized = materialize_configs(pipeline_config, skip_data_check=skip_data_check)
    downstream_trainer = pipeline_config.get("downstream", {}).get("trainer", "slot-ac")
    if materialized["episode_count"] is None:
        print(
            f"[prepare] Skipped downstream dataset scan for {pipeline_config['paths']['downstream_csv']}. "
            "CSV will be generated when downstream training starts.",
            flush=True,
        )
    else:
        print(f"[prepare] Wrote {materialized['episode_count']} episodes to {pipeline_config['paths']['downstream_csv']}", flush=True)
    print(f"[prepare] Wrote pipeline config to {materialized['pipeline_path']}", flush=True)
    print(f"[prepare] Wrote pretrain config to {materialized['pretrain_config_path']}", flush=True)
    print(f"[prepare] Wrote downstream config to {materialized['downstream_config_path']}", flush=True)
    print(
        "[prepare] Stage order:\n"
        f"1. generate config: {materialized['pipeline_path']}\n"
        f"2. pretrain: {materialized['pretrain_config_path']}\n"
        f"3. downstream ({downstream_trainer}): {materialized['downstream_config_path']}",
        flush=True,
    )

    devices = pipeline_config["devices"]
    debugmode = bool(pipeline_config.get("debugmode", False))
    downstream_checkpoint_path = materialized["downstream_checkpoint_path"]
    requires_slot_pretrain = downstream_trainer == "slot-ac"
    pretrain_cmd = build_pipeline_stage_command(materialized["pipeline_path"], "pretrain", repo_root, skip_data_check=True)
    downstream_cmd = build_pipeline_stage_command(materialized["pipeline_path"], "downstream", repo_root)
    if requires_slot_pretrain:
        print(f"[prepare] Pretrain command:\n{pretrain_cmd}", flush=True)
    else:
        print(f"[prepare] Pretrain command:\nnot required for downstream trainer `{downstream_trainer}`", flush=True)
    print(f"[prepare] Downstream command ({downstream_trainer}):\n{downstream_cmd}", flush=True)

    if stage == "prepare":
        return
    if requires_slot_pretrain and stage == "downstream" and not downstream_checkpoint_path.exists():
        print(
            f"[prepare] Downstream will load slot pretrain checkpoint from {downstream_checkpoint_path}. "
            "If you have not run pretraining yet, please run stage=pretrain first or provide --slot-pretrain-checkpoint.",
            flush=True,
        )
        if stage == "downstream":
            raise FileNotFoundError(f"Missing downstream slot pretrain checkpoint: {downstream_checkpoint_path}")

    if stage == "pretrain" and not requires_slot_pretrain:
        print(f"[prepare] Skipping pretrain: downstream trainer `{downstream_trainer}` does not use it.", flush=True)
        return
    if stage == "pretrain" or (stage == "all" and requires_slot_pretrain):
        launch_stage(materialized["pretrain_config_path"], devices, debugmode, repo_root)
    if stage in ("downstream", "all"):
        ensure_downstream_csv(pipeline_config)
        launch_stage(materialized["downstream_config_path"], devices, debugmode, repo_root)


if __name__ == "__main__":
    main()
