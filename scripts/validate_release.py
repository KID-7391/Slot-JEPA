#!/usr/bin/env python3
"""Fast structural validation for the curated release tree."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "requirement.txt",
    "assets/method_overview.jpg",
    "assets/planar_environments.png",
    "assets/uot_loss_landscapes.png",
    "assets/predicted_slot_rollouts.png",
    "assets/realm_jepa_tasks.png",
    "src/slot_jepa/encoders",
    "src/slot_jepa/slots",
    "src/slot_jepa/predictors",
    "src/slot_jepa/losses",
    "src/slot_jepa/losses/transport.py",
    "workflows/pretrain/README.md",
    "workflows/pretrain/train.py",
    "workflows/pretrain/scripts/build_kinetics_lmdb.py",
    "workflows/representation_eval/README.md",
    "workflows/representation_eval/scripts/test_depth.py",
    "workflows/representation_eval/configs/tasks/depth/kitti_slot_jepa.yaml",
    "workflows/representation_eval/configs/tasks/depth/nyuv2_slot_jepa.yaml",
    "workflows/representation_eval/configs/tasks/depth/sunrgbd_transfer_nyuv2_slot_jepa.yaml",
    "workflows/representation_eval/configs/tasks/linear_seg/ade20k_voc_slot_jepa.yaml",
    "workflows/representation_eval/configs/tasks/linear_seg/coco_slot_jepa.yaml",
    "workflows/representation_eval/configs/tasks/vos/davis_slot_jepa.yaml",
    "workflows/planar/README.md",
    "workflows/planar/scripts/train.py",
    "workflows/planar/scripts/plan.py",
    "workflows/planar/scripts/plan_multi_gpu.py",
    "workflows/planar/scripts/build_retrieval_cache.py",
    "workflows/planar/fixed_slot_references.py",
    "workflows/planar/conf/train_TwoRoom.yaml",
    "workflows/planar/conf/train_Maze.yaml",
    "workflows/planar/conf/train_PushT.yaml",
    "workflows/planar/conf/train_TwoRoomR.yaml",
    "workflows/planar/conf/train_MazeR.yaml",
    "workflows/planar/conf/train_PushObj.yaml",
    "workflows/planar/conf/plan_TwoRoom.yaml",
    "workflows/planar/conf/plan_Maze.yaml",
    "workflows/planar/conf/plan_PushT.yaml",
    "workflows/planar/conf/plan_TwoRoomR.yaml",
    "workflows/planar/conf/plan_MazeR.yaml",
    "workflows/planar/conf/plan_PushObj.yaml",
    "workflows/robot/README.md",
    "workflows/robot/train_slot_ac_camera_base.py",
    "workflows/robot/serve_slot.py",
    "workflows/robot/runtime/vjepa_planning.py",
    "workflows/robot/runtime/slot_planning.py",
    "workflows/robot/runtime/slot_sim_server.py",
    "workflows/robot/runtime/_output_permissions.py",
    "artifacts/data/planar/README.md",
    "artifacts/data/planar/SHA256SUMS",
    "artifacts/data/planar/TwoRoom",
    "artifacts/data/planar/Maze",
    "artifacts/data/planar/PushT",
    "artifacts/data/planar/TwoRoomR",
    "artifacts/data/planar/MazeR",
    "artifacts/data/planar/PushObj",
    "artifacts/data/robot/README.md",
    "artifacts/data/robot/successful_case",
    "artifacts/checkpoints/pretrain/slot_jepa_pretrained.pt",
    "artifacts/checkpoints/pretrain/slot_jepa_pretrained.yaml",
    "artifacts/checkpoints/pretrain/SHA256SUMS",
    "artifacts/checkpoints/downstream/robot/README.md",
    "artifacts/checkpoints/downstream/robot/slot_jepa_droid_finetuned.pt",
    "artifacts/checkpoints/downstream/robot/slot_jepa_droid_finetuned.yaml",
    "artifacts/checkpoints/downstream/planar/outputs/TwoRoom/hydra.yaml",
    "artifacts/checkpoints/downstream/planar/outputs/TwoRoom/checkpoints/model_latest.pth",
    "artifacts/checkpoints/downstream/planar/outputs/Maze/hydra.yaml",
    "artifacts/checkpoints/downstream/planar/outputs/Maze/checkpoints/model_latest.pth",
    "artifacts/checkpoints/downstream/planar/outputs/PushT/hydra.yaml",
    "artifacts/checkpoints/downstream/planar/outputs/PushT/checkpoints/model_latest.pth",
    "artifacts/checkpoints/downstream/planar/outputs/TwoRoomR/hydra.yaml",
    "artifacts/checkpoints/downstream/planar/outputs/TwoRoomR/checkpoints/model_latest.pth",
    "artifacts/checkpoints/downstream/planar/outputs/MazeR/hydra.yaml",
    "artifacts/checkpoints/downstream/planar/outputs/MazeR/checkpoints/model_latest.pth",
    "artifacts/checkpoints/downstream/planar/outputs/PushObj/hydra.yaml",
    "artifacts/checkpoints/downstream/planar/outputs/PushObj/checkpoints/model_latest.pth",
    "artifacts/checkpoints/downstream/visual_understanding/nyuv2_depth_head.pt",
    "artifacts/checkpoints/downstream/visual_understanding/kitti_depth_head.pt",
    "artifacts/checkpoints/downstream/visual_understanding/sunrgbd_transfer_nyuv2_depth_head.pt",
    "artifacts/checkpoints/downstream/SHA256SUMS",
    "artifacts/color_channel_weights/README.md",
    "artifacts/color_channel_weights/planar_block12_important_objects_v2.pt",
)

EXCLUDED_ARTIFACTS = (
    "artifacts/evaluations",
    "artifacts/training",
    "artifacts/caches",
    "artifacts/checkpoints/planar",
    "artifacts/checkpoints/downstream/planar/slot_cache",
    "workflows/representation_eval/outputs",
    "workflows/planar/plan_outputs",
    "workflows/robot/realm/outputs",
)


def main() -> int:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).exists()]
    link_roots = (ROOT / "src", ROOT / "workflows")
    broken = [
        str(path.relative_to(ROOT))
        for link_root in link_roots
        for path in link_root.rglob("*")
        if path.is_symlink() and not path.exists()
    ]
    unexpected = [
        relative
        for relative in EXCLUDED_ARTIFACTS
        if (ROOT / relative).exists() or (ROOT / relative).is_symlink()
    ]
    if missing or broken or unexpected:
        for relative in missing:
            print(f"[missing] {relative}")
        for relative in broken:
            print(f"[broken-link] {relative}")
        for relative in unexpected:
            print(f"[excluded-artifact] {relative}")
        return 1
    print(
        f"[ok] release layout: {len(REQUIRED)} required paths, "
        "current checkpoints/configs and planar data only, no generated outputs, and no broken links"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
