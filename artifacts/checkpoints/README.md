# Checkpoints

All workflows resolve released model weights from this directory.

| Path | Purpose |
| --- | --- |
| `pretrain/slot_jepa_pretrained.pt` | Shared final Kinetics-400 Slot-JEPA checkpoint used by every downstream task |
| `pretrain/slot_jepa_pretrained.yaml` | Minimal Slot-JEPA model config |
| `downstream/robot/slot_jepa_droid_finetuned.pt` | Final DROID-fine-tuned checkpoint used by visual-understanding evaluation and robot planning |
| `downstream/robot/slot_jepa_droid_finetuned.yaml` | Minimal robot model config |
| `downstream/robot/README.md` | Robot checkpoint contents and loading example |
| `downstream/planar/` | Six task-specific Slot-JEPA world models used for the paper's planar evaluation |
| `downstream/visual_understanding/` | Final depth heads for NYUv2, KITTI, and SUN RGB-D transfer |

Datasets, caches, optimizer states, TensorBoard logs, training runs, and evaluation outputs are intentionally not stored under `artifacts/`.

Run `sha256sum -c SHA256SUMS` inside either `pretrain/` or `downstream/` to verify the corresponding package.
