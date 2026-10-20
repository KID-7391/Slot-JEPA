# Visual-Understanding Evaluation

This workflow evaluates whether the pretrained Slot-JEPA decomposition preserves semantic information needed by downstream visual tasks. The visual encoder and slot decomposition remain frozen.

[![Segmentation](https://img.shields.io/badge/Task-Segmentation-4C8BF5.svg)](#-semantic-segmentation) [![Depth](https://img.shields.io/badge/Task-Depth-34A853.svg)](#-monocular-depth) [![VOS](https://img.shields.io/badge/Task-VOS-FBBC05.svg)](#-video-object-segmentation)

## 🚩 Benchmarks


| Probe                     | Dataset          | Primary metric    | Protocol                  |
| --------------------------- | ------------------ | ------------------- | --------------------------- |
| Semantic segmentation     | COCOStuff-27     | mIoU              | linear head               |
| Semantic segmentation     | ADE20K           | mIoU              | linear head               |
| Semantic segmentation     | PASCAL VOC       | mIoU              | linear head               |
| Monocular depth           | NYUv2            | δ1, AbsRel, RMSE | linear depth head         |
| Monocular depth           | KITTI            | δ1, AbsRel, RMSE | linear depth head         |
| Depth transfer            | NYUv2 → SUNRGBD | δ1, AbsRel, RMSE | NYUv2 head, no adaptation |
| Video object segmentation | DAVIS-2017       | J, F, J&F         | label propagation         |

Segmentation and depth train only task heads. DAVIS uses non-parametric label propagation without additional model training. The released Slot-JEPA configs use the paper's fixed feature mix, `(1 - alpha) Z + alpha Z_slot` with `alpha = 0.3`; this is an internal model setting and is not encoded in config or output names.

## 📁 Checkpoint and Datasets

All evaluation configs load the same centralized DROID-fine-tuned Slot-JEPA checkpoint:

```text
artifacts/checkpoints/downstream/robot/slot_jepa_droid_finetuned.pt
```

The path is resolved through [`configs/paths.yaml`](configs/paths.yaml); no workflow-local checkpoint copy is required.

Optional V-JEPA 2.1 baseline weights are not part of `artifacts/`; download them separately under `checkpoints/vjepa_2_1/` when reproducing baseline rows.

The upload package also includes the final trained depth heads in `artifacts/checkpoints/downstream/visual_understanding/`. Semantic-segmentation metrics are produced during linear-probe runs without a retained public head, while DAVIS is non-parametric and has no task checkpoint.

Datasets are not bundled. Configure their locations in [`configs/paths.yaml`](configs/paths.yaml):

```text
data/
├── coco/
├── ADEChallengeData2016/
├── VOCdevkit/VOC2012/
├── depth/nyu/
├── depth/KITTI/
├── depth/SUNRGBD/
└── davis-2017/DAVIS/
```

Absolute paths may be written directly in `configs/paths.yaml`, or the expected directories may be provided as symlinks under `data/`.

Download the datasets from their official sources:


| Dataset         | Download                                                                                                                                                  | Expected location            |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| COCOStuff-27    | [COCO-Stuff setup and downloads](https://github.com/nightrome/cocostuff#setup)                                                                            | `data/coco/`                 |
| ADE20K          | [ADEChallengeData2016](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)                                                           | `data/ADEChallengeData2016/` |
| PASCAL VOC 2012 | [VOC 2012 development kit and train/val data](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/#devkit)                                       | `data/VOCdevkit/VOC2012/`    |
| NYUv2           | [NYU Depth V2 downloads](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)                                                                           | `data/depth/nyu/`            |
| KITTI           | [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) and [annotated depth maps](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php) | `data/depth/KITTI/`          |
| SUNRGBD         | [SUNRGBD downloads](https://sun3d.cs.princeton.edu/data/SUNRGBD/)                                                                                         | `data/depth/SUNRGBD/`        |
| DAVIS-2017      | [DAVIS-2017 TrainVal 480p](https://davischallenge.org/davis2017/code.html)                                                                                | `data/davis-2017/DAVIS/`     |

The segmentation and DAVIS loaders use the standard extracted layouts shown above. For depth evaluation, prepare paired RGB/depth images and place the split files named in the task configs (`nyu_train.txt`, `nyu_test.txt`, `kitti_eigen_train.txt`, `kitti_eigen_test.txt`, and `SUNRGBD_val_splits.txt`) inside the corresponding dataset root.

## 🔧 Installation

```bash
pip install -r requirement.txt
pip install -e .
cd workflows/representation_eval
```

Validate the bundle before launching an evaluation:

```bash
python scripts/check_bundle.py --validate-configs --check-checkpoints --skip-data
python scripts/check_bundle.py --validate-data
```

## 📊 Semantic Segmentation

COCOStuff-27:

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/test_linear_seg.py \
  --config configs/tasks/linear_seg/coco_slot_jepa.yaml \
  --paths_config configs/paths.yaml
```

ADE20K and PASCAL VOC:

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/test_linear_seg.py \
  --config configs/tasks/linear_seg/ade20k_voc_slot_jepa.yaml \
  --paths_config configs/paths.yaml
```

## 📐 Monocular Depth

```bash
# NYUv2
python scripts/test_depth.py \
  --config configs/tasks/depth/nyuv2_slot_jepa.yaml \
  --paths_config configs/paths.yaml

# KITTI
python scripts/test_depth.py \
  --config configs/tasks/depth/kitti_slot_jepa.yaml \
  --paths_config configs/paths.yaml

# NYUv2-to-SUNRGBD transfer
python scripts/test_depth.py \
  --config configs/tasks/depth/sunrgbd_transfer_nyuv2_slot_jepa.yaml \
  --paths_config configs/paths.yaml
```

## 🎞️ Video Object Segmentation

```bash
python scripts/test_vos.py \
  --config configs/tasks/vos/davis_slot_jepa.yaml \
  --paths_config configs/paths.yaml
```

The final protocol retains the latest 20 frames, uses top-5 propagation neighbors, and initializes the shared slot bank from first-frame mask centers.

## 📦 Outputs

Each task writes to `outputs/` according to its `output_subdir`. Evaluation outputs are intentionally excluded from the release and from Git; rerunning a task creates a fresh local directory.

## 🌟 Acknowledgements

The evaluator suite is developed from [SSL_downstreams](https://github.com/EldercatSAM/SSL_downstreams), with Slot-JEPA checkpoint loading, slot reconstruction, feature mixing, and the paper's final task configurations added for this release. Shared Slot-JEPA code is imported from [`../../src/slot_jepa`](../../src/slot_jepa).
