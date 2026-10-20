# Slot-JEPA Pretraining

This workflow pretrains the reusable Slot-JEPA components on unlabeled Kinetics-400 video: the slot decomposition module and the Transition Memory Predictor (TMP).

[![V-JEPA 2](https://img.shields.io/badge/Based_on-V--JEPA_2-0467DF.svg)](https://github.com/facebookresearch/vjepa2) [![Dataset](https://img.shields.io/badge/Dataset-Kinetics--400-6A5ACD.svg)](https://www.deepmind.com/open-source/kinetics)

## 🚩 Overview

Starting from a V-JEPA 2.1 ViT-B/16 checkpoint, the final configuration:

- extracts 16 global semantic slots from local `2 x 4` spatiotemporal windows;
- decodes time-varying masks and reconstructs the dense feature field;
- pretrains TMP with pseudo-actions produced by inverse transition encoders;
- fine-tunes the final six visual-encoder blocks;
- predicts three target frames from four context frames in each seven-frame clip;
- combines feature reconstruction, mask L1, and UOT losses.

The inverse encoders are used only during pretraining. Downstream workflows replace them with task-specific state and action encoders while transferring the slot decomposition and TMP weights.

## 📁 Inputs

### V-JEPA 2.1 initialization

The release expects Meta's V-JEPA 2.1 ViT-B/16 distilled checkpoint at:

```text
workflows/pretrain/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
```

Download it from the [official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2#v-jepa-21-pretrained-checkpoints), or directly from Meta's model host:

```bash
mkdir -p workflows/pretrain/checkpoints
curl -L \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
  -o workflows/pretrain/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt

sha256sum workflows/pretrain/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
# 848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d
```

### Kinetics-400 videos

Kinetics-400 is not redistributed. The raw train, validation, and test archives are hosted by CVDF through the [Kinetics Dataset Downloader](https://github.com/cvdfoundation/kinetics-dataset). Follow its download and extraction scripts:

```bash
git clone https://github.com/cvdfoundation/kinetics-dataset.git
cd kinetics-dataset
bash k400_downloader.sh
bash k400_extractor.sh
```

This produces the original videos under `kinetics-dataset/k400/{train,val,test}`. The dataset is large, and some source videos may be unavailable or corrupted; the LMDB builder reports and skips videos that cannot be decoded.

### Build the training LMDB

From the Slot-JEPA release root, convert the extracted training split with:

```bash
python workflows/pretrain/scripts/build_kinetics_lmdb.py \
  --input /path/to/kinetics-dataset/k400/train \
  --output data/Kinetics-400/kinetics_train.lmdb \
  --sample-fps 2 \
  --workers 16
```

The released training recipe samples source videos at 2 FPS and requires at least seven sampled frames per video. The script recursively discovers videos, encodes sampled RGB frames as JPEG, writes loader-compatible keys, and creates the metadata and valid-video list. If conversion is interrupted, resume it with the same arguments plus `--resume`.

The default configuration expects:

```text
data/Kinetics-400/
└── kinetics_train.lmdb/
    ├── data.mdb
    ├── lock.mdb
    ├── metadata.pkl
    └── valid_video_names.txt
```

`failed_videos.txt` is also written when any videos are skipped. Edit [`configs/k400.yaml`](configs/k400.yaml) when the dataset or checkpoint is stored elsewhere. Relative paths are resolved from the directory in which the launcher is executed; the command below is run from the release root.

## 💻 Environment

```bash
pip install -r requirement.txt
pip install -e .
```

The paper run uses eight GPUs, bfloat16, a per-process batch size of 2, and 24GB of memory per GPU. Adjust the distributed and data-loader settings in the YAML for different hardware.

## 🚀 Training

Run from the release root:

```bash
torchrun --standalone --nproc_per_node=8 \
  workflows/pretrain/train.py \
  --fname workflows/pretrain/configs/k400.yaml
```

To resume training, set `meta.resume_checkpoint` in `configs/k400.yaml` and relaunch the same command. The launcher records the resolved configuration as `params-pretrain.yaml`.

## 📦 Outputs

```text
workflows/pretrain/outputs/slot_jepa/run/
├── latest.pt
└── params-pretrain.yaml
```

Periodic checkpointing is controlled by `meta.save_every_freq`. TensorBoard summaries are written when the TensorBoard writer is available:

The released final checkpoint and its minimal loading config are stored at `artifacts/checkpoints/pretrain/`. Fresh training outputs are ignored and are not part of the release.

```bash
tensorboard --logdir \
  workflows/pretrain/outputs/slot_jepa/run
```

## 🌟 Acknowledgements

This training stack is adapted from [V-JEPA 2](https://github.com/facebookresearch/vjepa2) and the project [VJSEPA fork](https://github.com/KID-7391/vjsepa). Slot-JEPA adds the slot-and-mask interface, inverse transition encoders, TMP, reconstruction and mask objectives, and UOT supervision. Shared code is organized under [`../../src/slot_jepa`](../../src/slot_jepa).
