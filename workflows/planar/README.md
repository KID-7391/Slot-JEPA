# Planar Visual Planning

This workflow trains task-specific Slot-JEPA world models from offline interactions and evaluates visual-goal planning with Model-Predictive Control (MPC).

[![DINO-WM](https://img.shields.io/badge/Based_on-DINO--WM-5C6BC0.svg)](https://github.com/gaoyuezhou/dino_wm)

The visual encoder and slot decomposition remain frozen. The transferred TMP and task-specific state/action encoders are optimized for each environment, while planning scores imagined outcomes with the mask-space UOT and L1 objective.

## 🚩 Environments


| Environment | Split | Description                         | Training config  | Planning config |
| ------------- | ------- | ------------------------------------- | ------------------ | ----------------- |
| `TwoRoom`   | ID    | fixed doorway navigation            | `train_TwoRoom`  | `plan_TwoRoom`  |
| `Maze`      | ID    | fixed U-shaped corridor             | `train_Maze`     | `plan_Maze`     |
| `PushT`     | ID    | T-shaped object manipulation        | `train_PushT`    | `plan_PushT`    |
| `TwoRoomR`  | OOD   | randomized wall and doorway layouts | `train_TwoRoomR` | `plan_TwoRoomR` |
| `MazeR`     | OOD   | randomized maze layouts             | `train_MazeR`    | `plan_MazeR`    |
| `PushObj`   | OOD   | unseen object shapes                | `train_PushObj`  | `plan_PushObj`  |

Each planning config uses fixed physical start/goal states. Images are rendered by the live simulator rather than loaded from saved videos.

## 📁 Data and Artifacts

```text
../../artifacts/data/planar/               # six Slot-JEPA task datasets
../../artifacts/checkpoints/pretrain/      # shared frozen Slot-JEPA checkpoint
../../artifacts/checkpoints/downstream/planar/outputs/
  <Task>/hydra.yaml                        # portable resolved training config
  <Task>/checkpoints/model_latest.pth       # released compact predictor
slot-ac/outputs/                           # locally generated training runs
slot-ac/slot_cache/                        # locally generated slot/retrieval caches
slot-ac/context_cache/                     # locally generated planning context cache
plan_outputs/                              # locally generated inference results
```

The upload-ready planar data package is stored at `../../artifacts/data/planar/`. It contains the released raw training/validation trajectories and fixed test source/target states for the six paper tasks. The six released predictor checkpoints and loading configs are stored under `../../artifacts/checkpoints/downstream/planar/outputs/`. Precomputed mask logits, slot/retrieval caches, training runs, and inference results are not published. In particular, the release does not include planning metrics, videos, proposal dumps, or per-case result directories.

### Cache roles

Caches are local, model-specific acceleration artifacts; they are not benchmark data and are not required downloads. The raw trajectories and fixed test endpoints remain the source of truth.


| Cache                    | Default location                                | Contents and purpose                                                                                                                                                                                                                                                                    |
| -------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Frame-slot cache         | `slot-ac/slot_cache/maskframe_*.pt`             | Mask logits and physical states for every usable frame in the raw training trajectories. It avoids repeatedly rendering and encoding the same frames during training and is the input used to construct a retrieval bank. This is usually the largest cache.                            |
| TMP retrieval bank       | `slot-ac/slot_cache/*_ret*_idxv1.pt`            | A balanced, deduplicated subset of training transitions containing source/target states, actions, source/target mask logits, layout IDs, and dataset indices. TMP retrieves its context transitions from this bank during training and planning. It contains no test source/goal cases. |
| Test context pool        | `slot-ac/context_cache/pool_*.pt`               | Deterministically generated transitions for a particular test layout when `test_context_num > 0`. It avoids regenerating and re-encoding the same online-context pool on later runs. The base default is 0; `TwoRoomR`, `MazeR`, and `PushObj` override it to 5.                        |
| Retrieval RGB-origin map | beside the retrieval bank as `*_rgb_origins.pt` | Maps retrieval rows back to raw training frames for visualization. It is not used by mask prediction or planning scores.                                                                                                                                                                |
| Predictor context cache  | GPU memory only                                 | Temporary encoded TMP context reused across CEM proposal chunks within one planning call. It is never a released or persistent file.                                                                                                                                                    |

Training and inference reuse a cache only when its identity matches the dataset,
window geometry, slot configuration, color-weight version, retrieval policy,
and relevant model settings. If the required retrieval bank is missing,
inference first builds the frame-slot cache from raw trajectories with the
downloaded model, then builds and loads the retrieval bank.

`scripts/plan.py` builds a missing cache on its single selected GPU. For a
multi-GPU build, run the cache builder directly with Accelerate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --multi_gpu \
  scripts/build_retrieval_cache.py --config-name plan_TwoRoomR
```

All six tasks also use one frozen color-channel reweighting vector from `../../artifacts/color_channel_weights/planar_block12_important_objects_v2.pt`. It reweights the final encoder-block channels before slot extraction, then restores every patch token's original L2 norm.

## 🔧 Installation

Install the shared package from this directory:

```bash
pip install -r ../../requirement.txt
pip install -e ../..
```

The simulator dependencies follow [DINO-WM](https://github.com/gaoyuezhou/dino_wm), and TwoRoom/Maze require MuJoCo.

```bash
cd workflows/planar
export WANDB_MODE=disabled
```

## 🚀 Training

```bash
# ID environments
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config-name train_TwoRoom
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config-name train_Maze
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config-name train_PushT

# OOD environments
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config-name train_TwoRoomR
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config-name train_MazeR
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config-name train_PushObj
```

Use Accelerate for the final multi-GPU setup:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --multi_gpu scripts/train.py --config-name train_TwoRoom
```

Hydra writes the resolved config, compact predictor checkpoints, logs, and training visualizations to `slot-ac/outputs/<Task>/<timestamp>/`, using the exact paper task name. Each checkpoint stores task-specific predictor/adaptor state only; it does not duplicate the frozen student or slot extractor. To publish a retrained model, replace the matching `hydra.yaml` and `checkpoints/model_latest.pth` under `../../artifacts/checkpoints/downstream/planar/outputs/<Task>/`.

## 🧭 Planning and Evaluation

The planning configs load the released checkpoints from `../../artifacts/checkpoints/downstream/planar/outputs/` by default. Test files provide physical `state_0`/`state_g` endpoints (plus layout or shape metadata where required); the simulator renders both images and the downloaded model generates their slot masks online.

Use `scripts/plan.py` for a single-GPU run or a quick smoke test. Choose the matching environment:

```bash
# TwoRoom
python scripts/plan.py --config-name plan_TwoRoom

# Maze
python scripts/plan.py --config-name plan_Maze

# PushT
python scripts/plan.py --config-name plan_PushT

# TwoRoomR
python scripts/plan.py --config-name plan_TwoRoomR

# MazeR
python scripts/plan.py --config-name plan_MazeR

# PushObj
python scripts/plan.py --config-name plan_PushObj
```

All six public checkpoints use the same `model_epoch=latest` argument. For full evaluation, `scripts/plan_multi_gpu.py` is the recommended launcher. It first builds a missing retrieval cache across the selected GPUs, then starts one persistent worker per GPU. Each worker loads the model once and consumes individual evaluation cases from a shared queue:

```bash
python scripts/plan_multi_gpu.py \
  --config-name plan_TwoRoom \
  --devices 0,1,2,3 \
  --no-visualization
```

`--devices` takes comma-separated physical GPU IDs. Replace `plan_TwoRoom` with any of the six planning configs listed above. Omit `--no-visualization` when per-case videos and diagnostic plots are needed.

Hydra overrides can be appended after the launcher options. For example, to evaluate four selected cases and use an explicit output directory:

```bash
python scripts/plan_multi_gpu.py --config-name plan_MazeR \
  --devices 0,1,2,3 \
  --output-dir plan_outputs/maze_r \
  --no-visualization
```

Results are stored as `case_XXXX/result.json` with attempt-specific logs and artifacts. The run also writes `config.yaml`, `manifest.json`, and the aggregate `summary.json`. Re-running the same command reuses completed cases and retries missing or errored cases. The default output directory is stable and includes a configuration fingerprint. To evaluate a locally retrained run instead, override both the checkpoint base and the timestamped model name, for example:

```bash
python scripts/plan.py --config-name plan_MazeR \
  ckpt_base_path=./slot-ac \
  model_name=TwoRoom/<timestamp>
```

## 🌟 Acknowledgements

The environments, dataset loaders, baseline world models, and MPC/CEM loop are based on [DINO-WM](https://github.com/gaoyuezhou/dino_wm). Slot-JEPA adds the structured state interface, transition-memory retrieval, mask prediction, UOT planning loss, and OOD environments.
