# Robotic Arm Planning on REALM-JEPA

This workflow fine-tunes Slot-JEPA on DROID trajectories and evaluates multi-view visual-goal planning in the REALM simulator.

[![REALM](https://img.shields.io/badge/Based_on-REALM-2E8B57.svg)](https://github.com/martin-sedlacek/REALM) [![DROID](https://img.shields.io/badge/Training_data-DROID-6A5ACD.svg)](https://droid-dataset.github.io/)

The planner receives synchronized observations from two fixed external cameras and one wrist camera, together with the current 7-DoF end-effector/gripper state. Each view maintains separate slots and transition memory while sharing the predictor.

## 🚩 REALM-JEPA Benchmark


| Family | Cases | Released task IDs               |
| -------- | ------: | --------------------------------- |
| Reach  |    18 | `0`–`17`                       |
| Place  |    30 | `18`–`23` across task variants |
| Pick   |    30 | `24`–`29` across task variants |
| Rotate |    30 | `30`–`35` across task variants |

Reach contains one case for each task definition. Pick, Rotate, and Place use six task definitions with five visual/scene variants. Tasks may contain one or two stage-wise visual goals.

Within each six-task block, the object order matches the corresponding reach block: place IDs `18`–`23` correspond to reach IDs `0`–`5`, pick IDs `24`–`29` to reach IDs `6`–`11`, and rotate IDs `30`–`35` to reach IDs `12`–`17`.

## 📁 Released Checkpoints

The release contains only the final weights and loading configs:

```text
../../artifacts/checkpoints/pretrain/slot_jepa_pretrained.pt
../../artifacts/checkpoints/pretrain/slot_jepa_pretrained.yaml
../../artifacts/checkpoints/downstream/robot/slot_jepa_droid_finetuned.pt
../../artifacts/checkpoints/downstream/robot/slot_jepa_droid_finetuned.yaml
```

Training runs, TensorBoard logs, DROID data, and full `successful_case` trajectories are not bundled. The release includes the minimal planning targets under `../../artifacts/data/robot/successful_case/`: manifests, compact goal metadata, and the three goal-camera images. Rollout traces, intermediate states, and collection diagnostics are omitted.

Download DROID from the [official dataset instructions](https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md). This workflow uses the three-camera raw release:

```bash
gsutil -m cp -r gs://gresearch/robotics/droid_raw /path/to/droid_raw
```

The downloaded episodes must be converted to the frame-based three-view layout expected at `data/droid_frame_camera_base_multiview/`. Pass a different location with `--downstream-dataset-root`.

## 🔧 Environment Setup

REALM planning uses two environments: a host Python environment for training and the model server, and a REALM Docker container for the OmniGibson simulation. The host must have an NVIDIA GPU driver, Docker, and the NVIDIA Container Toolkit.

Before building REALM, verify that the current user can access Docker:

```bash
docker info
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Install the shared Slot-JEPA package in the host Python environment:

```bash
pip install -r ../../requirement.txt
pip install -e ../..
```

Configure the REALM image and download the OmniGibson assets once:

```bash
export REALM_DATA_PATH=/absolute/path/to/REALM-data
./setup.sh --docker --dataset
```

Keep `REALM_DATA_PATH` exported in terminals used for this workflow; `--data-path` remains available as a one-command override. Accept the NVIDIA Omniverse EULA when prompted. The original [`martin-sedlacek/REALM`](https://github.com/martin-sedlacek/REALM) repository is a useful upstream reference, but it does not include the Slot-JEPA-specific simulation and planning changes released here.

The fork clone above is used to build `realm:latest` and fetch the assets. Run the evaluation commands below from this released `workflows/robot` directory; the Docker command mounts that directory at `/app`, so the container executes the released Slot-JEPA code rather than the clone used to build the image.

On systems where Docker is unavailable but Isaac Sim and OmniGibson are already installed, the retained cloud helper can prepare the Python dependencies, cache directories, and data paths:

```bash
bash scripts/setup_cloud_realm_env.sh --data-path /path/to/REALM-data
source .realm_cloud_env.sh
```

## 🚀 DROID Fine-Tuning

The downstream stage transfers the pretrained encoder, slot decomposition, and TMP, replaces the inverse pretraining encoders with robot state/action encoders, and fine-tunes the predictor on three-view DROID clips.

Create a fresh output namespace and materialize the training configs:

```bash
python train_slot_ac_camera_base.py \
  --stage prepare \
  --output-dir outputs/<RUN_NAME> \
  --downstream-dataset-root /path/to/droid_frame_camera_base_multiview \
  --slot-pretrain-checkpoint ../../artifacts/checkpoints/pretrain/slot_jepa_pretrained.pt

python train_slot_ac_camera_base.py \
  --config outputs/<RUN_NAME>/pipeline_config.yaml \
  --stage downstream
```

Run `--stage pretrain` first if no Slot-JEPA pretraining checkpoint is available, or use `--stage all` to run both stages sequentially.

```bash
tensorboard --logdir \
  outputs/<RUN_NAME>/downstream/run/tensorboard
```

## 🧩 Planning Architecture

Evaluation uses three cooperating processes:

```text
Planning client  ->  Slot-JEPA model server  ->  OmniGibson simulation server
```

The model server owns the checkpoint, slots, transition memory, TMP rollout, and CEM scoring. The simulation server owns the REALM scene and controller. The planning client supplies tasks and goals, coordinates replanning, and writes case results.

### 1. Start the Model Server

Run in the host GPU environment:

```bash
CUDA_VISIBLE_DEVICES=0 python serve_slot.py \
  --port 8031 \
  --ckpt-path ../../artifacts/checkpoints/downstream/robot/slot_jepa_droid_finetuned.pt \
  --config-path ../../artifacts/checkpoints/downstream/robot/slot_jepa_droid_finetuned.yaml
```

### 2. Start the Simulation Server

From the host, change to this workflow directory and start the simulation server. It reuses the `REALM_DATA_PATH` exported during setup:

```bash
cd /absolute/path/to/slot_jepa_release/workflows/robot
scripts/start_sim_server.sh
```

The launcher defaults to physical GPU 1 and port 8021, validates Docker/data/goal inputs, mounts persistent Isaac Sim caches, and directly runs `runtime/slot_sim_server.py`. Select another GPU or open a debugging shell with:

```bash
scripts/start_sim_server.sh --gpu 2 --port 8021
scripts/start_sim_server.sh --shell
```

`REALM_DATA_PATH`, `REALM_SIM_GPU`, and `REALM_DOCKER_IMAGE` provide the corresponding environment-variable defaults. The launcher intentionally does not use Docker `--privileged`, because privileged mode exposes every host GPU and defeats single-GPU isolation. Host GPU 1 appears as `cuda:0` inside the single-GPU container. The server should report `/artifacts/data/robot/successful_case` and 108 target cases during startup.

### 3. Run Planning

Verify the three-process connection with one case:

```bash
scripts/run_planning.sh --task-id 0 --task-variant default --sim-port 8021 --model-port 8031
```

The launcher creates a temporary CPU-only Docker container for the planning client, uses host networking to reach the model and simulation servers, and writes results back to the mounted workflow directory. Run the four paper task families with:

```bash
scripts/run_planning.sh --suite reach
scripts/run_planning.sh --suite pick
scripts/run_planning.sh --suite rotate
scripts/run_planning.sh --suite place
```

Use `scripts/run_planning.sh --help` for wrapper options. Pass native planning options after `--`, for example `scripts/run_planning.sh --suite pick -- --retries 1 --max-steps 20`. Use `python serve_slot.py --help` for model-side CEM and memory options.

## 📊 Results

The planning client writes fresh case data to `realm/outputs/`:

```bash
python runtime/summarize_planning_results.py --show-task-types
python runtime/summarize_planning_results.py \
  --json realm/outputs/planning_summary.json
```

Planning results are intentionally absent from the release. `realm/outputs/` is ignored by Git and created during evaluation.

## 🌟 Acknowledgements

- [REALM](https://github.com/martin-sedlacek/REALM) provides the OmniGibson environments, task definitions, perturbations, and simulator infrastructure.
- [V-JEPA 2](https://github.com/facebookresearch/vjepa2) provides the JEPA visual-world-model foundation.
- The [DROID dataset](https://droid-dataset.github.io/) provides downstream robot trajectories.
