# Release selection

## Included

- Current final model/training code for representation pretraining.
- TensorBoard-producing code for representation and robot training.
- Seven reported representation benchmarks: COCOStuff-27, ADE20K, VOC,
  NYUv2, KITTI, SUNRGBD transfer, and DAVIS.
- Six reported planar task configs and one released evaluation checkpoint per task; TMP retrieval banks are rebuilt from raw trajectories at inference time.
- Exact state/action datasets and fixed planning targets for the six reported planar tasks.
- Reach, Pick, Rotate, and Place REALM-JEPA code, configs, and final checkpoint.
- Selected final paper figures used by the top-level README.

## Excluded

- A1--A4 and other ablation configs, scripts, checkpoints, and outputs.
- Superseded model branches and checkpoints.
- VOS hyperparameter-search outputs after the final parameters were fixed.
- Exploratory analysis, inspection, and visualization scripts.
- All representation, planar-planning, and robot-control evaluation outputs;
  users generate results by running the released evaluation code.
- Training runs, intermediate checkpoints, standalone optimizer checkpoints, TensorBoard logs,
  datasets other than the curated planar package, full/validation transition caches, and
  successful-trajectory caches.
- `__pycache__`, editor state, Git metadata, PID files, and debug logs.
