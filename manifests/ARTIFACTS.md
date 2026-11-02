# Retained artifacts

Approximate on-disk sizes after release cleanup:

| Path | Size | Archive | Contents |
| --- | ---: | --- | --- |
| `artifacts/checkpoints/pretrain` | 1.2 GB | `pretrain.zip` | Slot-JEPA pretraining checkpoint and minimal loading config required by planar evaluation |
| `artifacts/checkpoints/downstream/planar` | 678 MB | `planar.zip` | Six planar checkpoints and loading configs |
| `artifacts/checkpoints/downstream/robot` | 573 MB | `robot.zip` | DROID-fine-tuned robot checkpoint and loading config |
| `artifacts/checkpoints/downstream/visual_understanding` | 42 MB | `visual_understanding.zip` | Three released monocular-depth heads |
| `artifacts/color_channel_weights` | 8 KB | `color_channel_weights.zip` | Frozen 512px-calibrated channel weights shared by all six planar tasks |
| `artifacts/data/planar` | 215 MB | `planar_data.zip` | Six Slot-JEPA planar training/validation datasets and fixed test targets |
| `artifacts/data/robot` | 553 MB | `robot_data.zip` | Minimal manifests, compact target poses, and three-view goal images for all 108 REALM-JEPA cases |

No frame-slot caches, retrieval banks, validation banks, full robot rollout
traces, training-run directories, TensorBoard logs, or generated evaluation outputs are bundled.
Planar inference rebuilds its retrieval bank from the raw released trajectories
when needed. The planar checkpoints reuse the shared pretraining checkpoint for
the frozen student and slot extractor.
