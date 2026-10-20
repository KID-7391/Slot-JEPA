# Retained artifacts

Approximate on-disk sizes after release cleanup:

| Path | Size | Contents |
| --- | ---: | --- |
| `artifacts/checkpoints/pretrain` | 1.2 GB | Slot-JEPA pretraining checkpoint and minimal loading config required by planar evaluation |
| `artifacts/checkpoints/downstream` | 1.3 GB | Robot checkpoint/config under `robot/`, six planar checkpoints/configs, three depth heads, and checksums |
| `artifacts/color_channel_weights` | 8 KB | Frozen 512px-calibrated channel weights shared by all six planar tasks |
| `artifacts/data/planar` | 220 MB | Six Slot-JEPA planar training/validation datasets and fixed test targets |
| `artifacts/data/robot` | 550 MB | Minimal manifests, compact target poses, and three-view goal images for all 108 REALM-JEPA cases |

No frame-slot caches, retrieval banks, validation banks, full robot rollout
traces, training-run directories, TensorBoard logs, or generated evaluation outputs are bundled.
Planar inference rebuilds its retrieval bank from the raw released trajectories
when needed. The planar checkpoints reuse the shared pretraining checkpoint for
the frozen student and slot extractor.
