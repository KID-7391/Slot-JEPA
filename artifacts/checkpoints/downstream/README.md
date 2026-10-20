# Downstream Checkpoints

This directory is the upload-ready downstream weight package.

| Task group | Released weights |
| --- | --- |
| Robot control and shared visual encoder | Checkpoint, config, and usage notes under `robot/` |
| Planar planning | Predictor checkpoints and resolved configs under `planar/outputs/` |
| Visual understanding | Depth heads for NYUv2, KITTI, and SUN RGB-D transfer under `visual_understanding/` |

The six planar checkpoints reuse the frozen student and slot extractor from `../pretrain/slot_jepa_pretrained.pt` at runtime. Retrieval banks are derived locally from the released raw trajectories when absent; precomputed mask caches are not part of the release. Generated planning metrics, videos, and per-case outputs are also excluded.

Linear segmentation and DAVIS evaluation do not produce independent fine-tuned checkpoints in this codebase. Verify downloaded files with:

```bash
sha256sum -c SHA256SUMS
```
