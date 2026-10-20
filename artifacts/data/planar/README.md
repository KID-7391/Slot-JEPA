# Planar Planning Data

This upload-ready package contains the released state/action datasets and fixed test targets for the six planar tasks. These task splits were generated and curated for Slot-JEPA; they are not downloaded from the DINO-WM data release.

| Directory | Task | Contents |
| --- | --- | --- |
| `TwoRoom/` | fixed two-room navigation | training/validation trajectories and 50 fixed planning targets |
| `Maze/` | fixed maze navigation | training trajectories and 50 fixed planning targets |
| `PushT/` | T-shaped object manipulation | uniform-density training/validation trajectories and 50 fixed planning targets |
| `TwoRoomR/` | randomized two-room layouts | train/eval manifests and 50 fixed planning targets |
| `MazeR/` | randomized maze layouts | train/eval manifests and 50 fixed planning targets |
| `PushObj/` | unseen object shapes | train/eval manifests, ID validation manifest, and 50 fixed planning targets |

Every task uses the same public layout:

```text
<Task>/
├── ... training and validation data ...
└── targets/
    └── test.pkl
```

Each `targets/test.pkl` stores the fixed benchmark source and target as
`state_0` and `state_g`. Random-layout or random-shape tasks additionally store
the corresponding `env_info` and deterministic evaluation seeds. These are raw
physical endpoints, not Slot-JEPA features: evaluation restores them in the
released simulator, renders source/target RGB observations, and lets the
selected downloaded model compute its own representations. This keeps the same
50 cases usable by other world models and planning methods.

Images are rendered from the stored physical states during training and evaluation. Pre-rendered videos, slot caches, and generated evaluation outputs are intentionally excluded.

TMP retrieval examples are likewise derived from the released raw training
trajectories. Slot-JEPA builds source/target mask logits into a local cache on
the first inference run; those model-specific logits are not dataset content.

PushT uses the same uniform-area-density rigid-body construction as PushObj: mass is proportional to polygon area, the center of mass is area weighted, and inertia is computed about that center. Its trajectories were rebuilt by replaying the source initial states and pusher commands in the released simulator; the historical unequal-part-mass dynamics are not included.

`PushT/normalization_stats.pt` is computed from valid frames of the rebuilt training split and is shared by training, validation, and planning targets. Relative pusher commands are preserved; absolute commands are re-derived from the rebuilt pusher states.

The workflow configs resolve this package directly from `artifacts/data/planar/`. If the data is hosted separately, extract the archive back to this directory or override the relevant dataset environment variable.

After downloading, verify the package from this directory with `sha256sum -c SHA256SUMS`.
