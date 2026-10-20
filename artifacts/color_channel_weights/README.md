# Planar color-channel weights

`planar_block12_important_objects_v2.pt` is the single released 768-dimensional
channel-weight vector for the final Slot-JEPA encoder block. It is calibrated at
512 px and shared unchanged by all six planar tasks, including 384 px Maze and
TwoRoom inputs.

The global palette contains all ten important colors: the TwoRoom red ball and
black wall; Maze green ball and orange obstacle; RoyalBlue pusher; and Push
block colors DarkTurquoise, Orange, Aquamarine3, DarkViolet, and
LightSlateGray. Scores use variance between all color means divided by
within-color nuisance variance; there is no paired or task-specific objective.

Every color uses the exact same four seeded irregular random masks, so the
calibration encodes no ball, wall, pusher, or block shape prior. White versus
PointMaze-like backgrounds and two brightness levels produce 160 probes. The
released weighting uses gamma 1.5 and bounds `[0.15, 6.0]`. At runtime the model
restores every weighted patch token's original L2 norm.

All release training and planning configs use mask softmax temperature 1.0 and
mask sharpening gamma 2.0. Changing either the weight version or mask gamma
requires rebuilding slot caches and fixed slot banks.

SHA-256:

```text
c0d1aa5b39a3c59cac351821ae431806b34dc06595fa39560bbaba4516e142ec
```
