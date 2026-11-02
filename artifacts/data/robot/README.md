# Robot Planning Targets

[`robot_data.zip`](https://drive.google.com/file/d/1x6LCzMLCUGpfqL9bC_i-G-Ms88rSZKUR/view?usp=sharing)
contains the minimal visual-goal package required by the REALM-JEPA planner for
all 108 released benchmark cases. Extract it from the Slot-JEPA release root so
that its contents are restored under `artifacts/data/robot/`.

Archive SHA256: `6973b7bcf47e4a0ced2385e1328a5fcc2effb6f8d1c3014a86fa3f9db1c2e208`

Each case directory contains only:

- `manifest.json`: task, scene, camera, and goal-stage metadata.
- `goal_targets.pt`: compact target poses and stage metadata. Large per-step
  observation records have been removed.
- `<target>_base.png`, `<target>_second.png`, and `<target>_wrist.png`: the
  synchronized goal images used by the three-view planner.

Full successful rollouts, intermediate states, controller traces, and
collection diagnostics are intentionally excluded.

The robot workflow resolves this directory automatically. Inside the REALM
container it is mounted at `/artifacts/data/robot/successful_case`. Set
`SLOT_JEPA_SUCCESS_CASE_ROOT` only to test a different target package.
