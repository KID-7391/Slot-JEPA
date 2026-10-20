# Robot Planning Targets

`successful_case/` contains the minimal visual-goal package required by the
REALM-JEPA planner for all 108 released benchmark cases.

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
