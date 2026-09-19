# belief-drawer-experiment — PR #17 (draft), base #16

Plan item `belief-drawer-experiment` of `aicon-belief-integration`. The
measurement the whole plan turns on.

## Plan

1. `experiments/src/experiments/belief_drawer_experiment/drawer_scenario.py`
   — `DrawerWorld.of(robot_world, cabinet_yaw)` adding a `Drawer` (case body,
   `Handle`, `Slider`) to a supplied robot world; `ArmConfiguration`;
   `DrawerCondition` (`UNCONDITIONAL_GRIP`, `BELIEVED_GRIP`, `FAILING_GRIP`);
   `ScriptedLikelihood`, a `GraspLikelihoodSource` publishing the share of
   hits the condition dictates; the statechart each condition builds
   (reach the handle, then `Open`, with collision avoidance on).
2. `.../sweep.py` — `DrawerRun`, `DrawerRunOutcome` (mechanism travel, arm
   travel, grip offset, contact, cycles, goals reached), aggregation into
   `DrawerConditionResult(ExperimentResult)`, `DrawerSweep.execute()` /
   `render_figure()` / `write_manifest`, and an argparse CLI — following
   `control_loop_experiments/benchmark.py`.
3. `test/experiments_test/test_belief_drawer_experiment.py` — tests first.
   Fast: what each condition builds, the scripted likelihood, the
   aggregation against hand-built outcomes, the sweep grid, a timed-out run.
   `@pytest.mark.slow`: one real run per condition.

## Decisions already recorded (roadmap section is pushed)

- `is_body_in_gripper` deduplicates its hits, so it can only answer `0` or
  `1/sample_size`. The likelihood is scripted; the defect is flagged for a
  separate bug PR off `main`, not fixed here.
- Three conditions, not four: stock `Open` reads no likelihood, so it is the
  baseline for both belief conditions at once.
- The mechanism DOF is commanded directly, so the outcome that separates the
  conditions is whether the arm is dragged, not whether the drawer opens.
- Contact comes from the world's collision detector; there is no physics
  engine in this loop.

## Done

- Branch off #16, empty bootstrap commit, draft PR #17.
- `plan.yaml` item flipped to `in_progress` with branch/session/PR; roadmap
  section appended.

## Next

- Write the tests, then the two modules.
- Republish the dashboard (`/plan-dashboard aicon-belief-integration`).
- Report the `is_body_in_gripper` defect to the user; consider a plan item
  for it and for scaling `mechanism_weight` by the belief.
