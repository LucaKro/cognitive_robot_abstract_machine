# `belief-weighted-open-goal` — PR #16 (draft)

Plan item of **AICON Belief Integration**. Branch `claude/adoring-bell-hilm79`,
based on #14 (`claude/plan-item-kickoff-belief-integration-6uj1wv`), which is where
`GraspBelief` lives. Tracking issue #7.

## Plan

Make `Open`'s hold-handle weight follow the grasp belief, opt-in, stock behaviour
unchanged.

1. `Task.constraint_weight` — a property returning `self.weight`, read by
   `CartesianPosition` (`cartesian_tasks.py:181`) and `CartesianOrientation`
   (`cartesian_tasks.py:626`). The field stays a `float`; retyping it to
   `sm.ScalarData` would drop the `weight` ORM column from every task DAO.
2. A belief-weighted Cartesian pose in `motion_statechart/beliefs/` — holds the
   `GraspBelief`, declares it in `prerequisite_nodes`, overrides
   `constraint_weight` as `self.weight * belief.probability`. Because
   `CartesianPose` is a `Parallel` goal, it also expands into belief-weighted
   position and orientation leaves.
3. An optional field on `Open`, unset by default; when set, the hold-handle child
   is the belief-weighted pose.
4. Tests first, in `test_other_tasks.py` beside the existing `Open` weight tests,
   reusing the `prismatic_bot` fixture and the `_expanded_nodes` helper.

## Why this shape

The wall is timing before typing: `Open.expand` runs in `_expand_goals`, which
completes before any `build_artifacts`, and `GraspBelief.probability` does not
exist until the belief builds. So the weight has to resolve at the task's own
build, which is what the property plus `prerequisite_nodes` buys.
`NotApproachingGoal` (`progress_monitors.py:75`) is the precedent.

## Done

- Read `roadmap.md` in full; dependency #14 reports `open_ready`.
- Established the QP re-evaluates `quadratic_weights` every cycle with
  `float_variables` among its parameters, so a variable-scaled weight is live.
- Seam breadth settled with the user: only the two Cartesian leaves, since `Open`
  builds just three constraint-adding tasks and only the grip's two carry
  `grasp_weight`. Remainder tracked as `task-weights-through-the-constraint-seam`,
  broadcast on #7.
- Branch, draft PR #16, manifest fields and roadmap section recorded.

## Next

- Write the failing tests, then the three production changes.
- Try the `test_motion_statechart` suite locally with #14's recorded container
  recipe; `test/giskardpy_test/conftest.py` imports `rclpy` through
  `GiskardTester`, which is what has blocked the fixture-using tests on this plan.
- Republish the dashboard.

## Open, carried into the PR description and roadmap

- Scaling the grip alone may not make the robot back off: `JointPositionList`
  constrains the environment connection's own position variable, so the solver can
  open the drawer directly while a slack grip leaves the arm uncoupled. Flagged for
  `belief-drawer-experiment`; scaling `mechanism_weight` too is outside this item's
  recorded scope.
- No floor under the scaled weight — a ruled-out grasp reaches
  `DefaultWeights.WEIGHT_MINIMUM`. A floor would be a tuned constant with nothing
  to cite.
