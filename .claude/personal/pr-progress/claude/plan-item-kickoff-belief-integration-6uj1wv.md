# grasp-belief-node (PR #14, plan aicon-belief-integration)

First concrete estimator: a recursive grasp belief over GraspLikelihood's
measurement. Draft PR #14, base `claude/plan-item-kickoff-aicon-0t046w` (#12),
with `claude/plan-item-kickoff-aicon-belief-84kl68` (#8) merged in as a second
parent. Settled design is in the plan's `roadmap.md` under this item.

## Plan

1. `motion_statechart/grasp_likelihood_source.py` - `GraspLikelihoodSource`, the
   abstraction the node reads its measurement through (published variable plus
   ray count). `GraspLikelihood` takes it as a second base class, no body change.
2. `motion_statechart/beliefs/grasp.py` - `GraspBelief(EstimatorNode)`. Quantity
   is the log-odds. Measurement: Haldane-Anscombe corrected fraction, variance
   from the ray count. Prediction: decay toward the prior while the gripper is
   open, half-life in seconds converted with `control_dt`. Publishes a third
   variable holding the probability, for `belief-weighted-open-goal`. Observation
   is `trinary_logic_from_continuous` over that probability, via an `on_tick`
   override.
3. `test/giskardpy_test/test_motion_statechart/test_grasp_belief.py` - driven by a
   recorded source mimic, no world and no raycast. Tests first, per TDD.

## Done

- Branch created off #12, #8 merged in clean (`git merge-tree` checked first), pushed.
- Draft PR #14 opened. `plan.yaml` item flipped to `in_progress` with branch,
  session and PR recorded; roadmap section appended.

## Next

- Write the failing tests, then the two modules.
- Verify locally: the container recipe recorded by #11's and #12's rounds
  (PyPI wheel of `random_events` for `random_events_lib`, each package's `src/` on
  `PYTHONPATH`, `--noconftest`) reaches `test_motion_statechart`.
- Update PR #14's description with what actually landed, keep it a draft.
- Republish the dashboard: `/plan-dashboard aicon-belief-integration`.

## Open

- Tracking-issue subscription to #7 was refused by this session's permission mode,
  as on every earlier round on this plan. Read issue #7 directly if structural
  changes matter.
