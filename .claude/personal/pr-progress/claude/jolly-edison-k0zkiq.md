# odometry-covariance-capture (PR #9)

Plan item `odometry-covariance-capture` of `aicon-belief-integration`, wave 1,
track *uncertainty plumbing*. No dependencies. Base: `main`.
Roadmap sections: kickoff, resolution, first review round (incl. the ORM break),
second review round.

## Plan

`OdometrySynchronizer.apply_message` dropped all 36 numbers of
`message.pose.covariance`. Capture them, and publish the total variance as a
registered `FloatVariable` so the statechart can condition on base uncertainty.

## Done

- Original implementation (`28dc12c2`): CI fully green, all 23 checks.
- First review round, four threads, addressed in `449abfcf`: type moved to
  `semantic_digital_twin/spatial_types/pose_covariance.py` with no ROS in it;
  `PoseWithCovarianceToSemDTConverter` added; ORM-ignored (user's call);
  `uncertainty_without_a_reading` field replaces the module constant;
  `npt.NDArray[np.float64]`.
- Merged `main` (`ab9ec2ea`); one conflict in sdt `exceptions.py`, all classes
  kept.
- `546fb2ab`: fixed the ORM regression that push caused. CI came back green on
  it, so that round is closed and all four of its threads are resolved.
- Second review round, four threads, addressed in `d8c6c10b` - below.

## Second review round (`d8c6c10b`)

Two threads named a change and are resolved; two asked a question, were answered,
and are left open for the author.

- **Dedupe (resolved).** `PoseAxis` restated `SpatialVariables`'s `x`/`y`/`z`, so
  the enum is gone. `SpatialVariables` gains `roll`/`pitch`/`yaw` plus `position`,
  `rotation` and `pose` orderings; `pose` is an ordered tuple, not a `SortedSet`,
  because a variable's place in it is its row and column. Order is unchanged, so
  the ROS row-major read still holds.
- **Datastructure (resolved).** `PoseCovariance.of` builds from pair-keyed
  entries and fills the mirror; `covariance_between`/`variance_of` read by
  variable; `values` stays numpy, as `Mean`/`Covariance` do on #10.
  `VariableNotInPoseError` added and ORM-ignored.
- **`PoseCovarianceSource` -> semdt? (open.)** Argued no: giskard's
  dependency-inversion seam, no sdt consumer or implementer.
- **Covariance on `Pose`? (open.)** A private field *does* dodge the ORM
  (`wrapped_table.py:628` skips `_`-prefixed fields) and the hand-written
  `to_json`. The real objection is propagation: poses compose and invert, a
  covariance is frame-dependent, and a field would silently survive or vanish.

Two new plan items recorded on #7: `pose-covariance-on-shared-quantities` and
`pose-uncertainty-through-transforms`.

## Next

- CI on `d8c6c10b` is running. The `PoseCovariance` tests pass locally; the
  converter, synchronizer and monitor tests need ROS packages and are CI's.
- The two open threads are the author's to close.
- The covariance frame is still unchecked, as the kickoff recorded.

## Notes

- Lesson worth keeping: adding a dataclass to sdt is never neutral, and the
  exceptions that come with it are dataclasses too. An unmappable field fails at
  *import of the generated module*, so it takes down every dependent package at
  once rather than failing locally.
- Cannot be caught here: regenerating the ORM needs the ROS message packages, so
  only CI exercises it.
- `random_events` still will not build here (antlr4 wheel). Stubbing
  `random_events.variable` down to a hashable `Continuous` and putting
  `krrood/src` + `semantic_digital_twin/src` on `PYTHONPATH` runs the sdt
  spatial-type tests against the real module source - 12 passed. Needed pip:
  numpy, casadi, scipy, sqlalchemy, rustworkx, mujoco, trimesh, ordered_set,
  pytest.
- The work stayed on this branch rather than the session's designated
  `claude/wonderful-edison-czfut6`, since the PR and its threads live here - the
  same call `belief-context-and-gaussian` made one round earlier.
- Per personal notes, this session does not watch the PR.
