# odometry-covariance-capture (PR #9)

Plan item `odometry-covariance-capture` of `aicon-belief-integration`, wave 1,
track *uncertainty plumbing*. No dependencies. Base: `main`.
Roadmap section: `.claude/personal/plans/aicon-belief-integration/roadmap.md`.

## Plan

`OdometrySynchronizer.apply_message` dropped all 36 numbers of
`message.pose.covariance`. Capture them, and publish the total variance as a
registered `FloatVariable` so the statechart can condition on base uncertainty.

## Done

- Branch opened, draft PR #9, manifest `in_progress`, roadmap section written.
- `PoseCovariance` + `PoseAxis` + `PoseCovarianceSource` in
  `motion_statechart/pose_covariance.py`.
- `CovarianceNotSixBySixError` and `PoseUncertaintyNotBuiltError` appended to
  `motion_statechart/exceptions.py`.
- `OdometrySynchronizer` implements `PoseCovarianceSource`, keeping the
  covariance of the message it applied.
- `PoseUncertainty` node in `monitors/uncertainty_monitors.py`, priming to
  infinity so an unread covariance never reads as certainty.
- 17 tests across three files. Implementation committed and pushed; PR
  description updated to match.

## Next

- Nothing outstanding on the branch. Waiting on CI.
- If CI is red, the first suspect is `OdometrySynchronizer`'s second base class
  interacting with `SubClassSafeGeneric.get_generic_type_parameters` - i.e.
  whether `message_type()` still returns `Odometry`. Reasoned through the
  `__orig_bases__` walk (non-parameterized bases are skipped) but could not
  execute it.

## Notes

- Container limits: `random_events` will not build (antlr4 wheel failure), and
  there is no `rclpy`. Installed numpy/casadi/scipy/pytest etc. and verified the
  7 `PoseCovariance` tests against the real module source with the exceptions
  module stubbed; everything else is CI-verified only.
- Overlaps #8 on `motion_statechart/exceptions.py` only (each appends its own
  read-before-built exception). Deliberately does not use #8's
  `trinary_logic_from_continuous` - no declared dependency on it.
- Subscribing to tracking issue #7 was denied by the permission classifier.
- Per personal notes, this session does not watch the PR.
