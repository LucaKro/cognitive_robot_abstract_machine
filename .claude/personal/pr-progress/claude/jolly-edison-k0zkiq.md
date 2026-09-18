# odometry-covariance-capture (PR #9)

Plan item `odometry-covariance-capture` of `aicon-belief-integration`, wave 1,
track *uncertainty plumbing*. No dependencies. Base: `main`.
Roadmap sections: the kickoff's, plus `odometry-covariance-capture - resolution`.

## Plan

`OdometrySynchronizer.apply_message` dropped all 36 numbers of
`message.pose.covariance`. Capture them, and publish the total variance as a
registered `FloatVariable` so the statechart can condition on base uncertainty.

## Done

- `PoseCovariance` + `PoseAxis` + `PoseCovarianceSource`, the two exceptions,
  `OdometrySynchronizer` implementing the source, and the `PoseUncertainty` node
  priming to infinity. Committed and pushed as `28dc12c2`.
- CI verified all 17 tests: `test_each_lib (giskardpy)` = 823 passed, 1 failed,
  and that failure is not in this diff.
- The kickoff's one flagged risk is closed: `message_type()` still returns
  `Odometry` with the second base class, confirmed by two passing tests.
- Re-ran the failed jobs once (run 35315156655) - the one red check is the known
  flaky `test_attached_self_collision_avoid_stick`. Test not touched.
- Roadmap resolution section written, PR description brought up to date.

## Next

- Nothing on the branch. Awaiting the author's own review; un-drafting is what
  records that, per the repo convention. PR stays a draft until then.
- If the re-run comes back red on the same test again, that is worth a second
  look rather than a third re-run - one re-run is the limit already spent.

## Notes

- Container limits (unchanged): `random_events` will not build here (antlr4
  wheel failure) and there is no `rclpy`, so local runs are limited to the 7
  `PoseCovariance` tests against the module source with exceptions stubbed.
- Three branches now append to `motion_statechart/exceptions.py`: this one, #8
  and #10. Whichever lands second and third resolves that file.
- No review threads, no PR comments, no tracking-issue discussion, no conflict;
  branch level with `main`.
- Subscribing to tracking issue #7 was denied by the permission classifier.
- Per personal notes, this session does not watch the PR.
