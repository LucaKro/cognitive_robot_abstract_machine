# odometry-covariance-capture (PR #9)

Plan item `odometry-covariance-capture` of `aicon-belief-integration`, wave 1,
track *uncertainty plumbing*. No dependencies. Base: `main`.
Roadmap sections: the kickoff's, the resolution, and the first review round.

## Plan

`OdometrySynchronizer.apply_message` dropped all 36 numbers of
`message.pose.covariance`. Capture them, and publish the total variance as a
registered `FloatVariable` so the statechart can condition on base uncertainty.

## Done

- Original implementation (`28dc12c2`): `PoseCovariance`/`PoseAxis`,
  `PoseCovarianceSource`, `OdometrySynchronizer` implementing it, the
  `PoseUncertainty` node. CI fully green on that commit, all 23 checks.
- First review round, four threads, all addressed in `449abfcf`:
  - type moved to `semantic_digital_twin/spatial_types/pose_covariance.py`
    with no ROS in it; `from_row_major` deleted;
  - `PoseWithCovarianceToSemDTConverter` added beside the other ROS converters
    and used by the synchronizer;
  - `PoseCovariance` added to sdt `generate_orm.py` `ignore_classes` (user's
    call, asked before doing it);
  - `UNCERTAINTY_WITHOUT_A_READING` is now the node field
    `uncertainty_without_a_reading`, default infinity;
  - `values: npt.NDArray[np.float64]`.
- Merged `main` (arrived on the branch as `b67533f7`); one conflict in sdt
  `exceptions.py` where both sides appended classes - all kept. Merge `ab9ec2ea`.
- Replied to all four threads; resolved three.

## Next

- CI on `ab9ec2ea` is pending - the review fixes are not verified yet. The move
  touches sdt, so `test_each_lib (semantic_digital_twin)` matters as much as
  giskardpy this time.
- One thread left open on purpose: the ORM-ignore call. It carries a question
  back to the author, so per the notes it is not resolved.

## Notes

- Container limits (unchanged): `random_events` will not build here (antlr4
  wheel failure), no `rclpy`. Verified the 8 `PoseCovariance`/reshape cases
  against the real module source with sdt exceptions stubbed; everything else is
  CI-only.
- The exceptions-file overlap with #8/#10 is smaller now: this branch's
  `CovarianceNotSixBySixError` left giskardpy's `motion_statechart/exceptions.py`
  for sdt, so only `PoseUncertaintyNotBuiltError` remains there.
- Per personal notes, this session does not watch the PR.
