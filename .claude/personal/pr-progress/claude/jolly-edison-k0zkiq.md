# odometry-covariance-capture (PR #9)

Plan item `odometry-covariance-capture` of `aicon-belief-integration`, wave 1,
track *uncertainty plumbing*. No dependencies. Base: `main`.
Roadmap sections: kickoff, resolution, first review round (incl. the ORM break).

## Plan

`OdometrySynchronizer.apply_message` dropped all 36 numbers of
`message.pose.covariance`. Capture them, and publish the total variance as a
registered `FloatVariable` so the statechart can condition on base uncertainty.

## Done

- Original implementation (`28dc12c2`): CI fully green, all 23 checks.
- Review round, four threads, addressed in `449abfcf`: type moved to
  `semantic_digital_twin/spatial_types/pose_covariance.py` with no ROS in it;
  `PoseWithCovarianceToSemDTConverter` added; ORM-ignored (user's call);
  `uncertainty_without_a_reading` field replaces the module constant;
  `npt.NDArray[np.float64]`. Replied to all four, resolved three.
- Merged `main` (`ab9ec2ea`); one conflict in sdt `exceptions.py`, all classes
  kept.
- `546fb2ab`: fixed the ORM regression that push caused - see below.

## The ORM regression (fixed, awaiting CI)

CI on `ab9ec2ea` was 10 of 23 red, one root cause:
`MappedAnnotationError` importing sdt's generated `ormatic_interface.py`.
Moving `PoseCovariance` into sdt moved `PoseCovarianceNotSixBySixError` with it,
and sdt ORM-maps every dataclass. `given_shape: tuple` has no column type, so
the generated DAO would not import and every package reaching sdt's ORM died.

The ORM-ignore decision covered the type but not its exception. Both are in
`ignore_classes` now; fields typed `tuple[int, ...]`.

## Next

- CI on `546fb2ab` is pending. Expect all 23 green; if anything is still red it
  is a *different* cause, since only one DAO was at fault.
- One review thread still open on purpose: the ORM-ignore call, which carries a
  question back to the author. Worth mentioning to them that the answer now also
  covers the exception.

## Notes

- Lesson worth keeping: adding a dataclass to sdt is never neutral, and the
  exceptions that come with it are dataclasses too. An unmappable field fails at
  *import of the generated module*, so it takes down every dependent package at
  once rather than failing locally.
- Cannot be caught here: regenerating the ORM needs the ROS message packages, so
  only CI exercises it. Local runs remain limited to the 8 `PoseCovariance`
  cases against the module source with sdt exceptions stubbed.
- Per personal notes, this session does not watch the PR.
