# `pose-covariance-on-shared-quantities` - PR #15 (draft)

Plan `aicon-belief-integration`, track *Belief core*. Base is
`claude/plan-item-kickoff-belief-pose-j36vt3` (#13), with #11's branch merged in -
`Quantities` lives only on #11's stack and `pose_covariance.py` only on #13's, and this
item needs both.

## The plan

Collapse `semantic_digital_twin`'s local rebuild of the quantity layout onto
`probabilistic_model.Quantities`, now that #11 has made it importable.

1. `SpatialVariables.pose` becomes a `Quantities` over the six degrees of freedom;
   `row_in_pose` is removed rather than reimplemented beside it.
2. `PoseCovariance` holds the uncertainty of each ordered pair of degrees of freedom
   with `of` / `from_array` / `as_array`, the shape `Covariance` took on #11, instead of
   `values: npt.NDArray[np.float64]`. Its `of()` symmetric fill becomes
   `Quantities.symmetric_matrix`.
3. `PoseDisplacementMap.as_array` builds through the layout instead of a comprehension.
4. `VariableNotInPoseError` goes; `Quantities.index_of` raises
   `VariableNotInQuantitiesError`. Drop it from `exceptions.py` and from
   `generate_orm.py`'s `ignore_classes`.
5. `PoseWithCovarianceToSemDTConverter` moves to `PoseCovariance.from_array` - the one
   caller that moves.

Tests first, per TDD: the new behaviour (reading by quantity, `from_array`/`as_array`
round trip keeping both directions of a pair apart, the rejection naming quantities)
before the refactor. Existing assertions about the ROS row-major read and the adjoint
identity must keep passing unchanged - they are what says behaviour did not move.

## Decided at kickoff

- **Base #13, not #9.** #13 already moved `_row_of` to `SpatialVariables.row_in_pose`
  for exactly this item, and added `PoseDisplacementMap` as the second place to collapse.
  `pose-uncertainty-through-transforms` added to `depends_on`; broadcast on issue #7.
- **Drop the bare array.** #11, #13 and #9 each recorded this item as where matching
  `Covariance`'s shape costs least. Taken up, with the user's answer.
- **`Mean`/`Covariance` are not reused.** #11 records them as the parameters a Gaussian
  is written in; a pose covariance is not a Gaussian. `Quantities` is the shared part.

## Done

- Branch created off #13 with #11 merged in cleanly; draft PR #15 opened.
- `plan.yaml` (status, branch, PR, session, the new dependency, corrected notes) and
  `roadmap.md` (the kickoff section) saved to the personal-notes branch.

## Next

- Comment the dependency change on issue #7.
- Republish the dashboard.
- Implement steps 1-5 above, tests first.

## Watch out for

- `semantic_digital_twin` already declares `probabilistic_model` in both `[project]
  dependencies` and `[dependency-groups] workspace`, so
  `test_imported_workspace_members_are_declared` needs no change here - checked, unlike
  on #10 where it caught an undeclared import.
- Removing `VariableNotInPoseError` touches `semantic_digital_twin/exceptions.py`, which
  every restack on this plan has found to be a shared append point that conflicts
  against `main`.
- `Quantities` has no `__getitem__` and no `.index`; three test sites call
  `SpatialVariables.pose.index(...)` and move to `index_of`.
