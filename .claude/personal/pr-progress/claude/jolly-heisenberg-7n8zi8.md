# `pose-covariance-on-shared-quantities` - PR #15 (draft)

Plan `aicon-belief-integration`, track *Belief core*. Base is
`claude/plan-item-kickoff-belief-pose-j36vt3` (#13), with #11's branch merged in -
`Quantities` lives only on #11's stack and `pose_covariance.py` only on #13's, and this
item needs both.

## Done - the whole item is implemented and pushed (`1b61ff7d`)

1. `SpatialVariables.pose` is a `Quantities` over the six degrees of freedom;
   `row_in_pose` deleted rather than reimplemented over it.
2. `PoseCovariance` holds the uncertainty of each ordered pair with
   `of` / `from_array` / `as_array`, instead of `values: npt.NDArray[np.float64]`.
   Its symmetric fill is `Quantities.symmetric_matrix`.
3. `PoseDisplacementMap.as_array` builds through the same layout (and is a property
   now, matching `Covariance.as_array` - one spelling per operation in the module).
4. `VariableNotInPoseError` removed from `exceptions.py` and `generate_orm.py`'s
   `ignore_classes`; the layout raises `VariableNotInQuantitiesError`.
5. `PoseWithCovarianceToSemDTConverter` moved onto `from_array` - the one caller.

Tests first: the five new tests and the four now naming the shared error were red
before the change. Every new assertion was then confirmed load-bearing by mutating the
implementation; the transpose mutation initially survived, so the round-trip test now
uses an asymmetric covariance.

## Verification

- sdt spatial types: 346 passed / 1 failed, against a stashed baseline of 342 / 1 - the
  same pre-existing `TestVector3::test_length_0` (this container's casadi 3.8.1).
- 109 pass across `test_quantities.py`, `test_multivariate_gaussian.py` and #10's
  `test_beliefs.py`, so the merged-in half costs nothing.
- `test/version_test`: 20 passed / 1 failed (missing local `coraplex`, pre-existing).
  `test_imported_workspace_members_are_declared[semantic_digital_twin]` passes - sdt
  already declared `probabilistic_model`.
- Cost measured, since `PoseUncertainty.on_tick` reads it per cycle: `total_variance`
  55 us -> 121 us, 0.13% of a 50 ms tick.

## Next

- Watch CI on `1b61ff7d`: the converter, monitor and synchronizer tests are CI's
  (`test/giskardpy_test/conftest.py` imports `rclpy`), as is regenerating the ORM.
- PR stays a draft until its author has reviewed it, per this repo's convention.

## Container recipe that worked here

PyPI wheel of `random_events` for `random_events_lib`, each package's `src/` on
`PYTHONPATH` (no editable install), `--noconftest`, plus numpy, scipy, casadi, mujoco,
trimesh, plyfile, sqlalchemy, ordered_set, rustworkx, sortedcontainers, platformdirs,
psutil, lxml, pandas, matplotlib, pydot, inflect, lemminflect, plotly, tqdm, piqp, daqp,
giskardpy_bullet_bindings. `pip install black docformatter` for
`scripts/format_docstrings.py`.
