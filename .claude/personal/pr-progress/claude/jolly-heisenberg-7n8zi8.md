# `pose-covariance-on-shared-quantities` - PR #15 (draft)

Plan `aicon-belief-integration`, track *Belief core*. Base is
`claude/plan-item-kickoff-belief-pose-j36vt3` (#13), with #11's branch merged in -
`Quantities` lives only on #11's stack and `pose_covariance.py` only on #13's, and this
item needs both.

## Done - the whole item is implemented and pushed (`1781fca6`)

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

## Resolution round - the one red check (`1781fca6`)

CI on `1b61ff7d` came back 22 of 23 green. The one failure was
`test_each_lib (semantic_digital_twin)`: **1 failed, 1764 passed, 52 skipped**, and it
was this branch's.

`test_convert_pose_covariance_reads_the_entries_row_by_row` raised
`AttributeError: 'Quantities' object has no attribute 'index'`. The rename was one site
short - its forward assertion used `pose.index_of`, and the transposed read on the next
line still used the tuple's `.index`, twice on one line. The item's own record says
three test sites moved; there was a fourth.

The local run could not have caught it: that test is under `test_ros/` and imports
`geometry_msgs.msg`, a ROS package with no PyPI distribution at all (checked, not
assumed), while the local verification ran `test_spatial_types`. The cheap guard for a
mechanical rename is a whole-branch grep for the removed name - `git grep 'pose\.index('`
reports the one line - rather than a local run.

Changing the test is normally the wrong move, and `AGENTS.md` says so. This is the case
that rule does not describe: the production code is right and the test called a method
that no longer exists, so it raised before asserting anything. What it asserts is
unchanged, and that was checked - the transposed read is still row `yaw`, column `x`,
which is 30 of a counting-up matrix, the same number #9's second review round recorded
for this layout. Both directions were run against the real `PoseCovariance.from_array`
here (5 and 30) and still differ, so a transposed `as_array` is still visible.

## Next

- **Read CI on `1781fca6`.** It was still running when this round closed: 3 green
  (`to-lowercase`, `check_generated_orm_interfaces_are_untracked`,
  `test_claude_dev_tooling`), 20 in flight including
  `test_each_lib (semantic_digital_twin)`, which is the one that matters.
- The converter, monitor and synchronizer tests stay CI's, as does regenerating the ORM.
- PR stays a draft until its author has reviewed it, per this repo's convention.
- The two-parent shape will draw a reviewer comment, as it did on #14. The answer is the
  empty `git diff <other parent> <head> -- <path>`, not the description's heading.

## Container recipe that worked here

PyPI wheel of `random_events` for `random_events_lib`, each package's `src/` on
`PYTHONPATH` (no editable install), `--noconftest`, plus numpy, scipy, casadi, mujoco,
trimesh, plyfile, sqlalchemy, ordered_set, rustworkx, sortedcontainers, platformdirs,
psutil, lxml, pandas, matplotlib, pydot, inflect, lemminflect, plotly, tqdm, piqp, daqp,
giskardpy_bullet_bindings. `pip install black docformatter` for
`scripts/format_docstrings.py`.
