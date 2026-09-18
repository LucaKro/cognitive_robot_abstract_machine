# `pose-uncertainty-through-transforms` — PR #13 (draft)

Plan item of **aicon-belief-integration**, wave 2 / belief core. Branch
`claude/plan-item-kickoff-belief-pose-j36vt3`, based on `claude/jolly-edison-k0zkiq`
(#9), re-base onto `main` once #9 lands.

## Plan

1. `PoseCovariance.transformed_by(new_reference_T_reference)` — the adjoint
   `[[R, cross_product_matrix(t) @ R], [0, R]]` over `SpatialVariables.pose`'s
   translation-then-rotation order. `HasFreeVariablesError` via `to_np` for a symbolic
   transform; no new exception.
2. `UncertainPose` in `spatial_types/uncertain_pose.py` — pose plus covariance, with
   `transformed_by` and `inverse` moving both halves. No private field on `Pose`;
   `spatial_types.py` untouched.
3. Export from `spatial_types/__init__.py`; `UncertainPose` into `generate_orm.py`'s
   `ignore_classes`.
4. Tests first, per TDD; `scripts/format_docstrings.py` on every modified file.

## Done — all of it, pushed as `a2d6dbe9`

- Branch off #9, draft PR #13 opened, manifest recorded (`in_progress`, branch, session,
  PR number), roadmap section appended.
- `PoseCovariance.transformed_by` + private `_adjoint_of` + module-level
  `_cross_product_matrix`.
- `UncertainPose` with `transformed_by` and `inverse`; `inverse` reuses the same
  propagation with the pose's own inverse rather than deriving a second formula.
- Export and ORM exclusion.
- 12 new tests. `test_spatial_types/` runs **335 passed, 1 failed**; the failure is
  `TestVector3::test_length_0`, confirmed identical on the base branch with this diff
  stashed (checked by `git stash`).
- PR description rewritten to match what landed.

### Container recipe — worth reusing, this plan's first locally-verified item

`numpy casadi scipy sortedcontainers typing_extensions rustworkx sqlalchemy trimesh
mujoco platformdirs pillow plyfile plotly psutil setuptools-scm pytest black docformatter
tqdm` from PyPI, then the way round the antlr4 wheel failure every earlier session hit:

    pip install --no-build-isolation antlr4-python3-runtime
    pip install --no-deps random_events

Run with `PYTHONPATH=krrood/src:semantic_digital_twin/src` and `--noconftest` — the root
`test/conftest.py` still needs `urdf_parser_py`, which is not on PyPI.

## Next

- Nothing outstanding on the branch. Awaiting first CI run on `a2d6dbe9`; the ORM
  regeneration and the `rclpy`/`nav_msgs` half of #9's own tests are still CI's to
  verify, since neither runs here.
- PR stays a draft awaiting its author's own review, per the repo convention that
  un-drafting is the record of having reviewed it.

## Open

- Dashboard not republished: the artifact reads as third-party-authored, so `Artifact`
  will not hand over a writable copy of the live version. User chose to skip rather than
  force-overwrite or mint a duplicate. `plan.yaml` and `roadmap.md` are correct; only the
  published page is stale.
- Frames unchecked on `transformed_by`, matching `HomogeneousTransformationMatrix.dot`.
- `inverse` inherits `HomogeneousTransformationMatrix.inverse`'s frame behaviour: the
  covariance round-trips through two inversions, the reference frame does not.
- `pose-covariance-on-shared-quantities` will also edit `pose_covariance.py`; whichever
  lands second resolves that file.
