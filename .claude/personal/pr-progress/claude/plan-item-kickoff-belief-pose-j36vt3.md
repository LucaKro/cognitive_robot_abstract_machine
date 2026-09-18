# `pose-uncertainty-through-transforms` — PR #13 (draft)

Plan item of **aicon-belief-integration**, wave 2 / belief core. Branch
`claude/plan-item-kickoff-belief-pose-j36vt3`, based on `claude/jolly-edison-k0zkiq`
(#9), re-base onto `main` once #9 lands.

## Plan

1. `PoseCovariance.transformed_by(new_reference_T_reference)` — re-express a covariance
   in another frame by the adjoint `[[R, skew(t) R], [0, R]]` of the transform, over
   `SpatialVariables.pose`'s translation-then-rotation order. Raises
   `HasFreeVariablesError` (krrood's, via `to_np`) for a transform whose numbers are not
   known; no new exception.
2. `UncertainPose` in a new `spatial_types/uncertain_pose.py` — a `Pose` with its
   `PoseCovariance`, with `transformed_by` and `inverse` moving both halves. No private
   field on `Pose`; `spatial_types.py` untouched.
3. Export both from `spatial_types/__init__.py`; add `UncertainPose` to
   `generate_orm.py`'s `ignore_classes`.
4. Tests first, per TDD:
   - `test_pose_covariance.py` gains a propagation section: a pure rotation rotates it;
     a translation turns yaw uncertainty into position uncertainty at the square of the
     lever arm; the rotation block is untouched by a translation; identity is a no-op;
     transforming twice equals transforming by the composed transform; a symbolic
     transform is rejected.
   - The adjoint identity itself, checked exactly:
     `T @ hat(perturbation) @ inverse(T) == hat(adjoint @ perturbation)`. This is what
     tests the model rather than the formula the code is written in.
   - New `test_uncertain_pose.py`: both halves move together, the result's reference
     frame follows the transform, inverse uses the pose's own inverse, two inversions
     round-trip the covariance, a symbolic pose is rejected.
5. `scripts/format_docstrings.py` on every modified file.

## Done

- Branch created off #9, empty bootstrap commit pushed, draft PR #13 opened.
- Manifest recorded: `in_progress`, branch/session/PR number, roadmap section appended.
- Container brought up far enough to run the spatial types locally — a first for this
  plan. `numpy casadi scipy sortedcontainers typing_extensions rustworkx sqlalchemy
  trimesh mujoco pytest` from PyPI, plus `pip install --no-build-isolation
  antlr4-python3-runtime` then `pip install --no-deps random_events`, which is the way
  round the antlr4 wheel failure every earlier session on this plan hit. Run with
  `PYTHONPATH=krrood/src:semantic_digital_twin/src`; the root `test/conftest.py` still
  needs `urdf_parser_py`, so run test files from a copy outside `test/` or with
  `--noconftest`.

## Next

- Write the failing tests, then the two production changes, then re-run.
- Republish `/plan-dashboard aicon-belief-integration` (done at kickoff; refresh again if
  status changes).

## Open

- Frames are not checked on `transformed_by`, matching `HomogeneousTransformationMatrix.dot`.
- `pose-covariance-on-shared-quantities` will also edit `pose_covariance.py`; whichever
  lands second resolves that file.
