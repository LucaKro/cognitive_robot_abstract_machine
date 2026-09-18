# `pose-uncertainty-through-transforms` — PR #13 (draft)

Plan item of **aicon-belief-integration**, wave 2 / belief core. Branch
`claude/plan-item-kickoff-belief-pose-j36vt3`, based on `claude/jolly-edison-k0zkiq`
(#9), re-base onto `main` once #9 lands.

## Where it stands

Kickoff landed as `a2d6dbe9`; CI came back **23 of 23 green** on it, which closed the
one gap the kickoff flagged — the ORM exclusion and the `rclpy`/`nav_msgs` half were
verified by `test_each_lib (semantic_digital_twin)` rather than only reasoned about.

First review round resolved as `ae1d92d0`. Two threads, both from the author.

## What the round changed

1. **`PoseDisplacementMap`** replaces the kickoff's `_adjoint_of` and the module-level
   `_cross_product_matrix`. Holds `factors: Mapping[PoseVariablePair, float]`;
   `of_transform` builds it, `as_array` is the one numpy boundary. (Thread 1, resolved.)
2. **`SpatialVariables.row_in_pose`** replaces `PoseCovariance._row_of`, so the two
   types share one lookup and `pose-covariance-on-shared-quantities` has one place to
   collapse instead of two.
3. **`UncertainPose.dot` / `@`** — the gap thread 2 found. Extending an uncertain pose
   by a certain transform carries the covariance *unchanged*, which is the
   bottle-in-an-uncertain-drawer case and is a different operation from
   `transformed_by`. Exact, no assumption.
4. **`UncertaintyCorrelationUnknownError`** — `uncertain @ uncertain` refuses rather
   than assuming independence. In `generate_orm.py`'s `ignore_classes` with its
   siblings.

## Verification

`test_spatial_types/` — **344 passed, 1 failed**, the failure being
`TestVector3::test_length_0`, confirmed identical with the diff stashed. The kickoff's
perturbation-identity test passes unchanged across the refactor, which is what says the
propagation still means what it meant.

### Container recipe — now reaches `test_datastructures` too

PyPI: `numpy casadi scipy sortedcontainers typing_extensions rustworkx sqlalchemy
trimesh mujoco platformdirs pillow plyfile plotly psutil setuptools-scm lxml daqp piqp
giskardpy_bullet_bindings pytest black docformatter tqdm`.

Round the antlr4 wheel failure every earlier session hit:

    pip install --no-build-isolation antlr4-python3-runtime
    pip install --no-deps random_events

`urdf_parser_py` and `xacro` fail the same way; per #12's finding, copy the package
directory out of their sdists onto `site-packages`. Then run with
`PYTHONPATH=krrood/src:semantic_digital_twin/src:giskardpy/src` and `--noconftest`.

Two `test_joint_state.py` tests still error here on a CasADi API mismatch
(`FunctionBuffer_set_res`) — this container's casadi 3.8.1, reproduces with the diff
stashed, not ours.

## Next

- Nothing to push. Awaiting CI on `ae1d92d0`.
- PR is back in draft, awaiting the author's own review.

## Open

- **Thread 2 left open on purpose**: asked whether to add the independent
  uncertain-on-uncertain composition, with the first-order formula on the thread. The
  author's call.
- `PoseCovariance.values` stays a bare array per #9's second round and #11's
  re-confirmation; offered on thread 1 to bring it forward if wanted.
- **Dashboard still not republished** — the `Artifact` tool treats this account's plan
  dashboard as third-party, so publish refuses without `force`. User chose to skip at
  kickoff; that choice stands. `plan.yaml` and `roadmap.md` are current.
- Tracking-issue subscription refused by this session's permission mode; issue #7's
  comments were read directly instead, nothing there touches this item.
