# `pose-uncertainty-through-transforms` — PR #13 (draft)

Plan item of **aicon-belief-integration**, wave 2 / belief core. Branch
`claude/plan-item-kickoff-belief-pose-j36vt3`, based on `claude/jolly-edison-k0zkiq`
(#9), re-base onto `main` once #9 lands.

## Where it stands

- Kickoff: `a2d6dbe9`, CI 23/23 green.
- First review round: `ae1d92d0` — `PoseDisplacementMap`,
  `SpatialVariables.row_in_pose`, `UncertainPose.dot`/`@`,
  `UncertaintyCorrelationUnknownError`. CI green on every completed check.
- Second review round: **no code change.** The one open thread was a deferral, and the
  author asked for it to become a plan item.

**Both review threads are now resolved. Nothing is outstanding on the branch.**

## What the second round did

The author on the uncertain-on-uncertain thread: *"Please make this a seperate plan item
in case i want to get back to this, but for now we should be able to continue without
right? or will the choice significantly change how we move forward?"*

1. **New plan item `uncertain-pose-composition`** — wave 2, belief core,
   `depends_on: [pose-uncertainty-through-transforms]`. In `plan.yaml`, in `roadmap.md`,
   and broadcast on issue #7 per the plan's structural-change convention.
2. **Answered the question: continuing without it is safe.** Nothing else in the plan
   composes two uncertain poses, and — the load-bearing reason — the refusal is a
   strictly narrower contract than any answer, so no working caller can exist today for
   a later implementation to break. Caveat recorded: if the eventual model needs
   `UncertainPose` to *carry* how its uncertainty relates to others rather than assuming
   independence at the call, that changes the type's shape (though still nothing built
   in the meantime).
3. **Scope check run rather than assumed.** `uncertain_pose.py` is absent from `main`
   and introduced by this branch, which usually argues for folding; on
   `scope-decision.md`'s own test it does not, because the work stands on its own once
   this item lands rather than existing to correct what the parent is about to ship.

## Verification

Unchanged from the first round: `test_spatial_types/` — **344 passed, 1 failed**, the
failure being `TestVector3::test_length_0`, confirmed identical with the diff stashed.

### Container recipe

PyPI: `numpy casadi scipy sortedcontainers typing_extensions rustworkx sqlalchemy
trimesh mujoco platformdirs pillow plyfile plotly psutil setuptools-scm lxml daqp piqp
giskardpy_bullet_bindings pytest black docformatter tqdm`.

Round the antlr4 wheel failure every earlier session hit:

    pip install --no-build-isolation antlr4-python3-runtime
    pip install --no-deps random_events

`urdf_parser_py` and `xacro` fail the same way; per #12's finding, copy the package
directory out of their sdists onto `site-packages`. Then run with
`PYTHONPATH=krrood/src:semantic_digital_twin/src:giskardpy/src` and `--noconftest`.

Two `test_joint_state.py` tests error here on a CasADi API mismatch
(`FunctionBuffer_set_res`) — this container's casadi 3.8.1, reproduces with the diff
stashed, not ours.

## Next

- Nothing to push. Three CI jobs on `ae1d92d0` were still running at the end of this
  round — `test_each_lib (giskardpy)`, `test_each_lib (coraplex)` and the
  `coraplex_real_tracy` demo; everything completed is green.
- PR stays a draft awaiting the author's own review.

## Open

- `PoseCovariance.values` stays a bare array per #9's second round and #11's
  re-confirmation; offered on the resolved thread to bring it forward if wanted.
- **Dashboard still not republished**, third round running — the `Artifact` tool treats
  this account's plan dashboard as third-party, so publish refuses without `force`. User
  chose to skip at kickoff; that choice stands. It now also means
  `uncertain-pose-composition` is in `plan.yaml` and on issue #7 but not on the
  published page.
- Tracking-issue subscription still refused by this session's permission mode; issue #7
  read directly instead.
