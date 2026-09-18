# belief-context-and-gaussian — PR #10 (ready for review)

Plan item `belief-context-and-gaussian` of `aicon-belief-integration`, wave 2,
track *belief core*. Base `main`. Kickoff and every resolve so far ran in `auto`
mode. Full reasoning: the item's roadmap sections on the personal-notes branch,
plus #10's own threads.

## State

Review round 2 is closed — all four threads on the pull request are resolved. CI
was 23 of 23 green on `bbee7915`. The branch then went stale: `main` moved on and
the stack maintenance routine hit a conflict integrating it, labelled the branch
`needs-resolution` and skipped it. That conflict is resolved and pushed as
`15a55dff`; CI on it has not been read yet.

The pull request is **out of draft** with `ichumuh` requested as reviewer — the
author un-drafted it, which in this repo's stack workflow is the record of having
reviewed it and the signal to promote upstream. The user was asked during this
round and chose to leave it ready rather than re-draft it after the push.

## Done

- Belief layer, 36 tests. `81f0aded`.
- `random_events` declared as a giskardpy dependency, after CI caught the first
  direct import of it from giskardpy's own source. `8e80853a`.
- **Round 1, both threads: proper datastructures instead of bare numpy arrays.**
  `Quantities` owns the layout and builds every array from the belief's own
  variables; all eight shape checks, the `BeliefArray` enum and
  `WrongBeliefShapeError` are gone; `predict`/`update` take quantity-keyed
  mappings and `Reading`s. `50047b27`, plus the `..note::` the author asked for in
  `822b3e59`.
- **Structural change agreed with the user**: the multivariate Gaussian,
  covariance and `Quantities` belong in `probabilistic_model`. New item
  `probability-concepts-in-probabilistic-model` in `plan.yaml`, depending on this
  one; recorded on issue #7 and answered on #10.
- **Round 2, both threads, `bbee7915`.** `Mean` and `Covariance` are types of
  their own, each carrying the `Quantities` it is laid out by and answering by
  quantity (`estimate_of`, `variance_of`, `between`) — round 1 had made the
  *inputs* quantity-keyed and left what the belief *holds* as bare arrays. Both
  `InitVar`s and the construction in `__post_init__` are gone; `GaussianBelief.of`
  is the builder. `BeliefQuantitiesDisagreeError` rejects an estimate and an
  uncertainty laid out by different quantities. 44 tests, run locally.
- **Restacked against `main`, `15a55dff`.** The one conflicting file was
  `giskardpy/src/giskardpy/motion_statechart/exceptions.py`: `main` gained
  `NodeStateVariableNotSerializableError` (from upstream #650, symbolic-math JSON
  serialization) while this branch appends its five belief exceptions, both at the
  end of the same file. Resolved by keeping both, with the union of the imports
  each side needs. No belief code changed.

## Next

- Read CI on `15a55dff`. The local run covers the belief tests, the
  dependency-declaration tests and krrood's symbolic math; the ROS-dependent half
  of the matrix and the ORM regeneration are still CI's to confirm.
- The `needs-resolution` label clears itself once the routine's next pass sees the
  branch merging cleanly again. If it is still there after a pass, check why.
- Waiting on `ichumuh`'s review.

## Notes for whoever picks this up

- **This branch is not the session's designated branch**, for the third round
  running. Round 2 ran from a session designated
  `claude/belief-integration-gaussian-8yaw8t`; this round from one designated
  `claude/belief-integration-gaussian-9youx0`. Both times the user was asked and
  chose to keep the work on `claude/belief-integration-gaussian-zi1o82`, where #10
  and its threads are. Neither designated branch was ever created.
- **Do not re-draft this pull request.** It is out of draft deliberately;
  `.claude/stack/stack.toml` reads the draft toggle as the approve-for-upstream
  signal, so re-drafting would silently withdraw it from the promotion queue. The
  standing "always convert back to draft after pushing" note was put to the user
  this round and they chose to leave it ready.
- The `..note::` on `Quantities` is a deliberate exception to `AGENTS.md`'s rule
  against documenting a design that was not chosen — the author asked for it, and
  it describes a live fallback with a stated trigger rather than history. Do not
  "clean it up". Round 2 is why it earns its place: the arithmetic under `Mean`
  and `Covariance` is still plain numpy.
- Two kickoff decisions are reversed and both reversals are in the roadmap rather
  than implicit in the diff: *"one shape error, not four"* (round 1) and *"do not
  force probabilistic_model to be a filter"* (partly, via the new item).
- `__post_init__` on `GaussianBelief` is validation only. The author's objection
  was to `InitVar`, not to `__post_init__`; the reply on that thread offers to
  move the check into `of` if they want the method gone entirely.
- **The container recipe has changed** and is easier than the one round 2
  recorded. `random_events` builds a C++ extension (`random_events_lib`) that does
  not build here; the workspace source fails with
  `module 'random_events_lib' has no attribute 'reals'`. Installing the **PyPI
  wheel** of `random_events` supplies a matching `random_events_lib`, and the
  workspace source then imports against it, so the real workspace code still runs.
  After that: `pip install numpy scipy casadi sqlalchemy rustworkx pytest mujoco
  trimesh ordered_set platformdirs pillow plyfile psutil lxml matplotlib pydot
  pandas inflect lemminflect plotly tqdm networkx anytree giskardpy_bullet_bindings
  piqp daqp`, and put each package's `src/` on `PYTHONPATH` rather than installing
  the workspace editable (the editable install of some members fails on its build
  dependencies here). The root `test/conftest.py` still needs `urdf_parser_py`, a
  ROS package not on PyPI, so run a test file from a copy outside `test/`, or
  `--noconftest` for a fixture-free one.
- **Two failures in this container are not the code's**, and both were proven so
  by running them on `origin/main` alone rather than assumed:
  `test_all_package_versions_match_root_version` (this container has no installed
  `coraplex`, so it has no `__version__`), and
  `TestExpression::test_jacobian_dot`/`test_jacobian_ddot` (a casadi API mismatch,
  `NotImplementedError` out of `casadi.py`). The latter is the same one #13's
  session recorded.
- **The dashboard cannot be republished from a session on this account.** The
  artifact the URL cache names (`WCfARob6AeALaMcNBCwmm8`) was published by a
  different account, does not appear in this one's artifact listing, and refuses a
  publish. The user chose to leave it stale rather than create a second dashboard
  for one plan; that choice stands for this round too.
- Nothing is being watched from here, per the standing note not to subscribe to
  PR activity. Subscribing to tracking issue #7 was attempted at the start of this
  round and denied by the permission classifier, which matches that note.
