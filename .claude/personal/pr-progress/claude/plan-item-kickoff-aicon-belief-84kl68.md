# grasp-likelihood-continuous (PR #8, draft)

Plan item `grasp-likelihood-continuous` of `aicon-belief-integration`, wave 1,
track *uncertainty plumbing*. Tracking mailbox: issue #7. Base: `main`.
No dependencies — this is the first work item of the plan, so there is no
landed sibling to copy from.

## Goal

Give the continuous grasp likelihood `is_body_in_gripper` already computes a
path into giskard's motion statechart, without disturbing the boolean
`is_body_gripped` that EQL preconditions use.

## The key design call

A continuous value must NOT be a node's observation. Four sites compare against
the trinary constants exactly and break on anything else:
`ObservationState.__getitem__` (motion_statechart.py:288), `goal_reached_state`
(graph_node.py:1284), `_create_verdict` (graph_node.py:926, falls through to
INTERRUPTED), and `is_true()`/`is_unknown()` in symbolic_math.py:1035, which
makes `_create_condition_holds` read a continuous 0.97 as false.

Carrier is instead a `FloatVariable` registered in
`context.float_variable_data` and written per tick. The observation updater
already compiles against those variables (motion_statechart.py:332), so
constraints and transitions can read it symbolically while the observation
stays trinary. Precedent: `WiggleInsert` (wiggle_insert.py:202, :250).

This corrects the item's own `notes`, which claimed the float was "type-
compatible with a giskard observation today" and flagged only `is_unknown()`.

## Plan

1. `trinary_logic_from_continuous` in krrood `symbolic_math`, joining the
   `trinary_logic_not/_and/_or/_to_str` family. Tests first.
2. Named constant for the gripped threshold — the bare `0.9` appears in both
   `robot_predicates.is_body_gripped` and coraplex `container.py:120`.
3. `GraspLikelihood` monitor, new module
   `giskardpy/.../motion_statechart/monitors/grasp_monitors.py`. Registers the
   variable in `build_artifacts`, writes it in `on_tick`, observation derived
   via the helper from step 1. Tests first.
4. Shared fixture for "box between the gripper fingers" — currently inline in
   `test_predicates.py::test_is_body_in_gripper`; move to `test/conftest.py`
   (where `pr2_world_copy` lives) so the new giskard test reuses it.
5. pytest the touched suites, `scripts/format_docstrings.py` on modified files,
   push, keep the PR description matching.

## Status

All steps implemented and pushed as commit `b5c4a811`. PR #8 still draft.

- [x] Branch, draft PR #8, manifest (`open`), roadmap section (`record`)
- [x] Step 1 — `trinary_logic_from_continuous` + `ThresholdsOutOfOrderError`
- [x] Step 2 — `GRIPPED_LIKELIHOOD_THRESHOLD`, used in sdt and coraplex
- [x] Step 3 — `GraspLikelihood` in `monitors/grasp_monitors.py`
- [x] Step 4 — `body_between_fingers` fixture in `test/conftest.py`,
      `test_predicates.py::test_is_body_in_gripper` refactored onto it
- [x] Step 5 — formatted, committed, pushed, PR description updated

## Verification state — RESOLVED, fully green

**All 23 checks pass on `b5c4a811`.** The verification gap the kickoff left open
is closed.

- krrood symbolic-math tests: run locally (285 passed; 3 pre-existing failures in
  that file, confirmed identical on a stashed clean tree) and green in CI.
- giskardpy + semantic_digital_twin tests: could not run locally (no
  numpy/casadi/trimesh; `uv sync` fails on the repo pyproject needing a newer uv
  than 0.8.17 plus absent graphviz headers) — but `test_each_lib (giskardpy)`
  and `test_each_lib (semantic_digital_twin)` have both since passed in CI.
  That covers the 8 node tests, the new root-conftest fixture and the refactored
  `test_is_body_in_gripper`.

The three calls made by reading the code rather than running it are confirmed by
that run: monitor-only statechart compiles/ticks with no constraints or DOFs;
`on_start` priming keeps the first observation off an unset variable; the
observation stays one of the three trinary constants.

## Design calls made (recorded so they aren't re-litigated)

- `false_below` is a required field; only `true_above` gets a default, because
  `GRIPPED_LIKELIHOOD_THRESHOLD` is established and nothing in the codebase
  says where "definitely not held" sits.
- Node measures in `on_start` as well as `on_tick`: `tick()` evaluates the
  observation expression *before* running nodes, so priming avoids reading an
  unset variable on the first observation.
- A monitor-only statechart compiles fine — `_compile_qp_controller` returns
  early when there are no constraints, so no EndMotion or DOF is needed.

## Next

Nothing outstanding. A `/plan-item-resolve` pass found no blocker: CI fully
green, no review threads, no PR comments, no tracking-issue discussion, no
merge conflict, branch level with `main`, never promoted upstream (no
`in-review` label) so there is no upstream review to read.

The PR is a draft waiting on its author's own review — un-drafting is this
repo's record of having done that review, and only the user does it. Nothing
for a session to do here until a review comment or a CI failure arrives.

Carry forward: this branch, #9 (`odometry-covariance-capture`) and #10
(`belief-context-and-gaussian`) each append their own exception class to
`giskardpy/.../motion_statechart/exceptions.py`. Independent work, but whichever
land second and third will have to resolve that file.

## Watch out

- Scope: memoryless on purpose. The recursive filter is `grasp-belief-node`
  (wave 2, depends on this). No base class here — wave 1 is "no new
  abstractions"; generalizing is `estimator-node-base`'s job.
- 100 raycasts synchronously on a 20 Hz tick is a real critical-path cost.
  Sample count stays a field; the threaded option (`ThreadedPredicateMonitor`)
  coerces with `bool()` so it would need the same treatment — deferred, not
  widened into this item.
- Do not push to the `cram2` remote. `origin` is `LucaKro/...`.
