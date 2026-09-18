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

## Verification state — read this before trusting the branch

- krrood symbolic-math tests: **run and passing** in a minimal venv
  (numpy/casadi/sqlalchemy only), 285 passed. 3 failures in that file
  (`test_jacobian_dot`, `test_jacobian_ddot`,
  `test_error_holding_a_variable_is_not_json_serializable`) are pre-existing —
  confirmed identical on a stashed clean tree.
- giskardpy + semantic_digital_twin tests: **not executed**. This container has
  no numpy/casadi/trimesh installed, and `uv sync` fails (repo pyproject needs a
  newer uv than 0.8.17, and pygraphviz needs system graphviz headers that
  aren't present). The 8 node tests and the refactored predicate test are
  CI-verified only. First CI run deserves a close look.

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

Nothing outstanding on this item beyond CI. If CI is red, the likely suspects
are the new giskard test's statechart setup, or the `body_between_fingers`
fixture's placement in the root conftest.

## Watch out

- Scope: memoryless on purpose. The recursive filter is `grasp-belief-node`
  (wave 2, depends on this). No base class here — wave 1 is "no new
  abstractions"; generalizing is `estimator-node-base`'s job.
- 100 raycasts synchronously on a 20 Hz tick is a real critical-path cost.
  Sample count stays a field; the threaded option (`ThreadedPredicateMonitor`)
  coerces with `bool()` so it would need the same treatment — deferred, not
  widened into this item.
- Do not push to the `cram2` remote. `origin` is `LucaKro/...`.
