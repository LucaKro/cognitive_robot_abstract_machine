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

- [x] Branch, draft PR #8, manifest (`open`), roadmap section (`record`)
- [ ] Step 1 — trinary helper
- [ ] Step 2 — threshold constant
- [ ] Step 3 — GraspLikelihood node
- [ ] Step 4 — shared fixture
- [ ] Step 5 — tests, formatting, push

## Watch out

- Scope: memoryless on purpose. The recursive filter is `grasp-belief-node`
  (wave 2, depends on this). No base class here — wave 1 is "no new
  abstractions"; generalizing is `estimator-node-base`'s job.
- 100 raycasts synchronously on a 20 Hz tick is a real critical-path cost.
  Sample count stays a field; the threaded option (`ThreadedPredicateMonitor`)
  coerces with `bool()` so it would need the same treatment — deferred, not
  widened into this item.
- Do not push to the `cram2` remote. `origin` is `LucaKro/...`.
