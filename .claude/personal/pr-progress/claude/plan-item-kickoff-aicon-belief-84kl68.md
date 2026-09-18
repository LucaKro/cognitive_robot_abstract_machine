# grasp-likelihood-continuous (PR #8, ready for review)

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

## Status — implemented, reviewed, restacked

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

## First review round — both threads resolved

Review arrived 2026-09-18. Two threads, both from LucaKro.

**1. Threshold constant — DONE, resolved (`f345eacd`).** "no global variables
like that. if you need some default, expose the parameter and give it the
default you think is sensible." `GRIPPED_LIKELIHOOD_THRESHOLD` removed; each
site carries its own default. Side effect worth knowing: `robot_predicates.py`
and `container.py` are byte-identical to `main` again, so **this branch now
touches no existing production code** — only new modules, new exceptions, the
shared test fixture and tests. Dropped
`test_the_true_threshold_defaults_to_the_one_the_boolean_predicate_uses` (it
pinned a coupling that no longer exists; `0.9 == 0.9` would only restate a
literal). 7 node tests now, not 8.

**2. Is `trinary_logic_from_continuous` still needed? — RESOLVED, kept.** Asked
whether the helper earns its place given the observation isn't the carrier.
Answered on the thread: the observation *is* used (it's the trinary view; it's
what lets other nodes gate on this one and lets it earn a verdict) — the question
is really where the mapping lives. The author's answer was *"okay good keep it"*,
and the thread is resolved. Both alternatives offered (inline the `if_cases`, or
drop the observation) are declined; nothing to carry out.

## Restack against `main` — DONE (`76bab6d3`)

The stack maintenance pass could not integrate `main` and labelled #8
`needs-resolution`, withholding it from promotion. That, not review, was the
stall: CI was 23/23 green on `fece1ffd` and both review threads were resolved.

Conflicts were in `symbolic_math.py` and `test_symbolic_math.py`, and were the
same import-list collision in both: `main`'s #650 (`e87f86da`) deleted
`SymbolicMathNotJsonSerializableError` and every use of it, while this branch had
added `ThresholdsOutOfOrderError` to the same block. Kept the addition, dropped
the deletion. Every other hunk of both files auto-merged, and no production code
changed.

Verified against a baseline rather than an expectation: the merged tree runs the
symbolic-math suite at 12 failed / 283 passed, plain `origin/main` in the same
container at 12 failed / 276 passed, and the two failure sets are *identical*
(eleven `TestJsonSerialization` cases plus `test_jacobian_ddot` — this
container's casadi 3.8.1, green in CI on both). The difference is exactly this
branch's 7 `TestTrinaryLogicFromContinuous` tests, passing.

The krrood suite does run in this container, which earlier rounds said it could
not: `--noconftest` sidesteps the root conftest's ORM build, and casadi, numpy,
scipy, typing_extensions, sqlalchemy, ordered_set and rustworkx all come from
PyPI with `krrood/src` on the path.

## Next

- Nothing is outstanding on the branch. Both review threads resolved, no merge
  conflict, no PR comments awaiting a reply.
- Watch CI on `76bab6d3`. The merge changes no production code, so 23/23 is
  expected; if it is not, that is the restack's to answer.
- The `needs-resolution` label is still on #8. The stack pass clears it once the
  branch merges cleanly again, which it now does, so the next pass should drop it
  and let the branch rejoin promotion. Nothing to do by hand.
- `GraspLikelihoodNotBuiltError` duplicates `NodeNotBuiltError`, which is already
  on `main` and is what `ConvergingTask.error_signal` raises for this exact case
  — `estimator-node-base`'s roadmap section records that pointing this branch's
  exception at it is *this* branch's change to make. Not done here: no reviewer
  asked, and widening a conflict resolution is the wrong place for it. Worth
  folding into this item's next code push, if there is one.

Carry forward: this branch, #9 (`odometry-covariance-capture`) and #10
(`belief-context-and-gaussian`) each append their own exception class to
`giskardpy/.../motion_statechart/exceptions.py`. Independent work, but whichever
land second and third will have to resolve that file. Also: both siblings publish a `FloatVariable` too, but
neither ended up needing this helper — `PoseUncertainty` observes whether a
reading arrived, and `EstimatorNode` observes whether it measured this cycle. So
the helper's single call site is settled rather than provisional.

## Watch out

- Scope: memoryless on purpose. The recursive filter is `grasp-belief-node`
  (wave 2, depends on this). No base class here — wave 1 is "no new
  abstractions"; generalizing is `estimator-node-base`'s job.
- 100 raycasts synchronously on a 20 Hz tick is a real critical-path cost.
  Sample count stays a field; the threaded option (`ThreadedPredicateMonitor`)
  coerces with `bool()` so it would need the same treatment — deferred, not
  widened into this item.
- Do not push to the `cram2` remote. `origin` is `LucaKro/...`.
- #8 is deliberately **not** a draft. Un-drafting is the author's record of having
  reviewed it and is what makes the branch promotable; the restack changed no
  production code, so re-drafting it would have withdrawn it from the promotion
  queue for a no-op merge. Confirmed in session. Do not re-draft it unasked.
- This session was designated `claude/plan-item-resolve-aicon-belief-9dx2nz`
  (created empty at `main`). The work belongs on this branch, where #8 is —
  asked and confirmed, per the precedent #9 and #10 both set.
