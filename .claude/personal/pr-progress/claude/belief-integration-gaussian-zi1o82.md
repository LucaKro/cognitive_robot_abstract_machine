# belief-context-and-gaussian — PR #10 (draft)

Plan item `belief-context-and-gaussian` of `aicon-belief-integration`, wave 2,
track *belief core*. Base `main`, no dependencies. Kickoff ran in `auto` mode.
Roadmap section with the full reasoning is on the personal-notes branch.

## Plan

1. Exceptions in `giskardpy/motion_statechart/exceptions.py`:
   `WrongBeliefShapeError`, `VariableNotInBeliefError`, `UnknownBeliefError`,
   `DuplicateBeliefError`.
2. `giskardpy/motion_statechart/beliefs/gaussian.py`: `BeliefArray` enum,
   `Measurement`, `GaussianBelief` with `predict` and `update` (Joseph form),
   dimensions named by `random_events` `Continuous` variables.
3. `giskardpy/motion_statechart/beliefs/context.py`: `BeliefContext`
   (`ContextExtension`) keyed by those variables, `add` / `require`.
4. Tests in `test/giskardpy_test/test_motion_statechart/test_beliefs.py`, written
   first, with hand-computed exact posteriors.
5. `scripts/format_docstrings.py` on everything touched, then push.

## Done

- Branch opened, draft PR #10 created, manifest + roadmap recorded.
- Local environment brought up (`uv sync --extra dev` after upgrading uv, plus
  system graphviz), so unlike the previous item's session this one can actually
  run the new tests.

## Next

- Write the tests, then the three modules, then run them.
- Keep the PR description matching what the diff does before the final push.

## Notes for whoever picks this up

- The root `test/conftest.py` regenerates the ORM interfaces at collection and
  that needs ROS message packages, which this container has not got. Run the new
  test file from a copy outside `test/` so no conftest is loaded; the belief
  tests need no fixtures.
- `subscribe_pr_activity` on tracking issue #7 was denied by the permission
  classifier this session. That matches the standing note not to subscribe to PR
  activity, so nothing is being watched here.
