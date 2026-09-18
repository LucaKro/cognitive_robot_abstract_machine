# belief-context-and-gaussian — PR #10 (draft)

Plan item `belief-context-and-gaussian` of `aicon-belief-integration`, wave 2,
track *belief core*. Base `main`. Kickoff and both resolves ran in `auto` mode.
Full reasoning: the item's roadmap sections on the personal-notes branch, plus
#10's own threads.

## State

Review round 1 is closed — both threads resolved, nothing outstanding on the
branch. CI green on `50047b27`; the doc-note commit `822b3e59` is the only thing
not yet CI-verified, and it touches a docstring only.

## Done

- Belief layer, 36 tests. `81f0aded`.
- `random_events` declared as a giskardpy dependency, after CI caught the first
  direct import of it from giskardpy's own source. `8e80853a`.
- **Round 1, both threads: proper datastructures instead of bare numpy arrays.**
  `Quantities` owns the layout and builds every array from the belief's own
  variables; all eight shape checks, the `BeliefArray` enum and
  `WrongBeliefShapeError` are gone; `predict`/`update` take quantity-keyed
  mappings and `Reading`s. `50047b27`. Thread 1 resolved by the author.
- **Thread 2 closed**: author chose to keep the datastructures and asked for a
  reminder that the pure-numpy option remains. `..note::` on `Quantities` in
  `822b3e59`, replied and resolved.
- **Structural change agreed with the user**: the multivariate Gaussian,
  covariance and `Quantities` belong in `probabilistic_model`. New item
  `probability-concepts-in-probabilistic-model` in `plan.yaml`, depending on this
  one; recorded on issue #7 and answered on #10.

## Next

- Nothing outstanding. Waiting on the author's next review pass.
- The PR stays a draft until its author reviews it.

## Notes for whoever picks this up

- The `..note::` on `Quantities` is a deliberate exception to `AGENTS.md`'s rule
  against documenting a design that was not chosen — the author asked for it, and
  it describes a live fallback with a stated trigger rather than history. Do not
  "clean it up".
- Two kickoff decisions are reversed and both reversals are in the roadmap rather
  than implicit in the diff: *"one shape error, not four"* (round 1) and *"do not
  force probabilistic_model to be a filter"* (partly, via the new item).
- Test running here: `pytest --noconftest` for a fixture-free file (the version
  tests), or run the file from a copy outside `test/` (the belief tests). The
  root `test/conftest.py` regenerates the ORM interfaces at collection and needs
  ROS message packages this container has not got.
- Local environment: `pip install -U uv`, `apt-get install graphviz
  libgraphviz-dev`, then `uv sync --extra dev`.
- The dashboard was not republished after the round-1-close roadmap append; the
  only stale part is that paragraph inside its collapsed roadmap section.
- Nothing is being watched from here, per the standing note not to subscribe to
  PR activity.
