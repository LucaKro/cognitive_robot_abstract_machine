# belief-context-and-gaussian — PR #10 (draft)

Plan item `belief-context-and-gaussian` of `aicon-belief-integration`, wave 2,
track *belief core*. Base `main`. Kickoff and both resolves ran in `auto` mode.
Full reasoning: the item's three roadmap sections on the personal-notes branch,
plus #10's own threads.

## State

First review round answered and pushed (`50047b27`). Waiting on CI for that
commit, and on the author's next pass.

## Done

- Belief layer, 36 tests. `81f0aded`.
- `random_events` declared as a giskardpy dependency, after CI caught the first
  direct import of it from giskardpy's own source. `8e80853a`.
- **Review round 1**, both threads the same ask — proper datastructures instead of
  bare numpy arrays, so the shape checks are unnecessary. `Quantities` now owns
  the layout and builds every array from the belief's own variables; all eight
  shape checks, the `BeliefArray` enum and `WrongBeliefShapeError` are gone, and
  `predict`/`update` take quantity-keyed mappings and `Reading`s. `50047b27`.
  Replied on both threads, including the honest downsides (per-call dict walks;
  a caller holding a real matrix has to state it as pair entries; `Quantities`
  fixes an order).
- **Structural change agreed with the user**: the multivariate Gaussian,
  covariance and `Quantities` belong in `probabilistic_model` —
  `ProbabilisticModel` is already multivariate and variable-keyed and its
  `conditional(point)` is the Kalman update in closed form. New item
  `probability-concepts-in-probabilistic-model` added to `plan.yaml`, depending
  on this one; recorded on issue #7 and answered on #10.

## Next

- Watch CI on `50047b27`, then the author's next review pass.
- Nothing else outstanding. The PR stays a draft until its author reviews it.

## Notes for whoever picks this up

- The two threads are answered but **not resolved** — thread 2 asked for the
  downsides of the change, so the author should judge whether the trade is worth
  it before either is closed.
- The kickoff's recorded decision *"one shape error, not four"* is reversed by
  round 1, and *"do not force probabilistic_model to be a filter"* is partly
  reversed by the new item. Both reversals are written up in the roadmap rather
  than left implicit in the diff.
- Test running here: `pytest --noconftest` for a fixture-free file (the version
  tests), or run the file from a copy outside `test/` (the belief tests). The
  root `test/conftest.py` regenerates the ORM interfaces at collection and needs
  ROS message packages this container has not got.
- Local environment: `pip install -U uv`, `apt-get install graphviz
  libgraphviz-dev`, then `uv sync --extra dev`.
- Nothing is being watched from here, per the standing note not to subscribe to
  PR activity.
