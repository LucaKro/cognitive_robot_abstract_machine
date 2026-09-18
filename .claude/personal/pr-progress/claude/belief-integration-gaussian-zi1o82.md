# belief-context-and-gaussian — PR #10 (draft)

Plan item `belief-context-and-gaussian` of `aicon-belief-integration`, wave 2,
track *belief core*. Base `main`. Kickoff and every resolve so far ran in `auto`
mode. Full reasoning: the item's roadmap sections on the personal-notes branch,
plus #10's own threads.

## State

Review round 2 is closed — all four threads on the pull request are resolved and
nothing is outstanding on the branch. `bbee7915` is pushed; CI on it has not been
read yet. The pull request stays a draft until its author reviews it.

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

## Next

- Nothing outstanding. Read CI on `bbee7915` when it finishes; the local run
  covers the belief tests and the version tests, not the rest of the matrix.
- Waiting on the author's review. The PR stays a draft until then.

## Notes for whoever picks this up

- **This branch is not the session's designated branch.** The round-2 resolve ran
  from a session designated `claude/belief-integration-gaussian-8yaw8t`; the user
  was asked and chose to keep the work on
  `claude/belief-integration-gaussian-zi1o82`, where #10 and its threads are.
  `8yaw8t` was never created and has no commits.
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
- **The belief tests do run in this container**, unlike in every earlier session on
  this item. What it takes: `pip install numpy sqlalchemy casadi scipy rustworkx
  mujoco trimesh ordered_set platformdirs pillow plyfile psutil lxml pygments piqp
  daqp objgraph plotly black docformatter tqdm pytest`, then `pip install
  --no-deps -e` each of `./random_events ./krrood ./semantic_digital_twin
  ./giskardpy ./probabilistic_model`. The root `test/conftest.py` still cannot be
  collected — it needs `urdf_parser_py`, a ROS package that is not on PyPI and
  does not build from source here — so run the test file from a copy outside
  `test/`, or `--noconftest` for a fixture-free file. ORM regeneration is blocked
  by the same package, so the ORM check stays CI's to confirm.
- Nothing is being watched from here, per the standing note not to subscribe to
  PR activity. Subscribing to tracking issue #7 was attempted at the start of the
  round-2 resolve and denied by the permission classifier, which matches that note.
