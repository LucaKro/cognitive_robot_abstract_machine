# belief-context-and-gaussian — PR #10 (draft)

Plan item `belief-context-and-gaussian` of `aicon-belief-integration`, wave 2,
track *belief core*. Base `main`. Kickoff and every resolve so far ran in `auto`
mode. Full reasoning: the item's roadmap sections on the personal-notes branch,
plus #10's own threads.

## State

Review round 2 is open: two unresolved threads, both on
`beliefs/gaussian.py`, both posted after the last push (`822b3e59`) and so not
yet addressed. Nothing else is blocking — CI is 23/23 green on `822b3e59`,
`mergeable_state` is clean, there is no merge conflict, and the branch is level
with `main`.

## Plan for round 2

1. **`InitVar` out, classmethod in** (thread on `gaussian.py:191`, *"hmm i dont
   like initvars at all. use classmethods instead if you want this kind of
   initialization"*). `GaussianBelief`'s `estimates`/`uncertainty` `InitVar`s and
   its `__post_init__` construction go; `GaussianBelief.of(quantities, estimates,
   uncertainty)` becomes the builder, next to the existing `of_one_variable`.
   `of` is already this module's word for a classmethod builder (`Quantities.of`).
2. **The belief's own fields become datastructures** (thread on
   `gaussian.py:202`, *"can these become actual datastructures instead of just
   numpy arrays?"*). Round 1 made the *inputs* quantity-keyed but left `mean` and
   `covariance` as bare `npt.NDArray[np.float64]`. They become `Mean` and
   `Covariance`, each carrying the `Quantities` it is laid out by and answering by
   quantity — `Mean.estimate_of`, `Covariance.variance_of`, `Covariance.between`.
   That follows the sibling `PoseCovariance` (#9), which the reviewer's *"we
   talked about this in another of the open draft PRs"* points at: a named type
   owning the array and reading it by name.
3. **One consistency check replaces what the types cannot make impossible.** With
   `Mean` and `Covariance` each carrying their own `Quantities`, the one way left
   to build a nonsense belief is to hand it two that disagree.
   `BeliefQuantitiesDisagreeError`, raised from `__post_init__` — validation
   there, which is what `PoseCovariance` does too, is not what the reviewer
   objected to; `InitVar` was.
4. **Tests first**, then the change: the new interface's tests are written and
   seen failing before `gaussian.py` moves.

Not in scope: `predict`'s `transition`/`process_noise`/`offset` and
`Reading.contributions` are already quantity-keyed mappings, and
`Quantities.vector`/`matrix`/`symmetric_matrix` stay the internal array builders
— the recorded note that the arithmetic underneath is plain numpy stays true.

No ORM change is needed: `generate_orm.py` already excludes the whole
`beliefs` package, so `Mean` and `Covariance` are covered. The new exception
lands in `motion_statechart/exceptions.py` beside the four belief exceptions
already mapped there and has the same field shapes, so it carries none of the
bare-`tuple` hazard that broke the sdt ORM in #9.

## Done

- Belief layer, 36 tests. `81f0aded`.
- `random_events` declared as a giskardpy dependency, after CI caught the first
  direct import of it from giskardpy's own source. `8e80853a`.
- **Round 1, both threads: proper datastructures instead of bare numpy arrays.**
  `Quantities` owns the layout and builds every array from the belief's own
  variables; all eight shape checks, the `BeliefArray` enum and
  `WrongBeliefShapeError` are gone; `predict`/`update` take quantity-keyed
  mappings and `Reading`s. `50047b27`. Both threads resolved by the author, the
  second after the `..note::` on `Quantities` in `822b3e59`.
- **Structural change agreed with the user**: the multivariate Gaussian,
  covariance and `Quantities` belong in `probabilistic_model`. New item
  `probability-concepts-in-probabilistic-model` in `plan.yaml`, depending on this
  one; recorded on issue #7 and answered on #10.

## Next

- Implement the four steps above, reply to both threads, resolve them, push.
- The PR stays a draft until its author reviews it.

## Notes for whoever picks this up

- **This branch is not the session's designated branch.** This resolve ran from a
  session designated `claude/belief-integration-gaussian-8yaw8t`; the user was
  asked and chose to keep the work on `claude/belief-integration-gaussian-zi1o82`,
  where #10 and its review threads are. `8yaw8t` is unused and has no commits.
- The `..note::` on `Quantities` is a deliberate exception to `AGENTS.md`'s rule
  against documenting a design that was not chosen — the author asked for it, and
  it describes a live fallback with a stated trigger rather than history. Do not
  "clean it up".
- Two kickoff decisions are reversed and both reversals are in the roadmap rather
  than implicit in the diff: *"one shape error, not four"* (round 1) and *"do not
  force probabilistic_model to be a filter"* (partly, via the new item).
- **The belief tests do run in this container now**, unlike in every earlier
  session on this item. What it takes: `pip install numpy sqlalchemy casadi scipy
  rustworkx mujoco trimesh ordered_set platformdirs pillow plyfile psutil lxml
  pygments piqp daqp objgraph pytest`, then `pip install --no-deps -e` each of
  `./random_events ./krrood ./semantic_digital_twin ./giskardpy`. The root
  `test/conftest.py` still cannot be collected — it needs `urdf_parser_py`, which
  is a ROS package and not on PyPI — so run the test file from a copy outside
  `test/`, or `--noconftest` for a fixture-free file.
- Nothing is being watched from here, per the standing note not to subscribe to
  PR activity. Subscribing to tracking issue #7 was attempted at the start of
  this resolve and denied by the permission classifier, which matches that note.
