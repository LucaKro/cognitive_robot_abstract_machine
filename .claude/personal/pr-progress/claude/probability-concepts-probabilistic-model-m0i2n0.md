# PR progress: `claude/probability-concepts-probabilistic-model-m0i2n0`

Plan item `probability-concepts-in-probabilistic-model` (aicon-belief-integration,
wave 2, track *belief core*). PR #11, **out of draft**, reviewer `ichumuh` requested.
Based on #10's branch `claude/belief-integration-gaussian-zi1o82`, not `main` —
re-base once #10 lands. Mode: `auto`; the settled plan and every round are in the
plan's `roadmap.md`.

## Status: restacked onto #10 and pushed (`ad9aa52b`). Stall fully cleared.

`mergeable_state` went `dirty` → `unstable`, and the `needs-resolution` label has
already cleared on its own — the stack pass drops it once the branch merges cleanly, so
#11 has rejoined promotion. CI re-running on the merge; every completed check green.

## Round 1 — "an actual datastructure instead of just np array"

Two threads, both from the author, on `Mean.values` and `Covariance.values`.

- `Mean` holds `estimates` (per quantity), `Covariance` holds `uncertainty` (per
  ordered pair). Neither holds an array.
- `as_array` / `from_array` are the only two places numpy appears on either type.
- `Covariance.from_array` keeps both directions of a pair apart — symmetrizing on the
  way in would have made the symmetry test true by construction.
- **Both threads are now resolved**, including the one this note previously recorded
  as left open (whether `ProbabilisticModel`'s abstract `log_likelihood`/`sample`
  signatures should stop taking arrays). The author answered it; nothing on the review
  side is outstanding.

## Round 2 — the restack onto #10

The stack pass could not integrate #10's own restack (`15a55dff`, which carries
`main`) and labelled #11 `needs-resolution`.

- One conflicting file: `giskardpy/src/giskardpy/motion_statechart/exceptions.py`.
  `main` inserts `NodeStateVariableNotSerializableError` at exactly the point this
  branch *removes* the three belief exceptions that moved into `probabilistic_model`,
  so git could not tell the insertion from the deletion.
- Resolved as the intersection: `main`'s exception kept, this branch's removals kept,
  `List` gone because the moved exceptions were its only users. Diffed against **both**
  parents to prove nothing else differs either way.
- **The trap worth remembering:** the `@dataclass` decorating `UnknownBeliefError` sat
  at the tail of the conflict hunk. Cutting the hunk cleanly drops it, leaving a class
  that imports fine and silently has no fields. Caught by asserting
  `dataclasses.fields()` on every class touching the hunk.

## Verified locally, against a pre-merge baseline

Every suite run twice, on `6da3fc22` and on the merge, same container:

| Suite | Pre-merge | Merged |
|---|---|---|
| `test/probabilistic_model_test` | 308 passed, 17 collection errors | identical |
| `test/giskardpy_test/test_motion_statechart` | 114 passed, 11 failed, 39 errors | identical |
| `test_beliefs.py` | 29 passed | 29 passed |
| `test/version_test` | 20 passed, 1 failed | 20 passed, 1 failed |

The motion-statechart failure sets are byte-identical across the merge (compared as
sorted lists, not counts). They are the container's — the same code pre-merge is
23/23 green in CI, which also closes the "CI has not run on `6da3fc22`" point.

Container recipe that reached the whole motion-statechart suite: PyPI wheel of
`random_events` for `random_events_lib`, each package's `src/` on `PYTHONPATH`,
`--noconftest`, plus mujoco/trimesh/plyfile/piqp/daqp/lxml/pandas/matplotlib/pydot/
inflect/lemminflect/rustworkx/sqlalchemy/ordered_set/giskardpy_bullet_bindings.

## Next

- Watch CI on `ad9aa52b`.
- Let the next stack pass promote (the label is already clear).
- After #10 lands: rebase onto `main`.

## The dashboard wall is gone

Every round since this item's kickoff recorded that the plan dashboard could not be
republished — the `Artifact` tool treated it as a third-party artifact. It no longer
does: the read returns the page source and the publish goes through, no `force`. The
dashboard is current again. The one non-obvious step: a publish that resends content an
earlier refusal rejected needs a second read of the artifact to confirm it.

## Deliberate non-actions

- **Left out of draft.** Un-drafting is this repo's record of author review; the push
  changed no production code, and re-drafting would withdraw it from the promotion
  queue for a no-op merge. Same call #8 confirmed with the author and #9 settled.
- **Did not touch the renamed exceptions** or widen the merge in any way.
