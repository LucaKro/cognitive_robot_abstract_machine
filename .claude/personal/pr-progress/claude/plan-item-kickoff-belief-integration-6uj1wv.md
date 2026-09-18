# grasp-belief-node (PR #14, plan aicon-belief-integration)

First concrete estimator: a recursive grasp belief over GraspLikelihood's
measurement. Draft PR #14, base `claude/plan-item-kickoff-aicon-0t046w` (#12),
with `claude/plan-item-kickoff-aicon-belief-84kl68` (#8) merged in as a second
parent. Design, implementation notes and the first review round are in the plan's
`roadmap.md` under this item, in three sections.

## Done

- Branch off #12 with #8 merged in clean; draft PR #14; `plan.yaml` item
  `in_progress` with branch, session and pull request recorded.
- `58b8cd66` - the whole item: `beliefs/grasp.py` (`GraspBelief`,
  `SampledLikelihood`), `grasp_likelihood_source.py`, `CertainPriorError`, the
  one-line base class on `GraspLikelihood`, and 23 tests.
- 23 tests pass, each confirmed load-bearing by six mutations; motion-statechart
  suite 223 passed against a 200-passed baseline with an identical 115-item
  failure/error set; `test/version_test` 20 passed.
- **Review round 1**: one thread, on `trinary_logic_from_continuous` - *"i also
  feel like we already have this here: #8"*. It is #8, appearing in this diff
  because #8 is a second parent not in the base. Verified (`git diff <#8> 58b8cd66
  -- krrood/` empty; my commit touches no krrood file; `GraspBelief` imports the
  helper rather than copying it), replied on the thread, **left unresolved** - it
  carries a question back (would the author rather review against a different
  base?). No code pushed.
- **ORM open point closed by checking rather than reasoning**:
  `GraspLikelihoodSource` is scanned (not in the excluded `beliefs/` package) but
  has `dataclasses.fields() == []`, so the unmappable-field failure this repo fears
  cannot occur. Left out of `ignore_classes` to match #9's `PoseCovarianceSource`.
- Local `regenerate_all_orm.py` fails at `CouldNotResolveType: MetaData` in a
  `semantic_digital_twin` exception - reproduced identically in a detached worktree
  at the base branch, so it is the container, not this diff.
- PR description rewritten: the two-parent caveat is now a reviewing note at the
  top, and the ORM point reflects what was checked.

## Next

- Nothing outstanding on this branch. `test_each_lib (giskardpy)` came back green on
  `58b8cd66`, which confirms the ORM point; 21 of 23 checks green, the two pending
  (`semantic_digital_twin`, coraplex) cover no file this diff touches.
- The pull request stays a draft until its author has reviewed it. No push was made
  this round, so nothing needed re-drafting.

## Open for the author

- Whether to restructure this branch's base so the diff does not carry #8 (asked on
  the review thread, thread left open).
- Whether `GraspLikelihoodSource` *and* #9's `PoseCovarianceSource` should both be
  in `generate_orm.py`'s `ignore_classes`. Consistency is the only argument either
  way; both map to empty tables.
- The dashboard, still unpublishable: the cached artifact `WCfARob6AeALaMcNBCwmm8`
  does not appear in this account's `Artifact` listing under `mine` or `shared`, so
  the publish precondition cannot be met. `force: true` would discard whatever #15's
  session published in parallel. Overwrite, or mint a fresh dashboard and repoint
  the cache?

## Container recipe that worked here

PyPI wheel of `random_events` (for `random_events_lib`) plus the checkout's
`random_events/src` on `PYTHONPATH`; `urdf_parser_py` from its sdist's package
directory and `xacro` from its wheel copied onto site-packages; every workspace
package's `src/` on `PYTHONPATH` (the repo root too, for
`cognitive_robot_abstract_machine.orm_interfaces`); `--noconftest`.
