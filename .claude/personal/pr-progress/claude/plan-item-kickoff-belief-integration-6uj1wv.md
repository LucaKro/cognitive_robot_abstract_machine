# grasp-belief-node (PR #14, plan aicon-belief-integration)

First concrete estimator: a recursive grasp belief over GraspLikelihood's
measurement. Draft PR #14, base `claude/plan-item-kickoff-aicon-0t046w` (#12),
with `claude/plan-item-kickoff-aicon-belief-84kl68` (#8) merged in as a second
parent. The settled design and what the implementation changed about it are in
the plan's `roadmap.md` under this item, in two sections.

## Done

- Branch off #12 with #8 merged in clean; draft PR #14 opened; `plan.yaml` item
  `in_progress` with branch, session and pull request recorded.
- `58b8cd66` - the whole item: `beliefs/grasp.py` (`GraspBelief`,
  `SampledLikelihood`), `grasp_likelihood_source.py`, `CertainPriorError`, the
  one-line base class on `GraspLikelihood`, and 23 tests.
- Verified here, not left to CI: 23 tests pass, each confirmed load-bearing by
  six mutations of the implementation; the collectible motion-statechart suite is
  223 passed against a 200-passed baseline with an identical 115-item
  failure/error set; `test/version_test` 20 passed.
- PR description and both roadmap sections match what actually landed.

## Next

- Read CI on `58b8cd66`. The one thing only CI can answer is whether
  `GraspLikelihoodSource` maps cleanly - it sits outside the `beliefs/` package
  `generate_orm.py` excludes. If `test_each_lib (giskardpy)` goes red on the
  generated interface, add it to `ignore_classes`.
- Republish the dashboard: `/plan-dashboard aicon-belief-integration`.
- The pull request stays a draft until its author has reviewed it.

## Container recipe that worked here

PyPI wheel of `random_events` (for `random_events_lib`) plus the checkout's
`random_events/src` on `PYTHONPATH`; `urdf_parser_py` from its sdist's package
directory and `xacro` from its wheel copied onto site-packages; every workspace
package's `src/` on `PYTHONPATH`; `--noconftest`.

## Open

- Tracking-issue subscription to #7 was refused by this session's permission mode,
  as on every earlier round on this plan. Issue #7 was not read this round.
