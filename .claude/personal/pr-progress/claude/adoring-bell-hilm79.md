# `belief-weighted-open-goal` — PR #16 (draft)

Plan item of **AICON Belief Integration**. Branch `claude/adoring-bell-hilm79`,
based on #14 (`claude/plan-item-kickoff-belief-integration-6uj1wv`), which is where
`GraspBelief` lives. Tracking issue #7.

## Status: implemented and pushed (`aca45716`), awaiting CI and the author's review

## What landed

1. `Task.constraint_weight` — a property returning `self.weight`, overridden where a
   weight follows something measured. `Task.weight` stays a `float`, so no ORM
   mapping changes.
2. `CartesianPosition` (`cartesian_tasks.py:181`) and `CartesianOrientation`
   (`:626`) read it — the two leaves the hold-handle `CartesianPose` expands into.
3. `motion_statechart/beliefs/grasp_weighted_tasks.py` — `GraspWeightedTask` holds
   the belief, declares it in `prerequisite_nodes`, returns
   `self.weight * belief.probability`; the two weighted leaves and the
   `CartesianPose` that expands into them sit beside it.
4. `Open.grasp_belief`, optional and unset by default. `Close` inherits it.
5. Ten tests in `test_grasp_weighted_open.py`, reusing #14's `grasp_belief` helper.

## Why this shape

Timing before typing: `Open.expand` runs in `_expand_goals`, which completes before
any `build_artifacts`, and `GraspBelief.probability` does not exist until the belief
builds. So the weight resolves at the task's own build, which the property plus
`prerequisite_nodes` buys. `NotApproachingGoal` (`progress_monitors.py:75`) is the
precedent. Retyping `Task.weight` to `sm.ScalarData` would drop the `weight` column
from every task DAO via `parse_field`'s fall-through.

## Verified

- 10/10 pass; every assertion confirmed load-bearing by mutation.
- Two test gaps found that way and fixed: reading the task's property left the seam
  itself unpinned, and a union across both halves hid a revert of either one.
- `test_motion_statechart` baseline: 3 failed / 246 passed with and without the
  diff, failure sets identical as sorted lists.
- `test/version_test` 21 passed.
- Container needs `casadi==3.7.0`; 3.8.1 breaks the FK memory binding.

## Next

- Nothing outstanding on the branch. CI on `aca45716` has not been read.
- Republish the dashboard (the read/publish cycle is expensive in context; left as
  the last step).

## Open, recorded in the roadmap and the PR description

- Scaling the grip alone may not make the robot back off: `JointPositionList`
  constrains the environment connection's own position variable, so the solver can
  open the drawer directly while a slack grip leaves the arm uncoupled. Flagged for
  `belief-drawer-experiment`; scaling `mechanism_weight` is outside this item's
  recorded scope.
- No floor under the scaled weight — a ruled-out grasp reaches
  `DefaultWeights.WEIGHT_MINIMUM`.
- The ORM point is read out of `wrapped_table.py`, not run. CI confirms it.

## Structural change made this session

`task-weights-through-the-constraint-seam` added to the `belief-application` track,
depending on this item: converting the other 17 `quadratic_weight=self.weight` sites.
Asked and approved by the user; broadcast on issue #7.

## Dashboard — the one thing left undone

Not republished. The publish precondition is a full Read of the live artifact
(4048 lines / 268KB), and spending ~110k tokens of context on it risked ending the
session before the work could be reported. `plan.yaml`, `roadmap.md` and this note
are current; the published page is not.

Recipe for the next session, from #11's restack round, which is the one that
succeeded: read the artifact, Read *every* line of the file the read saves, then
publish to the cached url. Budget the read first, before writing code.

While reading the first 1700 lines for that publish, five roadmap lines were found
rendering as headings — wrapped continuations beginning `#10`, `#8's`, `#11's`,
`#13)`, which Markdown parses as a heading at column 0. Fixed on the personal-notes
branch (`1b0ad148`): pulled up onto the previous line where it was a continuation,
prefixed with "Pull request" where it began a paragraph. 37 real `##` sections intact.
