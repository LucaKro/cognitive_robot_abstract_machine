# belief-drawer-experiment — PR #17 (draft), base #16

Plan item `belief-drawer-experiment` of `aicon-belief-integration`. The
measurement the whole plan turns on. **Implementation complete and pushed.**

## The result

Nine runs per cell (3 arm postures x 3 cabinet yaws), distances in mm:

| Condition | Rays | Drawer travel | Grip offset | Touching |
|---|---|---|---|---|
| Unconditional | 100 | 198.25 | 7.57 | 9/9 |
| Believed | 100/1k/10k | 198.25 | 7.57 | 9/9 |
| Failing | 100 | 201.26 | 8.72 | 9/9 |
| Failing | 1000 | 226.96 | 30.23 | 9/9 |
| Failing | 10000 | 286.12 | 86.50 | 0/9 |

Answers #16's handed-over question with a number: scaling the grip alone does
NOT make the robot back off at any evidence a 100-ray reading can produce.
Grip = 2500 x probability vs mechanism = 1.0, so the ordering flips below
p = 4e-4; no hits out of 100 rays only reaches p = 5e-3.

## Done

- Branch off #16, draft PR #17, manifest `in_progress`.
- Both roadmap sections pushed (kickoff plan + what the implementation
  settled, including the three corrections to the kickoff's own claims).
- `drawer_scenario.py`, `drawer_run.py`, `sweep.py`, 18 tests. Committed and
  pushed as `c7d73963`.
- 11 mutations, each failing only the tests that name it. One failed nothing
  (collision avoidance) and that code was removed.
- Baseline comparison: identical 10 pre-existing collection errors with and
  without the diff. `test/version_test` 21 passed.
- Dashboard republished (v21).
- PR description rewritten to match the result.

## Next

- Republish the dashboard once more (the roadmap gained a section since v21).
- CI has not run on `c7d73963`.
- Raise with the user: the `is_body_in_gripper` defect on `main` wants its own
  bug PR (needs the `bug` label, off the default branch), and the mis-scaled
  probability-to-weight mapping wants a plan item.
