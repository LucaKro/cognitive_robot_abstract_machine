# belief-drawer-experiment — PR #17 (draft), base #16

Plan item `belief-drawer-experiment` of `aicon-belief-integration`. The
measurement the whole plan turns on. **Sweep complete; a MuJoCo basis is pushed
and waiting on the author's confirmation.**

## The sweep's result

Nine runs per cell (3 arm postures x 3 cabinet yaws), distances in mm:

| Condition | Rays | Drawer travel | Grip offset | Touching |
|---|---|---|---|---|
| Unconditional | 100 | 198.25 | 7.57 | 9/9 |
| Believed | 100/1k/10k | 198.25 | 7.57 | 9/9 |
| Failing | 100 | 201.26 | 8.72 | 9/9 |
| Failing | 1000 | 226.96 | 30.23 | 9/9 |
| Failing | 10000 | 286.12 | 86.50 | 0/9 |

Answers #16's handed-over question with a number: scaling the grip alone does
not make the robot back off at any evidence a 100-ray reading can produce.
Grip = 2500 x probability vs mechanism = 1.0, so the ordering flips below
p = 4e-4; no hits out of 100 rays only reaches p = 5e-3.

## The blocker, and what was done about it

The author's PR comment rejected the *basis*, not the measurement: the sweep
integrates commanded velocities and reads contact off the world's own collision
detector. He asked for a MuJoCo setup with a robot arm where normal grasping
works under ideal conditions, a video of it, and his confirmation **before** the
experiment is rebuilt on it.

`experiments/simulated_grasp/` is that setup, pushed as `f2fbf16c`. Pick and
place with a six-joint arm and a parallel gripper; grip closed lifts the block
246 mm, holds it with both fingers and places it 3.1 mm from the target; grip
left open moves it 0.1 mm on the same motion at the same cycle count. Video sent
in chat.

## Done

- Branch off #16, draft PR #17, manifest `in_progress`.
- Three roadmap sections pushed (kickoff, what the implementation settled, and
  the resolution moving the basis to physics).
- `drawer_scenario.py`, `drawer_run.py`, `sweep.py`, 18 tests — `c7d73963`.
- `simulated_grasp/tabletop_world.py`, `grasp_attempt.py`, 12 tests —
  `f2fbf16c`.
- 11 mutations on the sweep, 13 on the MuJoCo setup, each failing only the tests
  that name it.
- Baseline comparison: identical 10 pre-existing collection errors with and
  without the diff. `test/version_test` 21 passed.
- CI 23 of 23 green on `c7d73963`.
- PR description rewritten; a comment on #17 reports the setup.

## Next

- **Waiting on the author to confirm the MuJoCo setup.** Nothing past that gate
  is this branch's to do — the sweep stays as it is until then.
- CI has not run on `f2fbf16c`.
- **Dashboard is at v21 and three roadmap sections behind.** Status fields are
  current except this item's new `blockers`.
  `/plan-dashboard aicon-belief-integration` refreshes it.
- Raise with the user: the `is_body_in_gripper` defect on `main` wants its own
  bug PR (needs the `bug` label, off the default branch), and the mis-scaled
  probability-to-weight mapping wants a plan item.
- Two adapter traps found while building the world, worth their own issues or
  PRs if anyone else builds one: `create_with_dofs` names every degree of
  freedom `dof`, which silently miswires MuJoCo actuators and emits mimic
  equality constraints; and contact exclusions only apply to bodies belonging to
  a robot.
