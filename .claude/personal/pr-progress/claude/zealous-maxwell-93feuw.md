# belief-drawer-experiment — PR #17 (draft), base #16

Plan item `belief-drawer-experiment` of `aicon-belief-integration`. **Sweep
complete; a MuJoCo Franka basis is pushed and waiting on the author's
confirmation.**

## The sweep's result

Nine runs per cell (3 arm postures x 3 cabinet yaws), distances in mm:

| Condition | Rays | Drawer travel | Grip offset | Touching |
|---|---|---|---|---|
| Unconditional | 100 | 198.25 | 7.57 | 9/9 |
| Believed | 100/1k/10k | 198.25 | 7.57 | 9/9 |
| Failing | 100 | 201.26 | 8.72 | 9/9 |
| Failing | 1000 | 226.96 | 30.23 | 9/9 |
| Failing | 10000 | 286.12 | 86.50 | 0/9 |

Grip = 2500 x probability vs mechanism = 1.0, so the ordering flips below
p = 4e-4; no hits out of 100 rays only reaches p = 5e-3. Scaling the grip alone
does not make the robot back off.

## The blocker, and what was done about it

The author's PR comment rejected the *basis*, not the measurement. He asked for
a MuJoCo setup where normal grasping works under ideal conditions, a video, and
his confirmation **before** the experiment is rebuilt on it. He then asked for
the Franka rather than a hand-built arm.

`experiments/simulated_grasp/` is that setup. It drives MuJoCo Menagerie's
Franka Emika Panda, vendored whole under
`semantic_digital_twin/resources/mjcf/franka_emika_panda` (33 MB; 112 KB of it
is the collision meshes the physics needs). Grip closed lifts the block 231 mm,
holds it with both fingers and places it 7.6 mm from the target; grip left open
moves it 0.06 mm on the same motion at the same cycle count. Video sent in chat.

## Done

- Branch off #16, draft PR #17, manifest `in_progress`.
- Four roadmap sections pushed (kickoff, what the implementation settled, the
  resolution moving the basis to physics, and the Franka port).
- `drawer_scenario.py`, `drawer_run.py`, `sweep.py`, 18 tests — `c7d73963`.
- Box-arm setup — `f2fbf16c`. Replaced by the Franka.
- Vendored Menagerie Panda — `156ed89f`; `panda_world.py` +
  `grasp_attempt.py` + 14 tests — `96ad4f98`.
- 11 mutations on the sweep, 15 on the Franka setup, each failing only the tests
  that name it. Three initially failed nothing and each exposed a real gap.
- Baseline comparison: identical 10 pre-existing collection errors.
  `test/version_test` 21 passed.
- CI 23 of 23 green on `c7d73963`.
- PR description rewritten; a comment on #17 reports the setup.

## Next

- **Waiting on the author to confirm the MuJoCo setup.** Nothing past that gate
  is this branch's to do — the sweep stays as it is until then.
- CI has not run on `f2fbf16c` or `96ad4f98`.
- **Dashboard is at v21 and four roadmap sections behind.**
  `/plan-dashboard aicon-belief-integration` refreshes it.
- Three bugs found, none fixed here, each wanting its own PR off the default
  branch: `is_body_in_gripper` deduplicates before counting rays;
  `MJCFParser.parse_actuator` cannot follow a tendon transmission to a joint;
  and a mimic coupling leaves both fingers carrying a degree of freedom named
  after the first, which miswires their servos.
- Also worth raising: the mis-scaled probability-to-weight mapping wants a plan
  item, and `mjx_single_cube_no_mesh.xml` has no geoms on any robot body.
