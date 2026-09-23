# Articulated Manipulation Under Uncertainty — roadmap

Plan-wide context. It is carried into every item brief, so keep it short. Per-item history
is appended below by the bootstrap under `` ## `item-id` `` headings.

## Why this plan exists

This plan supersedes `aicon-belief-integration` (retired 2026-09-23; see that plan's `## Retired` section). A critical reassessment of AICON found that its shipped estimators are mostly hand-shaped gradient devices, and that its task sequencing is hand-engineered rather than emergent. So its mechanism is not worth adopting.

**One real gap in CRAM survived:** articulated environment joints are QP decision variables, and their state is integrated from giskard's *own commands*.

- No synchroniser corrects them on a real robot.
- In MuJoCo they are hard-overwritten from physics.
- In a stuck-drawer probe, the model reported the drawer 0.30 m open, with the "opened" monitor TRUE from cycle 31, while the hand never moved.

The goal is to **match AICON's task-level capabilities with explicit machinery**, not its mechanism.

## Capability targets

| AICON capability | This plan |
|---|---|
| Opens under a wrong prior on the drawer *location* | handle-pose-estimator, active-perception-tasks |
| Opens under a wrong prior on the *joint parameters* | articulation-model-estimator, open-along-estimated-axis |
| Does not mistake "commanded" for "happened" | joint-state-estimator, mechanism-divergence-failure |
| Recovers when the cabinet is moved | handle-pose-estimator, reachievement-template |
| Recovers when the drawer is pulled from the hand | coupling-belief, reachievement-template |
| Re-opens a drawer closed again mid-task | reachievement-template |
| Re-establishes lost visibility | active-perception-tasks |
| Respects joint limits and avoids self-collision | already exceeded: giskard has hard constraints |

## Design principles

- **Transitions stay explicit.** Continuous values go into `float_variable_data`, never into observations: `is_true()` and the verdict test exact trinary values, so an observation of 0.95 never transitions.
- **Do not scale task weights by probabilities.** Measured: a ×0.01 weight changes nothing, and the task only switches off near zero. Scaling the hold-handle weight toward zero opens the *modelled* drawer without the hand. Belief-driven behaviour belongs in monitors and gating transitions.
- **Estimate; don't dead-reckon.** The environment joint stays a QP variable (closed-chain model), but its *state* is the estimator's posterior.
- **Grasp evidence comes from real signals** — finger width and effort, wrist wrench, whether the handle follows the hand — never from a raycast against the model, which confirms itself.
- **Re-achievement is generic**, not enumerated per failure class. This is the answer to AICON's "explicit designs must enumerate transitions" argument, and the benchmark tests it.
- **Ground truth is kept apart from belief** in simulation, or every test is circular.

## Benchmark and metrics

Tracy in MuJoCo, driven live by giskard (precedent: `test_mujoco_live_control.py`). The disturbance protocol follows AICON paper A:

- prior error on location and on joint parameters, at four levels;
- cabinet moved;
- drawer pulled from the hand;
- plus drawer re-closed mid-task, and the arm-pose and cabinet-yaw sweeps from the AICON ablation.

Metrics:

- success rate;
- **false-success rate**;
- time to completion;
- **task-specific recovery transitions authored** (target zero);
- control-cycle time, via `control_loop_profiler.py`.

AICON's paper-A real-robot numbers are indicative only, since the robot and simulator differ.

## Real data

Simulation validates an estimator's logic. It cannot validate the sensor models the estimator runs on, and those decide whether it works on hardware. Invented simulation noise matches the filter's own assumptions, so a filter looks justifiably confident there — exactly the confident-but-wrong failure AICON showed.

So real data comes early, as open-loop recordings replayed offline, not as closed-loop trials:

- `real-sensor-dataset` records real episodes with ground truth independent of the robot's perception;
- `sensor-model-calibration` fits the simulated sensors to those recordings;
- `estimator-replay-harness` scores each estimator's consistency (NEES/NIS) on them.

Closed-loop hardware trials stay in wave 4. `coupling-belief` and `handle-pose-estimator` are the most exposed: simulated contact forces are clean, and a synthetic detector flatters any filter.

## Gates

- **After `baseline-stock-cram`:** if stock CRAM already succeeds on most conditions and its failures are not about environment state, stop. That is a valid outcome.
- **After `mechanism-divergence-failure`:** if the false-success rate does not drop, do not build waves 2 and 3.

## Decisions taken

- **Robot: Tracy**, in simulation and on hardware.
- **#22** (multivariate Gaussian) is tracked as the belief foundation. `belief-core` is written clean, not salvaged from the closed #10/#12.
- **The condition-monitor rework** (`executables.py:175`) is owned by someone else. Whether `live-precondition-monitors` waits for it is decided at that item's kickoff.
- **Real-world ground truth** for `real-sensor-dataset` — a fiducial seen by a separate camera, a draw-wire encoder, or motion capture — is decided at that item's kickoff.
- **Out of scope:** gradient-based action selection; planning-free Blocks World; differentiating through beliefs; contact-rich pushing (AICON paper B).

## Conventions

- **Fork only.** Never push to, comment on, or open PRs against `cram2`; releasing there is the user's action.
- **Targeted test files only**; never the full suite. Do not read `ormatic_interface.py` files.
- **Items with two unlanded dependencies** (`joint-state-estimator`, `handle-pose-estimator`, `sensor-model-calibration`, `estimator-replay-harness`) cannot be stacked by `plan_stack`. Ask the user at kickoff.

Kicked off 2026-09-23 in auto mode. Branch `sim-drawer-scene` from `main`, draft PR #24.

**Finding that shapes the item.** `MujocoSynchronizer._write_connections_to_qpos` writes every world-state change of an unactuated 1-DOF joint straight into `qpos` (`_write_1dof_to_qpos`). So a drawer that giskard integrates as a decision variable is teleported in the physics every cycle, and the read-back after the step then confirms it. That breaks the acceptance rule directly. The *write* direction is this item's to fix.

**Decisions**
- In stepped simulation, a 1-DOF joint without a hardware interface (`is_controlled` is false) is owned by the physics, and the world never writes it into `qpos`. The world reaches the physics only through the servos, as it would on a real robot. Threaded simulation keeps today's behaviour, so existing users that set environment joints through the world are not affected. The rule is keyed on the existing `has_hardware_interface` flag rather than a new per-scene list.
- The scene is library code in `experiments`, not a test fixture, because `baseline-stock-cram` and the evaluator reuse it. It is built from the existing `Cabinet`/`Drawer`/`Door`/`Slider`/`Hinge`/`Handle` factories.
- The acceptance is shown twice: (a) giskard commanding the drawer joint leaves the physical drawer where it is; (b) Tracy's hand pushing the drawer front moves it, and the world reads the new position back. The same pair is checked for the door variant.

**Overlap.** `ground-truth-separation` owns the *read* direction (the physics overwriting the controller's belief). The two directions should converge into one concept when that item lands.

**Open.** MuJoCo tests run only in CI, and Tracy's description comes from the CI image, so the Tracy scene tests are verified in CI.
