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

## `belief-core`

The plan settled at kickoff (auto mode), and the calls it makes beyond the item's `notes`.

### Where it sits

- **Branch `belief-core`, stacked on #22** (`claude/epic-euler-o9o7v8`), which is where
  `MultivariateGaussianDistribution` exists. The session was designated
  `basstler_experiments`, but that branch holds the basstler tooling, so the user chose
  the item's own branch name at kickoff.
- **Package `giskardpy.motion_statechart.beliefs`**, the same place the retired #10/#12
  used. It is excluded from giskardpy's ORM by package in `generate_orm.py`, so later
  modules in it are excluded without touching the script. The code is written clean;
  only the location and the review lessons carry over.
- **giskardpy declares `probabilistic_model` and `random_events`** in `[project]
  dependencies` and `[dependency-groups] workspace`. `test_imported_workspace_members_are_declared`
  caught exactly this on #10.

### What it contains

- `Belief[PredictionT, EvidenceT]`: the abstract base class. It is keyed by `random_events`
  variables, predicts, updates from a list of evidence, and reports its `statistics()`
  as `VariableStatistic -> float`. `Statistic` is a `StrEnum` (mean, variance,
  probability), so the names of the published values exist once.
- `GaussianBelief`: holds #22's `MultivariateGaussianDistribution`. `predict(LinearPrediction)`
  is written here (mean `F m + offset`, covariance `F P Fᵀ + Q`), per the item notes.
  `update(readings)` builds the observation matrix from each `Reading`'s variable-keyed
  contributions and reuses #22's `product_with_gaussian_likelihood`. That is the
  conditioning on the distribution, which #11's review round settled belongs in
  probabilistic_model. The Kalman step itself (building F/H/Q/R from the named inputs)
  stays in the belief. Transitions, offsets, process noise and readings are all keyed by
  variable, not by position (lesson from #10's first review round).
- `BinaryBelief`: a discrete Bayes filter for one binary latent state, over a
  two-valued `Symbolic` variable. `BinaryTransition` (probability the state persists or
  arises) and `BinaryEvidence` (the likelihood of the observation if the state holds or
  does not). It keeps a plain probability rather than wrapping `SymbolicDistribution`:
  the two-state update is three lines, and the distribution's hash-keyed probabilities
  would add machinery without adding anything.
- `BeliefContext(ContextExtension)`: beliefs keyed by variable. `add` rejects a second
  belief about the same variable with `DuplicateBeliefError`. `belief_of` raises
  `VariableWithoutBeliefError`. `from_context(context)` returns the statechart's beliefs
  and adds the extension on first use (precedent: `SegmindContext`).
- `EstimatorNode[PredictionT, EvidenceT]`: the abstract node. A subclass states
  `create_initial_belief`, `create_prediction` and `measure`. The base registers the
  belief and one `FloatVariable` per statistic at build, and on each tick runs predict,
  measure, update and `float_variable_data.set_value` (precedent: `WiggleInsert.on_tick`).
  It observes TRUE when evidence arrived that cycle and FALSE when it ran on prediction
  alone. Only exact trinary values reach the observation; continuous values go only
  through `float_variable_data`. `published_variable(variable, statistic)` gives a goal
  or monitor the symbol to build on, and raises `NodeNotBuiltError` before build.

### Verification

Tests first, in `test/giskardpy_test/test_motion_statechart/test_beliefs/`, fixture-free,
so they run under `--noconftest`. The root conftest regenerates the ORM, which needs ROS
message packages this container cannot install. The Gaussian update is checked against
the closed-form scalar Kalman gain. The estimator is driven through an `Executor` on an
empty `World` with a mimic node whose readings the test decides.

### Open / assumptions

- **ORM exclusion is CI's to confirm.** Regenerating the ORM cannot run in this container.
- **Nothing reads the published variables yet.** `joint-state-estimator` is the first
  consumer. Reset behaviour (belief back to prior on `on_reset`) is not built. No item
  asks for it yet, and adding it later is additive.
- **Readings have independent noise** (diagonal R), and process noise is per variable
  (diagonal Q). A correlated sensor would need a full-covariance reading type. That is
  additive when a sensor needs it.

Kicked off 2026-09-23 in auto mode. Branch `tracy-sensor-mapping` from `main`, draft PR #27.

**What the drivers expose, according to their source and configs** (`iai_tracy@ros2-jazzy`, `ros2_robotiq_gripper@iai_dualarm`, `Universal_Robots_ROS2_Driver@jazzy`, `ros2_controllers@jazzy`). Not yet confirmed on the robot:
- **Wrist wrench:** `/<side>_arm/force_torque_sensor_broadcaster/wrench`, 500 Hz. The value is UR's `actual_TCP_force`, rotated into the controller's TCP frame. It is labelled `<side>_tool0`, which is only correct while the TCP offset set on the pendant is zero. The `topic_name: ft_data` setting is ignored in Jazzy.
- **Finger position** (`gPO`, 0–255) and **gripper motor current** (`gCU`) come as the position and effort of the knuckle joint on `/<side>_gripper/joint_states`. The driver maps the current linearly onto 0–235 and publishes it as "effort": it is a current, not a force. The velocity is a finite difference of the position, not a separate measurement. Messages are published at 100 Hz, but the serial port is polled every 10 ms plus the time of one Modbus transaction, so fresh values may arrive more slowly than that.
- **Object-detected flag** (`gOBJ`: 0 moving, 1 stopped by an object while opening, 2 stopped by an object while closing, 3 at the requested position) is only on `/<side>_gripper/dynamic_joint_states`. The config's `extra_interfaces` is not a Jazzy parameter; the flag gets through only because the broadcaster publishes every interface there by default. giskard's `TracyVelocityInterface` syncs only `joint_states`, so it never sees the flag.
- **Arm joint effort** on `/<side>_arm/joint_states` is motor current, either in amperes or converted to torque, depending on the driver's `use_currents_as_efforts`. It is an extra grasp-evidence candidate that the notes do not list.
- **Camera:** an Orbbec Femto Mega (color 1920×1080, registered depth), not a RealSense.

**Decisions**
- The deliverable is a typed signal inventory plus an at-rest measurement tool in `experiments/articulated_manipulation`. The tool reports each signal's rate, interval jitter and longest gap, and each channel's mean, standard deviation and quantisation step. A signal that sends no messages is reported as not exposed. The statistics use numpy only, and their tests need no ROS; the message-reading and recorder tests run in the CI ROS image.
- The camera is left out of the inventory. The estimators consume detections, not frames, and detector error is `real-sensor-dataset`'s and `sensor-model-calibration`'s to characterise.
- No force/torque sensor is added to the CRAM robot model here. Where the simulated wrench sits (the wrist, in the controller's TCP frame) is recorded for `simulated-sensors`, which owns the model change.

**Open.** This item is done only after the tool has been run on the real Tracy and its report committed. That needs someone at the robot. Bias drift and the noise under load are left to `sensor-model-calibration`.

**Overlap.** Both `articulated_manipulation/__init__.py` files are also added by `sim-drawer-scene` (#24), with identical content.

Resolved 2026-09-23 (auto mode). What was holding the PR up: the one unresolved review thread, "can't this be written with a WorldSpecification?". CI was green on 553b7a8b.

**What the implementation settled**
- The scene is a `CabinetSceneSpecification` (was `CabinetSceneBuilder`). `world_specification()` is a `WorldSpecification`: Tracy as a `RobotSpecification`, and the cabinet as one object whose drawer or door, with its slider or hinge and its handle, are nested part specifications. `to_domain_object()` builds the world, parks the arms and sets the joint dynamics, which connection specifications cannot carry.
- A `WorldSpecification` with no environment roots the world in `map`, which Tracy's URDF also defines, and MuJoCo rejects duplicate body names. The root is renamed with `World.force_root_name` (default `floor`).
- Objects are placed in the world frame, so the spec carries `world_T_table` (z = 0.88, `table_joint` in `tracy.urdf.xacro`). A test checks it against the built world.
- The cabinet is fixed to the world root rather than the table, so it is not part of Tracy's subtree.

**Tooling note.** `plan_item_brief` crashed in this cloud session because it reads review threads over GraphQL, which cloud sessions refuse. The thread was read over REST (`/pulls/{n}/ccr/review_threads`).
