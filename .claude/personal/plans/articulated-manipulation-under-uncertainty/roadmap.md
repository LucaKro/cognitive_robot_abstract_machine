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

## `tracy-sensor-mapping`

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

## `sim-drawer-scene`

Kicked off 2026-09-23 in auto mode. Branch `sim-drawer-scene` from `main`, draft PR #24.

**Finding that shapes the item.** `MujocoSynchronizer._write_connections_to_qpos` writes every world-state change of an unactuated 1-DOF joint straight into `qpos` (`_write_1dof_to_qpos`). So a drawer that giskard integrates as a decision variable is teleported in the physics every cycle, and the read-back after the step then confirms it. That breaks the acceptance rule directly. The *write* direction is this item's to fix.

**Decisions**
- In stepped simulation, a 1-DOF joint without a hardware interface (`is_controlled` is false) is owned by the physics, and the world never writes it into `qpos`. The world reaches the physics only through the servos, as it would on a real robot. Threaded simulation keeps today's behaviour, so existing users that set environment joints through the world are not affected. The rule is keyed on the existing `has_hardware_interface` flag rather than a new per-scene list.
- The scene is library code in `experiments`, not a test fixture, because `baseline-stock-cram` and the evaluator reuse it. It is built from the existing `Cabinet`/`Drawer`/`Door`/`Slider`/`Hinge`/`Handle` factories.
- The acceptance is shown twice: (a) giskard commanding the drawer joint leaves the physical drawer where it is; (b) Tracy's hand pushing the drawer front moves it, and the world reads the new position back. The same pair is checked for the door variant.

**Overlap.** `ground-truth-separation` owns the *read* direction (the physics overwriting the controller's belief). The two directions should converge into one concept when that item lands.

**Open.** MuJoCo tests run only in CI, and Tracy's description comes from the CI image, so the Tracy scene tests are verified in CI.

Resolved 2026-09-23 (auto mode). What was holding the PR up: the one unresolved review thread, "can't this be written with a WorldSpecification?". CI was green on 553b7a8b.

**What the implementation settled**
- The scene is a `CabinetSceneSpecification` (was `CabinetSceneBuilder`). `world_specification()` is a `WorldSpecification`: Tracy as a `RobotSpecification`, and the cabinet as one object whose drawer or door, with its slider or hinge and its handle, are nested part specifications. `to_domain_object()` builds the world, parks the arms and sets the joint dynamics, which connection specifications cannot carry.
- A `WorldSpecification` with no environment roots the world in `map`, which Tracy's URDF also defines, and MuJoCo rejects duplicate body names. The root is renamed with `World.force_root_name` (default `floor`).
- Objects are placed in the world frame, so the spec carries `world_T_table` (z = 0.88, `table_joint` in `tracy.urdf.xacro`). A test checks it against the built world.
- The cabinet is fixed to the world root rather than the table, so it is not part of Tracy's subtree.

**Tooling note.** `plan_item_brief` crashed in this cloud session because it reads review threads over GraphQL, which cloud sessions refuse. The thread was read over REST (`/pulls/{n}/ccr/review_threads`).

**Second review round (2026-09-23).** The review pointed out that `front_distance` and `sideways_offset` together were a 2D pose. They are now one field, `table_T_cabinet_front: Pose2D`: the centre of the cabinet's open front in Tracy's table frame, whose yaw turns the cabinet about that point. That gives the later cabinet-yaw sweeps their parameter. The push test places its waypoints in the cabinet's own frame (131d75ca).

## `belief-core` — first review round: the probability concepts move to probabilistic_model

Resolved 2026-09-23 (auto mode). CI was green on `23df0ae6`, ORM regeneration included.
What held the PR up was the author's review, with four unresolved threads:

- The review body: *"make sure that there are no duplications with Probabilistic model,
  and to adhere to its naming conventions"*.
- `belief.py` and `binary.py`: *"arent these things that could go into probabilistic model?"*
- `gaussian.py:63` (`Reading`): *"i dont like these names"*.
- `gaussian.py`, whole file: *"this as well"*.

**This reverses the item's recorded note** ("predict and update live here, not in
probabilistic_model"). The user decided to move them into probabilistic_model, within
this PR. They did not go into #22 or into a new item.

### What moves, and what it replaces

- **`MultivariateGaussianDistribution.linear_gaussian_transition(transition_matrix,
  offset, transition_covariance)`** is the Kalman prediction. It returns the
  distribution of `F x + b + w`, `w ~ N(0, Q)`, and takes arrays laid out by the
  variables, the same way #22's `product_with_gaussian_likelihood(observation_matrix,
  observed, observation_covariance)` does. The same `ShapeMismatchError` checks apply.
  #22's `_joint_with_observation` computed `H m` and `H P Hᵀ + R`, which is this
  transition with no offset, so it now reuses it instead of keeping a second copy.
  `LinearPrediction` and `Reading` go away, and so do their names: the update is #22's
  product, called directly.
- **`SymbolicDistribution.markov_transition(transition_model)`** is the discrete
  prediction. The transition model is a `MultinomialDistribution` over `(state,
  next_state)`, indexed `[state, next_state]` by domain order, as `markov_chain.py`'s
  `transition_model` is. **`SymbolicDistribution.product_with_likelihood(likelihoods)`**
  is the discrete update. The likelihoods are laid out by the variable's domain. A
  product that is zero everywhere raises `ImpossibleEvidenceError`, now in
  probabilistic_model. This covers any finite state, not only binary. `BinaryBelief`,
  `BinaryTransition` and `BinaryEvidence` go away.
- **`Belief`, `Statistic` and `VariableStatistic` go away.** A belief *is* a
  `ProbabilisticModel`. What gets published is read through probabilistic_model's own
  `expectation`, `variance` and `probability`.

### What stays in giskardpy

- `BeliefContext` holds the statechart's distributions by variable: `add`,
  `distribution_of` and `replace`. It keeps `DuplicateBeliefError` and
  `VariableWithoutBeliefError`.
- `EstimatorNode[ModelT]`. A subclass states `create_initial_distribution`, `predict` and
  `update`. `update` returns `None` in a cycle without evidence. The node publishes the
  mean and variance of every numeric variable and the probability of every value of every
  symbolic variable. It still observes TRUE only in a cycle with evidence.
- The giskardpy-side validation exceptions (`NegativeVarianceError`,
  `ProbabilityOutOfRangeError`, `NegativeLikelihoodError`) go with the types they checked.

### Overlap

This branch now also edits #22's `multivariate_gaussian.py` and its test file. When #22
changes, carry it up through the stack (`gh stack rebase --upstack`). Do not merge by hand.

## `belief-core` — second review round

Resolved 2026-09-23 (auto mode) on `e9ed7221`. Three new threads from the author:

1. `context.py:42`: *"Make sure that a ProbabilisticModel is really the right technology to
   use here regarding its capabilities and uses."*
2. `estimator.py:129` (`mean_variable`): *"is this also how probabilistic model names these
   kinds of methods?"*
3. `multivariate_gaussian.py:438`: should matrix, offset and covariance be one shared data
   structure, or would that be bloat?

### 1. `ProbabilisticModel` stays, and its per-cycle cost is fixed

It is the right type for what a belief *is*: a joint distribution over `random_events`
variables. It answers the queries a monitor will want (the probability of an event, the
mode, marginals, conditionals). Probabilistic circuits are `ProbabilisticModel`s too, and #22
added `MultivariateLeaf`, so a mixture belief (`handle-pose-estimator`'s "every pose
hypothesis") fits without another type. What it does not have is a filtering concept, so
prediction and update stay on the concrete distributions, and `EstimatorNode` leaves them
to its subclass.

**What measuring its uses in a control loop showed.** Per cycle, on six variables:
prediction 42 µs, update 294 µs. Reading the published values through the generic
queries, though, cost 0.93 ms for mean and variance (`moment` builds one univariate
Gaussian per variable, and `variance` computes the expectation again). Two symbolic
probabilities cost 158 µs, most of it building events. Fixes:

- `MultivariateGaussianDistribution` overrides `expectation` and `variance` in closed form,
  from the mean and the covariance's diagonal. This is still the generic interface, so
  every caller benefits, not only the estimator.
- `EstimatorNode` builds each published value's event once, at build time, and asks
  `probability_of_simple_event` each cycle (2 µs for two values).

The update's 294 µs is #22's conditioning. It is left as is, and noted.

### 2. Named as probabilistic_model names them

`mean_variable` becomes `expectation_variable`, and `PublishedValue.MEAN` becomes
`EXPECTATION`, matching `ProbabilisticModel.expectation`. `variance_variable` and
`probability_variable` already match. The `_variable` suffix is giskardpy's own convention
for accessors that return a `FloatVariable` (`observation_variable`,
`life_cycle_variable`). `predict` and `update` are filter vocabulary, which probabilistic_model
has none of.

### 3. `LinearGaussianModel`, for both the transition and #22's product

The user chose this at resolve time. `LinearGaussianModel(matrix, offset, covariance)` is
y = A x + b + N(0, Σ), with its shapes checked once, when it is built. It is used as
`linear_gaussian_transition(transition_model)` and
`product_with_gaussian_likelihood(observation_model, observed)`. The product therefore
gains a sensor bias through the offset. This changes the signature of #22's method, so #22's
tests for it move to the new signature in this PR. The term is Roweis & Ghahramani's
"linear Gaussian model".

## `tracy-sensor-mapping` — first review round: robot concepts move into semdt

Resolved 2026-09-23 (auto mode). CI was green on `a730ab8c`. What held the PR up was the author's review: 15 unresolved threads asking to move the concepts into semantic_digital_twin, to use probabilistic_model's Gaussian, pint for units, no module-level constants, and numpy typing.

**This reverses the item's recorded decision** ("no force/torque sensor is added to the CRAM robot model here; `simulated-sensors` owns the model change"). The user chose to **split by layer**:
- **semdt** (84a590bb) gets a generic `ForceTorqueSensor(Sensor)` part. Tracy's arms are `HasSensors[Tracy*WristForceTorqueSensor]`, one sensor at each `<side>_tool0`, so Tracy has three sensors. `ObjectDetectionStatus` moves next to `Robotiq85Gripper`. `simulated-sensors`' notes now say to *simulate* these parts rather than add a sensor model.
- **experiments** (9b87f33a) keeps the ROS binding and the measurement. Signals are built per semantic part (`WristWrench(sensor)`, `FingerPosition(gripper)`, `ArmJointEffort(arm)`, ...). `TracyDriverNamespace.of_part` finds a part's ROS namespace; semdt's robot models carry no topics.

**Units: pint** (user decision, after checking: 0.26.1 released 2026-09-10, Python ≥ 3.12, already a coraplex dependency). It is declared in `experiments`, with a `PintUnitJSONSerializer` for krrood. It is kept out of semdt's dataclasses because the ORM maps every semdt dataclass. The gripper current is read back as register counts, inverting the driver's mapping onto 235. The arm effort is in amperes (`use_currents_as_efforts` defaults to true).

**Statistics as distributions.** Each channel's values, and the time between samples, are a probabilistic_model `GaussianDistribution`, or a `DiracDeltaDistribution` when they never vary. There is no dependency on #22. A covariance across channels is `sensor-model-calibration`'s job.

**AGENTS.md** (eabf5059) records two rules: numpy arrays are typed with `numpy.typing`, and there are no module-level constants (a `ClassVar` on the owning class, or an enum member).

**Left open, with answers:**
- krrood's EQL aggregators don't fit arrays of samples and have no variance.
- segmind detects events from geometry, not a gripper's reported status; `coupling-belief` could feed it later.
- semdt's `Actuator`/`PositionServo` describe how the simulator drives a dof, not what a driver reports.

**Overlap.** `experiments/.../articulated_manipulation/__init__.py` is now empty here, while #24 adds it with a docstring. Whichever lands second reconciles it.

**Tooling note.** `plan_item_brief` still crashes in cloud sessions, which refuse GraphQL. The threads were read and resolved through the GitHub MCP tools.

## `belief-core` — merging #22 after its maintainer review

2026-09-23, at the user's request ("merge #22 into this branch … utilize the updates
from #22 where possible"). `cd21dea7`.

#22 took the probabilistic_model maintainer's review (`d10d969f`):
- the covariance became a `Covariance` class;
- the variables became a plain `variables` field (`7609b0de` made `ProbabilisticModel`
  allow that);
- `product_with_gaussian_likelihood(other)` now multiplies by another Gaussian over some
  of the same variables.

The conflicts were in `multivariate_gaussian.py` and its tests, and are resolved onto
#22's version:

- **The update is #22's product, unchanged.** This undoes part of round 2's choice: the
  observation form of `LinearGaussianModel` is gone, and with it observing a linear
  combination of the variables or a sensor bias. #22's own design wins in #22's file. An
  estimator that needs a general observation can linearise into a Gaussian over the
  variables it observes. If that turns out too narrow, the extension belongs in #22.
- `LinearGaussianModel` stays for the transition, with #22's `Covariance` as its noise.
- The closed-form `expectation`/`variance` read `Covariance.variances`. Per cycle on six
  variables: prediction 60 µs, update 90 µs (was 293 µs with the old product), publishing
  27 µs.

Kicked off 2026-09-23 in auto mode. Branch `ground-truth-separation`, stacked on `sim-drawer-scene` (#24).

**Plan**
- `MujocoSynchronizer.unobserved_connections`: a set of connections whose physical state is ground truth the world is not told. The sim→world read skips them, so the controller's world keeps its own value. The world→sim write skips them too, so a belief never teleports the physics. That second half is where this item converges with `sim-drawer-scene`'s write rule: one predicate (`_is_moved_only_by_physics`) now answers "does the physics own this connection?" for both the stepped-simulation rule on joints without a hardware interface and the unobserved set.
- Both 1-DOF joints (the drawer) and 6-DoF connections (a loose object's pose) can be unobserved. Nothing changes by default: the set starts empty, because robot-internal joints without a hardware interface must still be read back.
- Divergence is *recorded*, not only printed. On every read, the synchronizer appends a `DivergenceRecord` (simulation time plus, per degree of freedom, the world's position and the physics' position) to `divergence_log`. The evaluator and the later false-success metric read it directly. The physics' values come from the same qpos→state conversion the read direction uses, so both share one code path.
- `CabinetScene.environment_connections`: the cabinet's connections that carry degrees of freedom, including the cabinet's own parent connection. Today that is just the mechanism, since the cabinet is fixed; once the cabinet can be moved, its pose joins the set automatically.
- The evaluator reads MuJoCo directly (`simulator.get_joint_value`, as `sim-drawer-scene`'s tests already do); this item adds no separate reader.

**Acceptance tests** (TDD, MuJoCo ones CI-only): an unobserved hinge that the world sets is not moved in the physics; a falling free body that is unobserved stays put in the world while the physics moves it; the divergence log holds both values; observed connections are still read back. In the cabinet scene, giskard commanding the unobserved drawer leaves the world's drawer open and the physical drawer closed, and the log records the gap.

**Overlap.** Shares `multi_sim.py` and `cabinet_scene.py` with `sim-drawer-scene`. This item adds the read direction plus the set; the write rule it extends stays that item's.

**What the implementation settled (2026-09-23, 96073ff3, aa1b1708)**
- A connection that shares a degree of freedom with an unobserved one (a mimicking joint) is unobserved too, in both directions. The pendulum fixture's mirrored hinge showed the leak: its read-back overwrote the shared degree of freedom. Divergence is recorded only for the listed connections.
- Building a `MujocoSim` re-roots the world and gives free bodies new free connections. So connections are looked up after the simulation is built; `environment_connections` is a property for that reason.
- `_read_6dof_from_qpos`/`_read_1dof_from_qpos` keep their names (`test_multi_sim.py` patches one by name) but now return positions, which the read and the divergence share.
- Verified locally (no ROS): the semdt MuJoCo tests give 40 passed, 2 skipped. The Tracy cabinet tests are CI-only; the same giskard scenario on a cabinet-only world gave believed ≈ fully open, physics 0.0, log matching.

## `tracy-sensor-mapping` — second review round: no ClassVars

Resolved 2026-09-24 (auto mode). CI was green on `9b87f33a`. Eight new threads from the author:

- **No `ClassVar` constants either** ("also no classvars", "no. class. vars."). The rate constants became documentation, since nothing read them. The stamp conversion goes through pint. The gripper register's range comes from `np.iinfo(np.uint8)`. `driver_maximum_force` is a `kw_only` field defaulting to 235, because it is driver configuration. Queue depth, node name and report indentation are fields. A recording is timed once it has an interval between two samples, so the two-sample minimum is gone. AGENTS.md now rules out `ClassVar` constants too (`c11d417c`), which supersedes round 1's "a `ClassVar` on the owning class".
- **The pint serializer moved into krrood** (`df308b97`, `krrood/adapters/json_serializer.py`). krrood declares `pint`, a third-party package, so krrood stays self-contained.
- **`ObjectDetectionStatus` moved back to experiments**, because nothing in semdt used it. `robotiq_85_gripper.py` is identical to `main` again. semdt keeps only `ForceTorqueSensor` and Tracy's wrist sensors.
- **The measurement builds Tracy through `WorldSpecification`/`RobotSpecification`** (`b9d733e1`).
- **Left open:** "the ros stuff may move to semdt at some point". It stays in experiments for now, per the reviewer. This is a possible later move, not scheduled in any item.

## `ground-truth-separation` — first review round: a computed property, not an exclusion set

Resolved 2026-09-23 (auto mode) in 44b9ca07. CI was green on `aa1b1708`. The author left two threads, both now answered and resolved:

- `DivergenceRecord.simulation_time`: *"is this really a float, or is it a datetime or sth"*. It is now a `timedelta` (the simulated time elapsed since the start), like the durations `step_simulation` takes.
- `unobserved_connections`: *"this should be a computed property in the world class. but dont call it 'unobserved'"*.

**This reverses the item's recorded note** ("add an exclusion set"). `World.uncontrolled_connections` sits next to `controlled_connections`: the connections with degrees of freedom and no hardware interface. In a stepped simulation, `MujocoSynchronizer.physics_alone_moves_uncontrolled_connections` leaves them to the physics in both directions. That flag is #24's `physics_moves_uncommanded_joints` renamed and widened, which is the convergence `sim-drawer-scene`'s roadmap entry asked for.

**Why the computed rule is safe for robots.** A robot without a mobile base is attached by a `FixedConnection`. A mobile base's drive gets a hardware interface (`api.py`). Tracy sets one on every active connection of its arms and grippers. A mimicking joint shares its original's degree of freedom, and so its hardware interface, which made the separate shared-degree-of-freedom check unnecessary. `CabinetScene.environment_connections` went away with the set it fed.

**Divergence is recorded once per degree of freedom per read.** A mimic reports its original's degree of freedom again. Both sync directions ignore a mimic's `multiplier`/`offset` when converting to and from `qpos`; that was already the case before this item and is left as is.

**Overlap.** This branch renames #24's flag. If #24 changes, carry it up through the stack (`gh stack rebase --upstack`).

## `disturbance-protocol`

Kicked off 2026-09-23 in auto mode. Branch `disturbance-protocol` from `ground-truth-separation`, draft PR #30, the top layer of stack 29 (#24 → #28 → #30). The stack was registered over the Stacks REST endpoint: `gh stack link` needs GraphQL, which cloud sessions refuse.

**Plan**
- `experiments/.../articulated_manipulation/disturbance_protocol.py`: the conditions. Prior-error levels (none, low, medium, high) for the location and for the joint parameters; the disturbances (cabinet moved, drawer pulled from the hand, drawer re-closed); the arm-pose and cabinet-yaw sweeps; one seed per episode, so every episode replays exactly.
- **Prior error is belief-only.** The physics is built from the true scene; the controller's world from a believed scene whose specification differs by the sampled error (the cabinet's `table_T_cabinet_front`, and a tilt of the mechanism's axis). Both worlds share body and joint names, so the synchroniser pairs them. This needs `MultiSim` to accept the world the physics is built from separately from the world it synchronises with.
- **Disturbances act on the physics alone**, triggered by simulated time or by the physical opening. The controller learns of them only through what it senses, per the ground-truth separation #28 established.
- `episode.py`: runs a task under evaluation live against the stepped physics for one condition, up to a timeout, and records the outcome.
- `metrics.py`: success (the task reports done and the physics says the part is open); false success (reports done, physics disagrees); time to completion; task-specific recovery transitions authored; control-cycle time via `ControlLoopProfiler`, with `Executor.tick` as the cycle.
- Tests first. The protocol and metrics are tested without MuJoCo; every episode test runs only in CI, like the scene's own. The acceptance episode is the stuck-drawer probe: a task that commands the drawer joint directly must be counted as a false success.

**Decisions**
- The prior-error magnitudes are configurable; the defaults are placeholders, not sourced from paper A (user decision at kickoff). They are marked as such in the code.

**Overlap.** `cabinet_scene.py` (sim-drawer-scene, ground-truth-separation) gains the mechanism-axis parameter the believed scene needs; `multi_sim.py` (ground-truth-separation) gains the separate physics world. Carry any change to either up the stack with `gh stack rebase --upstack`.

**Open**
- How "task-specific recovery transitions authored" is counted: declared by the task under evaluation for now. `baseline-stock-cram` supplies the first real task and may settle a counting rule.
- Moving the cabinet, which is fixed to the world root, means moving a static body in MuJoCo; whether that needs a public method in `physics_simulators` is settled during implementation.
