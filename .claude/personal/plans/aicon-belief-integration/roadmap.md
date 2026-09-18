# AICON Belief Integration — roadmap

The narrative half of `plan.yaml`. Structured facts (branch, PR, status,
dependencies) live there; this file carries the *why*, the evidence behind the
judgment calls, and the standing conventions for the initiative.

Full written analysis this plan was drafted from:
<https://claude.ai/artifact/QjiVQGsKK58qCeajXrd1pk>

Per-branch working detail belongs in this repo's existing
`.claude/personal/pr-progress/<branch>.md` mechanism, which keeps working
independently of the plan — it is not duplicated here.

---

## Why this plan exists

Three facts, established by reading both codebases, decide the whole thing.

1. **CRAM is already a differentiable graph.** `krrood.symbolic_math` wraps
   CasADi and exposes `jacobian`, `total_derivative` (`ca.jtimes`),
   `jacobian_dot` and `hessian`. Forward kinematics is built symbolically on
   model change and compiled to a CasADi function bound directly to the state
   array's memory (`forward_kinematics.py:126`). Giskard takes ∂f/∂q of every
   task expression at build time to form its QP rows.

2. **CRAM has no uncertainty at all.** Grepping all seven packages for
   `covariance`, `kalman`, `belief`, `posterior` returns nothing that is state
   estimation. The world is one `(4, N)` float array — position, velocity,
   acceleration, jerk — and every body pose is derived from it. There is no
   filter anywhere in the stack.

3. **CRAM's differentiation stops exactly where AICON's begins.**
   `error_signals.py:17` deliberately treats every free variable that is not a
   joint position as a constant, and says so: *"Change caused by anything other
   than joint motion is invisible. An error whose goal is rewritten during
   execution, for example from perception, needs `SampledErrorSignal`."*

So the thesis: AICON's entire contribution, stripped of its framework, is one
edge — **the derivative of the goal with respect to the action, routed through a
belief that the action itself changes.** CRAM has the graph, the compiler, the
control loop and the 20 Hz tick. It is missing the beliefs and that one edge.

## What this plan deliberately does *not* do

**It does not adopt AICON as a framework.** Running AICON beside CRAM would mean
a second, weaker controller next to a QP that already handles joint, velocity,
acceleration and jerk limits plus collision avoidance, and two world models to
keep in sync. The report's level 4 rejects this reading and points instead at
`Context.query_backend` as the one genuinely well-shaped slot — deferred out of
this plan, since it is independent of waves 1–3 and can be picked up later.

**It does not import the required-signs mechanism.** AICON connections may carry
a `required_signs_dict` that overwrites autodiff elementwise
(`derivatives.py:71`: `new_derivatives = - required_sign * torch.abs(...)`).
Measured on the published drawer tutorial, changing only the initial arm pose:

| Configuration | With the sign dict | Without it |
|---|---|---|
| Arm pose `[0.2, 0.2, 0.1, −2.0, 0, 1.5, 0.7]` | opens at step 214 / 215 / 217 | never opens, 3 / 3 runs |
| Cabinet yawed −20° | never opens | opens |

It is load-bearing in both directions, and of the tutorial's three entries only
one ever fires (`position_drawer ← likelihood_grasped_drawer`, 297 times in a
run); the other two are dead in every configuration tested. It is also redundant
here: CRAM already encodes the same category of knowledge in `DefaultWeights`,
and more honestly — a wrong weight degrades visibly, a wrong forced sign inverts
a decision invisibly.

**It does not adopt the "multistage behaviour without a planner" framing.** True
of AICON and equally true of CRAM's statechart; in both cases a human authored
the ordering. CRAM's is explicit, inspectable, serializable and testable, which
is a strength to keep rather than rhetoric to match.

## Design decisions already taken

**Beliefs live in a `ContextExtension`, not in `WorldState`.** `WorldState`'s row
axis is the `Derivatives` enum hard-coded to 4, and that shape is assumed by the
ORM (`orm/model.py:95`), the ROS sync (position row only), the FK memory binding
and the trajectory recorder. Widening it touches at least eight files for no
gain.

**Nor on a `SemanticAnnotation`.** Mutating an annotation field properly goes
through `@synchronized_attribute_modification`, which JSON-diffs the entity,
requires an open `modify_world()`, bumps the *model* version, clears every
memoization cache and re-runs the ripple-down-rules reasoner. At 20 Hz that is
pathological. An annotation is the right place for the *declaration* — "this
handle has a grasp estimator with these parameters" — and the wrong place for
the value.

**Write the Kalman update by hand; do not force `probabilistic_model` to be a
filter.** Its `GaussianDistribution` is strictly univariate (`location`,
`scale`), there is no multivariate Gaussian and no covariance anywhere in the
package, and "conditioning" means truncation to an interval or collapse to a
Dirac — there is no measurement update, and `log_truncated_in_place` is
structure-changing. Use `random_events` for variables and events; write the
~30 lines of update.

**The precedent to copy is segmind, not AICON.** `AbstractDetector` subclasses
giskard's `MotionStatechartNode`, ticks every control cycle, and carries per-body
state across ticks in a `SegmindContext(ContextExtension)` retrieved with
`context.require_extension`. It already uses the statechart for something that is
not motion. An estimator is the same shape with a Gaussian where segmind has a
boolean. Two caveats: segmind is currently driven by offline episode replay, not
live execution, and nothing outside the package imports it — the machinery is
proven, the wiring to a live loop is not.

## Wave structure, and why

Strictly sequential, because each wave is a hard prerequisite for the next:

- **Wave 1** is independently shippable and is not really AICON at all — it is
  refusing to throw away information the stack already receives. Best
  effort-to-insight ratio available, and it produces the signals wave 2 filters.
- **Wave 2** is the real integration and is tractable: continuously-updated
  uncertain quantities usable in constraints and transitions, without touching
  the world model, the ORM or the control loop's critical path. It does not yet
  deliver AICON's actual contribution.
- **Wave 3** is the contribution, and the publishable claim. It surfaces most of
  its design questions only once wave 2 exists, which is why it is not started
  first.

Within wave 2 the work splits into two tracks that can proceed in parallel once
`estimator-node-base` lands: the core container/base class, and the first
concrete estimator plus the experiment that judges it.

## The measurement that actually decides this

`belief-drawer-experiment`, condition (iii): belief-weighted `Open` with the
grasp deliberately failing. Stock `Open` **cannot represent** a failed grasp — it
asserted one as a precondition (`goals/open_close.py:20`, *"Assumes that the
grasped part has already been grasped"*) — so it will drive the hinge goal
against a drawer it is not holding until `NotApproachingGoal` eventually fires.
The belief-weighted version should back off on its own. That is exactly the
behaviour AICON claims, and exactly the behaviour CRAM currently has no way to
express.

## Standing caveat for every demo in this plan

In our experiments the AICON drawer tutorial does not, on inspection, grasp the
handle. It wedges the gripper into the crease above it and drags. The estimator
reports `p_grasp ≈ 0.99` throughout, because the likelihood is consistent with
what it observes, not with what is physically happening.

**Validate against simulator contact state, never against the posterior.** A
confident belief is still only a belief, and this plan is in the business of
manufacturing confident beliefs.

## What the stack could give back

Worth keeping in view, because this is a collaboration case rather than a
critique. CRAM has, and AICON lacks: a QP with hard limits and a braking profile
that guarantees stopping at a joint limit (AICON normalizes the gradient and
takes a fixed-gain step); collision avoidance at all; semantics derived from
structure by ripple-down rules rather than a hardcoded cabinet at
`(0.2, 0.30, 0.03)`; `ActionTrial`, which runs the real controller in a
deep-copied world to ask whether something would work; a replayable modification
log; and a failure taxonomy where every exception implements
`suggest_correction()`.

The sharpest single offer: AICON's estimators need measurement models derived
from a semantic world model rather than hand-written per demo.
`compose_forward_kinematics_expression` gives an exact symbolic transform between
any two frames in the tree, and the RDR layer finds the drawer's joint without
being told where it is. That is the part of AICON that currently does not scale
past a tutorial.

## Conventions for sessions working this plan

- Subscribe to the tracking mailbox, **issue #7** on the fork.
- Structural changes — adding a wave, deferring a track, splitting an item,
  reprioritizing — are the user's call. Ask in-session first, then edit
  `plan.yaml` **and** comment on issue #7 describing the change.
- `status`, `notes` and `blockers` on an item you are actively working are normal
  edits; make them directly.
- Never push anything to the `cram2` remote. `origin` is
  `LucaKro/cognitive_robot_abstract_machine` and is where every branch and PR in
  this plan lives.

## `grasp-likelihood-continuous`

The plan settled at kickoff, and the one correction it makes to the item's own
recorded `notes`.

### The carrier for a continuous value is a `FloatVariable`, not an observation

The item's `notes` claim the likelihood is *"type-compatible with a giskard
observation today"*, and name `is_unknown()` (`symbolic_math.py:1035`,
`ca.eq(x, 0.5)`) as the single thing to guard against. The first half is right
about the *algebra* only: trinary not/and/or are `1-x`, `min` and `max`, which
are total on `[0, 1]`. It does not hold for the sites that compare against the
three constants exactly, and there are four of those rather than one:

- `motion_statechart.py:288`, `ObservationState.__getitem__`, coerces through
  `ObservationStateValues(value)` and raises `ValueError` on anything else.
- `graph_node.py:1284`, `goal_reached_state`, repeats that coercion.
- `graph_node.py:926`, `_create_verdict`, matches the three constants exactly
  and falls through to `INTERRUPTED`, so a node observing a continuous value
  could never be judged succeeded or failed.
- `_create_condition_holds` reads a condition through `is_true()`, which is
  `ca.eq(x, 1)`. A continuous `0.97` therefore reads as false, and a continuous
  observation would never fire a transition at all.

So the continuous value is carried as a `FloatVariable` registered in
`context.float_variable_data` and written each tick, while the node's
observation stays trinary. The observation updater already compiles against
`context.float_variable_data.variables` (`motion_statechart.py:332`), so the
value is symbolically readable by constraints and transition conditions without
any of the four sites changing. `WiggleInsert` (`wiggle_insert.py:202`, `:250`)
is the register-then-write-per-tick precedent this follows.

This also makes wave 1 agree with wave 2: `odometry-covariance-capture` and
`estimator-node-base` both already specify a registered `FloatVariable` as the
carrier.

### What is built

- `trinary_logic_from_continuous` in krrood's `symbolic_math`, joining the
  existing `trinary_logic_not`/`_and`/`_or`/`_to_str` family. It maps a
  continuous confidence onto the three constants with explicit thresholds, so
  the exact-equality hazard above is resolved in one place rather than
  hand-rolled per caller. This is what the item's note asks for when it says to
  guard `is_unknown()`.
- `GraspLikelihood`, a monitor in giskard's motion statechart, registering the
  likelihood variable, writing `is_body_in_gripper` into it each tick, and
  deriving its own observation from that variable through the helper.
- A named constant for the gripped-likelihood threshold, which was spelled as a
  bare `0.9` in both `is_body_gripped` and coraplex's container post-condition.

### Scope boundaries held

Memoryless by intent: this publishes the current measurement only. The recursive
filter over it is `grasp-belief-node`, which depends on this item, and the
statechart's existing predicate is a stateless Monte-Carlo raycast, so nothing
here has memory yet. No base class is introduced either — wave 1 is specified as
"no new abstractions", and `estimator-node-base` is where the pattern gets
generalized.

### Assumptions and open points

- The node evaluates a 100-ray raycast synchronously on the control loop. That
  is a real cost on a 20 Hz tick and is the critical-path concern
  `estimator-node-base` is already told to watch. The sample count is a field so
  it can be traded down, and the threaded alternative already exists in
  `ThreadedPredicateMonitor` — but that monitor coerces its result with `bool()`,
  so adopting it would need the same continuous treatment. Left for wave 2
  rather than widened into this item.
- The branch recorded for this item is the session's designated branch,
  `claude/plan-item-kickoff-aicon-belief-84kl68`, rather than the
  `grasp-likelihood-continuous` name the manifest carried as a placeholder. The
  session is constrained to develop on its designated branch.

## `odometry-covariance-capture`

The plan settled at kickoff, and the calls it makes beyond the item's recorded
`notes`.

### The covariance is captured where it is dropped, and published by a node

The item's `notes` name one site — `OdometrySynchronizer.apply_message`, which
reads `message.pose.pose` and never touches `message.pose.covariance` — and one
outcome, a registered `FloatVariable` the statechart can condition on. Those are
two different layers, and nothing joins them today:

- A `FloatVariable` is registered in `MotionStatechartContext.float_variable_data`
  during `build_artifacts`, so it exists per compile. A synchronizer outlives any
  one compiled statechart — `robot_interface_config.sync_odometry_topic` appends
  it to `motion_server.inputs` and `control_loop.inputs` once, at configuration
  time — so the synchronizer cannot own the variable.
- Therefore the synchronizer keeps the covariance, and a `MotionStatechartNode`
  registers the variable and writes it each control cycle. That is the same split
  `GraspLikelihood` uses, and the register-then-write-per-tick precedent is
  `WiggleInsert` (`wiggle_insert.py:202`, `:250`).

### The node depends on an abstraction, not on the synchronizer

In this package `giskardpy.middleware` imports `giskardpy.motion_statechart`
(`motion_goal.py:11`, `python_interface.py:17`) and never the reverse. A node
holding an `OdometrySynchronizer` would invert that direction, and would also be
untestable without a live ROS subscription.

So `PoseCovarianceSource` — one read-only property answering with the most recent
`PoseCovariance`, or nothing if none arrived — is declared in the statechart layer
and implemented by `OdometrySynchronizer` in the middleware layer. The node holds
the abstraction.

This is a new type, and wave 1 is specified as *"no new abstractions"*. That line
guards against introducing the estimator/belief base classes early, which
`estimator-node-base` is explicitly where the pattern gets generalized; a
one-property interface that keeps an existing layering rule is not that. The node
itself introduces no base class.

### An unread covariance is maximally uncertain, not zero

A registered `FloatVariable` starts at zero, and zero variance means perfect
certainty. A condition of the form the item asks for — *"do not begin the final
approach while base uncertainty exceeds X"* — would therefore pass before any
odometry message has been received, which is exactly backwards.

The variable is primed to infinity when the node starts and is only overwritten by
a real reading, so the condition fails closed. The node's own observation reports
whether a reading has arrived at all, following `WaitForMessage`.

### The 36 numbers become a type, not an array

`message.pose.covariance` is a flat, row-major 6×6 whose positions carry meaning
(x, y, z, then rotation about each axis). `PoseCovariance` holds it with a
`PoseAxis` enum indexing the rows and columns, and exposes the quantity the item
asks for — `total_variance`, the trace — alongside the position and rotation
halves.

It is deliberately frame-naive: an odometry covariance is expressed in the message's
own frame, and a frame-aware spatial uncertainty type belongs in
`semantic_digital_twin` under its spatial style guide rather than in wave 1. Wave 2's
`belief-context-and-gaussian` is the item that will want one.

### Scope boundaries held

- **No dependency on `grasp-likelihood-continuous`.** That sibling (#8, unmerged)
  adds `trinary_logic_from_continuous`, which would fit this node's observation.
  This item does not declare a dependency on it, so it does not use it; the
  observation answers the narrower question of whether a reading exists. The two
  branches overlap on `motion_statechart/exceptions.py` only, where each appends
  its own read-before-built exception. Generalizing that pair is
  `estimator-node-base`'s job.
- **No consumer is added.** The variable is published; no existing goal or
  condition is rewritten to read it. Conditioning on base uncertainty is the
  caller's to write, and the item asks only that it become possible.
- **Only the diagonal is interpreted.** The full 6×6 is kept, but the published
  quantity is its trace, which is what the item asks for ("even just its trace to
  begin with").

### Open points

- **The covariance frame is not checked.** `nav_msgs/Odometry` documents the pose
  covariance in the frame of `header.frame_id`, and the synchronizer already
  assumes that frame matches the drive connection's parent for the pose itself.
  This item inherits that assumption rather than fixing it.
- **Untested locally.** This container has no `numpy`, `casadi` or `rclpy`, so
  every test here is verified by CI rather than by the session that wrote it —
  the same constraint #8 reported.

The plan settled at kickoff, and what it decided that the item's own `notes`
left open.

### Where the belief layer lives

`ContextExtension` is giskard's (`motion_statechart/context.py:25`), so
`BeliefContext` can only live in giskardpy. The Gaussian itself goes beside it
in a new `motion_statechart/beliefs/` package rather than in krrood: krrood is a
knowledge-representation library whose numeric half is the CasADi wrapper, and a
recursive filter is neither. `beliefs/` also gives `estimator-node-base` a place
to land, matching how `goals/`, `monitors/` and `tasks/` are already organized.

### A belief names its dimensions with `random_events` variables

The roadmap's "use `random_events` for variables and events" is taken
literally: a `GaussianBelief` carries a list of `Continuous` variables, one per
row of its mean, and `BeliefContext` is keyed by those same variables. That
gives the context a structured key instead of a string, lets a vector belief
answer `variance_of(yaw)` without the caller counting rows, and is the same
vocabulary `symbolic-estimator-means` will need in wave 3.

One belief is registered under each of its variables, so a variable can belong
to exactly one belief — `add` raises `DuplicateBeliefError` rather than
silently splitting a quantity across two filters.

### Predict and update mutate in place

A belief is the filter's state, not a value object. Returning a new instance
would leave every holder of the old one — the context included — silently
stale, and the context would have to be re-written on every tick at 20 Hz.
`predict` and `update` therefore modify the belief the context already holds,
which is also what segmind's per-tick context state does.

### `predict` takes an offset

`x' = F x + offset`, not just `F x`. `grasp-belief-node` is already specified as
"a prediction step that decays toward the prior when the gripper is open", which
is affine rather than linear, so the offset is required by a recorded dependent
item rather than added speculatively.

### The update is written in Joseph form

`P' = (I - K H) P (I - K H)ᵀ + K R Kᵀ` rather than the shorter `(I - K H) P`.
Both are the same in exact arithmetic; the long form stays symmetric under
floating point, and a belief that drifts out of symmetry at 20 Hz is a failure
that shows up much later as an unexplained covariance.

### One shape error, not four

Every array the belief layer is handed can be the wrong shape — mean,
covariance, transition, process noise, offset, and a measurement's value, model
and noise. That is one situation with eight subjects, so it is one
`WrongBeliefShapeError` carrying a `BeliefArray` enum member naming which array,
the shape required and the shape given, rather than a separate exception class
per array or a message to match on.

### Scope boundaries held

No statechart node is introduced. This item is the container and the math;
`estimator-node-base` is the item that makes a `MotionStatechartNode` out of it
and registers a `FloatVariable` for the mean, and nothing here touches
`float_variable_data`. Nothing is added to `WorldState`, the ORM or the control
loop, per the design decisions already recorded above.

### Assumptions and open points

- Nothing adds a `BeliefContext` to a live context yet. Like `SegmindContext`,
  which `episode_segmenter.py:62` adds, that wiring belongs to whoever runs the
  statechart — here, `estimator-node-base`. Until then the extension is
  reachable only by a caller that adds it itself, which is what the tests do.
- The belief carries no prior over which variables are correlated: a caller
  states the full covariance. That is deliberate for a first version, but it
  means a two-variable belief built by two different estimators has no defined
  way to merge.
- This item's only file overlap with the unlanded `grasp-likelihood-continuous`
  branch is `motion_statechart/exceptions.py`, where both append new exception
  classes. Different classes in different places in the file, so the two are
  independent work; worth knowing when the second of the two rebases.
- The branch recorded is the session's designated branch,
  `claude/belief-integration-gaussian-zi1o82`, not the
  `belief-context-and-gaussian` placeholder the manifest carried, following the
  same correction `grasp-likelihood-continuous` made.

## `grasp-likelihood-continuous` — resolution

Nothing was blocking it. Resolving the item was a matter of closing the one gap
the kickoff left open rather than fixing a stall.

The kickoff recorded that the giskardpy and `semantic_digital_twin` tests could
not be executed in that session — the container had no numpy, casadi or trimesh,
and installing the workspace failed on the repository's `pyproject.toml` needing
a newer uv than 0.8.17 plus absent graphviz headers. Only the krrood
symbolic-math tests ran locally. The pull request description and the
PR-progress note both said so and flagged the first CI run as worth watching.

That run has now completed: all 23 checks on `b5c4a811` are green, including
`test_each_lib (giskardpy)` and `test_each_lib (semantic_digital_twin)`, which
are exactly the two suites the session could not run. So the eight
`GraspLikelihood` tests, the `body_between_fingers` fixture in the root
`test/conftest.py` and the refactored `test_is_body_in_gripper` are verified,
and the three design calls the kickoff had to make by reading the code rather
than by running it are confirmed by a passing suite:

- a monitor-only statechart compiles and ticks with no constraints, no
  `EndMotion` and no degrees of freedom;
- priming the likelihood in `on_start` means the first observation is read off a
  measured variable rather than an unset one;
- the trinary observation stays one of the three constants, so the four
  exact-equality sites keep working.

No review threads, no pull request comments, no tracking-issue discussion, no
merge conflict, and the branch is level with `main`. The pull request stays a
draft awaiting its author's own review, per this repository's convention that
un-drafting *is* the record of having reviewed it.

One thing to carry forward: this branch, `odometry-covariance-capture` (#9) and
`belief-context-and-gaussian` (#10) all append their own exception classes to
`giskardpy/src/giskardpy/motion_statechart/exceptions.py`. Different classes in
different places, so the work is independent, but whichever two land second and
third should expect to resolve that file.

## `odometry-covariance-capture` — resolution

Nothing was blocking it either. The kickoff left one gap and flagged one risk;
the first CI run closed both, and the single red check turned out not to be this
branch's.

### The container gap is closed

The kickoff recorded that `random_events` would not build in that session (an
antlr4 runtime wheel failure) and that there was no `rclpy`, so only the seven
`PoseCovariance` tests could be run — and those only against the module source
with the exceptions module stubbed. Everything else was written unverified.

The run on `28dc12c2` reports **823 passed, 1 failed** in
`test_each_lib (giskardpy)`, and the one failure is not in this diff. So all
seventeen tests this item added are verified: the seven `PoseCovariance` tests,
the seven `PoseUncertainty` tests against the `RecordedPoseCovariance` mimic and
the `mini_world` fixture, and the three `OdometrySynchronizer` covariance tests
that need `nav_msgs`.

### The flagged risk was unfounded

The kickoff could not execute the one thing it was unsure of: `OdometrySynchronizer`
gained a second base class, and `message_type()` resolves through
`SubClassSafeGeneric`'s walk over `__orig_bases__`. The reasoning was that the walk
skips a base that is not a parameterized generic, so `PoseCovarianceSource` would be
ignored and `TopicInputSynchronizer[Odometry]` would still answer.

`test_odometry_synchronizer_reads_odometry_messages` and
`test_odometry_synchronizer_buffers_odometry_messages` both pass on that run, so
the dependency-inversion split costs nothing at the generic-resolution layer. An
implementing class may carry the abstraction without disturbing its own generic
binding, which is worth knowing for `estimator-node-base`.

### The one red check is a known flake

`test_integration_pr2.py::TestSelfCollisionAvoidance::test_attached_self_collision_avoid_stick`
failed. It asserts a closest-point distance of at least 0.048 after running the
real QP controller with self-collision avoidance over an attached box — a
tolerance-sensitive integration test, and the developer confirms it is flaky.

It is not this branch's: the diff touches odometry covariance capture, two inert
exception classes and two modules that nothing on that code path imports, and
`test_each_lib (giskardpy)` is green on both sibling branches off the same base —
`grasp-likelihood-continuous` (#8) and `belief-context-and-gaussian` (#10). The
failed jobs were re-run once rather than the test being touched.

### Still open

- Nothing was reviewed. No review threads, no pull request comments, no
  tracking-issue discussion, no merge conflict, and the branch is level with
  `main`. The pull request stays a draft awaiting its author's own review.
- The covariance frame is still unchecked, as the kickoff recorded. CI passing
  says the code does what it says, not that the frame assumption it inherits
  from the synchronizer is right.

## `belief-context-and-gaussian` — resolution

Nothing was blocking the design. The first CI run found one real defect, in the
one part of the change the kickoff had flagged as unverifiable locally — though
not the part it expected.

### What CI confirmed

The run on `81f0aded` came back 22 of 23 green, `test_each_lib (giskardpy)`
among them. That covers the 34 belief tests *and* the ORM exclusion in
`giskardpy/scripts/generate_orm.py`, which the kickoff recorded as "verified by
CI rather than by me" because local ORM regeneration needs ROS message packages
this container has not got. Excluding the belief classes from the scan works;
`check_generated_orm_interfaces_are_untracked` is green too.

### What it caught

`test_each_lib (version)`, one failure:
`test_imported_workspace_members_are_declared[giskardpy]`, asserting
`{'random-events'} == set()`.

The belief layer is the first direct import of `random_events` from giskardpy's
own source. Nothing failed at runtime, because `krrood` and
`semantic_digital_twin` both declare that package and giskardpy depends on both
— but the test exists precisely for that case: a tool reading the workspace
without building it, such as uv or an IDE's project model, sees only what
`[project] dependencies` lists, so an undeclared sibling import drifts out of
the workspace graph.

`random_events` is now declared in giskardpy's `[project] dependencies` and in
its `[dependency-groups] workspace`, the second so uv keeps resolving it from
the checkout rather than from PyPI. That is how every other member declaring a
workspace sibling spells it — `semantic_digital_twin` lists five that way.

The failure was reproduced locally first (`pytest test/version_test
--noconftest`, which sidesteps the root `conftest.py`'s ORM build), then shown
passing: 21 passed. `uv sync --extra dev` still resolves, and
`random_events.__file__` points into the checkout rather than site-packages, so
the workspace group entry does what it is there for.

### Worth carrying forward

This is a cost of the recorded decision to name a belief's dimensions with
`random_events` variables, not an argument against it. Any later item that
imports a workspace sibling a package has not imported before will trip the same
check — `estimator-node-base` and `symbolic-estimator-means` both inherit this
import, but through giskardpy, which now declares it.

### Still open

- Nothing was reviewed. No review threads, no pull request comments, no
  tracking-issue discussion, no merge conflict, and the branch is level with
  `main`. The pull request stays a draft awaiting its author's own review.
- The rest of the suite still cannot be run in this container: the root
  `test/conftest.py` regenerates the ORM interfaces at collection and that needs
  the ROS message packages. `--noconftest`, or running a fixture-free test file
  from a copy outside `test/`, is the way around it.

## `grasp-likelihood-continuous` — first review round

The item picked up its first review. Two threads, both from the author.

### The shared threshold constant is gone

*"no global variables like that. if you need some default, expose the parameter
and give it the default you think is sensible."*

The kickoff had introduced `GRIPPED_LIKELIHOOD_THRESHOLD` in
`robot_predicates.py` and pointed three call sites at it — `is_body_gripped`'s
default, coraplex's container post-condition, and `GraspLikelihood.true_above`.
The reasoning recorded at the time was `AGENTS.md`'s rule against magic numbers;
the reviewer reads its rule against globals as the stronger one, and a default
belongs on the parameter that needs it.

Each site now carries its own default again (`f345eacd`). The side effect is
worth noting: `robot_predicates.py` and `container.py` are byte-identical to
`main` again, so this branch touches **no existing production code at all** —
it is two new modules, two new exception classes, a shared test fixture and
tests. A smaller blast radius than the item started with.

The test asserting the node's default equalled the constant went with it. It
existed to pin a coupling that the reviewer deliberately removed, and rewriting
it as `0.9 == 0.9` would only restate a literal.

### Whether the krrood helper earns a single call site — open

*"as the observation variable is not what was ultimately used, is
trinary_logic_from_continuous still something that is used/needed?"*

Answered on the thread, left unresolved pending the author's decision, because
removing it would drop one of the three deliverables this roadmap records.

The premise needs one correction: the observation *is* used — what the design
rejected was using it as the carrier for the continuous value. The likelihood
rides the `FloatVariable`; the observation is a trinary view of it, and it is
what lets another node gate on this one's `observation_variable` and lets this
one earn a verdict. A node without an observation is a pure publisher nothing
can branch on.

So the question is where the mapping lives, not whether the node observes. The
helper has exactly one production call site (`grasp_monitors.py:92`). It buys a
name beside `trinary_logic_not`/`_and`/`_or`, the threshold-order guard, and one
place holding the exact-equality hazard. Against it: new public krrood API for a
single use. The two alternatives offered were inlining the `if_cases` in the
node, or dropping the node's observation entirely.

Worth recording for the sibling items: `odometry-covariance-capture` and
`estimator-node-base` both publish a `FloatVariable` and will each face the same
"what is my observation, then?" question. If the helper is kept, it is the
answer for all three; if it is inlined here, each will decide separately.

## `odometry-covariance-capture` — first review round

The item picked up its first review: four threads, all from the author, all on
how the covariance was modelled rather than on what it does.

### The general concept carries no ROS assumption; an adapter does

Two of the four threads were the same objection in two places — `PoseAxis`'s
docstring said its order was *"the order the ROS pose covariance stores them"*,
and `PoseCovariance.from_row_major` encoded the flat row-major layout a ROS
message happens to use. *"there should a general concept that we use, and an
adapter pattern that transforms ros assumptions into ours."*

The kickoff had put the type in giskardpy and recorded that a spatial
uncertainty type in `semantic_digital_twin` was deferred out of wave 1. The
review overrides that deferral, so the type moved:

- `semantic_digital_twin/spatial_types/pose_covariance.py` holds `PoseAxis` and
  `PoseCovariance` with no ROS in them. The matrix is six by six over the
  degrees of freedom of a pose, validated on construction by
  `PoseCovarianceNotSixBySixError`. `from_row_major` is gone — the name itself
  encoded a serialization layout.
- `PoseWithCovarianceToSemDTConverter`, beside the other converters in
  `adapters/ros/ros2_to_semdt_converters.py`, is the only thing that knows ROS
  stores those numbers flat and row-major. `OdometrySynchronizer` asks it.
- `PoseCovarianceSource` stays in giskardpy, in its own module. It is giskard's
  dependency-inversion seam, not a spatial concept.

The tests split the same way: the axis and summary tests are now
`semantic_digital_twin` spatial-type tests, and the row-major ones became
converter tests that fail on a transposed read.

### Where it lives also decided its ORM fate

`ORMatic.from_package([semantic_digital_twin])` maps every dataclass it finds,
so moving the type there is not neutral — it either becomes a table or goes in
`ignore_classes`. That was put to the user rather than decided in passing, and
the answer was to ignore it: nothing stores a covariance in the world, it is
read from a live input and republished to the statechart per control cycle.

Worth carrying forward for `detection-confidence-field`, whose own notes already
flag the same mechanism, and for any later item that adds a dataclass to
`semantic_digital_twin`.

### The named default became a field

*"no global variables like that. if you need this default named either expose
the parameter where you use it, or create an enum for it."* — the same objection
`grasp-likelihood-continuous` got about `GRIPPED_LIKELIHOOD_THRESHOLD`, and
taken the same way.

`UNCERTAINTY_WITHOUT_A_READING` is now `PoseUncertainty.uncertainty_without_a_reading`,
defaulting to infinity. The reasoning for infinity is unchanged and is now in the
field's own docstring. It also buys something the constant did not allow: a
caller who can bound the uncertainty before the first reading may say so, and
there is a test for that case beside the default one.

### And the covariance is typed with numpy typing

*"always use numpy typing"* — `values: npt.NDArray[np.float64]`. `main` made the
same move in `symbolic_math` in the meantime (`d169530a`).

### The move into `semantic_digital_twin` broke the ORM, and the exception was why

The first CI run after the review fixes came back **10 of 23 red**, and every
failure was one root cause: `sqlalchemy.orm.exc.MappedAnnotationError` importing
the generated `semantic_digital_twin/orm/ormatic_interface.py`.

Moving `PoseCovariance` into `semantic_digital_twin` moved
`PoseCovarianceNotSixBySixError` there with it, and `ORMatic.from_package` maps
every dataclass it finds — exceptions included, which is normal here; sdt's other
exceptions all have DAOs. What is not normal is a field typed as a bare `tuple`:
a shape has no fixed length, so there is no column type for it, and the generated
`PoseCovarianceNotSixBySixErrorDAO` could not be imported. Nothing that reaches
sdt's ORM could start — giskardpy, coraplex, experiments, sdt itself and every
notebook and demo job.

The ignore-it decision recorded above covered the type but not its error. Both are
in `ignore_classes` now, for the same reason: a shape mismatch is not something a
world stores. The fields are typed `tuple[int, ...]` rather than bare `tuple`,
which is what they always meant.

The failing run is also what verifies the fix: `PoseCovariance` itself was already
excluded and produced no DAO, so the mechanism demonstrably works on the very run
that caught this.

Worth carrying forward, and sharper than the note already written above: adding a
dataclass to `semantic_digital_twin` is never neutral, and *the exceptions that
come with it count as dataclasses too*. `detection-confidence-field` and
`probability-concepts-in-probabilistic-model` both add types to shared packages.
An unmappable field type fails at import of the generated module, not at
generation, so it takes down every dependent package at once rather than
producing one local failure.

This also could not be caught in the authoring container: regenerating the ORM
needs the ROS message packages, so the existing sdt ORM tests are the only thing
that exercises it, and they only run in CI.

### How the round closed

All four threads are resolved. The ROS-assumption thread on `pose_covariance.py:19`
was left open at the time pending the author's call on the ORM exclusion; the
answer was to ignore the type, and the thread was resolved on that basis.

`main` was merged into the branch twice over this round. The one conflict was
`semantic_digital_twin/exceptions.py`, where both sides appended new exception
classes; all were kept.

CI was fully green on `28dc12c2`, the commit before these fixes, and green again
on `546fb2ab` once they landed.

..note:: This section was filed under `belief-context-and-gaussian`'s first review
    round until the second round below moved it here. That item's pull request
    (#10) touches no `semantic_digital_twin` file at all — `PoseCovariance`, its
    error and the `generate_orm.py` exclusion are all this item's.

## `belief-context-and-gaussian` — first review round

Two review threads and one pull request comment, all from the author. The first
two are one ask; the third reopens a design decision recorded above.

### The arrays are laid out by quantity, not checked for shape

*"are these arrays always different in size? or can we extract a datastructure
from that we dont need these 'expected shape' checks everywhere?"* and *"is
there any way we can use proper datastructures instead of just numpy arrays
everywhere?"*

They are not always different in size: every array in the module had one of five
shapes, all of them derived from two numbers — how many quantities the belief is
about, and how many numbers a reading carries. Eight shape checks, an
eight-member `BeliefArray` enum and `WrongBeliefShapeError` existed only because
the arrays were unlabelled and the caller built them.

`Quantities` now holds the variables a belief is about and builds every vector
and matrix over them. An array cannot describe a different set of quantities
than the belief it belongs to, so there is nothing left for a shape check to
check; naming a quantity the belief is not about is the only remaining way to
get it wrong, and `VariableNotInBeliefError` already covered that. The enum and
the exception are gone.

This also turned the interfaces from positions into names. A belief is built
from what each quantity is estimated at and how pairs of them co-vary; `predict`
takes its transition, process noise and offset the same way, with
`Quantities.unchanged` for quantities expected to stay put; and a measurement is
a list of `Reading`s carrying what a sensor reported, how much each quantity
contributes to that number, and how far that sensor scatters. Reading the *sum*
of two quantities went from a hand-built `[[1.0, 1.0]]` row to
`contributions={a: 1.0, b: 1.0}`.

The honest cost: the mappings are built per call rather than reused, so a large
belief on a fast loop does work a preallocated array would not. At the sizes
this plan needs — a scalar grasp belief, a six-quantity pose belief — it is far
below the raycast already on the same tick. It also means a caller that *has* a
matrix, such as the 6×6 `odometry-covariance-capture` lifts off a ROS message,
must state it as pair-keyed entries; when a consumer needs the array path, that
item adds it rather than this one guessing at it.

This reverses the *"one shape error, not four"* decision recorded at kickoff.
That decision was defensible while the arrays were unlabelled; once they are
built from the quantities, consolidating the checks is worse than not needing
them.

### The probability concepts belong in `probabilistic_model`

*"I talked to the maintainer of Probabilistic Model and he said that there is
just no Multivariate Gaussian because he didnt need them yet, but they would be
cool to have. is this possible"*, and — in session — that the same applies to
covariance and to any other probability concept.

The kickoff recorded *"write the Kalman update by hand; do not force
`probabilistic_model` to be a filter"*, on the evidence that the package has no
multivariate Gaussian, no covariance, and conditioning that is truncation or
Dirac collapse. The first two are true but are absence rather than obstacle. The
third was wrong about the relevant operation: `ProbabilisticModel` is already
multivariate and keyed by `random_events` variables, and it declares
`conditional(point)` — conditioning on a value of a subset of the variables,
which for a joint Gaussian *is* the Kalman measurement update in closed form.
`apply_translation` and `apply_scaling`, also keyed by variable, are the
prediction step's ingredients. So the distribution belongs there, and so do the
concepts around it.

What makes it real work rather than an afternoon: everything under
`ProbabilisticModel` today is a `UnivariateDistribution`, so a multivariate
Gaussian needs its own place in that hierarchy, and two of the eight abstract
methods are hard for a *correlated* Gaussian —
`probability_of_simple_event` (the probability of an axis-aligned box has no
closed form and needs numerical integration) and `log_truncated` (a truncated
correlated Gaussian is not Gaussian, so it cannot answer with `Self`). That is
the more likely reason it does not exist yet.

**Decided with the user:** this lands as a new plan item,
`probability-concepts-in-probabilistic-model`, depending on this one, rather
than holding wave 2 behind another package or widening this pull request across
two of them. `GaussianBelief`'s internals then swap onto it. Because this round
made the interface variable-keyed — the same vocabulary `ProbabilisticModel`
uses — that swap is internals only, and `estimator-node-base`,
`grasp-belief-node` and the drawer experiment do not have to move for it.

`Quantities` itself is a candidate to move with them: an ordered set of random
variables plus the layout of arrays over them is a probability concept, not a
motion-control one.

### How the datastructure round closed

Both threads are resolved. The author's answer on the second was *"lets do
datastructures for now, if we notice bottlenecks we can still remove them again.
but add a comment in the doc as a reminder that there is the pure numpy option
one could go back to if we think this is too expensive."*

That reminder is a `..note::` on `Quantities`, the class that does the per-call
building and so the one whose cost a reader would question. It says the walk
costs something, that it stays far below a control cycle at the sizes a belief
is used at, and — the part that makes this reversible rather than merely
regretted — that `predict` and `update` still do their arithmetic on plain
numpy. Undoing it means changing what the signatures accept, not rewriting the
filter.

It is a deliberate exception to `AGENTS.md`'s rule against documenting a design
that was not chosen: the alternative here is a live fallback with a stated
trigger, not history.

CI is green on `50047b27` across the matrix, `test_each_lib (giskardpy)`
included, so the 36 tests behind the new interface are verified rather than
claimed.

## `belief-context-and-gaussian` — second review round

Two threads, both from the author, both on `beliefs/gaussian.py`, both posted
after the round-1 close and so untouched by it. Nothing else was blocking: CI was
23 of 23 green on `822b3e59`, the branch was level with `main` with no conflict,
and the fork pull request carries no `in-review` label, so there is no upstream
review to read.

### The estimate and the uncertainty became types

*"can these become actual datastructures instead of just numpy arrays? i think we
talked about this in another of the open draft PRs"*

Round 1 made everything a caller *hands* a belief quantity-keyed and stopped
there; what the belief itself *holds* stayed two bare
`npt.NDArray[np.float64]` fields. That is the half the second thread names, and
the other draft pull request it points at is `odometry-covariance-capture` (#9),
where the same objection produced `PoseCovariance`: a named type owning the array
and reading it by name.

`Mean` and `Covariance` follow that shape. Each carries the `Quantities` it is
laid out by and answers by quantity rather than by row — `Mean.estimate_of`,
`Covariance.variance_of`, and `Covariance.between` for how two quantities co-vary.
`GaussianBelief.mean_of` and `variance_of` read through them, so nothing outside
the module moved. The clearest gain is the one the row indices hid: a shared
uncertainty is now asserted as `covariance.between(first, second) == 1.0` rather
than against a nested list.

The `..note::` the author asked for in round 1 still holds, and this round is why
it is worth keeping: `predict` and `update` still do their arithmetic on plain
numpy behind the two types, so undoing this is still a change to what the
signatures accept rather than a rewrite of the filter.

### Building a belief is a classmethod

*"hmm i dont like initvars at all. use classmethods instead if you want this kind
of initialization"*

`GaussianBelief` had taken `estimates` and `uncertainty` as `InitVar`s and built
`mean` and `covariance` from them in `__post_init__`. Both `InitVar`s are gone:
the belief now takes the two things it holds, and `of(quantities, estimates,
uncertainty)` builds both from the same quantities, beside the existing
`of_one_variable`. `of` was already this module's word for a classmethod builder,
so the vocabulary did not grow.

`__post_init__` itself stays, for validation rather than construction — which is
what `PoseCovariance` does on #9 too, and is not what the thread objected to. That
was said on the thread, with the alternative offered: the check can move into
`of`, at the cost of the raw constructor letting it through.

### One thing the types made possible to get wrong

With `Mean` and `Covariance` each carrying their own `Quantities`, a belief can be
handed two that disagree, and then neither says anything about the other.
`BeliefQuantitiesDisagreeError` rejects it. This is not a return of the shape
checks round 1 removed: it names quantities rather than counting rows, and it is
one situation rather than eight.

### Two things this round settled that later items inherit

- **The ORM exclusion is by package, so it scales.** `generate_orm.py` ignores
  `classes_of_package(giskardpy.motion_statechart.beliefs)`, so `Mean` and
  `Covariance` needed no change to that script. That was checked directly against
  `classes_of_package` rather than assumed — regenerating the ORM still cannot run
  in the authoring container. `estimator-node-base` adding types to this package
  inherits the same exclusion for free.
- **The belief tests now run in the authoring container**, for the first time on
  this item. The blocker was never the belief code: the root `test/conftest.py`
  imports `urdf_parser_py`, a ROS package that is not on PyPI and does not build
  from source here. Everything else the belief layer needs installs from PyPI
  (numpy, casadi, scipy, sqlalchemy, rustworkx, mujoco, trimesh and a handful
  more) with the workspace packages installed `--no-deps`. Running the test file
  from a copy outside `test/` then works, and `--noconftest` covers a fixture-free
  file. Worth doing at the start of any later item on this plan rather than
  writing a wave of code unverified.

### Still open

- Nothing is outstanding on the branch. All four review threads are resolved, and
  the pull request stays a draft awaiting its author's own review, per this
  repository's convention that un-drafting *is* the record of having reviewed it.
- The branch was not this session's designated branch. The session resolving this
  round was designated `claude/belief-integration-gaussian-8yaw8t`; the user chose
  to keep the work on `claude/belief-integration-gaussian-zi1o82`, where the pull
  request and its threads are. `8yaw8t` was never created and has no commits. Any
  later session designated a fresh branch for an item that already has one should
  ask the same question rather than opening a second pull request for one item.

## `odometry-covariance-capture` — second review round

Four threads, all from the author, posted after the round-1 close and so
untouched by it. Nothing else was blocking: CI was 22 of 23 green on `546fb2ab`
with `test_each_lib (coraplex)` still running, the branch was level with `main`
with no conflict, and the pull request carries no `in-review` label, so there was
no upstream review to read.

Two of the four asked questions rather than naming a change, and the answers
below were the author's, given in session.

### `PoseAxis` is gone; a pose covariance is indexed by spatial variables

*"this feels a bit like duplicated information. is there any way to dedupe?"*

`PoseAxis`'s three translational members restated `SpatialVariables`
(`datastructures/variables.py`), which already carries `x`, `y` and `z` as
`random_events` `Continuous` variables and is used across the package. The enum
is removed. `SpatialVariables` gains `roll`, `pitch` and `yaw` beside them, plus
`position`, `rotation` and `pose` orderings; `pose` is what every array over a
pose's degrees of freedom is laid out by, and it is an ordered tuple rather than
the `SortedSet` its neighbours return, because a variable's place in it is the
row and column it occupies.

The resulting order is `x, y, z, roll, pitch, yaw` — the same order `PoseAxis`
numbered, so the ROS row-major read is unchanged and the adapter's transposition
test still fails on a transposed read.

Two alternatives were offered and rejected. Deriving the six from `Pose`'s own
`.x`/`.y`/`.z`/`.roll`/`.pitch`/`.yaw` accessors removes no duplicate: `Pose` has
six separate properties and no list of them, so the list would have been invented
and then tied to property names by string. Leaving the docstring as the only thing
deduplicated would not have addressed the enum.

### Building and reading it is by name, not by position

*"is there a way we can use a datastructure here instead of numpy arrays? I think
we did sth similar in one of the other open draft already"*

The other draft is `belief-context-and-gaussian` (#10), whose own second round
points back at this item as its precedent. Both land on the same shape: a named
type owning the array and answering by name rather than by row.

`PoseCovariance.of` takes the covariance of each pair of degrees of freedom and
fills the mirror entry itself, so a caller states a correlation once;
`covariance_between` and `variance_of` read by variable. `values` stays
`npt.NDArray[np.float64]` and the arithmetic stays numpy, exactly as `Mean` and
`Covariance` do on #10. The adapter still constructs from the array it lifts off
the message, which is the one place a laid-out matrix is the honest input.

Naming a variable a pose does not have now raises `VariableNotInPoseError`
instead of indexing at random. It joins its sibling in `generate_orm.py`'s
`ignore_classes`, per the lesson recorded above.

**What could not be reused, and why.** The datastructure the thread points at is
`Quantities`, and it lives in giskardpy. giskardpy depends on
`semantic_digital_twin` and not the reverse, so `PoseCovariance` cannot import it;
the private row lookup here is that layout rebuilt locally. `semantic_digital_twin`
does depend on `probabilistic_model`, which is where
`probability-concepts-in-probabilistic-model` is already slated to move
`Quantities` — so the reuse becomes possible then, and is tracked as
`pose-covariance-on-shared-quantities` rather than left as a comment.

### `PoseCovarianceSource` stays in giskardpy

*"should this also move to semdt?"*

No. Its only consumer is the `PoseUncertainty` node and its only implementer is
`OdometrySynchronizer`, both in giskardpy; nothing in `semantic_digital_twin`
would use it. It is giskard's dependency-inversion seam, not a spatial concept,
which is the call the first round already recorded. Answered on the thread and
left for the author to close.

### A covariance on a `Pose` is propagation, not a field

*"this sounds a bit like something that should be associated directly with a pose
no? or does this not fit into the bigger picture of this plan?"*, and then *"if we
dont want it to be roundtripped or orm mapped, cant we just make it a private
field?"*

The private field does work, and that was checked rather than assumed: ORMatic
skips fields whose name starts with an underscore
(`krrood/ormatic/wrapped_table.py:628`), and `Pose.to_json`/`_from_json` are
hand-written over position and rotation only, so such a field is invisible to both
the ORM scan and the JSON round-trip.

Storage was not the real objection. `Pose` composes and inverts, a covariance is
frame-dependent, and transforming a pose must rotate its covariance — so a field
would either silently survive those operations, now wrong, or silently vanish. A
wrong covariance is worse than an absent one, which is this plan's own standing
caveat. Most `Pose` instances are also symbolic forward-kinematics expressions
with no measured uncertainty at all.

So the association is real work, not a field, and it is tracked as
`pose-uncertainty-through-transforms`: rotate the covariance with the transform,
decide what an operation between an uncertain and a certain pose yields, and
attach it only then.

### Verification

The authoring container still cannot build `random_events` — the same antlr4
wheel failure every session on this plan has hit — so the twelve
`PoseCovariance` tests were run against the real module source with
`random_events.variable` stubbed down to a hashable `Continuous`. They pass. The
converter, synchronizer and monitor tests need `geometry_msgs`, `nav_msgs` and
`rclpy` and remain CI's to verify.

The row-major layout was checked directly as well: a 36-entry matrix counting up
reads `covariance_between(x, yaw) == 5` and `covariance_between(yaw, x) == 30`, so
a transposed read is still visible.

### Still open

- The covariance frame is still unchecked, as the kickoff recorded.
- Two threads — `PoseCovarianceSource`'s home and the `Pose` association — were
  answered rather than acted on, so they stay open for the author. The two that
  named a change are resolved.
- The branch was not this session's designated branch. The session was designated
  `claude/wonderful-edison-czfut6`; the work stayed on `claude/jolly-edison-k0zkiq`,
  where the pull request and its threads are — the same call
  `belief-context-and-gaussian` records making one round earlier.

## `probability-concepts-in-probabilistic-model`

The plan settled at kickoff, and the calls it makes beyond the item's recorded
`notes`.

### The base is #10's branch, not `main`

`motion_statechart/beliefs/gaussian.py` does not exist on `main` — the scope check
confirms `belief-context-and-gaussian` (#10) is the only branch that introduces it. So
this item stacks on `claude/belief-integration-gaussian-zi1o82` and is re-based onto
`main` once #10 lands. Removing the overlapping edits still leaves a whole multivariate
Gaussian in another package, so this is real work on top of an unlanded parent rather
than something to fold into it — which is also the call the user already made on #10's
comment thread.

The one file both branches change is `beliefs/gaussian.py`, which this item empties of
probability math. That is this item's whole purpose, so the overlap is the work rather
than a collision.

### `Quantities` does not live with the Gaussian

The item's `notes` say `Quantities` moves; they do not say where. It goes in
`probabilistic_model/quantities.py`, its own module, rather than beside the
distribution — because the item that consumes it next,
`pose-covariance-on-shared-quantities`, needs the layout for a *pose covariance*, which
is not a Gaussian. A reader looking for the layout should not have to open a
distribution to find it.

`Mean` and `Covariance` do go beside the distribution: they are the parameters a
Gaussian is written in, and that is where a reader looks for them.

### The Kalman update is conditioning on a joint, so a reading needs no variable of its own

`ProbabilisticModel.conditional(point)` conditions on a value of a *subset of the
model's own variables*, so expressing a measurement update through it means building the
joint over the estimated quantities together with what the sensors report, and
conditioning that on the reported numbers. The posterior over the quantities is then the
Kalman posterior, in closed form, with no gain written by hand.

`Reading` keeps the three fields #10 gave it — what the sensor reported, how much each
quantity contributes to that number, and how far the sensor scatters. The variables the
joint needs for the readings are built where the joint is, not carried in the public
interface, so nothing #10 exposes changes shape.

This is what makes the swap internals-only, which is the promise the item's `notes`
make to `estimator-node-base`, `grasp-belief-node` and the drawer experiment.

### The two hard methods are answered, not deferred

The item's `notes` name `probability_of_simple_event` and `log_truncated` as the real
work. Both are implemented rather than raised on:

- **`probability_of_simple_event`** is the probability of an axis-aligned box under a
  correlated Gaussian, which has no closed form. `scipy.stats.multivariate_normal.cdf`
  takes a `lower_limit`, which is exactly the rectangle probability by Genz's algorithm,
  so the numerical integration the item anticipates is scipy's rather than ours. A
  variable's interval may be a union, so the probability is summed over the boxes the
  per-variable intervals make. This needs `scipy>=1.10`, which `probabilistic_model`
  does not currently pin.
- **`log_truncated`** cannot answer with `Self`, as the item records, so it answers with
  a `TruncatedMultivariateGaussianDistribution` — the base signature already permits a
  `ProbabilisticModel` that is not `Self`. It holds the untruncated Gaussian and the
  event, and answers likelihood, probability, sampling and further truncation. Its mode
  is the mean when the mean is inside the event and genuinely intractable otherwise, so
  that case raises `IntractableError`, which is what that exception is for.

### The exceptions lose the word "belief"

`VariableNotInBeliefError`, `RepeatedVariableInBeliefError` and
`BeliefQuantitiesDisagreeError` are raised by the code that moves, so they move with it
— and a probability package has no beliefs, so they are renamed for what they actually
reject. `UnknownBeliefError` and `DuplicateBeliefError` stay in giskardpy: they are
`BeliefContext`'s registry errors, not probability.

### Declaring the dependency is part of the change

giskardpy does not import `probabilistic_model` today. #10 established what that costs:
`test_imported_workspace_members_are_declared` fails on the first undeclared import of a
workspace sibling, and the fix is `[project] dependencies` plus `[dependency-groups]
workspace`. Done here as part of the change rather than after CI says so.

### Scope boundaries held

No statechart node, nothing registered in `float_variable_data`, nothing added to
`WorldState` or the control loop — the same boundaries #10 held, and for the same
reason. `GaussianBelief` keeps its public interface exactly; only what is behind it
moves. `PoseCovariance` is deliberately *not* rebuilt on `Quantities` here, even though
this item is what makes that possible: that is
`pose-covariance-on-shared-quantities`, which is a tracked item of its own.

This is one pull request, not several. It has a single purpose — the probability
concepts end up in the probability package — and splitting the move from the swap would
ship a state where the same concepts exist twice.

### Assumptions and open points

- **Nothing consumes the truncated distribution.** It exists because `log_truncated` is
  abstract and the item names it as the work, not because the belief layer truncates.
  Its conditioning is the one query left unanswered: conditioning a truncated Gaussian
  on a point is a truncated Gaussian over the slice, which is real work with no caller,
  so it raises with the reason stated.
- **`Mean` and `Covariance` are Gaussian parameters, not general moments.** They are
  named for what a Gaussian is written in. If a later distribution wants the same pair,
  that is when they earn a home of their own.
- **This item is verifiable in the authoring container**, which no earlier item on this
  plan was: `probabilistic_model` and its dependencies install from PyPI, and the
  existing 114 distribution tests pass here before any change. The giskardpy half still
  needs CI, for the reason #10 recorded — the root `test/conftest.py` imports
  `urdf_parser_py`, which is not on PyPI.

## `estimator-node-base`

The plan settled at kickoff, and the calls it makes beyond the item's recorded
`notes`.

### The base is #10's branch, and it is the only branch this one needs

`motion_statechart/beliefs/` exists on `claude/belief-integration-gaussian-zi1o82`
and nowhere else, so this item stacks on #10 and is re-based onto `main` once #10
lands — the same call `probability-concepts-in-probabilistic-model` made. The
dependency check reports #10 `open_ready`, so the stack is on an open, non-draft
parent rather than on something still being drafted.

`probability-concepts-in-probabilistic-model` (#11) is *not* in this item's
dependency chain and is not in its base. It swaps `GaussianBelief`'s internals onto
`probabilistic_model` while keeping its public interface, which is the promise that
item's own roadmap section makes to this one — so nothing here has to wait for it or
move when it lands. #11 touches `beliefs/context.py` not at all, so the one edit this
item makes to that file conflicts with nothing currently in flight.

### The estimator publishes its uncertainty as well as its estimate

The item's `notes` ask for *"a FloatVariable for the mean"*. The node registers one
variable per quantity for the estimate and a second one per quantity for how
uncertain it is.

The uncertainty is what the whole wave is for, and it is what a condition is actually
written against — `odometry-covariance-capture` already publishes a variance and
nothing else, for a condition of the form *"do not begin the final approach while the
base is less certain than this"*. An estimator that published only its mean would hide
the one thing that distinguishes a belief from a number, and every concrete estimator
would then register the uncertainty variable itself, which is the duplication a base
class exists to remove. Additive to the recorded contract rather than a departure from
it: nothing that wanted only the mean has to change.

### What a subclass states, and what the base does with it

Three abstract methods, one per thing only a concrete estimator knows:

- `create_initial_belief` — the prior, which also names the quantities the node is
  about and therefore the variables it registers.
- `create_prediction` — how the belief is expected to change over one cycle, as a
  `Prediction` carrying the transition, the process noise and the offset
  `GaussianBelief.predict` already takes. Abstract rather than defaulted to
  "nothing changes, no noise": zero process noise is a belief that never grows less
  certain, which is exactly the failure #10's `predict` docstring names, so a subclass
  has to say what its quantity does when nobody is looking at it. `grasp-belief-node`'s
  "decays toward the prior when the gripper is open" is the offset half of one.
- `measure` — what the sensors reported this cycle, as #10's `Reading`s. Reporting
  nothing is normal and leaves the estimate on prediction alone.

The base owns the order — predict, measure, update, publish — so no concrete estimator
can get the tick lifecycle wrong, which is the same shape segmind's `AbstractDetector`
uses where `on_tick` does the shared work and delegates one abstract method.

### The node observes whether it measured this cycle

`grasp-likelihood-continuous`'s open review thread records that this item would face
the *"what is my observation, then?"* question a `FloatVariable` publisher always
faces. The answer here does not need that thread resolved: the estimator observes
whether `measure` returned anything, true when a reading corrected it and false when
the belief ran on prediction alone. That is the direct analogue of `PoseUncertainty`
observing whether a pose arrived at all, it is one of the three truth values so the
four exact-equality sites keep working, and it needs neither `trinary_logic_from_continuous`
nor a threshold.

A threshold on the estimate itself — *"is the grasp confident enough"* — is a concrete
estimator's to observe, not the base class's, because only the concrete estimator knows
what its quantity means. A subclass that wants it overrides `on_tick`.

### There is already a general read-before-built exception

`grasp-likelihood-continuous` and `odometry-covariance-capture` each appended a
near-identical `…NotBuiltError` to `motion_statechart/exceptions.py`, and the second
item's section records that generalizing the pair is this item's job. The
generalization turns out to exist already: `NodeNotBuiltError` is on `main`, takes the
node itself, and is what `ConvergingTask.error_signal` raises for exactly this case.

So this item adds no exception at all — it raises `NodeNotBuiltError`, and reuses
`VariableNotInBeliefError` for a quantity the estimator is not about. `exceptions.py`
is therefore untouched here, which also removes this branch from the three-way
conflict #8, #9 and #10 already have in that file. Pointing the existing two at
`NodeNotBuiltError` is a change to their own branches rather than something this one
can do without stacking on both.

### The beliefs are wired to the statechart here

#10 recorded that nothing adds a `BeliefContext` to a live context yet and that the
wiring belongs to this item. `MotionStatechartContext` can only be asked for an
extension in a way that raises when it is absent, or given one in a way that raises
when it is present, so `BeliefContext.of` answers with the statechart's beliefs and
starts it carrying them if it had none. It lives on `BeliefContext` rather than on the
node because it is knowledge about where beliefs live, which a later goal reading a
belief wants just as much as an estimator writing one.

The belief is registered at build, which runs exactly once per compile, so the
registration cannot be repeated by a node that is started, reset and started again.

### Two estimators about one quantity is rejected where it is wired, not at 20 Hz

`BeliefContext.add` already raises `DuplicateBeliefError`. Registering at build means
a statechart holding two estimators of the same quantity fails to compile, rather than
producing two filters that silently disagree for a whole run.

### Scope boundaries held

- **No concrete estimator.** `grasp-belief-node` is the first one and is a tracked item
  of its own; the tests here drive the base class through a mimic whose readings the
  test decides.
- **Nothing on the critical path beyond the filter.** `float_variable_data.set_value`
  is a write into an array and notifies nobody, so the tick cost is whatever a subclass
  measures plus the numpy arithmetic — the node itself never touches world state and so
  never reaches `notify_state_change`'s forward-kinematics recompute, which is the cost
  the item's `notes` say to watch. The node keeps its own handle on the belief rather
  than looking it up per tick, which is the same object the context holds because
  `predict` and `update` mutate in place.
- **Nothing added to the ORM.** `generate_orm.py` ignores
  `classes_of_package(giskardpy.motion_statechart.beliefs)`, and `classes_of_package`
  walks sub-modules, so a new module in that package is excluded without touching the
  script — the inheritance #10's second round recorded, checked rather than assumed.

### Assumptions and open points

- **Nothing reads the published variables yet.** No goal or condition is rewritten to
  use an estimate; `belief-weighted-open-goal` is where one is. The item asks only that
  it become possible.
- **The vector case is registered but not exercised by a real estimator.** A belief
  about several quantities publishes one variable per quantity; the tests cover it, but
  the first real multi-quantity estimator is `pose-uncertainty-through-transforms`'
  territory rather than this plan's wave 2.
- **The branch is the session's designated branch**,
  `claude/plan-item-kickoff-aicon-0t046w`, rather than the `estimator-node-base` name
  the manifest carried as a placeholder — the same correction every earlier item on this
  plan made.

## `probability-concepts-in-probabilistic-model` — what the implementation settled

What the implementation settled that the kickoff plan did not anticipate.

### Adding types here is ORM-neutral, unlike the shared package this plan warned about

`odometry-covariance-capture`'s first review round recorded a sharp lesson — adding a
dataclass to `semantic_digital_twin` is never neutral, exceptions included, and an
unmappable field takes down every dependent package at import — and named this item as
one of the two that would inherit it.

It does not. The ORM scans are `ORMatic.from_package([semantic_digital_twin])` and
`ORMatic.from_package([giskardpy])`; nothing scans `probabilistic_model`. So the
multivariate Gaussian, the layout and the three exceptions need no `ignore_classes`
entry and cannot break a generated interface.

What does still apply is the giskardpy half, and it holds for free:
`generate_orm.py` ignores `classes_of_package(giskardpy.motion_statechart.beliefs)`,
and that package now contains exactly `BeliefContext` and `GaussianBelief` — checked
directly against `classes_of_package`, since regenerating the ORM still needs the ROS
message packages this container has not got.

### Symmetrizing replaces Joseph form, and covers more than Joseph form did

`belief-context-and-gaussian` chose the Joseph form for `update` so the uncertainty
would stay symmetric under rounding. Conditioning produces the shorter
`(identity - gain @ model) @ covariance`, so that decision could not carry over as
written.

It is replaced by averaging the conditional covariance with its transpose. An
uncertainty is symmetric by definition, so any difference there is rounding, and this
removes it exactly rather than merely resisting it. It also applies to *every*
conditioning rather than only to a measurement update, which Joseph form did not. The
test that pins it uses a deliberately ill-conditioned covariance and fails without the
symmetrization.

### `predict` needed two operations the package did not have

The item's `notes` say `apply_translation` and `apply_scaling` are "the prediction
step's ingredients". They are not sufficient: a transition is a general linear map,
which diagonal scaling cannot express, and process noise is not a transform at all.

Rather than leave that arithmetic in giskardpy — which would leave `GaussianBelief`
still doing probability — the distribution gained `apply_linear_map` and
`apply_added_uncertainty`, beside the existing `apply_translation`/`apply_scaling`
family. `predict` is then those three operations in order and holds no numpy of its
own.

### What later items on this plan have to change, and what they do not

`GaussianBelief`'s interface held: #10's 29 belief tests pass unchanged except for
imports and renamed exceptions, which is the evidence for the promise this item made to
`estimator-node-base`, `grasp-belief-node` and the drawer experiment.

Two mechanical changes do reach them, and `estimator-node-base` (#12) is stacked on #10
in parallel with this item:

- `Quantities`, `Mean`, `Covariance` and `Reading` are imported from
  `probabilistic_model`, not from `giskardpy.motion_statechart.beliefs.gaussian`. A
  re-export was rejected: it would leave two names for one thing, and
  `pose-covariance-on-shared-quantities` will import `Quantities` from
  `probabilistic_model` anyway, so both paths would exist at once.
- `VariableNotInBeliefError` became `VariableNotInQuantitiesError` (its
  `belief_variables` field is `quantities`), `RepeatedVariableInBeliefError` became
  `RepeatedVariableError`, and `BeliefQuantitiesDisagreeError` became
  `MeanAndCovarianceDisagreeError`.

The raw `GaussianBelief(mean=..., covariance=...)` constructor is gone; `of` and
`of_one_variable` are the builders, which is what #10's second review round established
anyway. The test pinning a disagreeing estimate and uncertainty moved to the
distribution, where the check now lives.

Removing the three exceptions also took `List` out of
`motion_statechart/exceptions.py`'s imports, which is worth knowing for the three
branches already appending to that file.

### Verification, for once, was not left to CI

Every earlier item on this plan recorded writing code it could not run. This one runs:
`probabilistic_model` and its dependencies install from PyPI, and the 114 pre-existing
distribution tests pass here before any change. 266 pass across the collectible suite
afterwards, 29 on the belief tests, and 20 on the dependency declarations including the
giskardpy check that caught #10.

The three assertions worth trusting were each confirmed load-bearing by mutating the
implementation: dropping the correlation from the box probability fails only the two
orthant tests, not narrowing the conditional covariance fails only the four conditioning
and measurement tests, and removing the dependency declaration fails only
`test_imported_workspace_members_are_declared[giskardpy]`.

### Still open

- **The dashboard was not republished.** The `Artifact` tool treats this account's plan
  dashboard as a public third-party artifact, so a read returns a summary rather than
  the page source and the publish refuses with "you haven't viewed the latest version".
  `force: true` would discard whatever `estimator-node-base`'s session last published,
  so it was left alone for the user to decide.
- **The tracking-issue subscription was refused** by this session's permission mode, so
  nothing here watched #7 for concurrent structural changes. `estimator-node-base`
  started in parallel during this session and was picked up from the manifest instead.
- Regenerating the ORM remains CI's to confirm, as on every earlier item.

## `pose-uncertainty-through-transforms`

The plan settled at kickoff, and the calls it makes beyond the item's recorded
`notes`.

### The base is #9's branch

`PoseCovariance` and `SpatialVariables.pose` exist on `claude/jolly-edison-k0zkiq`
and nowhere else, so this item stacks on `odometry-covariance-capture` (#9) and is
re-based onto `main` once #9 lands — the same call every stacked item on this plan has
made. The dependency check reports #9 `open_ready`, so the parent is an open, non-draft
pull request rather than something still being drafted.

Removing the edits the two branches share still leaves the whole propagation and a new
spatial type, so this is real work on top of an unlanded parent rather than something to
fold into #9 — which is also the call the user already made on #9's review thread.

`pose-covariance-on-shared-quantities` will also edit `pose_covariance.py`, to rebuild
its private row lookup on `Quantities` once `probability-concepts-in-probabilistic-model`
moves that type. The two do not collide in purpose — that item changes how the matrix is
laid out, this one adds an operation over it — but whichever lands second should expect
to resolve that file.

### The propagation is the adjoint, not a rotation

The item's `notes` say to *"rotate the covariance with the transform"*. Rotating it is
right for a transform that is a pure rotation and wrong for every other one, so the
operation implemented is the adjoint.

A pose covariance describes a small perturbation of the reported pose, applied in the
frame the pose is expressed in: `reference_T_pose_true = exp(perturbation)
reference_T_pose`. Re-expressing that in another frame with a certain transform
`new_reference_T_reference` moves the perturbation through it, and

    new_reference_T_reference exp(perturbation) reference_T_reference_inverse
        = exp(adjoint * perturbation)

is exact, with the adjoint of a transform whose rotation is `R` and whose translation is
`t` being

    [[R, skew(t) R],
     [0, R       ]]

over the degrees of freedom in `SpatialVariables.pose`, which are the translational
three followed by the rotational three. So the covariance becomes
`adjoint @ covariance @ adjoint.T`.

The half a plain rotation would miss is `skew(t) R`: a pose that is uncertain about its
yaw and is re-expressed about an origin a metre away is uncertain about its position
there, by the square of that lever arm. Silently dropping that term is exactly the
*"a wrong covariance is worse than an absent one"* failure this plan's standing caveat
names, so it is what the adjoint identity is tested against directly rather than only
through the formula the code is written in.

### Inverting is the same operation, applied to the inverse

`(exp(perturbation) T)_inverse = T_inverse exp(-perturbation)`, and moving that
perturbation through `T_inverse` gives `adjoint(T_inverse) * -perturbation`. The sign
squares away under `covariance = adjoint @ covariance @ adjoint.T`, so inverting an
uncertain pose is the propagation applied with the pose's own inverse, and there is one
primitive rather than two.

### The attachment is a pairing, not a private field on `Pose`

The review thread on #9 asked *"cant we just make it a private field?"*. The mechanism
works — #9's own section records that `ORMatic` skips underscore-prefixed fields and that
`Pose.to_json`/`_from_json` are hand-written over position and rotation only — and it is
still the wrong place, for the reason the item's `notes` already give: a field that
silently survives or silently vanishes across compose and invert is worse than no field.

That hazard is structural rather than incidental. Every path that produces a `Pose`
rebuilds it from a CasADi expression — `HomogeneousTransformationMatrix.dot` through
`type(other).from_casadi_sx`, `_copy_with_data`, `to_pose`, and each `Pose.from_*`
classmethod — so a private field is dropped by all of them unless every one is threaded
through, and any path missed drops a covariance without saying so. A field that did
survive `dot` unrotated would be worse, because it would be wrong rather than absent.
Most `Pose` instances are also symbolic forward-kinematics expressions with no measured
uncertainty at all, so the field would be `None` on nearly every one.

`UncertainPose` is the attachment instead: a pose together with how uncertain it is,
whose operations move both halves together. It cannot lose its covariance, because there
is no way to hold one without it. It is also where the covariance finally gets a frame —
`PoseCovariance` is deliberately frame-naive, and pairing it with the pose says which
frame it is expressed in without that type having to carry one.

### There is already a general not-yet-a-number exception

A covariance can only be moved by a transform whose numbers are known, and most poses in
this stack are symbolic. `SymbolicMathType.to_np` already raises
`HasFreeVariablesError` in exactly that case, carrying the free variables, so this item
adds no exception at all — the same finding `estimator-node-base` made about
`NodeNotBuiltError`.

That also keeps `semantic_digital_twin/exceptions.py` untouched here, so this branch
stays out of the append-an-exception conflict #8, #9 and #10 already have in their own
packages' exception modules.

### The new type is kept out of the ORM

`ORMatic.from_package([semantic_digital_twin])` maps every dataclass it finds, and #9's
section records what that costs when a field has no column type: it fails at import of
the generated module and takes down every dependent package at once. `UncertainPose`
joins `PoseCovariance` in `generate_orm.py`'s `ignore_classes`, for the same reason its
covariance is already there — nothing stores an uncertain pose in the world; it is read
from a live input and used within a control cycle.

### Scope boundaries held

- **No consumer is rewired.** The `PoseUncertainty` monitor and `OdometrySynchronizer`
  stay as #9 left them. The item asks that carrying uncertainty through a transform
  become possible, not that something start doing it.
- **Composing two uncertain poses is not offered.** Doing so would have to assume the two
  are independent, which is the standard assumption and not one this plan can make
  quietly; the mixed case the item actually asks about — an uncertain pose and a certain
  transform — is what `transformed_by` answers, and a certain transform contributes
  nothing but its adjoint. When a caller needs the uncertain-on-uncertain case, that
  caller's item states the correlation model rather than this one guessing at it.
- **Nothing is added to `Pose`.** `spatial_types.py` is not touched, so no existing
  spatial operation changes behaviour.

### Assumptions and open points

- **The rotational degrees of freedom are read as a rotation vector**, a small rotation
  about each axis, not as Euler angles composed in a fixed sequence. That is the reading
  the ROS odometry covariance #9 lifts already has, and it is what makes the adjoint
  identity exact rather than an approximation about the nominal orientation. It is
  first-order in the size of the uncertainty, which is the model a covariance is.
- **Frames are not checked.** `transformed_by` does not verify that the transform's child
  frame is the frame the pose is expressed in, because
  `HomogeneousTransformationMatrix.dot` does not either, and adding a check on one half
  of an operation the package performs unchecked elsewhere would be inconsistent rather
  than safer. The covariance frame `odometry-covariance-capture` left unchecked stays
  unchecked.
- **Inverting inherits `HomogeneousTransformationMatrix.inverse`'s frame behaviour.** A
  `Pose` carries a reference frame and no child frame, so the inverse of one has no
  reference frame to name. The covariance round-trips exactly through two inversions; the
  frame does not come back.

## `estimator-node-base` — what the implementation settled

What the implementation settled that the kickoff plan did not anticipate.

### The giskardpy half runs in the authoring container now

`belief-context-and-gaussian`'s second round recorded that the belief tests run here
and that it was worth doing at the start of a later item rather than after writing a
wave of code unverified. Taken literally, and it went further than that round managed:
the whole `test_motion_statechart` directory collects, so this is the first item on this
plan whose giskardpy work is verified by the session that wrote it rather than by CI.

Two of the packages that round named as blockers do install, just not through pip's
build path. `urdf_parser_py` and `xacro` both fail to build a wheel against this
container's Debian-patched setuptools (`AttributeError: install_layout`), but both are
pure Python, so the sdist's package directory copied onto `site-packages` is enough.
Everything else is PyPI plus the workspace packages installed `-e --no-deps`, with
`ordered_set`, `platformdirs`, `pillow`, `plyfile`, `psutil`, `lxml`, `piqp`, `daqp`,
`matplotlib`, `pydot`, `pandas`, `inflect`, `lemminflect`, `plotly`, `tqdm` and
`giskardpy_bullet_bindings` picked up by following the import errors.

58 tests pass — this item's 14 and #10's 44 unchanged, which is the evidence that the
one edit to `beliefs/context.py` costs its parent nothing.

### The assertions were checked by breaking the code

Each of the four behaviours the tests name was confirmed load-bearing by mutating the
implementation and watching which tests fell over: dropping `predict` fails only the
process-noise and offset tests, dropping `update` fails only the three reading tests,
always observing true fails only the prediction-alone test, and not publishing the
uncertainty fails only the three tests that read it. No mutation took down a test that
does not name it, so the tests are separable in the way `AGENTS.md` asks for.

### The two published values are an enum, not two spellings

The variable a quantity's estimate is written to and the one its uncertainty is written
to are named by suffix, and a suffix naming a fixed thing is what `AGENTS.md` says to
give a `StrEnum` member. `PublishedValue` holds the two, so the names exist once rather
than at each registration — the same objection the reviewer raised twice on this plan
about named constants, applied before it was raised a third time.

### The belief is assigned only once the context accepts it

`build_artifacts` builds the prior, registers it, and only then keeps it. Assigning
first would leave a node that reads as built after a `DuplicateBeliefError` had already
failed the compile, which is a node in a state nothing should be able to observe.

### Still open

- **Nothing was reviewed**, and the pull request stays a draft awaiting its author's own
  review, per this repository's convention that un-drafting *is* that record.
- **The dashboard was not republished.** The `Artifact` tool treats this account's plan
  dashboard as a public third-party artifact, so a read returns a prose summary rather
  than the page source, and the publish then refuses with "you haven't viewed the latest
  version". `force: true` would discard whatever was last published there, so it was
  left for the user to decide. `probability-concepts-in-probabilistic-model`'s session
  hit the same wall in parallel, so this is the tool's behaviour on this artifact rather
  than either session's doing.
- **The tracking-issue subscription was refused** by this session's permission mode, so
  nothing here watched #7 for concurrent structural changes. #11's session was working
  the plan in parallel throughout; its roadmap section was read from the branch instead.
- **Regenerating the ORM remains CI's to confirm**, as on every earlier item — the
  exclusion was checked against `classes_of_package` directly rather than by running the
  generator.

## `probability-concepts-in-probabilistic-model` — first review round

The item picked up its first review: two threads, both from the author, both the same
objection. Nothing else was blocking — CI was 23 of 23 green on `f431efa7`, the branch
was `clean` against its base with no conflict, there were no pull request comments, no
tracking-issue discussion about this item since it was created, and the fork pull request
carries no `in-review` label, so there is no upstream review to read.

### Nothing holds a numpy array any more

*"can these be an actual datastructure instead of just np array?"* on `Mean.values`, and
*"here as well. anywhere where np arrays are used in this diff"* on `Covariance.values`.

This is the fourth round across this plan to make the same ask, and the first to aim it
one layer deeper than the previous three. #10's first round made everything a caller
*hands* a belief quantity-keyed; #10's second round made what the belief *holds* into
`Mean` and `Covariance`; #9's second round did the same for `PoseCovariance`. All three
stopped at a named type owning a `values: npt.NDArray[np.float64]`. This round says the
array itself should not be what the type holds.

So `Mean` holds `estimates`, what each quantity is expected to be, and `Covariance` holds
`uncertainty`, how uncertain each ordered pair of them is. Neither holds an array.
`as_array` builds one for the arithmetic and `from_array` reads one back off the
arithmetic's result; those two are the only places numpy appears on either type, and the
distribution's internals go through them rather than indexing rows.

### Reading an array back must not symmetrize it

The one thing this could have quietly cost. `Covariance.from_array` records each
direction of a pair separately rather than averaging them. Had it symmetrized on the way
in, `test_the_conditioned_uncertainty_stays_exactly_symmetric` would have become true by
construction and stopped testing the symmetrization it exists for.

The test now states its assertion by quantity — `between(first, second) ==
between(second, first)` — which is what the datastructure buys over comparing an array to
its transpose, and it still fails when the symmetrization is removed. That was checked by
removing it, not assumed.

### What stays an array, and why

Answered on the open thread rather than decided unilaterally, because one of the three is
the author's call:

- **`ProbabilisticModel`'s own abstract signatures.** `log_likelihood(events:
  npt.NDArray) -> npt.NDArray` and `sample(amount) -> npt.NDArray` are the base class's
  contract (`probabilistic_model.py:118`, `:271`), `likelihood` calls straight into
  `log_likelihood`, and every other distribution in the package implements them that way.
  Changing them here would make this one distribution non-substitutable for its own base.
  Changing them everywhere is a piece of work across the package, not something to fold
  into this item — put to the author, thread left open.
- **`Quantities.vector`/`matrix`/`symmetric_matrix`.** Producing the array *is* their job;
  they are the boundary rather than a leak through it.
- **The private arithmetic helpers** — matrix inverse and block assembly.

### The cost note now covers two layers

The `..note::` on `Quantities` the author asked for in #10's first round said the layout
is walked per call and that the arithmetic stays numpy, so the decision is reversible. It
now also says that what holds the numbers walks it again to hand an array over, so one
operation may build the same array twice. That is the honest cost of this round, and it
is the thing to undo first if a profile ever shows it.

### `PoseCovariance` is left as it is

`Mean` and `Covariance` now hold named data while `PoseCovariance` on #9 still holds
`values: npt.NDArray[np.float64]`, which is the shape #9's own second round settled on.
That is not a problem for `pose-covariance-on-shared-quantities`: that item collapses
`PoseCovariance` onto `Quantities`, which this round did not change. Flagged on the thread
as the place where matching the two costs least, rather than widened into this item.

### Verification

`test_the_conditioned_uncertainty_stays_exactly_symmetric` and the two transform
round-trip tests were rewritten to read by quantity or through `as_array`; six new tests
pin what the two types now hold, that a quantity left out is still held, that an estimate
survives the trip through an array, and that reading a matrix back keeps both directions
of a pair apart.

272 passed across the collectible `probabilistic_model` suite, 29 on #10's belief tests,
20 on the dependency declarations. #10's belief tests needed only `.values` →
`.as_array` where they genuinely want the matrix — an inverse, a transpose and an
eigenvalue check, all properties of a matrix rather than of any named pair — which is
why that accessor is public rather than private.

### Still open

- The `npt.NDArray` on `ProbabilisticModel`'s abstract `log_likelihood` and `sample`,
  above. The second thread is left open for it; the first is resolved.
- CI has not yet run on `6da3fc22`.

## `pose-uncertainty-through-transforms` — first review round

The item picked up its first review: two threads, both from the author. Nothing else
was blocking — CI was 23 of 23 green on `a2d6dbe9`, including the
`test_each_lib (semantic_digital_twin)` run that exercises the ORM exclusion the
kickoff could only check by reading `classes_of_package`; the branch was level with
its base with no conflict; there were no pull request comments, no tracking-issue
discussion about this item since it was created, and no `in-review` label, so no
upstream review to read.

One thread asked for a change already familiar on this plan. The other found a gap.

### The frame change became a type, and the global function went with it

*"can we make the np arrays actual types?"* and *"that way we also dont need this global
function"*, on `_cross_product_matrix`.

This is the fifth round across this plan to ask for a named type instead of a bare
array, and the kickoff had left exactly one place still answering with one: the adjoint
was built and returned as a 6×6 by a private `_adjoint_of`, helped by a module-level
`_cross_product_matrix`.

`PoseDisplacementMap` replaces both. It holds `factors: Mapping[PoseVariablePair,
float]` — how much each degree of freedom seen in the new frame follows each one in the
frame the pose is reported in — `of_transform` reads that off a transform, the cross
product is a private static method on it, and `as_array` is the single place it becomes
numpy for the arithmetic. The same shape `Mean` and `Covariance` took on #11.

The gain shows in the tests: the lever-arm term was previously asserted through a
variance after a full round trip, and is now asserted directly as
`factor_of(y, yaw) == -lever_arm`.

### The row a degree of freedom occupies moved to where the ordering lives

Both types needed it, so keeping `PoseCovariance._row_of` private would have duplicated
the lookup. It is now `SpatialVariables.row_in_pose`, beside the `pose` ordering that
defines it, and both types read through it. `semantic_digital_twin.exceptions` does not
import `datastructures.variables`, so raising `VariableNotInPoseError` from there
introduces no cycle — checked rather than assumed.

This also helps `pose-covariance-on-shared-quantities`: one place to collapse onto
`Quantities`, rather than two.

### Uncertainty does travel up the chain, and that case was missing

*"but there may also be uncertain transforms no? for example if we have a drawer
connection whose state is uncertain? also if there is a bottle inside the drawer, doesnt
the uncertainty of the drawer position propagate upwards to the bottle pose as well?"*

It does, and the kickoff had implemented only half of it. `transformed_by` applies a
certain transform on the *left* — a change of reference frame. The drawer and bottle put
the uncertain pose on the left and the certain offset on the right, which is a different
operation with a different answer:

- `certain @ uncertain` re-expresses the pose in another frame, and the displacement has
  to be carried through that frame change.
- `uncertain @ certain` extends the pose along the chain, and the displacement is
  carried **unchanged**: `exp(perturbation) T B` relative to `T B` is still
  `exp(perturbation)`, exactly, with no approximation and no assumption.

`UncertainPose.dot`, with `__matmul__` beside it so the style guide's own
`a_T_c = a_T_b @ b_T_c` reads the same whether or not a pose is uncertain.

The part that could mislead, and so is a `..note::` on `dot`: unchanged does not mean
the far end is equally well located. The displacement acts about the reference frame's
origin, so the same perturbation moves a more distant frame further. Reading that as
uncertainty about the far end's own position is `transformed_by` — a drawer with
0.04 rad² of heading uncertainty gives a bottle a metre out 0.04 m² of lateral variance,
but only once the covariance is re-expressed about the bottle.

### Two uncertain poses are refused rather than assumed independent

The kickoff recorded not offering uncertain-on-uncertain composition, on the grounds
that it needs an independence assumption. That stands, but silently having no operation
was the wrong way to express it: a caller reaching for it got a `TypeError` naming
nothing.

`uncertain @ uncertain` now raises `UncertaintyCorrelationUnknownError`, whose message
says the answer depends on whether the two uncertainties are related and whose
`suggest_correction` points at extending by a certain transform instead. The first-order
formula and the offer to add the independent version went on the review thread; that
thread is left open for the author, per this repository's convention that a thread
carrying a question back is not resolved.

The exception has no fields — there is nothing informative to carry — and joins its
siblings in `generate_orm.py`'s `ignore_classes`. A field-less dataclass would map
harmlessly, but a DAO that fails to generate takes down every dependent package at
import, and only CI can check that, so the conservative entry is worth more than the
empty table it avoids.

### Verification

`test_spatial_types/` is 344 passed, 1 failed, the failure being the same
`TestVector3::test_length_0` the kickoff recorded and re-confirmed here by stashing the
diff. The refactor is covered by the kickoff's own perturbation-identity test, which
still passes unchanged and is what says the propagation still means what it meant.

The container reached further this round, following #12's finding that `urdf_parser_py`
and `xacro` install from their sdists' package directories even though their wheels fail
against this container's Debian-patched setuptools. `test_datastructures` collects with
those plus `lxml`, `daqp`, `piqp` and `giskardpy_bullet_bindings` and `giskardpy/src` on
the path. Two tests there error on a CasADi API mismatch —
`FunctionBuffer_set_res` rejects the arguments the forward-kinematics memory binding
passes — which reproduces identically with this diff stashed and is this container's
casadi 3.8.1 rather than anything in the change.

### Still open

- The uncertain-on-uncertain thread, above, awaiting the author's call.
- `PoseCovariance.values` is still a bare array, which the reply to the first thread says
  explicitly. #9's second round settled that shape and #11's session re-confirmed it,
  naming `pose-covariance-on-shared-quantities` as where matching it costs least. Offered
  to bring it forward if wanted.
- **The dashboard is still not republished.** The `Artifact` tool treats this account's
  plan dashboard as a third-party artifact, so a read returns a prose summary rather than
  the page source and the publish refuses with "you haven't viewed the latest version".
  The user was asked at kickoff and chose to skip rather than force-overwrite or mint a
  duplicate; that choice stands for this round. `plan.yaml` and this file are current.
- The tracking-issue subscription was refused by this session's permission mode, as at
  kickoff. Issue #7's comments were read directly instead; nothing there concerns this
  item beyond its creation.

## `pose-uncertainty-through-transforms` — second review round

One thread, and it closed a deferral rather than naming a change. Nothing else was
blocking: CI on `ae1d92d0` is green on every completed check, `test_each_lib
(semantic_digital_twin)` among them, with three jobs still running; the branch is level
with its base; there are no pull request comments and no `in-review` label, so no
upstream review to read. The first round's other thread is resolved and outdated.

### The deferred half became a plan item, at the author's request

The first round left `uncertain @ uncertain` raising
`UncertaintyCorrelationUnknownError`, with the first-order independent formula and an
offer to implement it put on the thread. The author's answer:

*"hmm okay i see. Please make this a seperate plan item in case i want to get back to
this, but for now we should be able to continue without right? or will the choice
significantly change how we move forward?"*

So `uncertain-pose-composition` is a new item in this track, depending on this one, and
is broadcast on issue #7 per this plan's own convention for a structural change.

### Why continuing without it changes nothing

The answer to the author's question, recorded because it is the reason the deferral is
safe rather than merely convenient.

- **Nothing else in the plan composes two uncertain poses.**
  `pose-covariance-on-shared-quantities` changes a matrix layout;
  `estimator-node-base`, `grasp-belief-node` and `belief-weighted-open-goal` work on
  scalar beliefs published as float variables; the wave-3 items differentiate estimator
  means over joint positions. None of them multiplies one uncertain pose by another.
- **The refusal is strictly narrower than any answer it could be replaced with.**
  Today that call raises, so no working caller can exist. Replacing the raise with a
  value later cannot break one. That is what makes this a deferral rather than a fork in
  the road.
- **The one thing that would reach further** is if the eventual model needed
  `UncertainPose` to *carry* how its uncertainty relates to others, rather than assuming
  independence at the call. That would change the type's shape — but still nothing built
  in the meantime, for the first reason above.

### Why it is a separate item and not folded back

`scope-decision.md`'s mechanical check reports `uncertain_pose.py` absent from `main`
and introduced by this branch, which is the shape that usually argues for folding. It
does not here, on that document's own test: the work stands on its own once this item
lands, because it adds an operation to a file that is then on `main`, rather than
existing only to correct what the parent is about to ship. Folding it would also hold
this pull request open until a design question the author has explicitly parked is
answered, which is the opposite of what was asked.

### Nothing was pushed to the branch

This round changed no code. The refusal, its message and its `suggest_correction`
already say what the new item will decide, and the item's own `notes` carry the formula,
so there is nothing for a comment in the source to add that would not go stale.

### Still open

- `PoseCovariance.values` is still a bare array, as the first round recorded. Unchanged
  and still offered.
- Three CI jobs on `ae1d92d0` had not finished at the time of this round —
  `test_each_lib (giskardpy)`, `test_each_lib (coraplex)` and the `coraplex_real_tracy`
  demo. Every completed check is green.
- **The dashboard is still not republished**, for the third round running: the
  `Artifact` tool treats this account's plan dashboard as a third-party artifact, so the
  publish refuses without `force`. The user chose to skip at kickoff rather than
  force-overwrite or mint a duplicate, and that choice stands. It now also means the new
  item is in `plan.yaml` and on issue #7 but not on the published page.
- The tracking-issue subscription is still refused by this session's permission mode;
  issue #7 was read directly.

## `grasp-likelihood-continuous` — restack resolution

The stall was mechanical, not a design question, and the one thing the first
review round left open has since been answered.

### The blocker was a restack conflict, and the recorded open question was stale

The item's recorded state said the first review round left
`trinary_logic_from_continuous`'s single call site undecided. That is no longer
true: the author answered on the thread — *"okay good keep it"* — and both of
#8's review threads are resolved. Nothing on the review side was blocking.

What was blocking is that the stack maintenance pass could not integrate `main`:
it reported a conflict in `krrood/src/krrood/symbolic_math/symbolic_math.py` and
`test/krrood_test/test_symbolic_math/test_symbolic_math.py`, left the branch
untouched, and labelled #8 `needs-resolution`, which withholds it from
promotion. CI was 23 of 23 green on `fece1ffd` throughout, so a dashboard reading
checks and review state alone would have shown nothing wrong.

### Both conflicts were the same import-list collision

`main`'s #650 (`e87f86da`, *"remove JSON serialization exceptions for symbolic
and spatial types"*) deleted `SymbolicMathNotJsonSerializableError` along with
every use of it. This branch had added `ThresholdsOutOfOrderError` to the same
import block in both files. Git could not tell the deletion and the addition
apart, so it conflicted on the two adjacent lines; every other hunk of both files
auto-merged.

The resolution keeps the added exception and drops the removed one. Both halves
were checked rather than assumed: `SymbolicMathNotJsonSerializableError` has no
remaining reference anywhere in the merged tree, and `ThresholdsOutOfOrderError`
is still defined in `symbolic_math/exceptions.py` (which auto-merged cleanly) and
still raised by the helper.

### The merge was verified against a baseline, not against an expectation

The symbolic-math suite on the merged tree reports 12 failed, 283 passed. Plain
`origin/main` in the same container reports 12 failed, 276 passed, and the two
failure sets are *identical* — eleven `TestJsonSerialization` cases plus
`test_jacobian_ddot`, all pre-existing and all green in CI, so they are this
container's casadi 3.8.1 rather than anything in the merge. The difference is
exactly this branch's seven `TestTrinaryLogicFromContinuous` tests, all passing.

Diffing the failure set against a `main` baseline is what makes "the merge broke
nothing" a measurement rather than a claim, and it is cheap enough to be worth
doing on every restack this plan has left.

CI on the merge commit `76bab6d3` agrees: all 23 checks are green, including
`test_each_lib (krrood)` — the suite that covers both conflicted files — and
`test_each_lib (giskardpy)`, which covers the node the helper is called from.

### The container reached the krrood suite this time

Earlier rounds on this item recorded that nothing could be run locally. The
symbolic-math tests do run here: `--noconftest` sidesteps the root
`test/conftest.py`'s ORM build, and `casadi`, `numpy`, `scipy`,
`typing_extensions`, `sqlalchemy`, `ordered_set` and `rustworkx` all install from
PyPI with `krrood/src` on the path. That is the same finding
`belief-context-and-gaussian` and `estimator-node-base` recorded for their own
packages, and it now covers krrood too.

### Scope boundaries held

- **No production code changed.** The merge commit is the only change; the diff
  against `main` is what it was before the restack.
- **`GraspLikelihoodNotBuiltError` was left alone.** `estimator-node-base`'s
  section records that `NodeNotBuiltError` already exists on `main` and that
  pointing this branch's exception at it is this branch's change to make. It is a
  real simplification and no reviewer has asked for it, so it stays flagged here
  rather than folded into a conflict resolution. Worth doing on this item's next
  code push, if there is one.

### Still open

- **The label cleared and the branch was promoted, within the session.** A stack
  maintenance pass ran once CI went green, dropped `needs-resolution`, and
  replaced it with `cram2-link-sent` plus a "Promote" section on #8 carrying the
  upstream compare-and-create link. `mergeable_state` is `clean`. So the item is
  out of the stall entirely and is waiting on the author to click Create
  upstream — the one step the pass never takes itself.
- **#8 was deliberately left out of draft.** Un-drafting is this repository's
  record of the author having reviewed it, the push changed no production code,
  and re-drafting would have withdrawn it from the promotion queue for a no-op
  merge. Confirmed with the author in session rather than assumed from the
  standing convention.
- **The work stayed on the item's own branch.** This session was designated
  `claude/plan-item-resolve-aicon-belief-9dx2nz`, which was created empty at
  `main`; the merge belongs on `claude/plan-item-kickoff-aicon-belief-84kl68`
  where #8 is. Asked and confirmed, per the note
  `belief-context-and-gaussian`'s second round left for exactly this case.
- **The dashboard was not republished**, for the same reason every round since
  `probability-concepts-in-probabilistic-model` has recorded: the `Artifact` tool
  treats this account's plan dashboard as a public artifact from outside the
  organization, so the read returns nothing and the publish cannot proceed
  without `force`. Forcing it would discard whatever the sessions restacking #9
  and #10 in parallel published, so it was left alone. `plan.yaml` and this file
  are current; the page is not.
- **The tracking-issue subscription was refused** by this session's permission
  mode, as on every earlier round. Issue #7's comments were read directly
  instead; the three structural changes recorded there concern other items.

## `odometry-covariance-capture` — restack resolution

The stall was mechanical, and the same one `grasp-likelihood-continuous` hit: the
stack maintenance pass could not integrate `main`, so it left the branch alone
and labelled #9 `needs-resolution`, which withholds it from promotion. Nothing on
the review side was blocking — all eight review threads are resolved, including
the two the second round left open for the author, and CI was 23 of 23 green on
`d8c6c10b` throughout. A dashboard reading checks and review state alone would
have shown nothing wrong.

### The conflict was the append-an-exception collision this plan predicted

`giskardpy/src/giskardpy/motion_statechart/exceptions.py`, and it is exactly the
three-way overlap the resolution round recorded a warning about — except the
other side is now `main` rather than a sibling branch. `main`'s #650 added
`NodeStateVariableNotSerializableError` at the end of the file; this branch had
added `PoseUncertaintyNotBuiltError` at the same point. Git could not tell the two
appends apart.

Both are kept, and neither is optional — that was checked rather than assumed.
`graph_node.py:47` imports `NodeStateVariableNotSerializableError` and raises it in
`to_json`; `uncertainty_monitors.py:10` imports `PoseUncertaintyNotBuiltError` and
raises it when the published variable is read before the node is built. Dropping
either side would have broken an import. The file's import block auto-merged, so
`JSONSerializationError` and `NodeStateVariable` were already in place.

`semantic_digital_twin/exceptions.py` auto-merged this time; `main` added no class
to it, and this branch's two survived.

### The merge was verified against a baseline, not against an expectation

Following the recommendation `grasp-likelihood-continuous`'s own restack left.
The `semantic_digital_twin` spatial-type suite on the merged tree reports 1 failed,
321 passed; plain `origin/main` in the same container reports 1 failed, 309 passed,
and the failure is the same `TestVector3::test_length_0` both times — the one
`pose-uncertainty-through-transforms` already recorded as this container's casadi
3.8.1 rather than anything in a diff. The difference is exactly this branch's twelve
`PoseCovariance` tests, all passing.

That the failure is `main`'s is also true by construction: after the merge,
`spatial_types.py`, `test_spatial_types.py` and `reference_implementations.py` are
byte-identical to `origin/main`.

The ten dependency-declaration tests pass, giskardpy's among them, so the merge
introduced no undeclared workspace import.

### `random_events` builds here now, which no earlier round managed

Every previous session on this plan recorded the same antlr4 wheel failure and
worked around it by stubbing `random_events.variable` down to a hashable
`Continuous`. The wheel still fails, but the fix is not to build it: `pip install
--no-deps random_events` takes the manylinux wheel, which ships the compiled
`random_events_lib` the checkout's own source needs. With that installed and
`random_events/src` on the path, the checkout's `random_events` — `plotting`
included, which the wheel lacks — imports and works.

So the twelve `PoseCovariance` tests ran against the real module this round rather
than against a stub. `antlr4` itself, if a later item needs it, installs the same
way `urdf_parser_py` and `xacro` do: the sdist's package directory copied onto
`site-packages`.

### What still cannot run here, and why it is a harder boundary than it looks

The monitor, converter and synchronizer tests remain CI's. Two separate walls,
and only the first is avoidable:

- The root `test/conftest.py` regenerates the ORM, which needs the ROS message
  packages. `--orm-build=never` is the supported way past it — a real option on the
  option parser, not a workaround — and it works.
- `test/giskardpy_test/conftest.py` imports `GiskardTester`, which imports `rclpy`.
  The `mini_world` fixture those tests want is pure `semantic_digital_twin` and
  touches no ROS at all, so only that one unused import stands between this
  container and the monitor tests. Stubbing it was tried and abandoned: the
  stub has to satisfy `from rclpy.x import a, b` across the middleware layer, at
  which point it is reimplementing ROS rather than standing in for it.

Worth knowing rather than re-deriving: the giskardpy suites on this plan are
blocked by one import in a conftest, not by the code under test.

### Scope boundaries held

- **No production code changed.** The merge commit is the only change, and the diff
  against `main` is what it was before the restack.
- **`PoseUncertaintyNotBuiltError` was left alone.** `estimator-node-base`'s section
  records that `NodeNotBuiltError` already exists on `main` and that pointing this
  branch's exception at it is this branch's change to make — the same call
  `grasp-likelihood-continuous` made about its own. It is a real simplification, no
  reviewer has asked for it, and folding it into a conflict resolution would widen a
  no-op merge. Worth doing on this item's next code push, if there is one.

### Still open

- **The `needs-resolution` label is still on #9.** The stack pass clears it once the
  branch merges cleanly again, which it now does, so the next pass should drop it and
  let the branch rejoin promotion.
- **#9 was deliberately left out of draft**, for the reason `grasp-likelihood-continuous`
  recorded and confirmed with the author one round earlier: un-drafting is this
  repository's record of the author having reviewed it, the push changed no production
  code, and re-drafting would have withdrawn it from the promotion queue for a no-op
  merge. #9 also carries a requested reviewer, which #8 did not.
- **The work stayed on the item's own branch.** This session was designated
  `claude/odometry-covariance-capture-frq0bu`; the merge belongs on
  `claude/jolly-edison-k0zkiq` where #9 and its eight threads are. Settled rather than
  asked again — `belief-context-and-gaussian`'s second round left the note, this item's
  own second round made the same call, and `grasp-likelihood-continuous` confirmed it
  with the author.
- **The covariance frame is still unchecked**, as every round since the kickoff has
  recorded.
- **`PoseCovariance.values` is still a bare array**, as #11 and #13 both record.
  `pose-covariance-on-shared-quantities` is where matching it costs least.
- **The tracking-issue subscription was refused** by this session's permission mode,
  as on every earlier round. Issue #7's comments were read directly instead; the three
  structural changes recorded there concern other items.

## `belief-context-and-gaussian` — restack against `main`

The first round on this item where the blocker was neither a design question nor a
review thread. Nothing was wrong with the branch; `main` moved and took the one
file this item shares with it.

### The recorded blocker was right, and it was mechanical

`plan.yaml` carried no blocker, and the item's own notes describe settled design.
The stall was on the pull request itself: `mergeable_state` was `dirty`, the fork
pull request carried the `needs-resolution` label, and the stack maintenance
routine had commented saying it hit a conflict integrating `main`, left the branch
untouched and skipped it so later passes would not re-report the same thing.

The conflicting file was the one the roadmap has been predicting since round 1:
`giskardpy/src/giskardpy/motion_statechart/exceptions.py`. What it did not predict
is which branch would collide. Three sibling items — #8, #9 and #10 — were each
recorded as appending their own exception classes there, and the standing note said
whichever two landed second and third should expect to resolve it. In the event the
collision was not with a sibling at all: `main` itself gained
`NodeStateVariableNotSerializableError` from upstream #650 (symbolic-math JSON
serialization), appended at the end of the same file this item appends its five
belief exceptions to.

So the note generalizes: that file is a shared append point for the *whole
repository*, not only for this plan's siblings, and any branch touching it should
expect to resolve it against `main` rather than only against a sibling.

### The resolution is the union, and it was checked as one

Both sides only appended. The resolution keeps `main`'s exception and this branch's
five, with the union of the imports each needs — `List` and `Continuous` for the
belief exceptions, `JSONSerializationError` and `NodeStateVariable` for `main`'s.
git had already auto-merged the import block correctly; only the two appended
blocks conflicted, and only because git interleaved them.

Rather than hand-edit the conflict markers, the tail was rebuilt from the two
parents directly and then diffed against each: the only line not carried over from
`main`'s version is its narrower `typing_extensions` import, and nothing at all is
dropped from the branch's. That is what says the resolution is a union rather than a
merge that happens to compile.

The repository's default is merge rather than rebase (`.claude/stack/stack.toml`'s
`rebase_label` is opt-in and #10 does not carry it), so this is a merge commit and
the branch's own history is untouched.

### Verification ran here, and the two failures were proven not to be ours

The belief tests pass on the merged tree — all 44, unchanged. The 20
dependency-declaration tests pass, including the
`test_imported_workspace_members_are_declared[giskardpy]` that caught this branch's
first round. Both merged exception classes import, and `graph_node.py`'s use of
`main`'s one resolves against the merged module.

Two failures in this container were each run on `origin/main` alone before being
attributed: `test_all_package_versions_match_root_version` fails because no
`coraplex` is installed here, and `test_jacobian_dot`/`test_jacobian_ddot` fail on a
casadi API mismatch — the same one #13's session recorded. Both reproduce
identically without this merge.

### The container recipe changed, and is worth carrying forward

Round 2 recorded that the belief tests run here. They no longer run the way it
described: `random_events` builds a C++ extension that fails here, and the workspace
source now raises `module 'random_events_lib' has no attribute 'reals'` against
whatever stale library is present. Installing the **PyPI wheel** of `random_events`
supplies a matching `random_events_lib`, and the workspace source then imports
against it — so the real workspace code still runs, rather than a stub of it.

The other change is to skip the editable install entirely: some workspace members
fail on their build dependencies here, and putting each package's `src/` on
`PYTHONPATH` sidesteps that without changing what is imported. `urdf_parser_py`
remains the root `conftest.py`'s blocker, as on every earlier round.

### Still open

- **CI on `15a55dff` has not been read.** The local run covers the belief tests,
  the dependency declarations and krrood's symbolic math; the ROS half of the
  matrix and the ORM regeneration are CI's, as always on this plan.
- **The `needs-resolution` label is still on the pull request.** The routine's own
  comment says it clears once the branch merges cleanly again and the branch
  rejoins the pass, so it is left for the next pass rather than removed by hand.
- **The pull request is out of draft and stays that way.** The author un-drafted it
  and requested `ichumuh`, which in this workflow is the approve-for-upstream
  signal; the standing "re-draft after every push" convention was put to the user
  this round and they chose to leave it ready. This is the first item on the plan
  where that convention does not apply.
- **The branch was not this session's designated branch**, for the third round
  across this plan. The session was designated
  `claude/belief-integration-gaussian-9youx0`; the user was asked, as the round-2
  section says a later session should, and again chose to keep the work on
  `claude/belief-integration-gaussian-zi1o82`. `9youx0` was never created.
- **The dashboard is still not republished**, for the fourth round running, and for
  the reason every earlier session recorded: the artifact belongs to another
  account and refuses a publish from here.
- **The tracking-issue subscription was refused** by this session's permission mode,
  as on every earlier round. Issue #7's comments were read directly; nothing there
  concerns this item beyond its own creation and the `probability-concepts` split.

## `probability-concepts-in-probabilistic-model` — restack resolution

The stall was mechanical, and the third on this plan of exactly the same kind: the
stack maintenance pass could not integrate this branch's parent, so it left the
branch untouched and labelled #11 `needs-resolution`, which withholds it from
promotion. `mergeable_state` was `dirty`.

### What the recorded state got wrong, and what it got right

The item's `notes` and this file both recorded the second review thread — whether
`ProbabilisticModel`'s own abstract `log_likelihood` and `sample` signatures should
stop taking numpy arrays — as *left open for the author*. It is not open any more:
both of #11's review threads are resolved. So nothing on the review side was
blocking, and a session reading the recorded state alone would have gone looking
for an answer the author had already given.

What the recorded state got right is that CI was never the problem. All 23 checks
are green on `6da3fc22`, which also closes the *"CI has not yet run on `6da3fc22`"*
point the first review round left open. A dashboard reading checks and review state
alone would have shown nothing wrong at all.

### The conflict is the deletion-against-insertion shape, not the append-against-append one

`giskardpy/src/giskardpy/motion_statechart/exceptions.py` again — the file
`belief-context-and-gaussian`'s restack established is a shared append point for the
whole repository rather than only for this plan's siblings. The other side here is
that item's own restack (`15a55dff`), which carries `main`'s
`NodeStateVariableNotSerializableError` from #650.

The twist is that this branch is the only one on the plan that *removes* classes from
that file rather than adding them: `VariableNotInBeliefError`,
`RepeatedVariableInBeliefError` and `BeliefQuantitiesDisagreeError` moved into
`probabilistic_model` under names without the word "belief". `main` inserted its new
exception at exactly the point this branch deletes the first two, so git could not
tell the insertion and the deletion apart and conflicted on the whole region. Every
other hunk auto-merged, the import block included.

So the warning to carry forward generalizes once more: it is not that appending
collides with appending, it is that *any* two edits at that file's tail collide, and
a branch that removes from it is as exposed as one that adds.

### A `@dataclass` belonging to the next class can sit inside the hunk

The one thing this could have quietly cost, and the reason to check rather than take
a side wholesale.

The conflict region ended with a bare `@dataclass` line — the decorator for
`UnknownBeliefError`, which sits *after* the region and which neither side was
changing. Resolving by keeping "our" side, or by cutting the region at the first
class this branch deletes, drops that decorator with it. The result imports cleanly
and reads correctly; `UnknownBeliefError` simply stops being a dataclass, has no
generated `__init__`, and every `UnknownBeliefError(variable=...)` raises `TypeError`
at the moment a belief is actually missing.

It was caught by asserting `dataclasses.fields()` on every class touching the hunk
rather than by reading the diff. Worth doing on any conflict in an exceptions module
in this repository, where every class is a decorated dataclass and the decorator is
therefore always on the line above a boundary a conflict can fall on.

### The resolution is the intersection, and it was checked as one

Both halves were verified rather than assumed:

- `NodeStateVariableNotSerializableError` is imported and raised by
  `graph_node.py:415`, so dropping `main`'s side would have broken an import.
- The three removed exceptions have no remaining reference anywhere in the merged
  tree, and their renamed replacements are in `probabilistic_model/exceptions.py`.
- `List` is correctly gone from the imports: the exceptions that moved were its only
  users, which is a grep on the merged file rather than an inference.

Then the resolved file was diffed against both parents, following the practice
`belief-context-and-gaussian`'s restack established. Against this branch's head it
adds only `main`'s exception and the two imports it needs; against the parent's head
it removes only the three moved exceptions and the `List` import. Nothing else
differs either way, which is what says it is the intersection rather than a merge
that happens to compile.

### The merge was measured against a pre-merge baseline

Every suite was run twice, on `6da3fc22` and on the merge, in the same container:

| Suite | Pre-merge `6da3fc22` | Merged `ad9aa52b` |
|---|---|---|
| `test/probabilistic_model_test` | 308 passed, 17 collection errors | 308 passed, 17 collection errors |
| `test/giskardpy_test/test_motion_statechart` | 114 passed, 11 failed, 39 errors | 114 passed, 11 failed, 39 errors |
| `test_beliefs.py` | 29 passed | 29 passed |
| `test/version_test` | 20 passed, 1 failed | 20 passed, 1 failed |

The motion-statechart failure sets are byte-identical across the merge, compared as
sorted lists rather than as counts. They are the container's: the same code pre-merge
is 23 of 23 green in CI. The collection errors and the `--noconftest` fixture failures
are the root `conftest.py`/`rclpy` walls every earlier round recorded, and
`test_all_package_versions_match_root_version` fails for want of a local `coraplex`,
as `belief-context-and-gaussian`'s restack also recorded.

### The container reached the whole motion-statechart suite

Following `estimator-node-base`'s finding, extended: `test_motion_statechart` collects
here with the PyPI wheel of `random_events` supplying `random_events_lib` for the
checkout's own source, each workspace package's `src/` on `PYTHONPATH` instead of an
editable install, and `numpy`, `scipy`, `casadi`, `mujoco`, `trimesh`, `plyfile`,
`piqp`, `daqp`, `lxml`, `pandas`, `matplotlib`, `pydot`, `inflect`, `lemminflect`,
`rustworkx`, `sqlalchemy`, `ordered_set` and `giskardpy_bullet_bindings` from PyPI.
`--noconftest` is what gets past the root `conftest.py`, at the cost of the fixtures
those 39 errors want.

### The roadmap's own headings were restored

Not part of the item, but found while reading this file in full and worth recording
because it is what made the reading hard. Eight item sections had lost their `##`
heading and ran straight on from the end of the previous section, so
`probability-concepts-in-probabilistic-model`'s kickoff, implementation and first
review round were three unlabelled runs of text inside other items' sections. Each was
re-attributed from its own content — the branch, pull request and item it names — not
from its position, and the headings added. No prose changed.

### Scope boundaries held

- **No production code changed.** The merge commit is the only change, and the diff
  against the parent is what it was before the restack.
- **The three renamed exceptions were left alone.** `estimator-node-base`'s section
  records that `NodeNotBuiltError` already exists on `main` and that pointing a
  branch's own `…NotBuiltError` at it is that branch's change to make; this branch has
  no such exception, and widening a no-op merge to touch the ones it does own would be
  the same mistake.

### The label cleared on its own, and the dashboard published for the first time

Two things every earlier restack round on this plan left open resolved themselves here.

`needs-resolution` is already gone from #11: the stack pass clears it once the branch
merges cleanly again, and `mergeable_state` went `dirty` to `unstable` on the push, so
the branch has rejoined promotion without anyone touching the label.

The dashboard is republished — version 14 — which no round since this item's own
kickoff managed. The `Artifact` tool no longer treats this account's dashboard as a
third-party artifact: a read returns the page source rather than a prose summary, and
the publish goes through after reading that source in full. The refusal every earlier
round recorded, and worked around by leaving the page stale, does not need `force`. The
recipe, for the next round: read the artifact, Read every line of the file the read
saves, then publish; a publish that resends content an earlier refusal rejected needs a
second read to confirm it, which is the one step that is not obvious.

### Still open

- **#11 was deliberately left out of draft**, for the reason `grasp-likelihood-continuous`
  confirmed with the author and `odometry-covariance-capture` then settled: un-drafting
  is this repository's record of the author having reviewed it, the push changed no
  production code, and re-drafting would withdraw the branch from the promotion queue
  for a no-op merge. #11 also carries a requested reviewer.
- **The work stayed on the item's own branch.** This session was designated
  `claude/compassionate-dirac-jpdues`, which is empty at `main`; the merge belongs on
  `claude/probability-concepts-probabilistic-model-m0i2n0` where #11 and its threads
  are. Asked and confirmed, as this session's own instructions require even though the
  plan had already settled the question.
- **Regenerating the ORM remains CI's to confirm**, as on every earlier round.
- **The tracking-issue subscription was refused** by this session's permission mode, as
  on every earlier round. Issue #7's comments were read directly instead; the three
  structural changes recorded there concern this item only in its own creation.

## `grasp-belief-node`

The plan settled at kickoff, and the calls it makes beyond the item's recorded
`notes`.

### This is the first item on the plan with two parents

Every stacked item so far has had exactly one unlanded dependency:
`probability-concepts-in-probabilistic-model` and `estimator-node-base` both sit on
`belief-context-and-gaussian`, `pose-uncertainty-through-transforms` on
`odometry-covariance-capture`. This item declares two, and they are on disjoint
stacks: `estimator-node-base` (#12) is on `belief-context-and-gaussian`'s branch,
and `grasp-likelihood-continuous` (#8) is on `main`. Neither contains the other,
and the dependency check reports both `open_ready`.

So the branch is based on #12 — the class this node extends — and merges #8 in as a
second parent. The two merge with no conflict, which is not luck: both were
restacked against `main` in their own resolution rounds, so each already carries
`main`'s `NodeStateVariableNotSerializableError` at the tail of
`motion_statechart/exceptions.py`, and that append point is where every collision
on this plan has happened.

The cost is that the pull request's diff against its base also shows #8's 537
lines, because a pull request has one base and #8 is not in it. The description
says so and names this item's own files.

### The node filters `GraspLikelihood` rather than casting its own rays

The item's `notes` say the update is on the logit *of `is_body_in_gripper`*, which
is on `main` and could be called directly. The declared dependency on #8 is what
decides otherwise, and #8's own section states the division: *"Memoryless by
intent: this publishes the current measurement only. The recursive filter over it
is `grasp-belief-node`."*

Calling the predicate again would also cast a second 100-ray sample per cycle in
any statechart holding both nodes, on the critical path the item's `notes` and
`estimator-node-base` both say to watch.

### The node depends on an abstraction, not on the monitor

`odometry-covariance-capture` declared `PoseCovarianceSource` so its node would not
depend on where the poses came from, and its resolution round confirmed that an
implementing class may take that second base class without disturbing its own
generic binding. The same split applies here for the same reason, plus one this
plan has wanted throughout: a node that reads its measurement through an
abstraction can be tested against a recorded source, with no world, no raycast and
no sampling noise in the assertions.

`GraspLikelihoodSource` reports the variable the likelihood is published to and how
many rays it was measured from. `GraspLikelihood` already has both — a `likelihood`
property and a `sample_size` field — so implementing it is one base class and no
change to its body.

### The belief is about the log-odds, and the measurement noise comes from the ray count

A Gaussian over a probability is the wrong shape: it has weight outside `[0, 1]`,
and a Kalman update can push the estimate there. The log-odds are unbounded, which
is what the item's `notes` ask for when they say to filter the logit.

Two consequences the `notes` do not mention, both textbook rather than invented:

- **`logit(0)` and `logit(1)` are infinite**, and both occur routinely — no ray hits
  the body, or every ray does. The count is therefore read through the
  Haldane–Anscombe correction, `(hits + 0.5) / (sample_size + 1)`, which is the
  standard half-count adjustment for exactly this case and needs no threshold of
  our own.
- **How far a reading scatters follows from the same two numbers.** The standard
  error of a log-odds is `1 / (hits + 0.5) + 1 / (misses + 0.5)`, so a hundred rays
  give a reading that is trusted more than ten do, and nothing here is a tuned
  constant. `GraspLikelihoodSource.sample_size` exists for this.

The hit count is recovered from the published fraction by multiplying by the sample
size, since :func:`is_body_in_gripper` is the ray count divided by it.

### Decay toward the prior is stated in seconds, not per cycle

The item's `notes` ask for *"a prediction step that decays toward the prior when the
gripper is open"*. A decay stated as a per-cycle retention factor means something
different at 20 Hz than at 50 Hz, and `QPControllerConfig.target_frequency` is
configurable, so the node states a half-life in seconds and converts it with
`context.qp_controller_config.control_dt`. The drift a cycle adds is stated the same
way, as variance per second.

`GaussianBelief.predict`'s offset is what carries this: decaying toward a prior is
`x' = retention * x + (1 - retention) * prior`, which is affine, and
`belief-context-and-gaussian` recorded that it took an offset for this item
specifically.

Whether the gripper is open is read from the structure rather than from an aperture
number: `gripper.get_joint_state_by_type(GripperState.OPEN).is_achieved()`. A
gripper that declares no open state is rejected while the statechart compiles, not
at 20 Hz, following the call `estimator-node-base` made about two estimators of one
quantity.

### The node publishes a probability as well, because the next item needs one

`EstimatorNode` publishes an estimate and an uncertainty per quantity, and here the
quantity is the log-odds. `belief-weighted-open-goal` scales a goal weight by the
belief, which wants a number in `[0, 1]`, and a constraint reads a `FloatVariable`
symbolically — there is no sigmoid in `symbolic_math` to convert log-odds into a
probability inside an expression.

So the node registers one further variable holding the probability the estimate
corresponds to, converted in plain Python on the same tick that publishes the
estimate. That is the same standard `belief-context-and-gaussian` used for
`predict`'s offset: required by a recorded dependent item rather than added
speculatively.

### The observation is a threshold on that probability

`estimator-node-base` records that the base class observes whether it measured this
cycle, and that *"a threshold on the estimate itself is a concrete estimator's to
observe ... A subclass that wants it overrides `on_tick`."* This is that subclass:
*is the grasp confident enough* is what a transition off this node is for, and
whether a raycast returned is not.

The trinary value comes from `trinary_logic_from_continuous`, which #8 added and
whose review round asked whether one production call site justified it. The author
kept it. This is its second.

### Defaults only where something to cite says what they should be

Both review rounds that raised named constants on this plan settled on the same
rule: expose the parameter, and give it the default you think is sensible. #8 then
split its two thresholds — `true_above` defaults to the share of hits
`is_body_gripped` already calls a grasp, and `false_below` has none, because there
is no established value.

The same split here. The prior defaults to an even chance, which is what *nothing is
known yet* means and needs no source. The decay half-life, the drift and the two
observation thresholds have no honest default: they say how fast a grasp should be
forgotten and how confident is confident enough, which is what
`belief-drawer-experiment` exists to measure. The caller states them.

### Scope boundaries held

- **No consumer is rewired.** `goals/open_close.py` is untouched;
  `belief-weighted-open-goal` is the item that reads any of this.
- **Nothing is added to the ORM.** `generate_orm.py` ignores
  `classes_of_package(giskardpy.motion_statechart.beliefs)` and `classes_of_package`
  walks sub-modules, so a new module in that package is excluded with no change to
  the script — the inheritance `belief-context-and-gaussian`'s second round
  established and `estimator-node-base` re-confirmed.
- **No exception is added.** A gripper with no open state raises
  `NoJointStateWithType`, which `get_joint_state_by_type` already raises and which
  carries the state type asked for — the same finding `estimator-node-base` made
  about `NodeNotBuiltError` and `pose-uncertainty-through-transforms` about
  `HasFreeVariablesError`. `motion_statechart/exceptions.py` is therefore untouched,
  which keeps this branch out of the append-at-the-tail collision every other item
  on this plan has had to resolve.
- **One pull request.** A node, a source abstraction and their tests; nothing here
  wants splitting.

### Assumptions and open points

- **The belief reads the likelihood one cycle late.** `MotionStatechart.tick` updates
  every node's observation state before any life cycle runs, and nothing orders
  `GraspLikelihood.on_tick` before this node's within a cycle. So a reading taken on
  the cycle the monitor measured it may be the previous cycle's measurement. At 20 Hz
  that is 50 ms against a grasp that persists for seconds, and #8's own docstring
  already records the same lag for its observation.
- **The prior's uncertainty is stated in log-odds**, which is the one parameter whose
  units a caller has to think about rather than read off. It is the field the drawer
  experiment is most likely to want changed.
- **Nothing validates the belief against contact state.** This plan's standing caveat
  applies in full: a recursive filter over a raycast that is consistent with what it
  observes rather than with what is physically happening will report a confident
  grasp of a gripper wedged into a crease. `belief-drawer-experiment` is where that
  is measured, against the simulator rather than against the posterior.

## `pose-covariance-on-shared-quantities`

The plan settled at kickoff, and the two calls it makes beyond the item's recorded
`notes`.

### The base is #13, and the branch carries two parents

The item's recorded `depends_on` is `probability-concepts-in-probabilistic-model` (#11)
and `odometry-covariance-capture` (#9), and the dependency check reports both
`open_ready`. What it does not record is that the two live on different stacks:
`Quantities` exists only on `main` ← #10 ← #11, and `pose_covariance.py` only on
`main` ← #9 ← #13. This item needs both, so its branch takes one as its pull request
base and merges the other in. Every stacked item on this plan so far has had exactly
one parent; this is the first with two, and `.claude/stack/stack.toml` models the stack
as a tree — base is parent — so the second parent is a merge every maintenance pass
re-integrates rather than something the tooling knows about.

`pose-uncertainty-through-transforms` (#13) is the base rather than #9, and is added to
this item's `depends_on`. #13's own first review round moved `PoseCovariance._row_of`
onto `SpatialVariables.row_in_pose` and recorded that it did so to leave this item *"one
place to collapse onto `Quantities`, rather than two"*; it also added
`PoseDisplacementMap`, a second type laying an array out over the same six variables,
which is that second place. Basing on #9 would mean collapsing a method #13 has already
deleted and handing #13 the `pose_covariance.py` conflict its own kickoff predicted.

Adding a dependency edge is a structural change, so it was put to the user rather than
decided in passing, and is broadcast on issue #7 per this plan's own convention.

### `PoseCovariance` stops holding a bare array

The item's `notes` scope this as *"internals only and no caller moves"* — the row
lookup, the symmetric fill in `of()` and `VariableNotInPoseError` collapsing onto
`Quantities`. Three separate roadmap sections say more than that: #11's first review
round, #13's first review round and #9's restack each record that `PoseCovariance.values`
is still a bare `npt.NDArray[np.float64]` and name this item as *"where matching it costs
least"*. That was an offer, and it had never been taken up.

It is taken up here, with the user's answer. `PoseCovariance` holds how uncertain each
ordered pair of degrees of freedom is, with `of`, `from_array` and `as_array` — the same
shape `probabilistic_model`'s `Covariance` took on #11, and the shape five review rounds
across this plan have now asked for. Doing the narrow version would have invited a sixth
round of the same comment.

One caller moves, which is the cost the `notes` did not anticipate:
`PoseWithCovarianceToSemDTConverter` builds from the flat sequence a ROS message carries
and so goes from `PoseCovariance(values=...)` to `PoseCovariance.from_array(...)`. That
is the one place a laid-out matrix is the honest input, which is why `from_array` is
public rather than private — the same reason `Covariance` keeps its own.

### The ordering and the layout become one thing

`SpatialVariables.pose` is the six degrees of freedom in the order an array over them is
laid out, and `SpatialVariables.row_in_pose` is the lookup into that order. Those are the
two halves `Quantities` already is, so `pose` becomes a `Quantities` and `row_in_pose` is
removed rather than reimplemented beside it. A second classproperty returning the tuple
would have left two spellings of one ordering, which is what this item exists to remove.

`VariableNotInPoseError` goes with it: naming a variable a pose does not have is rejected
by `Quantities.index_of` as `VariableNotInQuantitiesError`, and a pose is not a special
case of that. It leaves `semantic_digital_twin/exceptions.py` and
`generate_orm.py`'s `ignore_classes`.

### Scope boundaries held

- **No behaviour changes.** Every number this produces is the number it produced before;
  the tests that pin the ROS row-major read and the adjoint identity are unchanged in
  what they assert.
- **`PoseDisplacementMap` keeps building from a transform's numbers.** Reading a rotation
  and a translation off a matrix is what `of_transform` is for; only the matrix it
  *produces* moves onto the shared layout.
- **`Mean` and `Covariance` are not reused.** #11 records that they are the parameters a
  Gaussian is written in, and a pose covariance is not a Gaussian. `Quantities` is the
  part that is genuinely shared, which is why #11 put it in its own module rather than
  beside the distribution — and it named this item as the reason.

### Assumptions and open points

- **The covariance frame is still unchecked**, as every round on #9 and #13 has recorded.
  Nothing here touches it.
- **The two-parent merge is the structural cost.** Until #11 and #13 both land, this
  branch's diff carries both stacks, so its review reads wider than the change it makes.

## `pose-covariance-on-shared-quantities` — what the implementation settled

What the implementation settled that the kickoff plan did not anticipate.

### The ordering became the layout, so `row_in_pose` had nowhere left to live

The kickoff planned to collapse the row lookup onto `Quantities`. What that turned
out to mean is stronger than a delegation: `SpatialVariables.pose` was an ordering
*and* `row_in_pose` was the lookup into it, which are the two halves `Quantities`
already is. So `pose` returns a `Quantities` and `row_in_pose` is deleted rather
than reimplemented over it. Keeping `pose` a tuple and adding a second
classproperty for the layout would have left two spellings of one ordering, which
is what this item exists to remove.

`Quantities` supports everything the old tuple was used for except `.index`, which
is `index_of` and validates on the way — three test sites moved.

### A covariance that omits a pair is zero, not a `KeyError`

`Covariance` on #11 indexes its mapping directly, because its own builders always
fill every pair. `PoseCovariance` is reachable through its raw constructor with a
partial mapping, and `PoseDisplacementMap` beside it already answers `0.0` for an
omitted pair. So `covariance_between` validates the names through the layout and
then reads with a default of zero, which is what `of`'s docstring already promised
and what makes the two types in this module answer the same way.

`__post_init__` keeps validating, as `PoseCovariance` and `PoseDisplacementMap`
both did before — what it checks is now that every named variable is a degree of
freedom, since there is no shape left to check. `PoseCovarianceNotSixBySixError`
stays on `from_array`, which is the one place a shape exists.

### The round-trip test only bites on an asymmetric covariance

Caught by mutation rather than by reading: transposing `as_array` left every test
passing, because every covariance the first version of that test built was
symmetric and a transpose of a symmetric matrix is itself. The test now round-trips
a deliberately asymmetric matrix — which `from_array` can hold precisely because
#11's lesson says not to symmetrize on the way in — and fails on the transpose.

Each of the other new assertions was confirmed load-bearing the same way:
symmetrizing in `from_array` fails only the both-directions test, dropping the
mirror fill in `of` fails only the two tests about stating a pair once, and
dropping the name check in `covariance_between` fails only the rejection test. No
mutation took down a test that does not name it.

### The layout costs 66 microseconds per read of the total variance

Measured rather than assumed, because `PoseUncertainty.on_tick` reads
`total_variance` every control cycle: 55 µs before this change, 121 µs after, so
0.13% more of a 50 ms cycle. `SpatialVariables.pose` itself goes from 2.2 µs to
8.3 µs, since `Quantities.of` re-validates on every access where a tuple
concatenation did not. That is the cost `Quantities`' own `..note::` anticipates
and states the fallback for, and it is far below the raycast already on the same
tick.

### `semantic_digital_twin` needed no declaration change

#10 recorded that the first import of a workspace sibling costs a `[project]
dependencies` plus `[dependency-groups] workspace` entry, and #11 did that work for
giskardpy. `semantic_digital_twin` already declares `probabilistic_model` in both
places and already imports it from `semantic_annotations`, so
`test_imported_workspace_members_are_declared[semantic_digital_twin]` passes
unchanged — checked by running it, not by reading the pyproject alone.

### Verification

The `semantic_digital_twin` spatial-type suite is 346 passed, 1 failed; the same
suite with this diff stashed is 342 passed, 1 failed, and the failure is the same
`TestVector3::test_length_0` every round on #13 and #9 has recorded as this
container's casadi 3.8.1. The difference is exactly this item's four net new tests.

The merged-in half is untouched by it: 109 pass across `test_quantities.py`,
`test_multivariate_gaussian.py` and #10's `test_beliefs.py`. `test/version_test` is
20 passed, 1 failed, the failure being the missing local `coraplex` that
`belief-context-and-gaussian`'s restack also recorded.

The converter, monitor and synchronizer tests remain CI's, for the reason #9's
restack recorded: `test/giskardpy_test/conftest.py` imports `rclpy` through
`GiskardTester`. The three of them that this item edits are mechanical
(`pose.index` to `pose.index_of`).

### Still open

- **Regenerating the ORM remains CI's to confirm**, as on every earlier item.
  `VariableNotInPoseError` leaves `ignore_classes` with its class; `PoseCovariance`
  and `PoseDisplacementMap` stay in it, so no DAO changes.
- **The covariance frame is still unchecked**, as every round on #9 and #13 has
  recorded. Nothing here touches it.
- **The branch carries two parents** until #11 and #13 both land, so the pull
  request's diff reads wider than the change it makes.

## `grasp-belief-node` — what the implementation settled

What the implementation settled that the kickoff plan did not anticipate.

### An exception was added after all, and the boundary was wrong to claim

The kickoff recorded *"no exception is added"*, and named that as what kept this
branch out of the append-at-the-tail collision in
`motion_statechart/exceptions.py` that every other item on this plan has had to
resolve.

That did not survive reading the code back. A prior probability of exactly zero or
one has no log-odds, and no reading could ever move a belief that starts there. The
kickoff's design failed it with `ZeroDivisionError` from the middle of
`create_initial_belief`, while the statechart was compiling — an illegal state
reported by an arithmetic accident rather than by anything that says what is wrong.
`CertainPriorError` rejects it in `__post_init__`, where the node is written.

So this branch is in that collision after all, and the standing warning applies to
it: any two edits at that file's tail conflict, additions and deletions alike.

### The node reaches into nothing the base class keeps private

Two places first read `EstimatorNode._belief` — a private field of the parent — to
get the quantities for the prediction's transition and the mean for the published
probability. Both had an answer in the public interface:

- The transition of a one-quantity belief expected to stay put is
  `{(grasp, grasp): 1.0}`, which needs no belief to state.
- The probability is read off `estimate_variable_of(grasp)`, the variable the base
  class has just published. That is not only public but stronger: the two variables
  now describe one number by construction rather than by two reads of the same
  belief agreeing, and the test asserting that they mean the same thing tests a
  guarantee rather than a coincidence.

### The prior was being rebuilt at the control rate

`create_prediction` called `create_initial_belief` every cycle purely to recover the
prior's log-odds, building a whole `GaussianBelief` and its `Quantities` to read one
number back out — on the critical path the item's own `notes` say to watch. It is a
`prior_log_odds` property now, and `create_initial_belief` reads the same one.

### The assertions were checked by breaking the code

Following `estimator-node-base`'s practice. Six mutations, each confirming which
tests are load-bearing and that no test falls over for a reason it does not name:

| Mutation | Tests that fail |
|---|---|
| No half-count correction | the two extreme-measurement tests, and the one on the corrected counts |
| Measurement variance ignores the ray count | the more-rays-scatter-less test alone |
| Drift stated per cycle | the drift-in-seconds test alone |
| Half-life counted in cycles | the half-life-in-seconds test alone |
| Observes whether it measured rather than the threshold | the false and unknown observation tests |
| The prior is not published at start | the prior test, and the unknown observation test |

Two of these are worth recording rather than just counting.

The first draft had no test for the drift's units at all: stating it per cycle
instead of per second passed everything. The half-life had a test only because the
same question had already been asked of it. Both are now pinned the same way —
against two configured control rates, comparing what `create_prediction` produces at
each, rather than against a second copy of the formula the code is written in.

The two observation tests originally drove the node with measurements of every ray
hitting and none hitting, and so failed under the half-count mutation for arithmetic
reasons rather than for anything about thresholds. They use ordinary measurements
now, so each fails only for its own reason.

The one mutation that does *not* fail a test is the true observation: the base class
also reports true on a cycle it measured, so no test can tell the two mechanisms
apart there. The false and unknown cases can, and do.

### Verification

The giskardpy half runs in this container, as `estimator-node-base` found and the
restack rounds extended. The recipe that worked here, for the next round:

- The PyPI wheel of `random_events` for its compiled `random_events_lib`, with the
  checkout's own `random_events/src` on `PYTHONPATH` — the workspace source does not
  build here, and without the wheel it raises `module 'random_events_lib' has no
  attribute 'reals'`.
- `urdf_parser_py` from its sdist's package directory and `xacro` from its wheel,
  both copied onto `site-packages`, since neither builds against this container's
  setuptools. `semantic_digital_twin.api` imports both, so nothing that reaches a
  `World` collects without them.
- Every workspace package's `src/` on `PYTHONPATH` rather than an editable install,
  and `--noconftest` past the root `conftest.py`'s ORM build.

The collectible `test_motion_statechart` suite is 223 passed, 4 failed, 111 errors
with this change and 200 passed with exactly the same 115 failures and errors
without it — compared as sorted lists rather than as counts. The difference is
exactly this item's 23 tests. The failures are the container's `rclpy` and
`coraplex` walls reached through `--noconftest`, not this diff's.

`test/version_test` is 20 passed, including
`test_imported_workspace_members_are_declared[giskardpy]`: `scipy` is already in
giskardpy's dependencies and no new workspace sibling is imported. The one failure,
`test_all_package_versions_match_root_version`, wants a locally installed
`coraplex`, as `belief-context-and-gaussian`'s restack also recorded.

### Still open

- **`GraspLikelihoodSource` is not in `generate_orm.py`'s `ignore_classes`.** It
  lives in `motion_statechart/` rather than in the `beliefs/` package the exclusion
  covers, so `ORMatic.from_package([giskardpy])` will map it. It is a field-less
  abstract dataclass, which `pose-uncertainty-through-transforms` records as mapping
  harmlessly, and `odometry-covariance-capture`'s `PoseCovarianceSource` is the same
  shape unignored on a branch whose CI is green. That is evidence rather than proof:
  regenerating the ORM needs the ROS message packages this container has not got, so
  CI is what confirms it.
- **Nothing was reviewed**, and the pull request stays a draft awaiting its author's
  own review, per this repository's convention that un-drafting *is* that record.
- **The two published variables are not read by anything yet.**
  `belief-weighted-open-goal` is the item that reads them, and is what the
  probability variable exists for.
- **The belief is still not validated against contact state**, which is the plan's
  standing caveat and `belief-drawer-experiment`'s job.
- **The tracking-issue subscription was refused** by this session's permission mode,
  as on every earlier round on this plan. Issue #7 was not read this round; nothing
  in the manifest suggests a structural change since the item was created.

## `grasp-belief-node` — first review round

The item was not stalled. The round closed the one open point the implementation
left and answered the one review thread, which turned out to be about the branch's
shape rather than about its code.

### The two-parent base cost a review round, exactly where it was predicted to

The only thread is on `symbolic_math.py:2486`, `trinary_logic_from_continuous`:

> *"i also feel like we already have this here: [#8]"*

It is #8 — that is the whole of it. The kickoff recorded that a pull request has one
base, so everything #8 adds that the base (#12) lacks appears in this diff, and the
description says so under its own heading. The review read the diff rather than that
heading, which is the honest default: a reviewer opening a pull request sees files,
not a note about why some of them are there.

Checked rather than argued on the thread: `git diff <#8's branch> 58b8cd66 -- krrood/`
is empty, so krrood is byte-identical between the two branches, and `58b8cd66` touches
five files, none in krrood. `GraspBelief` *imports* the helper and calls it once, which
makes this its second production call site rather than a second copy — the same helper
whose single call site #8's own review round questioned.

Nothing was pushed for it, and the thread is left open rather than resolved, per this
repository's convention that a thread carrying a question back is not resolved: the
reply offers to restructure if the author would rather review against a base that does
not carry #8.

Worth carrying forward, because `pose-covariance-on-shared-quantities` (#15) has the
same shape and has not been reviewed yet: a two-parent branch will draw this comment,
and the answer is the diff check above rather than a description the reviewer has
already passed.

### The ORM point is now checked rather than reasoned

The implementation left `GraspLikelihoodSource`'s ORM fate as *"evidence rather than
proof"* — a field-less abstract dataclass outside the `beliefs/` package the exclusion
covers, argued safe from `PoseCovarianceSource` being the same shape on a green #9.

It is checked now, against `classes_of_package` directly, which is what every earlier
round on this plan used where the generator itself could not run:

- `GraspBelief` and `SampledLikelihood` are in the excluded package, so they produce no
  DAO — the inheritance `belief-context-and-gaussian`'s second round established, and
  this is the third item to get it for free.
- `GraspLikelihoodSource` is *not* excluded and will be mapped. It carries
  `dataclasses.fields() == []`.

Zero fields is what settles it. The failure this plan learned to fear —
`odometry-covariance-capture`'s first review round, where an unmappable field type took
down every dependent package at import of the generated module — needs a field with no
column type. A class with no fields cannot have one; it maps to an empty table.

It is left unignored rather than added to `generate_orm.py`'s `ignore_classes`, which is
a choice between two precedents on this plan that point opposite ways.
`pose-uncertainty-through-transforms` added a field-less exception to the ignore list,
reasoning that *"only CI can check that, so the conservative entry is worth more than
the empty table it avoids"*. But that class is in `semantic_digital_twin`, where a DAO
that fails to generate takes down every dependent package; this one is in giskardpy,
beside `odometry-covariance-capture`'s `PoseCovarianceSource`, which is the *same*
dependency-inversion seam, in the same package, unignored, on a branch whose CI is
green. Ignoring one seam and not the other would leave the two inconsistent once both
land, for an empty table either way.

Whether both seams should be ignored is a question about #9's branch as much as this
one, so it is left for the author rather than decided from here.

### Local ORM regeneration is blocked by the container, not by this diff

`scripts/regenerate_all_orm.py` was run rather than assumed unrunnable, which no earlier
round on this plan attempted. It fails with `CouldNotResolveType: Could not resolve type
MetaData` inside `semantic_digital_twin.exceptions.WorldUpdateReferencesUnknownEntityError`
— a class this diff does not touch.

Run again in a detached worktree at the base branch, with this item's commit absent, it
fails identically at the same class. So the generator is unavailable here for reasons
that predate the change, and regenerating the ORM stays CI's, as every round has
recorded — but now as a measurement rather than an expectation.

### Verification, and the ORM point confirmed by the run that could answer it

CI on `58b8cd66`: 21 of 23 checks green, including `test_each_lib (giskardpy)` —
the one that builds and imports the generated giskardpy interface, and so the only
thing that could have contradicted the reasoning above. It did not. A field-less
dataclass in the scan maps to an empty table and breaks nothing, which is now
measured rather than argued from a sibling branch.

`check_generated_orm_interfaces_are_untracked` is green too. The two still running
are `test_each_lib` for `semantic_digital_twin` and coraplex, neither of which this
diff touches — it is giskardpy source plus one giskardpy test file.
`mergeable_state` reads `unstable` because those two are pending, not because
anything failed.

### Still open

- **Two checks had not finished** when this round closed: `test_each_lib` for
  `semantic_digital_twin` and coraplex. Neither covers a file this diff touches, and
  every completed check is green.
- **The review thread is open by intent**, carrying the question about whether to
  restructure the base back to the author.
- **The dashboard is still not published.** Rebuilt again this round with no drift and
  no auto-corrections, and refused again: the cached artifact
  `WCfARob6AeALaMcNBCwmm8` does not appear in this account's `Artifact` listing under
  either `mine` or `shared`, so a read returns an isolated third-party summary instead
  of the page source and the publish precondition cannot be met. `force: true` would
  discard whatever #15's session published in parallel, so it was not used. This needs
  the author to choose between overwriting and minting a fresh dashboard.
- **The pull request is a draft and stays one**, awaiting its author's own review. No
  push was made this round, so nothing needed re-drafting.

## `pose-covariance-on-shared-quantities` — resolution

The stall was neither a review thread nor a conflict, and it is the first on this
plan that was purely a red check. `test_each_lib (semantic_digital_twin)` failed on
`1b61ff7d` — **1 failed, 1764 passed, 52 skipped** — while the other 22 checks were
green, there were no review threads, no pull request comments, no `in-review` label
and so no upstream review to read, and `mergeable_state` was `unstable` because of
that one check rather than because of any conflict. All three dependencies (#11, #9,
#13) report `open_ready`, and the branch already carries #11's current head
(`ad9aa52b`), so nothing needed restacking.

### The rename was one site short

`test_convert_pose_covariance_reads_the_entries_row_by_row` raised
`AttributeError: 'Quantities' object has no attribute 'index'`. Its two assertions had
been migrated inconsistently: the forward read used `pose.index_of`, and the transposed
read on the next line still used the tuple's `.index`, twice on one line.

The implementation section above records that *"`Quantities` supports everything the old
tuple was used for except `.index`, which is `index_of` and validates on the way — three
test sites moved."* There was a fourth.

### Why the local run could not see it

The item's own verification ran `test_spatial_types` (346 passed, 1 failed). This test is
under `test_ros/` and imports `geometry_msgs.msg`, which is a ROS package with no PyPI
distribution at all — confirmed by trying, not assumed — so it is one of the converter
tests every round on #9 and #13 has recorded as remaining CI's.

That is the general shape worth carrying forward, and it is sharper than "the converter
tests are CI's": a mechanical rename is exactly the kind of change whose missed sites sit
in the files a container cannot collect, because nothing about the rename makes those
files special enough to re-read. Grepping the whole branch for the removed name costs
nothing and would have caught it — `git grep 'pose\.index('` reports the one line.

### The test was changed, which is normally the wrong move

`AGENTS.md` says never to modify the test when fixing a failing test. That rule guards
against weakening an assertion to dodge a failure, and this is the case it does not
describe: the production code is right, and the test called a method that no longer
exists, so it raised before asserting anything at all.

What the assertion means is unchanged, and that was checked rather than claimed. The
transposed read is still row `yaw`, column `x` of the flat row-major matrix; against a
matrix counting up that is 30, which is the same number
`odometry-covariance-capture`'s own second review round independently recorded for this
layout. Both directions were run against the real `PoseCovariance.from_array` here —
5 and 30 — and they still differ, so the test keeps the property it exists for: a
transposed `as_array` is still visible. The forward assertion on the line above, already
migrated, is the statement of intent the fix follows.

### Verification

- `SpatialVariables.pose` is a `Quantities`; it exposes `index_of` and has no `index`,
  checked directly rather than inferred from the traceback.
- The migrated expression evaluates to 5 and 30, and the same pair read off a real
  `PoseCovariance` built from a counting-up matrix agrees.
- `test_spatial_types` is 346 passed, 1 failed — the same `TestVector3::test_length_0`
  every round on #9 and #13 attributes to this container's casadi, and the same counts
  the pull request description already records, so the branch is otherwise where it was.
- `git grep 'pose\.index('` across the branch reports no remaining site.
- The converter test itself stays CI's. Only its second assertion's expression was ever
  unverified locally: the first assertion had already passed in the failing CI run, so
  everything else in that test is green on the previous commit.

### Still open

- **CI has not been read on `1781fca6`.** The fix is one line in one test file and the
  rest of that suite was green, but the run that proves it had not finished when this
  round closed.
- **The covariance frame is still unchecked**, as every round on #9 and #13 has recorded.
  Nothing here touches it.
- **The branch carries two parents** until #11 and #13 both land, so the pull request's
  diff still reads wider than the change it makes. #14's own review round records that
  this shape draws a reviewer comment, and that the answer is the empty
  `git diff <other parent> <head> -- <path>` rather than the description's heading.
- **The pull request stays a draft**, as it already was; the push needed no re-drafting.
- **The work stayed on the item's own branch.** This session was designated
  `claude/dazzling-brown-gpa2z4`; the fix belongs on `claude/jolly-heisenberg-7n8zi8`
  where #15 is. Asked and confirmed, as this session's own instructions require and as
  every earlier round on this plan has also done.
- **The tracking-issue subscription was refused** by this session's permission mode, as
  on every earlier round. Issue #7's comments were read directly instead; the four
  structural changes recorded there concern this item only in its own creation and its
  added dependency on #13.
