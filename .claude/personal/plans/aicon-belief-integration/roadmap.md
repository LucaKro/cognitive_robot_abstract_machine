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
