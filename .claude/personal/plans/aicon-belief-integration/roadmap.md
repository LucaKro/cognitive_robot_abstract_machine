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
