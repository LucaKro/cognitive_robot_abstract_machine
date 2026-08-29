from __future__ import annotations

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    Goal,
    NodeArtifacts,
    CancelMotion,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
    Pulse,
)
from giskardpy.data_types.exceptions import GiskardException


@dataclass
class TestNodeAssertionError(GiskardException):
    """
    Raised by test motion statechart nodes when a behaviour they assert on is violated.
    """

    reason: str
    """
    Description of the violated assertion.
    """

    def error_message(self) -> str:
        return self.reason

    def suggest_correction(self) -> str:
        return ""


@dataclass(eq=False, repr=False)
class ConstTrueNode(MotionStatechartNode):
    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(eq=False, repr=False)
class ConstFalseNode(MotionStatechartNode):
    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_false())


@dataclass(repr=False, eq=False)
class ChangeStateOnEvents(MotionStatechartNode):
    state: str | None = None

    def on_start(self, context: MotionStatechartContext):
        self.state = "on_start"

    def on_pause(self, context: MotionStatechartContext):
        self.state = "on_pause"

    def on_unpause(self, context: MotionStatechartContext):
        self.state = "on_unpause"

    def on_end(self, context: MotionStatechartContext):
        self.state = "on_end"

    def on_reset(self, context: MotionStatechartContext):
        self.state = "on_reset"


@dataclass(repr=False, eq=False)
class TestGoal(Goal):
    sub_node1: ConstTrueNode = field(init=False)
    sub_node2: ConstTrueNode = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.sub_node1 = ConstTrueNode(name="sub muh1")
        self.add_node(self.sub_node1)
        self.sub_node2 = ConstTrueNode(name="sub muh2")
        self.add_node(self.sub_node2)
        self.sub_node1.success_condition = self.sub_node1.observation_variable
        self.sub_node2.start_condition = self.sub_node1.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.sub_node2.observation_variable)


@dataclass(repr=False, eq=False)
class TestNestedGoal(Goal):
    sub_node1: TestGoal = field(init=False)
    sub_node2: TestGoal = field(init=False)
    inner: TestGoal = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.inner = TestGoal(name="inner")
        self.add_node(self.inner)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.inner.observation_variable))


@dataclass(repr=False, eq=False)
class TestRunAfterStop(Goal):
    """
    Goal that tests if a child node runs after the parent node has stopped.

    Uses a CancelMotion node to raise an exception if the child node runs after the
    parent has stopped.
    """

    ticking1: CountControlCycles = field(init=False)
    ticking2: CountControlCycles = field(init=False)
    cancel: CancelMotion = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.ticking1 = CountControlCycles(name="3ticks", control_cycles=3)
        self.ticking2 = CountControlCycles(name="2ticks", control_cycles=2)
        self.cancel = CancelMotion(
            name="Cancel_on_tick_after_done",
            exception=TestNodeAssertionError(
                reason="Node ticked after template stopped"
            ),
        )

        self.add_nodes(
            nodes=[
                self.ticking1,
                self.ticking2,
                self.cancel,
            ]
        )
        self.cancel.start_condition = self.ticking1.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.ticking2.observation_variable))


@dataclass(repr=False, eq=False)
class TestEndBeforeStart(Goal):
    """
    Test if a child node can end before it was started.

    node1 waits 1 tick, then starts node 3. node2 fulfills the success condition of node
    3 immediately. node3 should start when node1 is True and transition to RUNNING with
    Observationstate UNKNOWN. On the next tick, node3 should succeed because its success
    condition is already fulfilled by node2.
    """

    node1: CountControlCycles = field(init=False)
    node2: ConstTrueNode = field(init=False)
    node3: ConstTrueNode = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.node1 = CountControlCycles(control_cycles=1)
        self.node2 = ConstTrueNode()
        self.node3 = ConstTrueNode()

        self.add_nodes(nodes=[self.node1, self.node2, self.node3])

        self.node3.start_condition = self.node1.observation_variable
        self.node3.success_condition = self.node2.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.node3.observation_variable))


@dataclass(repr=False, eq=False)
class TestRunAfterStopFromPause(Goal):
    """
    Test if child node can transition to RUNNING from PAUSED after parent node is DONE.

    Uses a CancelMotion node to raise an exception if the child node runs after the
    parent has stopped.
    """

    ticking1: CountControlCycles = field(init=False)
    ticking2: CountControlCycles = field(init=False)
    ticking3: CountControlCycles = field(init=False)
    pulse: Pulse = field(init=False)
    cancel: CancelMotion = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.ticking1 = CountControlCycles(name="3ticks", control_cycles=3)
        self.ticking2 = CountControlCycles(
            name="trigger_cancel_after_unpause", control_cycles=4
        )
        self.ticking3 = CountControlCycles(name="2ticks", control_cycles=2)
        self.pulse = Pulse()
        self.cancel = CancelMotion(
            name="Cancel_on_tick_after_done",
            exception=TestNodeAssertionError(
                reason="Node ticked after template stopped"
            ),
        )

        self.add_nodes(
            nodes=[self.ticking1, self.ticking2, self.ticking3, self.cancel, self.pulse]
        )
        self.pulse.start_condition = self.ticking3.observation_variable
        self.ticking2.pause_condition = self.pulse.observation_variable
        self.cancel.start_condition = self.ticking2.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar(self.ticking1.observation_variable))


@dataclass(repr=False, eq=False)
class TestUnpauseUnknownFromParentPause(Goal):
    """
    Tests if a child node can transition from PAUSED back to RUNNING when
    child.pause_condition is UNKNOWN.

    Child was paused by parent node being paused and child.pause_condition is UNKNOWN.
    When parent unpauses, child should transition back to RUNNING.
    """

    count_ticks1: CountControlCycles = field(init=False)
    count_ticks2: CountControlCycles = field(init=False)
    cancel: CancelMotion = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.count_ticks1 = CountControlCycles(control_cycles=2)
        self.count_ticks2 = CountControlCycles(control_cycles=5)
        self.cancel = CancelMotion(
            name="check_unpause_failed",
            exception=TestNodeAssertionError(reason="Node did not unpause correctly"),
        )

        self.add_node(self.count_ticks1)
        self.add_node(Sequence(nodes=[self.count_ticks2, self.cancel]))

        self.count_ticks1.pause_condition = sm.Scalar.const_trinary_unknown()
        self.count_ticks1.success_condition = self.count_ticks1.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.Scalar(self.count_ticks1.observation_variable)
        )


# %% goals that decide the verdict of their child


@dataclass(repr=False, eq=False)
class GoalCuttingOffItsChild(Goal):
    """
    Goal whose child never reaches a verdict of its own, so the child is only ever ended
    by this goal ending.
    """

    child: ConstFalseNode = field(init=False)
    """
    The child that keeps running until it is cut off.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = ConstFalseNode()
        self.add_node(self.child)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(repr=False, eq=False)
class GoalWithChildFailingOnItsOwn(Goal):
    """
    Goal whose child fails on its own terms on the first tick, so that a caller ending
    this goal on that same tick makes the child's own verdict compete with being cut
    off.
    """

    trigger: ConstTrueNode = field(init=False)
    """
    Turns true on the first tick, which is what fails the child.
    """

    child: ConstFalseNode = field(init=False)
    """
    The child that fails on its own terms.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.trigger = ConstTrueNode()
        self.child = ConstFalseNode()
        self.add_nodes(nodes=[self.trigger, self.child])
        self.child.failure_condition = self.trigger.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())
