from __future__ import division

from dataclasses import dataclass, field
from typing import List

from typing_extensions import Optional

from krrood.symbolic_math.symbolic_math import Scalar, sum
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
    TerminalNode,
)


@dataclass(repr=False, eq=False)
class Sequence(Goal):
    """
    Takes a list of nodes and wires their start and end conditions such that they are
    executed in order.

    Its observation is whether the last node in the sequence reached its goal.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)

    def expand(self, context: MotionStatechartContext) -> None:
        """
        A step ends itself once it observes its goal, which succeeds it, and the next
        step reads that verdict rather than the observation behind it, because only the
        verdict outlasts the step that reached it.
        """
        self._check_has_children()
        last_node: Optional[MotionStatechartNode] = None
        for i, node in enumerate(self.nodes):
            self.add_node(node)
            if last_node is not None:
                node.start_condition = last_node.is_succeeded
            # A node that ends the motion has nothing left to transition to.
            if not isinstance(node, TerminalNode):
                node.end_condition = node.observation_variable
            last_node = node

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=Scalar(self.nodes[-1].goal_reached))


@dataclass(repr=False, eq=False)
class Parallel(Goal):
    """
    Takes a list of nodes and executes them in parallel.

    Its observation turns True once at least :attr:`minimum_success` of them reached
    their goals.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    minimum_success: Optional[int] = field(default=None, kw_only=True)
    """
    How many nodes must have reached their goals for this goal to be achieved.

    Defaults to None, which means all of them.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self._check_has_children()
        for node in self.nodes:
            self.add_node(node)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Count the nodes that reached their goals, and compare that against
        :attr:`minimum_success`.

        This goal ends none of its nodes, so a node that keeps running is counted by
        what it observes now and stops counting once it drifts away from its goal again.
        A node something *else* ended keeps counting, because its verdict outlasts it.
        """
        nodes_at_their_goal = [node.goal_reached == True for node in self.nodes]
        minimum_success = (
            self.minimum_success
            if self.minimum_success is not None
            else len(self.nodes)
        )
        return NodeArtifacts(observation=minimum_success <= sum(*nodes_at_their_goal))
