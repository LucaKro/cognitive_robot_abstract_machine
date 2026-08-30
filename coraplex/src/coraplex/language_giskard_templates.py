from __future__ import division

from dataclasses import dataclass, field
from typing import List

from typing_extensions import Optional

from krrood.symbolic_math.symbolic_math import (
    Scalar,
    trinary_logic_not,
    trinary_logic_or,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
)
from giskardpy.motion_statechart.monitors.progress_monitors import (
    DEFAULT_STALL_TIMEOUT,
    StillProgressing,
)


@dataclass(repr=False, eq=False)
class TryAll(Goal):
    """
    Takes a list of nodes and executes them in parallel.

    Its observation turns True as soon as any node is True and turns False only when all
    nodes are False, i.e. it only fails if every node fails.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes executed in parallel.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add all child nodes to this goal so they run in parallel.
        """
        for node in self.nodes:
            self.add_node(node)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True as soon as any child node reached its goal.

        This goal ends none of its children, so a child that keeps running is judged by
        what it observes now rather than by a verdict it never reaches.
        """
        return NodeArtifacts(
            observation=_any_of([node.goal_reached for node in self.nodes]),
        )


@dataclass(repr=False, eq=False)
class TryInOrder(Goal):
    """
    Takes a list of nodes and tries them one after another, short-circuiting on the
    first success.

    The next alternative only starts once the previous one has actually failed, not
    merely while it is still short of its goal. Its observation turns True as soon as an
    alternative succeeds and False only once every one of them has failed, so it stays
    unknown while any of them is still being tried.
    """

    alternatives: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes tried one after another, in order.

    Kept apart from :attr:`~giskardpy.motion_statechart.graph_node.Goal.nodes`, which
    also holds the progress monitor :meth:`expand` adds for each of them.
    """

    give_up_after: float = field(default=DEFAULT_STALL_TIMEOUT, kw_only=True)
    """
    Seconds of simulated time an alternative may make no progress before it is abandoned
    and the next one is tried.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add the child nodes and wire them so each one starts only after the previous one
        failed, short-circuiting on the first success.

        An alternative is ended once it reaches its goal or once it stops making
        progress; which of the two happened is decided by what the alternative observes
        as it ends, not here. An observation that is merely still false means the
        alternative has not arrived yet, and is no reason to abandon it.
        """
        previous_node: Optional[MotionStatechartNode] = None
        for node in self.alternatives:
            self.add_node(node)
            if previous_node is not None:
                node.start_condition = previous_node.is_failed
            still_progressing = StillProgressing(
                name=f"{self.name}/progress_of_{node.name}",
                monitored_node=node,
                timeout=self.give_up_after,
            )
            self.add_node(still_progressing)
            still_progressing.start_condition = node.is_running
            node.end_condition = trinary_logic_or(
                node.observation_variable,
                trinary_logic_not(still_progressing.observation_variable),
            )
            previous_node = node

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True as soon as any alternative succeeded, and
        False only once every one of them failed.
        """
        return NodeArtifacts(
            observation=_any_of([node.goal_reached for node in self.alternatives]),
        )


# %% combining a variable number of children


def _any_of(expressions: List[Scalar]) -> Scalar:
    """
    :param expressions: The trinary expressions to combine, at least one.
    :return: The disjunction of the expressions, or the single expression itself.
    """
    if len(expressions) == 1:
        return Scalar(expressions[0])
    return trinary_logic_or(*expressions)
