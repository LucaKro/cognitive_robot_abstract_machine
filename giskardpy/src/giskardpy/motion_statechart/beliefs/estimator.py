from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from krrood.symbolic_math.symbolic_math import FloatVariable
from typing_extensions import TYPE_CHECKING, Dict, Generic, List, Optional

from giskardpy.motion_statechart.beliefs.belief import (
    Belief,
    EvidenceT,
    PredictionT,
    Statistic,
    VariableStatistic,
)
from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import (
    NodeNotBuiltError,
    UnpublishedStatisticError,
)
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    NodeArtifacts,
)

if TYPE_CHECKING:
    from random_events.variable import Variable


@dataclass(eq=False, repr=False)
class EstimatorNode(MotionStatechartNode, Generic[PredictionT, EvidenceT], ABC):
    """
    Estimates some variables of the world as a belief, refined every control cycle, and
    publishes the statistics of that belief as float variables other nodes can build on.

    It observes true in a cycle in which evidence arrived, and false in one in which the
    belief ran on prediction alone.
    """

    _belief: Optional[Belief[PredictionT, EvidenceT]] = field(
        default=None, init=False, repr=False
    )
    """
    The belief this node refines, None until the node is built.
    """

    _published_variables: Dict[VariableStatistic, FloatVariable] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The float variable each statistic of the belief is published to.
    """

    @abstractmethod
    def create_initial_belief(
        self, context: MotionStatechartContext
    ) -> Belief[PredictionT, EvidenceT]:
        """
        :param context: The context the node is built in.
        :return: What is believed before any cycle, which also fixes the variables the
            node estimates.
        """

    @abstractmethod
    def create_prediction(self, context: MotionStatechartContext) -> PredictionT:
        """
        :param context: The context the node is ticked in.
        :return: What happens to the estimated variables over this cycle.
        """

    @abstractmethod
    def measure(self, context: MotionStatechartContext) -> List[EvidenceT]:
        """
        :param context: The context the node is ticked in.
        :return: The evidence gathered in this cycle, empty if there is none.
        """

    @property
    def belief(self) -> Belief[PredictionT, EvidenceT]:
        """
        :return: The belief this node refines.
        :raises NodeNotBuiltError: If the node has not been built yet.
        """
        if self._belief is None:
            raise NodeNotBuiltError(node=self)
        return self._belief

    def published_variable(
        self, variable: Variable, statistic: Statistic
    ) -> FloatVariable:
        """
        :param variable: One of the variables this node estimates.
        :param statistic: A statistic its belief reports about it.
        :return: The float variable the statistic is published to.
        :raises NodeNotBuiltError: If the node has not been built yet.
        :raises UnpublishedStatisticError: If the belief does not report the statistic.
        """
        if self._belief is None:
            raise NodeNotBuiltError(node=self)
        key = VariableStatistic(variable, statistic)
        if key not in self._published_variables:
            raise UnpublishedStatisticError(node=self, statistic=key)
        return self._published_variables[key]

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Add the initial belief to the beliefs of the statechart and register a float
        variable for each of its statistics.

        :raises DuplicateBeliefError: If the statechart already believes something about
            one of the estimated variables.
        """
        belief = self.create_initial_belief(context)
        BeliefContext.from_context(context).add(belief)
        for key in belief.statistics():
            published = FloatVariable(
                f"{self.unique_name}/{key.variable.name}/{key.statistic}"
            )
            context.float_variable_data.register_expression(published)
            self._published_variables[key] = published
        self._belief = belief
        self._publish(context)
        return NodeArtifacts()

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        self.belief.predict(self.create_prediction(context))
        evidence = self.measure(context)
        self.belief.update(evidence)
        self._publish(context)
        return ObservationStateValues.TRUE if evidence else ObservationStateValues.FALSE

    def _publish(self, context: MotionStatechartContext):
        """
        Write the current statistics of the belief to their float variables.
        """
        for key, value in self.belief.statistics().items():
            context.float_variable_data.set_value(self._published_variables[key], value)
