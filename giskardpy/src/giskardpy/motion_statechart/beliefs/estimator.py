from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from krrood.symbolic_math.symbolic_math import FloatVariable
from probabilistic_model.probabilistic_model import ProbabilisticModel
from random_events.product_algebra import SimpleEvent
from random_events.variable import Symbolic, Variable
from typing_extensions import Dict, Generic, Hashable, Optional, TypeVar

from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import (
    NodeNotBuiltError,
    UnpublishedValueError,
)
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    NodeArtifacts,
)

ModelT = TypeVar("ModelT", bound=ProbabilisticModel)
"""
The kind of distribution an estimator refines.
"""


class PublishedValue(StrEnum):
    """
    What an estimator publishes about one variable of its distribution.
    """

    MEAN = "mean"
    """
    The expectation of a numeric variable.
    """

    VARIANCE = "variance"
    """
    The variance of a numeric variable.
    """

    PROBABILITY = "probability"
    """
    The probability of one value of a symbolic variable.
    """


@dataclass(eq=False, repr=False)
class EstimatorNode(MotionStatechartNode, Generic[ModelT], ABC):
    """
    Estimates some variables of the world as a distribution, refined every control
    cycle, and publishes what it believes as float variables other nodes can build on:
    the mean and variance of every numeric variable, and the probability of every value
    of every symbolic one.

    It observes true in a cycle in which evidence arrived, and false in one in which the
    distribution ran on prediction alone.
    """

    _distribution: Optional[ModelT] = field(default=None, init=False, repr=False)
    """
    The distribution this node refines, None until the node is built.

    It is the one the statechart's :class:`BeliefContext` holds for its variables.
    """

    _means: Dict[Variable, FloatVariable] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The float variable the mean of each numeric variable is published to.
    """

    _variances: Dict[Variable, FloatVariable] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The float variable the variance of each numeric variable is published to.
    """

    _probabilities: Dict[Variable, Dict[Hashable, FloatVariable]] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The float variable the probability of each value of each symbolic variable is
    published to.
    """

    @abstractmethod
    def create_initial_distribution(self, context: MotionStatechartContext) -> ModelT:
        """
        :param context: The context the node is built in.
        :return: What is believed before any cycle, which also fixes the variables the
            node estimates.
        """

    @abstractmethod
    def predict(self, context: MotionStatechartContext, distribution: ModelT) -> ModelT:
        """
        :param context: The context the node is ticked in.
        :param distribution: What was believed at the end of the previous cycle.
        :return: What is believed one cycle later, before this cycle's evidence.
        """

    @abstractmethod
    def update(
        self, context: MotionStatechartContext, distribution: ModelT
    ) -> Optional[ModelT]:
        """
        :param context: The context the node is ticked in.
        :param distribution: What is believed before this cycle's evidence.
        :return: What is believed after it, or None if this cycle gathered no evidence.
        """

    @property
    def distribution(self) -> ModelT:
        """
        :return: The distribution this node refines.
        :raises NodeNotBuiltError: If the node has not been built yet.
        """
        if self._distribution is None:
            raise NodeNotBuiltError(node=self)
        return self._distribution

    def mean_variable(self, variable: Variable) -> FloatVariable:
        """
        :param variable: A numeric variable this node estimates.
        :return: The float variable its mean is published to.
        :raises NodeNotBuiltError: If the node has not been built yet.
        :raises UnpublishedValueError: If no mean of the variable is published.
        """
        return self._published(self._means, variable, PublishedValue.MEAN)

    def variance_variable(self, variable: Variable) -> FloatVariable:
        """
        :param variable: A numeric variable this node estimates.
        :return: The float variable its variance is published to.
        :raises NodeNotBuiltError: If the node has not been built yet.
        :raises UnpublishedValueError: If no variance of the variable is published.
        """
        return self._published(self._variances, variable, PublishedValue.VARIANCE)

    def probability_variable(
        self, variable: Variable, value: Hashable
    ) -> FloatVariable:
        """
        :param variable: A symbolic variable this node estimates.
        :param value: One of the values in its domain.
        :return: The float variable the probability of that value is published to.
        :raises NodeNotBuiltError: If the node has not been built yet.
        :raises UnpublishedValueError: If no probability of the value is published.
        """
        by_value = self._published(
            self._probabilities, variable, PublishedValue.PROBABILITY
        )
        if value not in by_value:
            raise UnpublishedValueError(
                node=self, variable=variable, value=PublishedValue.PROBABILITY
            )
        return by_value[value]

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Add the initial distribution to the beliefs of the statechart and register a
        float variable for everything published about it.

        :raises DuplicateBeliefError: If the statechart already believes something about
            one of the estimated variables.
        """
        distribution = self.create_initial_distribution(context)
        BeliefContext.from_context(context).add(distribution)
        for variable in distribution.variables:
            self._register_published_values_of(variable, context)
        self._distribution = distribution
        self._publish(context)
        return NodeArtifacts()

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        predicted = self.predict(context, self.distribution)
        updated = self.update(context, predicted)
        self._distribution = predicted if updated is None else updated
        BeliefContext.from_context(context).replace(self._distribution)
        self._publish(context)
        if updated is None:
            return ObservationStateValues.FALSE
        return ObservationStateValues.TRUE

    def _register_published_values_of(
        self, variable: Variable, context: MotionStatechartContext
    ):
        """
        Register a float variable for everything published about one variable.
        """
        if variable.is_numeric:
            self._means[variable] = self._registered(
                context, variable, PublishedValue.MEAN
            )
            self._variances[variable] = self._registered(
                context, variable, PublishedValue.VARIANCE
            )
        if isinstance(variable, Symbolic):
            self._probabilities[variable] = {
                element.element: self._registered(
                    context, variable, PublishedValue.PROBABILITY, element.element
                )
                for element in variable.domain.simple_sets
            }

    def _registered(
        self,
        context: MotionStatechartContext,
        variable: Variable,
        published_value: PublishedValue,
        value: Optional[Hashable] = None,
    ) -> FloatVariable:
        """
        :return: A new float variable for one published value, registered in the
            context's float variable data.
        """
        name = f"{self.unique_name}/{variable.name}/{published_value}"
        if value is not None:
            name = f"{name}/{value}"
        float_variable = FloatVariable(name)
        context.float_variable_data.register_expression(float_variable)
        return float_variable

    def _published(
        self,
        published: Dict[Variable, FloatVariable],
        variable: Variable,
        published_value: PublishedValue,
    ):
        """
        :return: The entry of *published* for *variable*.
        :raises NodeNotBuiltError: If the node has not been built yet.
        :raises UnpublishedValueError: If *published* has no entry for the variable.
        """
        if self._distribution is None:
            raise NodeNotBuiltError(node=self)
        if variable not in published:
            raise UnpublishedValueError(
                node=self, variable=variable, value=published_value
            )
        return published[variable]

    def _publish(self, context: MotionStatechartContext):
        """
        Write what is currently believed to the published float variables.
        """
        data = context.float_variable_data
        numeric = list(self._means)
        if numeric:
            means = self.distribution.expectation(numeric)
            variances = self.distribution.variance(numeric)
            for variable in numeric:
                data.set_value(self._means[variable], float(means[variable]))
                data.set_value(self._variances[variable], float(variances[variable]))
        for variable, by_value in self._probabilities.items():
            for value, float_variable in by_value.items():
                event = SimpleEvent.from_data({variable: value}).as_composite_set()
                data.set_value(float_variable, self.distribution.probability(event))
