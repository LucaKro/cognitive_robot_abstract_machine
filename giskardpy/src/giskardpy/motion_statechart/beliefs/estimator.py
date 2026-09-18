from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from random_events.variable import Continuous
from typing_extensions import Dict, List, Mapping, Optional

from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.beliefs.gaussian import (
    GaussianBelief,
    QuantityPair,
    Reading,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import (
    NodeNotBuiltError,
    VariableNotInBeliefError,
)
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from krrood.symbolic_math.symbolic_math import FloatVariable

# %% what an estimator publishes about each of its quantities


class PublishedValue(StrEnum):
    """
    What one of an estimator's variables says about the quantity it is about, which also
    names that variable.
    """

    ESTIMATE = "estimate"
    """
    What the quantity is currently estimated at.
    """

    UNCERTAINTY = "uncertainty"
    """
    How uncertain that estimate is.
    """


# %% how a belief is expected to change over one control cycle


@dataclass
class Prediction:
    """
    How an estimator expects its belief to change over one control cycle, before
    anything is measured.
    """

    transition: Mapping[QuantityPair, float]
    """
    How much each quantity's estimate carries into each quantity's next one.
    """

    process_noise: Mapping[QuantityPair, float]
    """
    How much uncertainty the cycle itself adds, which is what keeps a quantity nobody is
    measuring from staying as certain as it was when it was last seen.
    """

    offset: Mapping[Continuous, float] = field(default_factory=dict)
    """
    What each quantity gains regardless of the estimate, such as a pull toward a prior.
    """


# %% the node that keeps a belief up to date


@dataclass(eq=False, repr=False)
class EstimatorNode(MotionStatechartNode, ABC):
    """
    Keeps a belief about continuous quantities up to date, one control cycle at a time,
    and publishes what it currently estimates.

    Every cycle the belief is carried forward by :meth:`create_prediction`, corrected by
    whatever :meth:`measure` reports, and written out: one variable per quantity for the
    estimate and one for how uncertain it is, so constraints and transition conditions
    can read either. The belief itself is registered in the statechart's
    :class:`~giskardpy.motion_statechart.beliefs.context.BeliefContext`, which is where
    anything that is not this node reads it from.

    The node observes whether it measured anything this cycle, so another node can
    branch on an estimator that has stopped being corrected. A subclass whose quantity
    has a meaningful threshold observes that instead, by returning its own answer from
    :meth:`on_tick` once ``super().on_tick(context)`` has run the cycle.

    .. warning:: :meth:`measure` runs on the control loop. Whatever it costs is paid
        every cycle.
    """

    _belief: Optional[GaussianBelief] = field(default=None, init=False, repr=False)
    """
    The belief being maintained, created while building.
    """

    _estimate_variables: Dict[Continuous, FloatVariable] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The variable each quantity's estimate is written to, created while building.
    """

    _uncertainty_variables: Dict[Continuous, FloatVariable] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The variable each quantity's uncertainty is written to, created while building.
    """

    def estimate_variable_of(self, variable: Continuous) -> FloatVariable:
        """
        :param variable: The quantity to read.
        :return: The variable carrying its estimate, for use in constraints and
            conditions.
        :raises NodeNotBuiltError: If the node has not been built yet.
        :raises VariableNotInBeliefError: If this estimator is not about that quantity.
        """
        return self._published_variable(self._estimate_variables, variable)

    def uncertainty_variable_of(self, variable: Continuous) -> FloatVariable:
        """
        :param variable: The quantity to read.
        :return: The variable carrying how uncertain its estimate is, for use in
            constraints and conditions.
        :raises NodeNotBuiltError: If the node has not been built yet.
        :raises VariableNotInBeliefError: If this estimator is not about that quantity.
        """
        return self._published_variable(self._uncertainty_variables, variable)

    def _published_variable(
        self, variables: Dict[Continuous, FloatVariable], variable: Continuous
    ) -> FloatVariable:
        """
        :param variables: The published variables to read one of.
        :param variable: The quantity to read.
        :return: The variable published for it.
        :raises NodeNotBuiltError: If nothing has been published yet.
        :raises VariableNotInBeliefError: If this estimator is not about that quantity.
        """
        if self._belief is None:
            raise NodeNotBuiltError(node=self)
        if variable not in variables:
            raise VariableNotInBeliefError(
                variable=variable, belief_variables=list(variables)
            )
        return variables[variable]

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        belief = self.create_initial_belief(context)
        BeliefContext.of(context).add(belief)
        self._belief = belief
        for variable in self._belief.quantities:
            self._estimate_variables[variable] = self._register(
                context, variable, PublishedValue.ESTIMATE
            )
            self._uncertainty_variables[variable] = self._register(
                context, variable, PublishedValue.UNCERTAINTY
            )
        return NodeArtifacts()

    def _register(
        self,
        context: MotionStatechartContext,
        variable: Continuous,
        published: PublishedValue,
    ) -> FloatVariable:
        """
        :param context: The context holding the float variable data to register in.
        :param variable: The quantity the new variable is about.
        :param published: What the new variable says about that quantity.
        :return: The registered variable.
        """
        registered = FloatVariable(f"{self.unique_name}_{variable.name}_{published}")
        context.float_variable_data.register_expression(registered)
        return registered

    def on_start(self, context: MotionStatechartContext) -> None:
        self._publish(context)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        prediction = self.create_prediction(context)
        self._belief.predict(
            transition=prediction.transition,
            process_noise=prediction.process_noise,
            offset=prediction.offset,
        )
        readings = self.measure(context)
        self._belief.update(readings)
        self._publish(context)
        if not readings:
            return ObservationStateValues.FALSE
        return ObservationStateValues.TRUE

    def _publish(self, context: MotionStatechartContext) -> None:
        """
        Writes the current estimate and uncertainty of every quantity into the variables
        carrying them.

        :param context: The context holding the float variable data to write to.
        """
        for variable in self._belief.quantities:
            context.float_variable_data.set_value(
                self._estimate_variables[variable], self._belief.mean_of(variable)
            )
            context.float_variable_data.set_value(
                self._uncertainty_variables[variable],
                self._belief.variance_of(variable),
            )

    @abstractmethod
    def create_initial_belief(self, context: MotionStatechartContext) -> GaussianBelief:
        """
        Build the belief to start from, which also names the quantities this estimator
        is about.

        :param context: The context that contains data that can be used to build it.
        :return: The belief before anything has been measured.
        """

    @abstractmethod
    def create_prediction(self, context: MotionStatechartContext) -> Prediction:
        """
        Say how the belief is expected to change over the coming control cycle.

        :param context: The context that contains data that can be used to decide it.
        :return: The change to apply before this cycle's readings are taken into
            account.
        """

    @abstractmethod
    def measure(self, context: MotionStatechartContext) -> List[Reading]:
        """
        Take this control cycle's readings.

        .. warning:: This runs inside the control loop, make sure it is fast.

        :param context: The context that contains data that can be measured.
        :return: What the sensors reported, or nothing to leave the estimate on
            prediction alone.
        """
