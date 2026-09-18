"""
Tests for the node that keeps a belief up to date and publishes what it estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from random_events.variable import Continuous
from typing_extensions import List, Mapping

from giskardpy.executor import Executor
from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.beliefs.estimator import EstimatorNode, Prediction
from giskardpy.motion_statechart.beliefs.gaussian import (
    GaussianBelief,
    Quantities,
    QuantityPair,
    Reading,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import (
    DuplicateBeliefError,
    NodeNotBuiltError,
    VariableNotInBeliefError,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.world import World

# %% an estimator whose prior, prediction and readings the test decides


@dataclass(eq=False, repr=False)
class RecordedReadingEstimator(EstimatorNode):
    """
    Reports the readings the test put there, standing in for whatever a real estimator
    measures.
    """

    quantities: Quantities = field(kw_only=True)
    """
    The quantities the belief is about.
    """

    estimates: Mapping[Continuous, float] = field(kw_only=True)
    """
    What each of them starts out estimated at.
    """

    uncertainty: Mapping[QuantityPair, float] = field(kw_only=True)
    """
    How uncertain those starting estimates are.
    """

    process_noise: Mapping[QuantityPair, float] = field(kw_only=True)
    """
    How much uncertainty each control cycle adds on its own.
    """

    offset: Mapping[Continuous, float] = field(default_factory=dict, kw_only=True)
    """
    What each quantity gains per cycle regardless of the estimate.
    """

    readings: List[Reading] = field(default_factory=list, kw_only=True)
    """
    What the sensors are to report on every cycle; empty means nothing was measured.
    """

    def create_initial_belief(self, context: MotionStatechartContext) -> GaussianBelief:
        return GaussianBelief.of(
            quantities=self.quantities,
            estimates=self.estimates,
            uncertainty=self.uncertainty,
        )

    def create_prediction(self, context: MotionStatechartContext) -> Prediction:
        return Prediction(
            transition=self.quantities.unchanged,
            process_noise=self.process_noise,
            offset=self.offset,
        )

    def measure(self, context: MotionStatechartContext) -> List[Reading]:
        return self.readings


# %% running a single estimator


@dataclass
class TickedEstimator:
    """
    One estimator that has been built into a statechart and ticked.
    """

    node: RecordedReadingEstimator
    """
    The node under test.
    """

    context: MotionStatechartContext
    """
    The context it was ticked with, holding the values it published.
    """

    executor: Executor
    """
    The executor driving it, so a test can tick it again.
    """

    def published_estimate_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: The estimate the node published for it on its last cycle.
        """
        return self.context.float_variable_data.get_value(
            self.node.estimate_variable_of(variable)
        )

    def published_uncertainty_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: The uncertainty the node published for it on its last cycle.
        """
        return self.context.float_variable_data.get_value(
            self.node.uncertainty_variable_of(variable)
        )


def build(*nodes) -> Executor:
    """
    Compiles a statechart holding the given nodes without ticking it.

    :param nodes: The nodes to put in it.
    :return: The executor driving them.
    """
    motion_statechart = MotionStatechart()
    for node in nodes:
        motion_statechart.add_node(node)
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(motion_statechart=motion_statechart)
    return executor


def tick(node: RecordedReadingEstimator, times: int = 1) -> TickedEstimator:
    """
    Compiles a statechart holding a single estimator and ticks it.

    :param node: The estimator under test.
    :param times: How many control cycles to run.
    :return: The ticked node, its context and its executor.
    """
    executor = build(node)
    for _ in range(times):
        executor.tick()
    return TickedEstimator(node=node, context=executor.context, executor=executor)


def scalar_estimator(
    variable: Continuous,
    mean: float,
    variance: float,
    process_noise: float,
    **node_arguments,
) -> RecordedReadingEstimator:
    """
    An estimator of a single quantity.

    :param variable: The quantity it is about.
    :param mean: The estimate it starts from.
    :param variance: How uncertain that starting estimate is.
    :param process_noise: How much uncertainty each cycle adds on its own.
    :param node_arguments: Further arguments for the node.
    :return: The estimator.
    """
    return RecordedReadingEstimator(
        quantities=Quantities.of(variable),
        estimates={variable: mean},
        uncertainty={(variable, variable): variance},
        process_noise={(variable, variable): process_noise},
        **node_arguments,
    )


# %% what the estimator publishes


def test_the_prior_is_published_before_anything_is_measured():
    """
    A constraint reading the estimate on the first cycle has to see the prior rather
    than the zero an unwritten variable holds.
    """
    grasp = Continuous("grasp")
    node = scalar_estimator(grasp, mean=0.25, variance=0.5, process_noise=0.0)

    executor = build(node)
    node.on_start(executor.context)

    assert (
        executor.context.float_variable_data.get_value(node.estimate_variable_of(grasp))
        == 0.25
    )
    assert (
        executor.context.float_variable_data.get_value(
            node.uncertainty_variable_of(grasp)
        )
        == 0.5
    )


def test_a_reading_moves_the_published_estimate_toward_it():
    grasp = Continuous("grasp")
    reading = Reading.of_one_variable(grasp, value=1.0, variance=1.0)
    node = scalar_estimator(
        grasp, mean=0.0, variance=1.0, process_noise=0.0, readings=[reading]
    )

    ticked = tick(node)

    expected = GaussianBelief.of_one_variable(grasp, mean=0.0, variance=1.0)
    expected.predict(transition=Quantities.of(grasp).unchanged, process_noise={})
    expected.update([reading])
    assert ticked.published_estimate_of(grasp) == expected.mean_of(grasp)


def test_a_reading_makes_the_published_estimate_less_uncertain():
    grasp = Continuous("grasp")
    node = scalar_estimator(
        grasp,
        mean=0.0,
        variance=1.0,
        process_noise=0.0,
        readings=[Reading.of_one_variable(grasp, value=1.0, variance=1.0)],
    )

    ticked = tick(node)

    assert ticked.published_uncertainty_of(grasp) < 1.0


def test_a_cycle_without_a_reading_grows_the_published_uncertainty():
    """
    The process noise is what keeps a quantity nobody is measuring from staying as
    certain as it was when it was last seen.
    """
    grasp = Continuous("grasp")
    node = scalar_estimator(grasp, mean=0.0, variance=1.0, process_noise=0.25)

    ticked = tick(node)

    assert ticked.published_uncertainty_of(grasp) == 1.25


def test_the_offset_moves_the_estimate_without_any_reading():
    """
    ``grasp-belief-node``'s decay toward a prior is an offset, so the base class has to
    carry one through to the belief.
    """
    grasp = Continuous("grasp")
    node = scalar_estimator(
        grasp, mean=0.5, variance=1.0, process_noise=0.0, offset={grasp: -0.1}
    )

    ticked = tick(node)

    assert ticked.published_estimate_of(grasp) == pytest.approx(0.4)


def test_each_quantity_of_a_vector_belief_gets_its_own_variables():
    first, second = Continuous("base_x"), Continuous("base_yaw")
    quantities = Quantities.of(first, second)
    node = RecordedReadingEstimator(
        quantities=quantities,
        estimates={first: 1.0, second: 2.0},
        uncertainty={(first, first): 0.5, (second, second): 0.25},
        process_noise={},
    )

    ticked = tick(node)

    assert ticked.published_estimate_of(first) == 1.0
    assert ticked.published_estimate_of(second) == 2.0
    assert ticked.published_uncertainty_of(first) == 0.5
    assert ticked.published_uncertainty_of(second) == 0.25
    assert node.estimate_variable_of(first) is not node.estimate_variable_of(second)


# %% the belief outliving the cycle that produced it


def test_the_belief_is_carried_from_one_cycle_to_the_next():
    """
    The estimate having a memory is the whole point: two cycles of the same reading move
    it further than one, which a stateless measurement could not do.
    """
    grasp = Continuous("grasp")
    reading = Reading.of_one_variable(grasp, value=1.0, variance=1.0)
    after_one = tick(
        scalar_estimator(
            grasp, mean=0.0, variance=1.0, process_noise=0.0, readings=[reading]
        )
    )
    after_two = tick(
        scalar_estimator(
            grasp, mean=0.0, variance=1.0, process_noise=0.0, readings=[reading]
        ),
        times=2,
    )

    assert after_two.published_estimate_of(grasp) > after_one.published_estimate_of(
        grasp
    )


def test_the_belief_is_reachable_by_the_quantity_it_is_about():
    """
    A goal reading an estimate finds it through the statechart's beliefs, not through
    the node that happens to maintain it.
    """
    grasp = Continuous("grasp")
    node = scalar_estimator(grasp, mean=0.25, variance=0.5, process_noise=0.0)

    executor = build(node)

    assert BeliefContext.of(executor.context).require(grasp).mean_of(grasp) == 0.25


def test_beliefs_the_statechart_already_carries_are_kept():
    """
    Two estimators in one statechart share its beliefs rather than the second replacing
    what the first registered.
    """
    grasp, reach = Continuous("grasp"), Continuous("reach")

    executor = build(
        scalar_estimator(grasp, mean=0.25, variance=0.5, process_noise=0.0),
        scalar_estimator(reach, mean=0.75, variance=0.5, process_noise=0.0),
    )

    beliefs = BeliefContext.of(executor.context)
    assert beliefs.require(grasp).mean_of(grasp) == 0.25
    assert beliefs.require(reach).mean_of(reach) == 0.75


def test_two_estimators_of_one_quantity_are_rejected_while_compiling():
    """
    Two filters over one quantity would disagree silently for a whole run, so the
    statechart refuses to compile rather than ticking them both.
    """
    grasp = Continuous("grasp")

    with pytest.raises(DuplicateBeliefError) as error:
        build(
            scalar_estimator(grasp, mean=0.25, variance=0.5, process_noise=0.0),
            scalar_estimator(grasp, mean=0.75, variance=0.5, process_noise=0.0),
        )
    assert error.value.variable == grasp


# %% what the estimator observes


def test_an_estimator_that_measured_this_cycle_observes_true():
    grasp = Continuous("grasp")
    node = scalar_estimator(
        grasp,
        mean=0.0,
        variance=1.0,
        process_noise=0.0,
        readings=[Reading.of_one_variable(grasp, value=1.0, variance=1.0)],
    )

    ticked = tick(node)

    assert ticked.node.observation_state == ObservationStateValues.TRUE


def test_an_estimator_running_on_prediction_alone_observes_false():
    grasp = Continuous("grasp")
    node = scalar_estimator(grasp, mean=0.0, variance=1.0, process_noise=0.25)

    ticked = tick(node)

    assert ticked.node.observation_state == ObservationStateValues.FALSE


# %% reading the published variables


def test_the_variables_are_unavailable_before_the_node_is_built():
    grasp = Continuous("grasp")
    node = scalar_estimator(grasp, mean=0.0, variance=1.0, process_noise=0.0)

    with pytest.raises(NodeNotBuiltError) as error:
        node.estimate_variable_of(grasp)
    assert error.value.node is node


def test_a_quantity_the_estimator_is_not_about_is_rejected():
    grasp = Continuous("grasp")
    unestimated = Continuous("reach")
    node = scalar_estimator(grasp, mean=0.0, variance=1.0, process_noise=0.0)
    build(node)

    with pytest.raises(VariableNotInBeliefError) as error:
        node.uncertainty_variable_of(unestimated)
    assert error.value.variable == unestimated
