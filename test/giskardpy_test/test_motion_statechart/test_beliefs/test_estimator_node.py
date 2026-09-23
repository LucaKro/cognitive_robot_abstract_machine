from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from random_events.variable import Continuous
from typing_extensions import List

from giskardpy.executor import Executor
from giskardpy.motion_statechart.beliefs.belief import Statistic
from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.beliefs.estimator import EstimatorNode
from giskardpy.motion_statechart.beliefs.gaussian import (
    GaussianBelief,
    LinearPrediction,
    Reading,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import (
    DuplicateBeliefError,
    NodeNotBuiltError,
    UnpublishedStatisticError,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
)
from semantic_digital_twin.world import World

# %% an estimator whose readings the test decides


@dataclass(eq=False, repr=False)
class ScriptedReadingsEstimator(EstimatorNode[LinearPrediction, Reading]):
    """
    Estimates one variable from readings the test decides, one list per control cycle.
    """

    variable: Continuous = field(kw_only=True)
    """
    The variable estimated.
    """

    prior_variance: float = field(kw_only=True, default=1.0)
    """
    How uncertain the estimate is before any cycle.
    """

    process_noise: float = field(kw_only=True, default=0.0)
    """
    How much less certain the estimate becomes each cycle.
    """

    readings_per_cycle: List[List[Reading]] = field(kw_only=True, default_factory=list)
    """
    What the sensors report in each cycle; cycles past its end report nothing.
    """

    cycles_measured: int = field(init=False, default=0)
    """
    How many cycles have been measured so far.
    """

    def create_initial_belief(self, context: MotionStatechartContext) -> GaussianBelief:
        return GaussianBelief(
            distribution=MultivariateGaussianDistribution.from_mean_and_covariance(
                distribution_variables=[self.variable],
                mean=[0.0],
                covariance=[[self.prior_variance]],
            )
        )

    def create_prediction(self, context: MotionStatechartContext) -> LinearPrediction:
        return LinearPrediction(process_noise={self.variable: self.process_noise})

    def measure(self, context: MotionStatechartContext) -> List[Reading]:
        readings = (
            self.readings_per_cycle[self.cycles_measured]
            if self.cycles_measured < len(self.readings_per_cycle)
            else []
        )
        self.cycles_measured += 1
        return readings


def compiled_executor(*estimators: EstimatorNode) -> Executor:
    """
    :return: An executor that has compiled a statechart holding the estimators.
    """
    motion_statechart = MotionStatechart()
    for estimator in estimators:
        motion_statechart.add_node(estimator)
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(motion_statechart=motion_statechart)
    return executor


# %% the tick


def test_published_variables_hold_the_belief_statistics():
    x = Continuous("x")
    estimator = ScriptedReadingsEstimator(
        variable=x,
        process_noise=0.1,
        readings_per_cycle=[[Reading(value=1.0, contributions={x: 1.0}, variance=0.5)]],
    )
    executor = compiled_executor(estimator)

    executor.tick()
    executor.tick()

    for statistic, value in estimator.belief.statistics().items():
        published = estimator.published_variable(
            statistic.variable, statistic.statistic
        )
        assert executor.context.float_variable_data.get_value(published) == value


def test_estimator_observes_true_exactly_in_cycles_with_evidence():
    x = Continuous("x")
    reading = Reading(value=1.0, contributions={x: 1.0}, variance=0.5)
    readings_per_cycle = [[], [reading], [], [reading], []]
    estimator = ScriptedReadingsEstimator(
        variable=x, readings_per_cycle=readings_per_cycle
    )
    executor = compiled_executor(estimator)

    for _ in range(len(readings_per_cycle) - 1):
        executor.tick()
        cycle = estimator.cycles_measured - 1
        expected = (
            ObservationStateValues.TRUE
            if readings_per_cycle[cycle]
            else ObservationStateValues.FALSE
        )
        assert estimator.observation_state == expected


def test_evidence_measured_in_a_cycle_corrects_the_belief():
    x = Continuous("x")
    reading = Reading(value=1.0, contributions={x: 1.0}, variance=0.5)
    estimator = ScriptedReadingsEstimator(variable=x, readings_per_cycle=[[reading]])
    executor = compiled_executor(estimator)
    expected = estimator.create_initial_belief(executor.context)

    executor.tick()

    expected.update([reading])
    assert estimator.belief.mean_of(x) == pytest.approx(expected.mean_of(x))
    assert estimator.belief.variance_of(x) == pytest.approx(expected.variance_of(x))


def test_prediction_runs_every_cycle():
    x = Continuous("x")
    prior_variance, process_noise = 1.0, 0.25
    estimator = ScriptedReadingsEstimator(
        variable=x, prior_variance=prior_variance, process_noise=process_noise
    )
    executor = compiled_executor(estimator)

    executor.tick()
    executor.tick()

    assert estimator.belief.variance_of(x) == pytest.approx(
        prior_variance + estimator.cycles_measured * process_noise
    )


# %% the build


def test_estimator_registers_its_belief_in_the_context():
    x = Continuous("x")
    estimator = ScriptedReadingsEstimator(variable=x)

    executor = compiled_executor(estimator)

    beliefs = executor.context.require_extension(BeliefContext)
    assert beliefs.belief_of(x) is estimator.belief


def test_two_estimators_of_one_variable_fail_to_compile():
    x = Continuous("x")

    with pytest.raises(DuplicateBeliefError):
        compiled_executor(
            ScriptedReadingsEstimator(variable=x), ScriptedReadingsEstimator(variable=x)
        )


def test_published_variable_is_unavailable_before_the_build():
    estimator = ScriptedReadingsEstimator(variable=Continuous("x"))

    with pytest.raises(NodeNotBuiltError):
        estimator.published_variable(estimator.variable, Statistic.MEAN)


def test_a_statistic_the_belief_does_not_report_is_not_published():
    estimator = ScriptedReadingsEstimator(variable=Continuous("x"))
    compiled_executor(estimator)

    with pytest.raises(UnpublishedStatisticError):
        estimator.published_variable(estimator.variable, Statistic.PROBABILITY)
