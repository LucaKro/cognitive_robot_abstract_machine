from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    LinearGaussianModel,
    MultivariateGaussianDistribution,
)
from probabilistic_model.utils import MissingDict
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from typing_extensions import List, Optional

from giskardpy.executor import Executor
from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.beliefs.estimator import EstimatorNode
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import (
    DuplicateBeliefError,
    NodeNotBuiltError,
    UnpublishedValueError,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.world import World

# %% estimators whose evidence the test decides


@dataclass(eq=False, repr=False)
class ScriptedObservationsEstimator(EstimatorNode[MultivariateGaussianDistribution]):
    """
    Estimates one continuous variable with a Kalman filter, from observations the test
    decides, one per control cycle.
    """

    variable: Continuous = field(kw_only=True)
    """
    The variable estimated.
    """

    prior_variance: float = field(kw_only=True, default=1.0)
    """
    How uncertain the estimate is before any cycle.
    """

    transition_variance: float = field(kw_only=True, default=0.0)
    """
    How much less certain the estimate becomes each cycle.
    """

    observation_variance: float = field(kw_only=True, default=0.5)
    """
    How far each observation scatters around the true value.
    """

    observations_per_cycle: List[Optional[float]] = field(
        kw_only=True, default_factory=list
    )
    """
    What is observed in each cycle, None for a cycle without an observation; cycles past
    its end observe nothing.
    """

    cycles_updated: int = field(init=False, default=0)
    """
    How many cycles have asked for an update so far.
    """

    def create_initial_distribution(
        self, context: MotionStatechartContext
    ) -> MultivariateGaussianDistribution:
        return MultivariateGaussianDistribution.from_mean_and_covariance(
            variables=[self.variable],
            mean=[0.0],
            covariance=[[self.prior_variance]],
        )

    def predict(
        self,
        context: MotionStatechartContext,
        distribution: MultivariateGaussianDistribution,
    ) -> MultivariateGaussianDistribution:
        return distribution.linear_gaussian_transition(
            LinearGaussianModel(
                matrix=np.eye(1),
                offset=np.zeros(1),
                covariance=Covariance.from_matrix([[self.transition_variance]]),
            )
        )

    def update(
        self,
        context: MotionStatechartContext,
        distribution: MultivariateGaussianDistribution,
    ) -> Optional[MultivariateGaussianDistribution]:
        observed = self.observation_in(self.cycles_updated)
        self.cycles_updated += 1
        if observed is None:
            return None
        return self.corrected(distribution, observed)

    def observation_in(self, cycle: int) -> Optional[float]:
        """
        :param cycle: The index of a control cycle.
        :return: What is observed in it, None if nothing is.
        """
        if cycle >= len(self.observations_per_cycle):
            return None
        return self.observations_per_cycle[cycle]

    def corrected(
        self, distribution: MultivariateGaussianDistribution, observed: float
    ) -> MultivariateGaussianDistribution:
        """
        :return: The distribution corrected by one observation of the variable.
        """
        return distribution.product_with_gaussian_likelihood(
            MultivariateGaussianDistribution.from_mean_and_covariance(
                variables=[self.variable],
                mean=[observed],
                covariance=[[self.observation_variance]],
            )
        )


@dataclass(eq=False, repr=False)
class ScriptedLikelihoodsEstimator(EstimatorNode[SymbolicDistribution]):
    """
    Estimates whether a binary state holds with a discrete Bayes filter, from the
    likelihoods the test decides for the first cycle.
    """

    variable: Symbolic = field(kw_only=True)
    """
    The state estimated.
    """

    likelihoods: List[float] = field(kw_only=True)
    """
    How likely the first cycle's observation is under each value of the state.
    """

    def create_initial_distribution(
        self, context: MotionStatechartContext
    ) -> SymbolicDistribution:
        probabilities = MissingDict(float)
        for element in self.variable.domain.simple_sets:
            probabilities[hash(element)] = 1 / len(self.variable.domain.simple_sets)
        return SymbolicDistribution(variable=self.variable, probabilities=probabilities)

    def predict(
        self, context: MotionStatechartContext, distribution: SymbolicDistribution
    ) -> SymbolicDistribution:
        return distribution

    def update(
        self, context: MotionStatechartContext, distribution: SymbolicDistribution
    ) -> Optional[SymbolicDistribution]:
        return distribution.product_with_likelihood(self.likelihoods)


@dataclass(eq=False, repr=False)
class JointStatesEstimator(EstimatorNode[MultinomialDistribution]):
    """
    Holds one joint distribution over two binary states, without evidence.
    """

    first: Symbolic = field(kw_only=True)
    """
    One of the states.
    """

    second: Symbolic = field(kw_only=True)
    """
    The other state.
    """

    probabilities: np.ndarray = field(kw_only=True)
    """
    The joint table, indexed by the two states' domains in order.
    """

    def create_initial_distribution(
        self, context: MotionStatechartContext
    ) -> MultinomialDistribution:
        return MultinomialDistribution(
            distribution_variables=(self.first, self.second),
            probabilities=self.probabilities,
        )

    def predict(
        self, context: MotionStatechartContext, distribution: MultinomialDistribution
    ) -> MultinomialDistribution:
        return distribution

    def update(
        self, context: MotionStatechartContext, distribution: MultinomialDistribution
    ) -> Optional[MultinomialDistribution]:
        return None


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


def published_value(executor: Executor, variable) -> float:
    """
    :return: The value currently written to a published float variable.
    """
    return executor.context.float_variable_data.get_value(variable)


# %% the tick


def test_the_expectation_and_variance_of_a_numeric_variable_are_published():
    x = Continuous("x")
    estimator = ScriptedObservationsEstimator(
        variable=x, transition_variance=0.1, observations_per_cycle=[1.0]
    )
    executor = compiled_executor(estimator)

    executor.tick()
    executor.tick()

    distribution = estimator.distribution
    assert published_value(
        executor, estimator.expectation_variable(x)
    ) == pytest.approx(distribution.expectation([x])[x])
    assert published_value(executor, estimator.variance_variable(x)) == pytest.approx(
        distribution.variance([x])[x]
    )


def test_the_probability_of_each_value_of_a_symbolic_variable_is_published():
    state = Symbolic("coupled", domain=Set.from_iterable((False, True)))
    estimator = ScriptedLikelihoodsEstimator(variable=state, likelihoods=[0.2, 0.8])
    executor = compiled_executor(estimator)

    executor.tick()

    for value in (False, True):
        event = SimpleEvent.from_data({state: value}).as_composite_set()
        assert published_value(
            executor, estimator.probability_variable(state, value)
        ) == pytest.approx(estimator.distribution.probability(event))


def test_the_probability_of_a_value_is_its_marginal_in_a_joint_distribution():
    coupled = Symbolic("coupled", domain=Set.from_iterable((False, True)))
    moving = Symbolic("moving", domain=Set.from_iterable((False, True)))
    probabilities = np.array([[0.1, 0.2], [0.3, 0.4]])
    estimator = JointStatesEstimator(
        first=coupled, second=moving, probabilities=probabilities
    )
    executor = compiled_executor(estimator)

    executor.tick()

    for index, value in enumerate((False, True)):
        assert published_value(
            executor, estimator.probability_variable(coupled, value)
        ) == pytest.approx(probabilities[index].sum())
        assert published_value(
            executor, estimator.probability_variable(moving, value)
        ) == pytest.approx(probabilities[:, index].sum())


def test_estimator_observes_true_exactly_in_cycles_with_evidence():
    x = Continuous("x")
    observations_per_cycle = [None, 1.0, None, 2.0, None]
    estimator = ScriptedObservationsEstimator(
        variable=x, observations_per_cycle=observations_per_cycle
    )
    executor = compiled_executor(estimator)

    for _ in range(len(observations_per_cycle) - 1):
        executor.tick()
        cycle = estimator.cycles_updated - 1
        expected = (
            ObservationStateValues.FALSE
            if estimator.observation_in(cycle) is None
            else ObservationStateValues.TRUE
        )
        assert estimator.observation_state == expected


def test_evidence_in_a_cycle_corrects_the_distribution():
    x = Continuous("x")
    observed = 1.0
    estimator = ScriptedObservationsEstimator(
        variable=x, observations_per_cycle=[observed]
    )
    executor = compiled_executor(estimator)
    expected = estimator.corrected(
        estimator.create_initial_distribution(executor.context), observed
    )

    executor.tick()

    assert estimator.distribution.mean == pytest.approx(expected.mean)
    assert estimator.distribution.covariance.matrix == pytest.approx(
        expected.covariance.matrix
    )


def test_prediction_runs_every_cycle():
    x = Continuous("x")
    prior_variance, transition_variance = 1.0, 0.25
    estimator = ScriptedObservationsEstimator(
        variable=x,
        prior_variance=prior_variance,
        transition_variance=transition_variance,
    )
    executor = compiled_executor(estimator)

    executor.tick()
    executor.tick()

    assert estimator.distribution.variance([x])[x] == pytest.approx(
        prior_variance + estimator.cycles_updated * transition_variance
    )


# %% the build


def test_estimator_keeps_its_distribution_in_the_context():
    x = Continuous("x")
    estimator = ScriptedObservationsEstimator(variable=x, observations_per_cycle=[1.0])
    executor = compiled_executor(estimator)

    executor.tick()

    beliefs = executor.context.require_extension(BeliefContext)
    assert beliefs.distribution_of(x) is estimator.distribution


def test_two_estimators_of_one_variable_fail_to_compile():
    x = Continuous("x")

    with pytest.raises(DuplicateBeliefError):
        compiled_executor(
            ScriptedObservationsEstimator(variable=x),
            ScriptedObservationsEstimator(variable=x),
        )


def test_the_distribution_is_unavailable_before_the_build():
    estimator = ScriptedObservationsEstimator(variable=Continuous("x"))

    with pytest.raises(NodeNotBuiltError):
        estimator.expectation_variable(estimator.variable)


def test_a_value_the_distribution_does_not_have_is_not_published():
    estimator = ScriptedObservationsEstimator(variable=Continuous("x"))
    compiled_executor(estimator)

    with pytest.raises(UnpublishedValueError):
        estimator.probability_variable(estimator.variable, True)
