import pytest
from random_events.variable import Continuous

from giskardpy.motion_statechart.beliefs.binary import BinaryBelief
from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.beliefs.gaussian import GaussianBelief
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import (
    DuplicateBeliefError,
    VariableWithoutBeliefError,
)
from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
)
from semantic_digital_twin.world import World


def gaussian_belief_over(*variables: Continuous) -> GaussianBelief:
    """
    :return: A standard normal belief over the variables.
    """
    return GaussianBelief(
        distribution=MultivariateGaussianDistribution.from_mean_and_covariance(
            distribution_variables=variables,
            mean=[0.0] * len(variables),
            covariance=[
                [float(row == column) for column in range(len(variables))]
                for row in range(len(variables))
            ],
        )
    )


def test_from_context_adds_the_beliefs_once_and_returns_them_after():
    context = MotionStatechartContext(world=World())

    first = BeliefContext.from_context(context)
    second = BeliefContext.from_context(context)

    assert first is second
    assert context.require_extension(BeliefContext) is first


def test_each_variable_of_an_added_belief_leads_to_it():
    x, y = Continuous("x"), Continuous("y")
    beliefs = BeliefContext()
    belief = gaussian_belief_over(x, y)

    beliefs.add(belief)

    assert beliefs.belief_of(x) is belief
    assert beliefs.belief_of(y) is belief


def test_beliefs_of_different_kinds_live_side_by_side():
    x = Continuous("x")
    beliefs = BeliefContext()
    gaussian = gaussian_belief_over(x)
    binary = BinaryBelief.about("coupled", probability=0.5)

    beliefs.add(gaussian)
    beliefs.add(binary)

    assert beliefs.belief_of(binary.variable) is binary
    assert beliefs.belief_of(x) is gaussian


def test_a_second_belief_about_the_same_variable_is_rejected():
    x, y = Continuous("x"), Continuous("y")
    beliefs = BeliefContext()
    beliefs.add(gaussian_belief_over(x, y))

    with pytest.raises(DuplicateBeliefError):
        beliefs.add(gaussian_belief_over(y))


def test_asking_about_a_variable_nothing_believes_in_is_rejected():
    beliefs = BeliefContext()

    with pytest.raises(VariableWithoutBeliefError):
        beliefs.belief_of(Continuous("x"))
