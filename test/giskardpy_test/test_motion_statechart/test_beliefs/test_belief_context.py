import numpy as np
import pytest
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
)
from probabilistic_model.utils import MissingDict
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import (
    DuplicateBeliefError,
    VariableWithoutBeliefError,
)
from semantic_digital_twin.world import World

# %% helpers


def standard_normal_over(*variables: Continuous) -> MultivariateGaussianDistribution:
    """
    :return: A standard normal distribution over the variables.
    """
    return MultivariateGaussianDistribution.from_mean_and_covariance(
        variables=variables,
        mean=np.zeros(len(variables)),
        covariance=np.eye(len(variables)),
    )


def even_odds_about(name: str) -> SymbolicDistribution:
    """
    :return: A distribution giving a binary state even odds of holding.
    """
    variable = Symbolic(name, domain=Set.from_iterable((False, True)))
    probabilities = MissingDict(float)
    probabilities[hash(False)] = 0.5
    probabilities[hash(True)] = 0.5
    return SymbolicDistribution(variable=variable, probabilities=probabilities)


# %% the context


def test_from_context_adds_the_beliefs_once_and_returns_them_after():
    context = MotionStatechartContext(world=World())

    first = BeliefContext.from_context(context)
    second = BeliefContext.from_context(context)

    assert first is second
    assert context.require_extension(BeliefContext) is first


def test_each_variable_of_an_added_distribution_leads_to_it():
    x, y = Continuous("x"), Continuous("y")
    beliefs = BeliefContext()
    distribution = standard_normal_over(x, y)

    beliefs.add(distribution)

    assert beliefs.distribution_of(x) is distribution
    assert beliefs.distribution_of(y) is distribution


def test_distributions_of_different_kinds_live_side_by_side():
    x = Continuous("x")
    beliefs = BeliefContext()
    gaussian = standard_normal_over(x)
    symbolic = even_odds_about("coupled")

    beliefs.add(gaussian)
    beliefs.add(symbolic)

    assert beliefs.distribution_of(symbolic.variable) is symbolic
    assert beliefs.distribution_of(x) is gaussian


def test_a_second_distribution_over_the_same_variable_is_rejected():
    x, y = Continuous("x"), Continuous("y")
    beliefs = BeliefContext()
    beliefs.add(standard_normal_over(x, y))

    with pytest.raises(DuplicateBeliefError):
        beliefs.add(standard_normal_over(y))


def test_asking_about_a_variable_nothing_believes_in_is_rejected():
    beliefs = BeliefContext()

    with pytest.raises(VariableWithoutBeliefError):
        beliefs.distribution_of(Continuous("x"))


def test_replacing_a_distribution_leads_each_of_its_variables_to_the_new_one():
    x, y = Continuous("x"), Continuous("y")
    beliefs = BeliefContext()
    beliefs.add(standard_normal_over(x, y))
    replacement = standard_normal_over(x, y)

    beliefs.replace(replacement)

    assert beliefs.distribution_of(x) is replacement
    assert beliefs.distribution_of(y) is replacement


def test_replacing_a_distribution_over_a_variable_nothing_believes_in_is_rejected():
    beliefs = BeliefContext()

    with pytest.raises(VariableWithoutBeliefError):
        beliefs.replace(standard_normal_over(Continuous("x")))
