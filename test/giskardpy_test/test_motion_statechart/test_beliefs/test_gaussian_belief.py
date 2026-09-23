import numpy as np
import pytest
from random_events.variable import Continuous

from giskardpy.motion_statechart.beliefs.belief import Statistic, VariableStatistic
from giskardpy.motion_statechart.beliefs.gaussian import (
    GaussianBelief,
    LinearPrediction,
    Reading,
)
from giskardpy.motion_statechart.exceptions import NegativeVarianceError
from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
)
from probabilistic_model.exceptions import VariableNotInDistributionError

# %% helpers


def belief_over(variables, mean, covariance) -> GaussianBelief:
    """
    :return: A Gaussian belief with the given mean and covariance over the variables.
    """
    return GaussianBelief(
        distribution=MultivariateGaussianDistribution.from_mean_and_covariance(
            distribution_variables=variables, mean=mean, covariance=covariance
        )
    )


# %% prediction


def test_prediction_without_transition_keeps_the_estimate_and_adds_process_noise():
    x = Continuous("x")
    belief = belief_over([x], [1.0], [[2.0]])

    belief.predict(LinearPrediction(process_noise={x: 0.5}))

    assert belief.mean_of(x) == 1.0
    assert belief.variance_of(x) == 2.5


def test_prediction_applies_the_transition_and_the_offset():
    position, velocity = Continuous("position"), Continuous("velocity")
    mean, covariance = np.array([0.0, 1.0]), np.diag([1.0, 0.25])
    belief = belief_over([position, velocity], mean, covariance)
    time_step = 0.1

    belief.predict(
        LinearPrediction(
            transitions={position: {position: 1.0, velocity: time_step}},
            offsets={velocity: -0.5},
        )
    )

    transition = np.array([[1.0, time_step], [0.0, 1.0]])
    expected_mean = transition @ mean + np.array([0.0, -0.5])
    expected_covariance = transition @ covariance @ transition.T
    assert belief.mean_of(position) == pytest.approx(expected_mean[0])
    assert belief.mean_of(velocity) == pytest.approx(expected_mean[1])
    assert belief.variance_of(position) == pytest.approx(expected_covariance[0, 0])
    assert belief.covariance_between(position, velocity) == pytest.approx(
        expected_covariance[0, 1]
    )


def test_prediction_rejects_negative_process_noise():
    x = Continuous("x")
    with pytest.raises(NegativeVarianceError):
        LinearPrediction(process_noise={x: -0.1})


def test_prediction_about_an_unknown_variable_is_rejected():
    x, unknown = Continuous("x"), Continuous("unknown")
    belief = belief_over([x], [0.0], [[1.0]])

    with pytest.raises(VariableNotInDistributionError):
        belief.predict(LinearPrediction(process_noise={unknown: 1.0}))


# %% update


def test_update_with_one_reading_follows_the_scalar_kalman_gain():
    x = Continuous("x")
    prior_mean, prior_variance, reading_value, reading_variance = 0.0, 4.0, 2.0, 1.0
    belief = belief_over([x], [prior_mean], [[prior_variance]])

    belief.update(
        [
            Reading(
                value=reading_value, contributions={x: 1.0}, variance=reading_variance
            )
        ]
    )

    gain = prior_variance / (prior_variance + reading_variance)
    assert belief.mean_of(x) == pytest.approx(
        prior_mean + gain * (reading_value - prior_mean)
    )
    assert belief.variance_of(x) == pytest.approx((1 - gain) * prior_variance)


def test_reading_of_a_sum_correlates_the_variables_it_sums():
    x, y = Continuous("x"), Continuous("y")
    belief = belief_over([x, y], [0.0, 0.0], np.eye(2))

    belief.update([Reading(value=2.0, contributions={x: 1.0, y: 1.0}, variance=1.0)])

    observation = np.array([[1.0, 1.0]])
    gain = observation.T / (observation @ observation.T + 1.0)
    expected_covariance = (np.eye(2) - gain @ observation) @ np.eye(2)
    assert belief.covariance_between(x, y) == pytest.approx(expected_covariance[0, 1])
    assert belief.mean_of(x) == pytest.approx((gain * 2.0)[0, 0])


def test_update_without_readings_leaves_the_belief_unchanged():
    x = Continuous("x")
    belief = belief_over([x], [3.0], [[2.0]])

    belief.update([])

    assert belief.mean_of(x) == 3.0
    assert belief.variance_of(x) == 2.0


def test_reading_about_an_unknown_variable_is_rejected():
    x, unknown = Continuous("x"), Continuous("unknown")
    belief = belief_over([x], [0.0], [[1.0]])

    with pytest.raises(VariableNotInDistributionError):
        belief.update([Reading(value=1.0, contributions={unknown: 1.0}, variance=1.0)])


def test_reading_rejects_negative_variance():
    x = Continuous("x")
    with pytest.raises(NegativeVarianceError):
        Reading(value=1.0, contributions={x: 1.0}, variance=-1.0)


# %% statistics


def test_statistics_are_the_mean_and_variance_of_each_variable():
    x, y = Continuous("x"), Continuous("y")
    belief = belief_over([x, y], [1.0, 2.0], [[3.0, 0.5], [0.5, 4.0]])

    assert belief.statistics() == {
        VariableStatistic(x, Statistic.MEAN): belief.mean_of(x),
        VariableStatistic(x, Statistic.VARIANCE): belief.variance_of(x),
        VariableStatistic(y, Statistic.MEAN): belief.mean_of(y),
        VariableStatistic(y, Statistic.VARIANCE): belief.variance_of(y),
    }
