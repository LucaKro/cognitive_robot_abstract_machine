from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
)
from random_events.variable import Continuous
from typing_extensions import Dict, Sequence, Tuple

from giskardpy.motion_statechart.beliefs.belief import (
    Belief,
    Statistic,
    VariableStatistic,
)
from giskardpy.motion_statechart.exceptions import NegativeVarianceError

# %% what a Gaussian belief is told


@dataclass
class LinearPrediction:
    """
    What happens to the variables of a Gaussian belief over one control cycle: each
    becomes a weighted sum of their current values, shifted by an offset, and grows less
    certain by the process noise.
    """

    transitions: Dict[Continuous, Dict[Continuous, float]] = field(
        default_factory=dict, kw_only=True
    )
    """
    For each variable that changes, how much each current value contributes to its next
    value.

    A variable without an entry keeps its value.
    """

    offsets: Dict[Continuous, float] = field(default_factory=dict, kw_only=True)
    """
    What is added to each variable on top of its transition; zero for a variable without
    an entry.
    """

    process_noise: Dict[Continuous, float] = field(default_factory=dict, kw_only=True)
    """
    The variance each variable gains over the cycle; zero for a variable without an
    entry.
    """

    def __post_init__(self):
        """
        :raises NegativeVarianceError: If any process noise is negative.
        """
        for variance in self.process_noise.values():
            if variance < 0:
                raise NegativeVarianceError(variance=variance)


@dataclass
class Reading:
    """
    A number a sensor reported: a weighted sum of the variables plus Gaussian noise,
    independent of any other reading.
    """

    value: float
    """
    The number reported.
    """

    contributions: Dict[Continuous, float]
    """
    How much each variable contributes to the number; zero for a variable without an
    entry.
    """

    variance: float
    """
    How far the sensor scatters around the true weighted sum.
    """

    def __post_init__(self):
        """
        :raises NegativeVarianceError: If the variance is negative.
        """
        if self.variance < 0:
            raise NegativeVarianceError(variance=self.variance)


# %% the belief


@dataclass
class GaussianBelief(Belief[LinearPrediction, Reading]):
    """
    A Gaussian belief about continuous variables, with full covariance, filtered by a
    Kalman filter.
    """

    distribution: MultivariateGaussianDistribution
    """
    What is currently believed.
    """

    @property
    def variables(self) -> Tuple[Continuous, ...]:
        return self.distribution.variables

    def mean_of(self, variable: Continuous) -> float:
        """
        :param variable: One of the variables of this belief.
        :return: What the variable is estimated at.
        """
        return float(self.distribution.mean[self.distribution.index_of(variable)])

    def variance_of(self, variable: Continuous) -> float:
        """
        :param variable: One of the variables of this belief.
        :return: How uncertain the estimate of the variable is.
        """
        return self.distribution.covariance_between(variable, variable)

    def covariance_between(self, first: Continuous, second: Continuous) -> float:
        """
        :param first: One of the variables of this belief.
        :param second: Another one of them.
        :return: How the estimates of the two co-vary.
        """
        return self.distribution.covariance_between(first, second)

    def predict(self, prediction: LinearPrediction):
        transition = self._transition_matrix(prediction.transitions)
        self.distribution = MultivariateGaussianDistribution.from_mean_and_covariance(
            distribution_variables=self.variables,
            mean=transition @ self.distribution.mean
            + self._vector_over(prediction.offsets),
            covariance=transition @ self.distribution.covariance @ transition.T
            + np.diag(self._vector_over(prediction.process_noise)),
        )

    def update(self, evidence: Sequence[Reading]):
        if not evidence:
            return
        self.distribution = self.distribution.product_with_gaussian_likelihood(
            observation_matrix=np.array(
                [self._vector_over(reading.contributions) for reading in evidence]
            ),
            observed=np.array([reading.value for reading in evidence]),
            observation_covariance=np.diag([reading.variance for reading in evidence]),
        )

    def statistics(self) -> Dict[VariableStatistic, float]:
        statistics = {}
        for variable in self.variables:
            statistics[VariableStatistic(variable, Statistic.MEAN)] = self.mean_of(
                variable
            )
            statistics[VariableStatistic(variable, Statistic.VARIANCE)] = (
                self.variance_of(variable)
            )
        return statistics

    def _vector_over(self, values: Dict[Continuous, float]) -> npt.NDArray:
        """
        :param values: A number for some of the variables of this belief.
        :return: The numbers laid out by the variables, zero where none was given.
        :raises VariableNotInDistributionError: If a number is given for a variable
            this belief is not about.
        """
        vector = np.zeros(len(self.variables))
        for variable, value in values.items():
            vector[self.distribution.index_of(variable)] = value
        return vector

    def _transition_matrix(
        self, transitions: Dict[Continuous, Dict[Continuous, float]]
    ) -> npt.NDArray:
        """
        :param transitions: For each variable that changes, how much each current value
            contributes to its next value.
        :return: The matrix taking the current values to the next ones, laid out by the
            variables; a variable without a transition keeps its value.
        """
        matrix = np.eye(len(self.variables))
        for variable, contributions in transitions.items():
            matrix[self.distribution.index_of(variable)] = self._vector_over(
                contributions
            )
        return matrix
