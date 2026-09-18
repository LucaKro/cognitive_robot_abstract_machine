from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
)
from random_events.variable import Continuous
from typing_extensions import List, Mapping, Optional, Self, Tuple

VariablePair = Tuple[Continuous, Continuous]
"""
Two quantities whose relationship an entry describes.
"""

# %% what a sensor reported


@dataclass
class Reading:
    """
    One number a sensor reported, and what it says about a belief's quantities.
    """

    value: float
    """
    The number the sensor reported.
    """

    contributions: Mapping[Continuous, float]
    """
    How much each quantity adds to that number if the estimate is exactly right, so a
    sensor reading one quantity itself contributes one of it and nothing else.
    """

    variance: float
    """
    How far this sensor's readings scatter around the truth, which is what decides how
    far the reading is allowed to move the estimate.
    """

    @classmethod
    def of_one_variable(
        cls, variable: Continuous, value: float, variance: float
    ) -> Self:
        """
        Build the reading of a sensor that reports one quantity itself.

        :param variable: The quantity that was read.
        :param value: What the sensor reported for it.
        :param variance: How far that sensor's readings scatter.
        :return: The reading.
        """
        return cls(value=value, contributions={variable: 1.0}, variance=variance)


# %% a belief about continuous quantities


@dataclass
class GaussianBelief:
    """
    An estimate of one or more continuous quantities, together with how uncertain it is,
    kept up to date from one control cycle to the next.

    A caller states what it knows by quantity; the distribution behind this belief is
    laid out as arrays over those quantities, and this is where the two meet.

    :meth:`predict` carries the estimate to the next cycle and :meth:`update` corrects it
    with readings. Both change this belief rather than returning a new one, so everything
    holding it — the
    :class:`~giskardpy.motion_statechart.beliefs.context.BeliefContext` included — reads
    the same estimate.
    """

    distribution: MultivariateGaussianDistribution
    """
    What is currently believed about the quantities.
    """

    @classmethod
    def of(
        cls,
        variables: Tuple[Continuous, ...],
        estimates: Mapping[Continuous, float],
        uncertainty: Mapping[VariablePair, float],
    ) -> Self:
        """
        Build a belief about several quantities at once.

        :param variables: The quantities being estimated, in the order every array over
            them is laid out in. That order is the caller's own and is kept as given,
            since a domain's own ordering is rarely alphabetical.
        :param estimates: The estimate to start each of them at; one left out starts at
            zero.
        :param uncertainty: How uncertain those estimates are: a quantity paired with
            itself is its own variance, and two different quantities are how far their
            errors move together. A pair stated once fills its mirror as well.
        :return: The belief about them.
        :raises VariableNotInDistributionError: If either names a quantity that is not
            one of them.
        """
        variables = tuple(variables)
        belief = cls(
            distribution=MultivariateGaussianDistribution(
                distribution_variables=variables,
                mean=np.zeros(len(variables)),
                covariance=np.zeros((len(variables), len(variables))),
            )
        )
        belief.distribution.mean = belief.vector(estimates)
        belief.distribution.covariance = belief.symmetric_matrix(uncertainty)
        return belief

    @classmethod
    def of_one_variable(
        cls, variable: Continuous, mean: float, variance: float
    ) -> Self:
        """
        Build a belief about a single quantity.

        :param variable: The quantity being estimated.
        :param mean: The estimate to start from.
        :param variance: How uncertain that starting estimate is.
        :return: The belief about it.
        """
        return cls.of(
            variables=(variable,),
            estimates={variable: mean},
            uncertainty={(variable, variable): variance},
        )

    @property
    def variables(self) -> Tuple[Continuous, ...]:
        """
        :return: The quantities this belief is about, in their layout order.
        """
        return self.distribution.variables

    def mean_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: Its current estimate.
        :raises VariableNotInDistributionError: If this belief is not about it.
        """
        return self.distribution.mean_of(variable)

    def variance_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: How uncertain its estimate is.
        :raises VariableNotInDistributionError: If this belief is not about it.
        """
        return self.distribution.variance_of(variable)

    def covariance_between(self, first: Continuous, second: Continuous) -> float:
        """
        :param first: One of the quantities.
        :param second: The other one.
        :return: How far their errors move together.
        :raises VariableNotInDistributionError: If this belief is not about either of
            them.
        """
        return self.distribution.covariance_between(first, second)

    # %% stating a matrix over the quantities by name

    def matrix(self, entries: Mapping[VariablePair, float]) -> npt.NDArray[np.float64]:
        """
        Build one number per ordered pair of quantities.

        :param entries: The number for each pair that has one; the rest are zero. The
            first quantity of a pair is the row, the second the column.
        :return: Them, laid out by this belief's quantities.
        :raises VariableNotInDistributionError: If an entry names a quantity this belief
            is not about.
        """
        built = np.zeros((len(self.variables), len(self.variables)))
        for (row, column), value in entries.items():
            built[
                self.distribution.index_of(row), self.distribution.index_of(column)
            ] = value
        return built

    def symmetric_matrix(
        self, entries: Mapping[VariablePair, float]
    ) -> npt.NDArray[np.float64]:
        """
        Build one number per unordered pair of quantities, for a covariance.

        Two quantities vary together by one number rather than two, so each pair given
        fills its mirror as well and a caller states it once.

        :param entries: The number for each pair that has one; the rest are zero.
        :return: Them, laid out by this belief's quantities.
        :raises VariableNotInDistributionError: If an entry names a quantity this belief
            is not about.
        """
        built = self.matrix(entries)
        for (row, column), value in entries.items():
            built[
                self.distribution.index_of(column), self.distribution.index_of(row)
            ] = value
        return built

    @property
    def unchanged(self) -> Mapping[VariablePair, float]:
        """
        :return: The transition of quantities expected to stay as they are.
        """
        return {(variable, variable): 1.0 for variable in self.variables}

    # %% carrying it forward and correcting it

    def predict(
        self,
        transition: Mapping[VariablePair, float],
        process_noise: Mapping[VariablePair, float],
        offset: Optional[Mapping[Continuous, float]] = None,
    ) -> None:
        """
        Carry the estimate to the next control cycle.

        Each quantity becomes what the transition makes of the others, plus the offset,
        and grows less certain by the process noise, which is what keeps a belief nobody
        is observing from staying confident forever.

        :param transition: How much each quantity's estimate carries into each
            quantity's next one. :attr:`unchanged` is the one for quantities expected to
            stay put.
        :param process_noise: How much uncertainty the step itself adds.
        :param offset: What each quantity gains regardless of the estimate, such as the
            pull toward a prior. Defaults to nothing.
        :raises VariableNotInDistributionError: If any of them names a quantity this
            belief is not about.
        """
        self.distribution.apply_linear_map(self.matrix(transition))
        self.distribution.apply_added_covariance(self.symmetric_matrix(process_noise))
        self.distribution.apply_translation(dict(offset or {}))

    def update(self, readings: List[Reading]) -> None:
        """
        Correct the estimate with what a sensor reported.

        The readings and the estimate are weighed against each other by how uncertain
        each is, so a sensor that is unsure of itself barely moves the estimate.

        :param readings: What the sensor reported, and what each number says about this
            belief. Reporting nothing leaves the estimate alone.
        :raises VariableNotInDistributionError: If a reading names a quantity this
            belief is not about.
        """
        if not readings:
            return
        self.distribution = self.distribution.conditional_on_measurement(
            model=np.array(
                [self.vector(reading.contributions) for reading in readings]
            ),
            measured=np.array([reading.value for reading in readings]),
            noise=np.diag([reading.variance for reading in readings]),
        )

    def vector(self, values: Mapping[Continuous, float]) -> npt.NDArray[np.float64]:
        """
        Build one number per quantity.

        :param values: The number for each quantity that has one; the rest are zero.
        :return: Them, laid out by this belief's quantities.
        :raises VariableNotInDistributionError: If an entry names a quantity this belief
            is not about.
        """
        built = np.zeros(len(self.variables))
        for variable, value in values.items():
            built[self.distribution.index_of(variable)] = value
        return built
