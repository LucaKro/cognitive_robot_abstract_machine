from __future__ import annotations

from dataclasses import dataclass

from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    Mean,
    MultivariateGaussianDistribution,
    Reading,
)
from probabilistic_model.quantities import Quantities, QuantityPair
from random_events.variable import Continuous
from typing_extensions import List, Mapping, Optional, Self

# %% a belief about continuous quantities


@dataclass
class GaussianBelief:
    """
    An estimate of one or more continuous quantities, together with how uncertain it is,
    kept up to date from one control cycle to the next.

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
        quantities: Quantities,
        estimates: Mapping[Continuous, float],
        uncertainty: Mapping[QuantityPair, float],
    ) -> Self:
        """
        Build a belief about several quantities at once.

        :param quantities: The quantities being estimated.
        :param estimates: The estimate to start each of them at; one left out starts at
            zero.
        :param uncertainty: How uncertain those estimates are: a quantity paired with
            itself is its own variance, and two different quantities are how far their
            errors move together.
        :return: The belief about them.
        :raises VariableNotInQuantitiesError: If either names a quantity that is not one
            of them.
        """
        return cls(
            distribution=MultivariateGaussianDistribution.of(
                quantities=quantities, estimates=estimates, uncertainty=uncertainty
            )
        )

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
        return cls(
            distribution=MultivariateGaussianDistribution.of_one_variable(
                variable=variable, mean=mean, variance=variance
            )
        )

    @property
    def quantities(self) -> Quantities:
        """
        :return: The quantities this belief is about.
        """
        return self.distribution.quantities

    @property
    def mean(self) -> Mean:
        """
        :return: What each quantity is currently estimated at.
        """
        return self.distribution.mean

    @property
    def covariance(self) -> Covariance:
        """
        :return: How uncertain those estimates are.
        """
        return self.distribution.covariance

    def mean_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: Its current estimate.
        :raises VariableNotInQuantitiesError: If this belief is not about it.
        """
        return self.distribution.mean_of(variable)

    def variance_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: How uncertain its estimate is.
        :raises VariableNotInQuantitiesError: If this belief is not about it.
        """
        return self.distribution.variance_of(variable)

    def predict(
        self,
        transition: Mapping[QuantityPair, float],
        process_noise: Mapping[QuantityPair, float],
        offset: Optional[Mapping[Continuous, float]] = None,
    ) -> None:
        """
        Carry the estimate to the next control cycle.

        Each quantity becomes what the transition makes of the others, plus the offset,
        and grows less certain by the process noise, which is what keeps a belief nobody
        is observing from staying confident forever.

        :param transition: How much each quantity's estimate carries into each
            quantity's next one. :attr:`Quantities.unchanged` is the one for quantities
            expected to stay put.
        :param process_noise: How much uncertainty the step itself adds.
        :param offset: What each quantity gains regardless of the estimate, such as the
            pull toward a prior. Defaults to nothing.
        :raises VariableNotInQuantitiesError: If any of them names a quantity this
            belief is not about.
        """
        self.distribution.apply_linear_map(transition)
        self.distribution.apply_added_uncertainty(process_noise)
        self.distribution.apply_translation(dict(offset or {}))

    def update(self, readings: List[Reading]) -> None:
        """
        Correct the estimate with what a sensor reported.

        The readings and the estimate are weighed against each other by how uncertain
        each is, so a sensor that is unsure of itself barely moves the estimate.

        :param readings: What the sensor reported, and what each number says about this
            belief. Reporting nothing leaves the estimate alone.
        :raises VariableNotInQuantitiesError: If a reading names a quantity this belief
            is not about.
        """
        self.distribution = self.distribution.conditional_on_readings(readings)
