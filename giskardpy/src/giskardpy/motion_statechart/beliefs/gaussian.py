from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import numpy as np
import numpy.typing as npt
from random_events.variable import Continuous
from typing_extensions import Dict, List, Mapping, Optional, Self, Tuple

from giskardpy.motion_statechart.exceptions import (
    RepeatedVariableInBeliefError,
    VariableNotInBeliefError,
)

QuantityPair = Tuple[Continuous, Continuous]
"""
Two quantities whose errors, or whose influence on each other, an entry describes.
"""

# %% the quantities a belief is about


@dataclass(frozen=True)
class Quantities:
    """
    The continuous quantities a belief is about, and the layout of every array over
    them.

    Everything a belief computes with is built here, from quantities rather than from
    row and column counts, so an array cannot end up describing a different set of
    quantities than the belief it belongs to. Naming a quantity the belief is not about
    is then the only way left to get it wrong, and that is what :meth:`index_of`
    rejects.
    """

    variables: Tuple[Continuous, ...]
    """
    The quantities, in the order they index a belief's mean and covariance.
    """

    @classmethod
    def of(cls, *variables: Continuous) -> Self:
        """
        :param variables: The quantities a belief is to be about.
        :return: Them, in the order given.
        :raises RepeatedVariableInBeliefError: If one of them is named more than once.
        """
        for position, variable in enumerate(variables):
            if variable in variables[:position]:
                raise RepeatedVariableInBeliefError(variable=variable)
        return cls(variables=variables)

    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)

    def __contains__(self, variable: Continuous) -> bool:
        return variable in self.variables

    def index_of(self, variable: Continuous) -> int:
        """
        :param variable: The quantity to locate.
        :return: The row every array over these quantities holds it in.
        :raises VariableNotInBeliefError: If it is not one of them.
        """
        if variable not in self.variables:
            raise VariableNotInBeliefError(
                variable=variable, belief_variables=list(self.variables)
            )
        return self.variables.index(variable)

    def vector(self, values: Mapping[Continuous, float]) -> npt.NDArray[np.float64]:
        """
        Build one number per quantity.

        :param values: The number for each quantity that has one; the rest are zero.
        :return: Them, in this layout.
        """
        built = np.zeros(len(self))
        for variable, value in values.items():
            built[self.index_of(variable)] = value
        return built

    def matrix(self, entries: Mapping[QuantityPair, float]) -> npt.NDArray[np.float64]:
        """
        Build one number per ordered pair of quantities.

        :param entries: The number for each pair that has one; the rest are zero. The
            first quantity of a pair is the row, the second the column.
        :return: Them, in this layout.
        """
        built = np.zeros((len(self), len(self)))
        for (row, column), value in entries.items():
            built[self.index_of(row), self.index_of(column)] = value
        return built

    def symmetric_matrix(
        self, entries: Mapping[QuantityPair, float]
    ) -> npt.NDArray[np.float64]:
        """
        Build one number per unordered pair of quantities, for a covariance.

        Two quantities vary together by one number rather than two, so each pair given
        fills its mirror as well and a caller states it once.

        :param entries: The number for each pair that has one; the rest are zero.
        :return: Them, in this layout.
        """
        built = self.matrix(entries)
        for (row, column), value in entries.items():
            built[self.index_of(column), self.index_of(row)] = value
        return built

    @property
    def unchanged(self) -> Dict[QuantityPair, float]:
        """
        :return: The transition of quantities expected to stay as they are.
        """
        return {(variable, variable): 1.0 for variable in self.variables}


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
    An estimate of one or more continuous quantities, together with how uncertain it is.

    :meth:`predict` carries the estimate to the next control cycle and :meth:`update`
    corrects it with readings. Both change this belief rather than returning a new one,
    so everything holding it — the
    :class:`~giskardpy.motion_statechart.beliefs.context.BeliefContext` included — reads
    the same estimate.
    """

    quantities: Quantities
    """
    The quantities this belief is about.
    """

    estimates: InitVar[Mapping[Continuous, float]]
    """
    The estimate to start each quantity at; one left out starts at zero.
    """

    uncertainty: InitVar[Mapping[QuantityPair, float]]
    """
    How uncertain those estimates are: a quantity paired with itself is its own
    variance, and two different quantities are how far their errors move together.
    """

    mean: npt.NDArray[np.float64] = field(init=False)
    """
    The current estimate of each quantity, laid out by :attr:`quantities`.
    """

    covariance: npt.NDArray[np.float64] = field(init=False)
    """
    How uncertain that estimate is, in the same layout.
    """

    def __post_init__(
        self,
        estimates: Mapping[Continuous, float],
        uncertainty: Mapping[QuantityPair, float],
    ):
        self.mean = self.quantities.vector(estimates)
        self.covariance = self.quantities.symmetric_matrix(uncertainty)

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
            quantities=Quantities.of(variable),
            estimates={variable: mean},
            uncertainty={(variable, variable): variance},
        )

    def mean_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: Its current estimate.
        """
        return float(self.mean[self.quantities.index_of(variable)])

    def variance_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: How uncertain its estimate is.
        """
        index = self.quantities.index_of(variable)
        return float(self.covariance[index, index])

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
        :raises VariableNotInBeliefError: If any of them names a quantity this belief is
            not about.
        """
        if offset is None:
            offset = {}
        transition_matrix = self.quantities.matrix(transition)

        self.mean = transition_matrix @ self.mean + self.quantities.vector(offset)
        self.covariance = (
            transition_matrix @ self.covariance @ transition_matrix.T
            + self.quantities.symmetric_matrix(process_noise)
        )

    def update(self, readings: List[Reading]) -> None:
        """
        Correct the estimate with what a sensor reported.

        The readings and the estimate are weighed against each other by how uncertain
        each is, so a sensor that is unsure of itself barely moves the estimate.

        The covariance is rebuilt as a sum of two symmetric, positive semi-definite terms
        rather than by the shorter ``(identity - gain @ model) @ covariance``. The two
        agree in exact arithmetic, but only this form stays a covariance under rounding,
        and one that stops being a covariance at a control cycle's rate does so long
        before anything looks wrong.

        :param readings: What the sensor reported, and what each number says about this
            belief. Reporting nothing leaves the estimate alone.
        :raises VariableNotInBeliefError: If a reading names a quantity this belief is not
            about.
        """
        if not readings:
            return

        model = np.array(
            [self.quantities.vector(reading.contributions) for reading in readings]
        )
        reported = np.array([reading.value for reading in readings])
        noise = np.diag([reading.variance for reading in readings])

        predicted_reading = model @ self.mean
        prediction_error_covariance = model @ self.covariance @ model.T + noise
        gain = self.covariance @ model.T @ np.linalg.inv(prediction_error_covariance)
        uncorrected = np.eye(len(self.quantities)) - gain @ model

        self.mean = self.mean + gain @ (reported - predicted_reading)
        self.covariance = (
            uncorrected @ self.covariance @ uncorrected.T + gain @ noise @ gain.T
        )
