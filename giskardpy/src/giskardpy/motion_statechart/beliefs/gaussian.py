from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from random_events.variable import Continuous
from typing_extensions import List, Optional, Self, Tuple

from giskardpy.motion_statechart.exceptions import (
    RepeatedVariableInBeliefError,
    VariableNotInBeliefError,
    WrongBeliefShapeError,
)

# %% the arrays a belief is built from


class BeliefArray(StrEnum):
    """
    The arrays a Gaussian belief is built from.
    """

    MEAN = "mean"
    """
    The estimate of every quantity a belief is about.
    """

    COVARIANCE = "covariance"
    """
    How uncertain those estimates are, and how their errors are related.
    """

    TRANSITION = "transition"
    """
    The part of a cycle's expected change that scales the estimate.
    """

    PROCESS_NOISE = "process noise"
    """
    The uncertainty that expected change adds.
    """

    OFFSET = "offset"
    """
    The part of a cycle's expected change that is the same whatever the estimate is.
    """

    MEASUREMENT_VALUE = "measurement value"
    """
    What a sensor reported.
    """

    MEASUREMENT_MODEL = "measurement model"
    """
    What that sensor would report if the estimate were exactly right.
    """

    MEASUREMENT_NOISE = "measurement noise"
    """
    How far a sensor's readings scatter around the truth.
    """

    def require_shape(
        self, array: npt.NDArray[np.float64], expected_shape: Tuple[int, ...]
    ) -> None:
        """
        Reject an array that cannot describe the quantities it is meant to.

        :param array: The array given for this part of a belief.
        :param expected_shape: The shape the belief's quantities require of it.
        :raises WrongBeliefShapeError: If `array` has any other shape.
        """
        if array.shape == expected_shape:
            return
        raise WrongBeliefShapeError(
            array=self, expected_shape=expected_shape, actual_shape=array.shape
        )


# %% what a sensor reported


@dataclass
class Measurement:
    """
    One reading of a sensor, together with what that reading says about a belief.
    """

    value: npt.NDArray[np.float64]
    """
    The numbers the sensor reported.
    """

    model: npt.NDArray[np.float64]
    """
    Maps a belief's estimate onto the reading the sensor would give if that estimate
    were exactly right, one row per reported number.
    """

    noise: npt.NDArray[np.float64]
    """
    The covariance of the sensor's error, which is what decides how far the reading is
    allowed to move the estimate.
    """

    def __post_init__(self):
        BeliefArray.MEASUREMENT_VALUE.require_shape(self.value, (self.readings,))
        BeliefArray.MEASUREMENT_NOISE.require_shape(
            self.noise, (self.readings, self.readings)
        )

    @property
    def readings(self) -> int:
        """
        :return: How many numbers the sensor reported.
        """
        return self.value.size


# %% a belief about continuous quantities


@dataclass
class GaussianBelief:
    """
    An estimate of one or more continuous quantities, together with how uncertain it is.

    :meth:`predict` carries the estimate to the next control cycle and :meth:`update`
    corrects it with a reading. Both change this belief rather than returning a new one,
    so everything holding it — the
    :class:`~giskardpy.motion_statechart.beliefs.context.BeliefContext` included — reads
    the same estimate.
    """

    variables: List[Continuous]
    """
    The quantities this belief is about, one per row of :attr:`mean`.
    """

    mean: npt.NDArray[np.float64]
    """
    The current estimate of each of them.
    """

    covariance: npt.NDArray[np.float64]
    """
    How uncertain that estimate is, and how the quantities' errors are related.
    """

    def __post_init__(self):
        for position, variable in enumerate(self.variables):
            if variable in self.variables[:position]:
                raise RepeatedVariableInBeliefError(variable=variable)
        BeliefArray.MEAN.require_shape(self.mean, (self.dimensions,))
        BeliefArray.COVARIANCE.require_shape(
            self.covariance, (self.dimensions, self.dimensions)
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
            variables=[variable],
            mean=np.array([mean], dtype=float),
            covariance=np.array([[variance]], dtype=float),
        )

    @property
    def dimensions(self) -> int:
        """
        :return: How many quantities this belief is about.
        """
        return len(self.variables)

    def index_of(self, variable: Continuous) -> int:
        """
        :param variable: The quantity to locate.
        :return: The row of :attr:`mean` holding it.
        :raises VariableNotInBeliefError: If this belief is not about `variable`.
        """
        if variable not in self.variables:
            raise VariableNotInBeliefError(
                variable=variable, belief_variables=self.variables
            )
        return self.variables.index(variable)

    def mean_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: Its current estimate.
        """
        return float(self.mean[self.index_of(variable)])

    def variance_of(self, variable: Continuous) -> float:
        """
        :param variable: The quantity to read.
        :return: How uncertain its estimate is.
        """
        index = self.index_of(variable)
        return float(self.covariance[index, index])

    def measurement_of(
        self, variable: Continuous, value: float, variance: float
    ) -> Measurement:
        """
        Build the reading of a sensor that reports one of this belief's quantities
        directly.

        :param variable: The quantity that was read.
        :param value: What the sensor reported for it.
        :param variance: How far that sensor's readings scatter.
        :return: The reading, ready to be passed to :meth:`update`.
        """
        model = np.zeros((1, self.dimensions))
        model[0, self.index_of(variable)] = 1.0
        return Measurement(
            value=np.array([value], dtype=float),
            model=model,
            noise=np.array([[variance]], dtype=float),
        )

    def predict(
        self,
        transition: npt.NDArray[np.float64],
        process_noise: npt.NDArray[np.float64],
        offset: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """
        Carry the estimate to the next control cycle.

        The estimate becomes ``transition @ mean + offset`` and grows less certain by
        `process_noise`, which is what keeps a belief nobody is observing from staying
        confident forever.

        :param transition: The part of the expected change that scales the estimate.
        :param process_noise: The covariance of what that change cannot account for.
        :param offset: The part of the expected change that does not depend on the
            estimate, such as the pull toward a prior. Defaults to no offset.
        :raises WrongBeliefShapeError: If any of them does not fit this belief.
        """
        BeliefArray.TRANSITION.require_shape(
            transition, (self.dimensions, self.dimensions)
        )
        BeliefArray.PROCESS_NOISE.require_shape(
            process_noise, (self.dimensions, self.dimensions)
        )
        if offset is None:
            offset = np.zeros(self.dimensions)
        BeliefArray.OFFSET.require_shape(offset, (self.dimensions,))

        self.mean = transition @ self.mean + offset
        self.covariance = transition @ self.covariance @ transition.T + process_noise

    def update(self, measurement: Measurement) -> None:
        """
        Correct the estimate with a sensor's reading.

        The reading and the estimate are weighed against each other by how uncertain each
        is, so a sensor that is unsure of itself barely moves the estimate.

        The covariance is rebuilt as a sum of two symmetric, positive semi-definite terms
        rather than by the shorter ``(identity - gain @ model) @ covariance``. The two
        agree in exact arithmetic, but only this form stays a covariance under rounding,
        and one that stops being a covariance at a control cycle's rate does so long
        before anything looks wrong.

        :param measurement: What the sensor reported, and what it says about this belief.
        :raises WrongBeliefShapeError: If the reading does not describe this belief.
        """
        BeliefArray.MEASUREMENT_MODEL.require_shape(
            measurement.model, (measurement.readings, self.dimensions)
        )

        predicted_reading = measurement.model @ self.mean
        prediction_error = measurement.value - predicted_reading
        prediction_error_covariance = (
            measurement.model @ self.covariance @ measurement.model.T
            + measurement.noise
        )
        gain = (
            self.covariance
            @ measurement.model.T
            @ np.linalg.inv(prediction_error_covariance)
        )
        uncorrected = np.eye(self.dimensions) - gain @ measurement.model

        self.mean = self.mean + gain @ prediction_error
        self.covariance = (
            uncorrected @ self.covariance @ uncorrected.T
            + gain @ measurement.noise @ gain.T
        )
