from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from semantic_digital_twin.exceptions import PoseCovarianceNotSixBySixError

# %% the degrees of freedom a pose covariance relates


class PoseAxis(IntEnum):
    """
    The six degrees of freedom of a pose, in the order :class:`PoseCovariance` indexes
    them.

    The members double as indices into :attr:`PoseCovariance.values`.
    """

    POSITION_X = 0
    """
    Translation along the x axis.
    """

    POSITION_Y = 1
    """
    Translation along the y axis.
    """

    POSITION_Z = 2
    """
    Translation along the z axis.
    """

    ROTATION_X = 3
    """
    Rotation about the x axis.
    """

    ROTATION_Y = 4
    """
    Rotation about the y axis.
    """

    ROTATION_Z = 5
    """
    Rotation about the z axis.
    """

    @classmethod
    def position_axes(cls) -> tuple[Self, ...]:
        """
        :return: The three translational axes.
        """
        return cls.POSITION_X, cls.POSITION_Y, cls.POSITION_Z

    @classmethod
    def rotation_axes(cls) -> tuple[Self, ...]:
        """
        :return: The three rotational axes.
        """
        return cls.ROTATION_X, cls.ROTATION_Y, cls.ROTATION_Z


# %% the covariance itself


@dataclass
class PoseCovariance:
    """
    How uncertain a pose is, as the covariance between its six degrees of freedom.

    .. note:: The covariance is expressed in the frame the pose itself is expressed in;
        this type does not carry that frame.
    """

    values: npt.NDArray[np.float64]
    """
    The covariance, as a matrix whose rows and columns are indexed by :class:`PoseAxis`.
    """

    def __post_init__(self):
        """
        :raises PoseCovarianceNotSixBySixError: If the matrix does not relate all six
            degrees of freedom to each other.
        """
        expected_shape = (len(PoseAxis), len(PoseAxis))
        if self.values.shape != expected_shape:
            raise PoseCovarianceNotSixBySixError(
                given_shape=tuple(self.values.shape), expected_shape=expected_shape
            )

    def variance_of(self, axis: PoseAxis) -> float:
        """
        :param axis: The degree of freedom to read.
        :return: How uncertain that one degree of freedom is.
        """
        return float(self.values[axis, axis])

    @property
    def position_variance(self) -> float:
        """
        :return: The uncertainty of the position alone, summed over its three axes.
        """
        return sum(self.variance_of(axis) for axis in PoseAxis.position_axes())

    @property
    def rotation_variance(self) -> float:
        """
        :return: The uncertainty of the orientation alone, summed over its three axes.
        """
        return sum(self.variance_of(axis) for axis in PoseAxis.rotation_axes())

    @property
    def total_variance(self) -> float:
        """
        :return: The uncertainty of the whole pose, summed over all six axes.
        """
        return self.position_variance + self.rotation_variance
