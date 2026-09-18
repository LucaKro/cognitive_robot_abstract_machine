from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from random_events.variable import Continuous
from typing_extensions import Mapping, Self, Tuple

from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import (
    PoseCovarianceNotSixBySixError,
    VariableNotInPoseError,
)

PoseVariablePair = Tuple[Continuous, Continuous]
"""
Two degrees of freedom of a pose whose joint uncertainty an entry describes.
"""

# %% how uncertain a pose is


@dataclass
class PoseCovariance:
    """
    How uncertain a pose is, as the covariance between its six degrees of freedom.

    .. note:: The covariance is expressed in the frame the pose itself is expressed in;
        this type does not carry that frame.
    """

    values: npt.NDArray[np.float64]
    """
    The covariance, as a matrix whose rows and columns are the degrees of freedom in
    :attr:`SpatialVariables.pose`, in that order.
    """

    def __post_init__(self):
        """
        :raises PoseCovarianceNotSixBySixError: If the matrix does not relate all six
            degrees of freedom to each other.
        """
        side = len(SpatialVariables.pose)
        expected_shape = (side, side)
        if self.values.shape != expected_shape:
            raise PoseCovarianceNotSixBySixError(
                given_shape=tuple(self.values.shape), expected_shape=expected_shape
            )

    @classmethod
    def of(cls, covariances: Mapping[PoseVariablePair, float]) -> Self:
        """
        Build one from named pairs rather than from a matrix a caller lays out.

        Two degrees of freedom vary together by one number rather than two, so each pair
        given fills its mirror as well and a caller states it once.

        :param covariances: How much each pair of degrees of freedom varies together; a
            pair of one with itself is its own variance, and omitted pairs are zero.
        :return: The covariance those entries describe.
        :raises VariableNotInPoseError: If a named variable is not a degree of freedom
            of a pose.
        """
        side = len(SpatialVariables.pose)
        values = np.zeros((side, side), dtype=np.float64)
        for (one, other), covariance in covariances.items():
            row, column = cls._row_of(one), cls._row_of(other)
            values[row, column] = covariance
            values[column, row] = covariance
        return cls(values=values)

    @staticmethod
    def _row_of(variable: Continuous) -> int:
        """
        :param variable: The degree of freedom to locate.
        :return: The row and column it occupies.
        :raises VariableNotInPoseError: If it is not a degree of freedom of a pose.
        """
        pose = SpatialVariables.pose
        if variable not in pose:
            raise VariableNotInPoseError(
                variable_name=variable.name,
                pose_variable_names=tuple(each.name for each in pose),
            )
        return pose.index(variable)

    def covariance_between(self, one: Continuous, other: Continuous) -> float:
        """
        :param one: The first degree of freedom.
        :param other: The second one.
        :return: How much the two vary together.
        :raises VariableNotInPoseError: If either is not a degree of freedom of a pose.
        """
        return float(self.values[self._row_of(one), self._row_of(other)])

    def variance_of(self, variable: Continuous) -> float:
        """
        :param variable: The degree of freedom to read.
        :return: How uncertain that one degree of freedom is.
        :raises VariableNotInPoseError: If it is not a degree of freedom of a pose.
        """
        return self.covariance_between(variable, variable)

    @property
    def position_variance(self) -> float:
        """
        :return: The uncertainty of the position alone, summed over its three axes.
        """
        return sum(self.variance_of(variable) for variable in SpatialVariables.position)

    @property
    def rotation_variance(self) -> float:
        """
        :return: The uncertainty of the orientation alone, summed over its three axes.
        """
        return sum(self.variance_of(variable) for variable in SpatialVariables.rotation)

    @property
    def total_variance(self) -> float:
        """
        :return: The uncertainty of the whole pose, summed over all six axes.
        """
        return self.position_variance + self.rotation_variance
