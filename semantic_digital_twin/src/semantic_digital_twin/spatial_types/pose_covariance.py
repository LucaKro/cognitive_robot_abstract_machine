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
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
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

    def transformed_by(
        self, new_reference_T_reference: HomogeneousTransformationMatrix
    ) -> Self:
        """
        Re-express this uncertainty in another frame.

        A pose covariance is about how far the true pose is from the reported one, as a
        displacement in the frame the pose is expressed in. Seen from another frame that
        displacement is a different one, and it is the *whole* transform that decides
        which: turning the frame turns the displacement, and moving the frame away turns
        being uncertain about which way the pose faces into being uncertain about where
        it is, by the distance moved.

        ..note:: The rotational degrees of freedom are read as a small turn about each
            axis, which is what makes this exact rather than an approximation about a
            particular orientation. The model is first order in how uncertain the pose
            is, which is what a covariance describes.

        :param new_reference_T_reference: The transform from the frame this uncertainty
            is expressed in to the frame to express it in, whose own numbers are taken
            to be certain.
        :return: The same uncertainty, read in the new frame.
        :raises HasFreeVariablesError: If the transform is still symbolic, so that its
            numbers are not known.
        """
        adjoint = self._adjoint_of(new_reference_T_reference)
        return type(self)(values=adjoint @ self.values @ adjoint.T)

    @staticmethod
    def _adjoint_of(
        transform: HomogeneousTransformationMatrix,
    ) -> npt.NDArray[np.float64]:
        """
        :param transform: The transform to carry a displacement through.
        :return: The matrix that carries a displacement of a pose through it, over the
            degrees of freedom in :attr:`SpatialVariables.pose`.
        :raises HasFreeVariablesError: If the transform is still symbolic.
        """
        matrix = transform.to_np()
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        side = len(SpatialVariables.pose)
        first_rotation_row = len(SpatialVariables.position)
        adjoint = np.zeros((side, side), dtype=np.float64)
        adjoint[:first_rotation_row, :first_rotation_row] = rotation
        adjoint[:first_rotation_row, first_rotation_row:] = (
            _cross_product_matrix(translation) @ rotation
        )
        adjoint[first_rotation_row:, first_rotation_row:] = rotation
        return adjoint

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


# %% the cross product as a matrix


def _cross_product_matrix(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    :param vector: The left-hand side of the cross products.
    :return: The matrix that takes the cross product of ``vector`` with whatever it is
        applied to.
    """
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
