from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from probabilistic_model.quantities import QuantityPair
from random_events.variable import Continuous
from typing_extensions import Dict, Mapping, Self

from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import PoseCovarianceNotSixBySixError
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)

# %% how a displacement of a pose is read in another frame


@dataclass
class PoseDisplacementMap:
    """
    How far each degree of freedom of a pose moves, seen from another frame, per unit it
    moves in the frame it is reported in.

    Turning the frame turns a displacement into the axes it turns into; moving the frame
    away additionally reads a turn as a displacement, by the distance moved. Both are
    stated per named pair rather than as a matrix a caller counts rows in.

    ..note:: This is what robotics calls the adjoint of the transform. The name is not
        used here because what the type does is readable without it.
    """

    factors: Mapping[QuantityPair, float]
    """
    How much each degree of freedom seen in the new frame, named first, follows each one
    in the frame the pose is reported in, named second.

    Omitted pairs do not follow each other at all.
    """

    def __post_init__(self):
        """
        :raises VariableNotInQuantitiesError: If a named variable is not a degree of
            freedom of a pose.
        """
        for pair in self.factors:
            for variable in pair:
                SpatialVariables.pose.index_of(variable)

    @classmethod
    def of_transform(
        cls, new_reference_T_reference: HomogeneousTransformationMatrix
    ) -> Self:
        """
        Read off the transform between the two frames.

        :param new_reference_T_reference: The transform from the frame a pose is
            reported in to the frame to read its displacement in.
        :return: The map that transform describes.
        :raises HasFreeVariablesError: If the transform is still symbolic, so that its
            numbers are not known.
        """
        matrix = new_reference_T_reference.to_np()
        rotation = matrix[:3, :3]
        turn_into_shift = cls._cross_product_matrix(matrix[:3, 3]) @ rotation
        factors = {}
        for row, shift in enumerate(SpatialVariables.position):
            for column, source_shift in enumerate(SpatialVariables.position):
                factors[(shift, source_shift)] = float(rotation[row, column])
            for column, source_turn in enumerate(SpatialVariables.rotation):
                factors[(shift, source_turn)] = float(turn_into_shift[row, column])
        for row, turn in enumerate(SpatialVariables.rotation):
            for column, source_turn in enumerate(SpatialVariables.rotation):
                factors[(turn, source_turn)] = float(rotation[row, column])
        return cls(factors=factors)

    @staticmethod
    def _cross_product_matrix(
        vector: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        :param vector: The left-hand side of the cross products.
        :return: The matrix that takes the cross product of ``vector`` with whatever it
            is applied to.
        """
        x, y, z = vector
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

    def factor_of(self, seen: Continuous, reported: Continuous) -> float:
        """
        :param seen: The degree of freedom in the frame being read in.
        :param reported: The degree of freedom in the frame the pose is reported in.
        :return: How much the first follows the second.
        :raises VariableNotInQuantitiesError: If either is not a degree of freedom of a
            pose.
        """
        SpatialVariables.pose.index_of(seen)
        SpatialVariables.pose.index_of(reported)
        return self.factors.get((seen, reported), 0.0)

    @property
    def as_array(self) -> npt.NDArray[np.float64]:
        """
        :return: The same map as a matrix over :attr:`SpatialVariables.pose`, for the
            arithmetic that carries a covariance through it.
        """
        return SpatialVariables.pose.matrix(self.factors)


# %% how uncertain a pose is


@dataclass
class PoseCovariance:
    """
    How uncertain a pose is, as the covariance between its six degrees of freedom.

    The numbers are kept against the pairs of degrees of freedom they describe rather
    than as a matrix a reader indexes by row, and :attr:`as_array` lays them out over
    :attr:`SpatialVariables.pose` where the arithmetic needs a matrix.

    .. note:: The covariance is expressed in the frame the pose itself is expressed in;
        this type does not carry that frame.
    """

    uncertainty: Dict[QuantityPair, float]
    """
    How much each ordered pair of degrees of freedom varies together, with one paired
    with itself being its own variance.

    Omitted pairs do not vary together at all.
    """

    def __post_init__(self):
        """
        :raises VariableNotInQuantitiesError: If a named variable is not a degree of
            freedom of a pose.
        """
        for pair in self.uncertainty:
            for variable in pair:
                SpatialVariables.pose.index_of(variable)

    @classmethod
    def of(cls, covariances: Mapping[QuantityPair, float]) -> Self:
        """
        Build one from named pairs.

        Two degrees of freedom vary together by one number rather than two, so each pair
        given fills its mirror as well and a caller states it once.

        :param covariances: How much each pair of degrees of freedom varies together; a
            pair of one with itself is its own variance, and omitted pairs are zero.
        :return: The covariance those entries describe.
        :raises VariableNotInQuantitiesError: If a named variable is not a degree of
            freedom of a pose.
        """
        return cls.from_array(SpatialVariables.pose.symmetric_matrix(covariances))

    @classmethod
    def from_array(cls, values: npt.NDArray[np.float64]) -> Self:
        """
        Read an uncertainty back off the matrix the arithmetic produced.

        Each direction of a pair is read on its own rather than averaged, so an
        uncertainty that has drifted out of symmetry stays visible.

        :param values: The covariance as a matrix laid out over
            :attr:`SpatialVariables.pose`.
        :return: The uncertainty that matrix describes.
        :raises PoseCovarianceNotSixBySixError: If the matrix does not relate all six
            degrees of freedom to each other.
        """
        pose = SpatialVariables.pose
        expected_shape = (len(pose), len(pose))
        if values.shape != expected_shape:
            raise PoseCovarianceNotSixBySixError(
                given_shape=tuple(values.shape), expected_shape=expected_shape
            )
        return cls(
            uncertainty={
                (one, other): float(values[row, column])
                for row, one in enumerate(pose)
                for column, other in enumerate(pose)
            }
        )

    @property
    def as_array(self) -> npt.NDArray[np.float64]:
        """
        :return: The uncertainty as a matrix over :attr:`SpatialVariables.pose`, for the
            arithmetic that needs one.
        """
        return SpatialVariables.pose.matrix(self.uncertainty)

    def covariance_between(self, one: Continuous, other: Continuous) -> float:
        """
        :param one: The first degree of freedom.
        :param other: The second one.
        :return: How much the two vary together.
        :raises VariableNotInQuantitiesError: If either is not a degree of freedom of a
            pose.
        """
        SpatialVariables.pose.index_of(one)
        SpatialVariables.pose.index_of(other)
        return self.uncertainty.get((one, other), 0.0)

    def variance_of(self, variable: Continuous) -> float:
        """
        :param variable: The degree of freedom to read.
        :return: How uncertain that one degree of freedom is.
        :raises VariableNotInQuantitiesError: If it is not a degree of freedom of a
            pose.
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
        carried = PoseDisplacementMap.of_transform(new_reference_T_reference).as_array
        return type(self).from_array(carried @ self.as_array @ carried.T)

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
