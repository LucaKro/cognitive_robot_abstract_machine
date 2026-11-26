"""
Observed poses of a child relative to its parent, and the noise they carry.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from typing_extensions import Self

from krrood.patterns.classproperty import classproperty
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.articulation_estimation.exceptions import (
    NotHomogeneousTransformationError,
    TooFewObservationsError,
)


# %% Observed poses
@dataclass(kw_only=True)
class PoseObservations:
    """
    Poses of a child relative to its parent, in the order they were observed.

    Consecutive poses are less than half a revolution apart, so that the rotation
    between them is unambiguous.
    """

    parent_T_child_stack: npt.NDArray[np.float64]
    """
    One homogeneous transformation matrix per observation, stacked to shape (n, 4, 4).

    A copy of the given poses, with each last row set to exactly [0, 0, 0, 1], as
    :class:`scipy.spatial.transform.RigidTransform` requires.
    """

    def __post_init__(self) -> None:
        """
        :raises TooFewObservationsError: if fewer than
            :attr:`required_observation_count` poses are given.
        :raises NotHomogeneousTransformationError: if the last row of a pose is not
            [0, 0, 0, 1], to within :func:`numpy.isclose`'s default tolerance.
        """
        if len(self.parent_T_child_stack) < self.required_observation_count:
            raise TooFewObservationsError(
                observation_count=len(self.parent_T_child_stack),
                required_observation_count=self.required_observation_count,
            )
        homogeneous_row = np.array([0.0, 0.0, 0.0, 1.0])
        inexact_indices = np.flatnonzero(
            ~np.all(
                np.isclose(self.parent_T_child_stack[:, 3], homogeneous_row), axis=1
            )
        )
        if len(inexact_indices) > 0:
            raise NotHomogeneousTransformationError(
                observation_indices=inexact_indices.tolist()
            )
        self.parent_T_child_stack = np.array(self.parent_T_child_stack, dtype=float)
        self.parent_T_child_stack[:, 3] = homogeneous_row

    @classproperty
    def required_observation_count(cls) -> int:
        """
        The fewest poses that can show whether the child moves.
        """
        return 2

    @classmethod
    def from_transformations(
        cls, *, transformations: list[HomogeneousTransformationMatrix]
    ) -> Self:
        """
        :param transformations: The observed poses of the child relative to its parent.
        :return: The observations of those poses.
        """
        return cls(
            parent_T_child_stack=np.stack(
                [transformation.to_np() for transformation in transformations]
            )
        )

    @property
    def displacements_between_observations(self) -> npt.NDArray[np.float64]:
        """
        The step to each observation from the one before, stacked to shape (n - 1, 4, 4).

        Each is the transformation in the parent frame between the two poses.
        """
        return self.parent_T_child_stack[1:] @ np.linalg.inv(
            self.parent_T_child_stack[:-1]
        )


# %% Observation noise
@dataclass(kw_only=True)
class ObservationNoise:
    """
    How far an observed pose may lie from the true one, as the standard deviations of
    independent, normally distributed errors.
    """

    position_standard_deviation: float
    """
    The standard deviation of each coordinate of an observed position, in meters.
    """

    orientation_standard_deviation: float
    """
    The standard deviation of each component of the rotation vector between an observed
    orientation and the true one, in radians.
    """

    @property
    def exponential_coordinate_standard_deviations(self) -> npt.NDArray[np.float64]:
        """
        The standard deviations in the order of the exponential coordinates of a
        displacement: three times that of the orientation, for the rotation vector,
        then three times that of the position, for the translation.
        """
        return np.repeat(
            [self.orientation_standard_deviation, self.position_standard_deviation], 3
        )

    def standardized_pose_errors(
        self,
        *,
        observed_stack: npt.NDArray[np.float64],
        predicted_stack: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        :param observed_stack: The observed poses, stacked to shape (n, 4, 4).
        :param predicted_stack: The poses a model predicts for the same observations,
            stacked the same way.
        :return: The position and orientation errors of each observation, each divided
            by its standard deviation, flattened to :math:`6 n` numbers.
        """
        position_errors = observed_stack[:, :3, 3] - predicted_stack[:, :3, 3]
        orientation_errors = (
            Rotation.from_matrix(predicted_stack[:, :3, :3]).inv()
            * Rotation.from_matrix(observed_stack[:, :3, :3])
        ).as_rotvec()
        return np.concatenate(
            [
                position_errors / self.position_standard_deviation,
                orientation_errors / self.orientation_standard_deviation,
            ],
            axis=1,
        ).ravel()
