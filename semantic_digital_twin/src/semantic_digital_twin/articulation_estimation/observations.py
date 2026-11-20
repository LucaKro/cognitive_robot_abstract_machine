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

    parent_T_child: npt.NDArray[np.float64]
    """
    One homogeneous transformation matrix per observation, stacked to shape (n, 4, 4).
    """

    def __post_init__(self):
        """
        :raises TooFewObservationsError: if fewer than
            :attr:`required_observation_count` poses are given.
        """
        if len(self.parent_T_child) < self.required_observation_count:
            raise TooFewObservationsError(
                observation_count=len(self.parent_T_child),
                required_observation_count=self.required_observation_count,
            )

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
            parent_T_child=np.stack(
                [transformation.to_np() for transformation in transformations]
            )
        )

    @property
    def displacements_between_observations(self) -> npt.NDArray[np.float64]:
        """
        The step to each observation from the one before, stacked to shape (n - 1, 4, 4).

        Each is the transformation in the parent frame between the two poses.
        """
        return self.parent_T_child[1:] @ np.linalg.inv(self.parent_T_child[:-1])


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

    def standardized_pose_errors(
        self,
        *,
        observed: npt.NDArray[np.float64],
        predicted: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        :param observed: The observed poses, stacked to shape (n, 4, 4).
        :param predicted: The poses a model predicts for the same observations.
        :return: The position and orientation errors of each observation, each divided
            by its standard deviation, flattened to 6 n numbers.
        """
        position_errors = observed[:, :3, 3] - predicted[:, :3, 3]
        orientation_errors = (
            Rotation.from_matrix(predicted[:, :3, :3]).inv()
            * Rotation.from_matrix(observed[:, :3, :3])
        ).as_rotvec()
        return np.concatenate(
            [
                position_errors / self.position_standard_deviation,
                orientation_errors / self.orientation_standard_deviation,
            ],
            axis=1,
        ).ravel()
