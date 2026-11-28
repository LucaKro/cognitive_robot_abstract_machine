"""
Derivatives of rotations and rigid transformations with respect to their exponential
coordinates, for stacks of them at once, as Barfoot derives them in *State Estimation
for Robotics*, Chapter 7.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing_extensions import Callable

from krrood.patterns.classproperty import classproperty


# %% Vectors
@dataclass(kw_only=True)
class Vector3Stack:
    """
    Vectors of three entries, stacked along the last axis to shape (..., 3).
    """

    entries: npt.NDArray[np.float64]
    """
    The vectors.
    """

    @property
    def skew_symmetric_matrices(self) -> npt.NDArray[np.float64]:
        r"""
        The matrix :math:`a^\wedge` of each vector :math:`a`, with
        :math:`a^\wedge x = a \times x`, stacked to shape (..., 3, 3).
        """
        x, y, z = np.moveaxis(self.entries, -1, 0)
        zero = np.zeros_like(x)
        return np.stack(
            [
                np.stack([zero, -z, y], axis=-1),
                np.stack([z, zero, -x], axis=-1),
                np.stack([-y, x, zero], axis=-1),
            ],
            axis=-2,
        )


# %% Rotation vectors
@dataclass(kw_only=True)
class RotationVectorStack(Vector3Stack):
    r"""
    Rotation vectors :math:`\phi`, whose length is the angle of the rotation.
    """

    @classproperty
    def series_angle(cls) -> float:
        """
        The angle in radians below which a coefficient is taken from its Taylor series
        up to the fourth power, where its closed form would lose digits to cancellation.
        """
        return 0.1

    @property
    def angles(self) -> npt.NDArray[np.float64]:
        """
        The angle of each rotation, the length of its vector.
        """
        return np.linalg.norm(self.entries, axis=-1)

    def coefficients(
        self,
        *,
        closed_form: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        series: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """
        :param closed_form: The coefficient as a function of the angle.
        :param series: Its Taylor series, used below :attr:`series_angle`.
        :return: The coefficient at each angle, shaped (..., 1, 1) to scale matrices.
        """
        angles = self.angles
        small = angles < self.series_angle
        return np.where(
            small, series(angles), closed_form(np.where(small, 1.0, angles))
        )[..., None, None]

    @property
    def left_jacobians(self) -> npt.NDArray[np.float64]:
        r"""
        The left Jacobian :math:`J(\phi)` of each rotation, stacked to shape (..., 3,
        3): :math:`\exp((\phi + \delta)^\wedge) \approx \exp((J \delta)^\wedge)
        \exp(\phi^\wedge)`.
        """
        skew = self.skew_symmetric_matrices
        return (
            np.eye(3)
            + self.coefficients(
                closed_form=lambda angle: (1 - np.cos(angle)) / angle**2,
                series=lambda angle: 1 / 2 - angle**2 / 24 + angle**4 / 720,
            )
            * skew
            + self.coefficients(
                closed_form=lambda angle: (angle - np.sin(angle)) / angle**3,
                series=lambda angle: 1 / 6 - angle**2 / 120 + angle**4 / 5040,
            )
            * skew
            @ skew
        )

    @property
    def inverse_left_jacobians(self) -> npt.NDArray[np.float64]:
        """
        The inverse of each :attr:`left_jacobians`, in closed form, for angles of up to
        half a revolution.
        """
        skew = self.skew_symmetric_matrices
        return (
            np.eye(3)
            - skew / 2
            + self.coefficients(
                closed_form=lambda angle: (
                    1 / angle**2 - np.cos(angle / 2) / (2 * angle * np.sin(angle / 2))
                ),
                series=lambda angle: 1 / 12 + angle**2 / 720 + angle**4 / 30240,
            )
            * skew
            @ skew
        )


# %% Exponential coordinates of rigid transformations
@dataclass(kw_only=True)
class ExponentialCoordinateStack:
    r"""
    Exponential coordinates of rigid transformations, the rotation vector :math:`\phi`
    and then the translation part :math:`\rho`, stacked to shape (..., 6).
    """

    entries: npt.NDArray[np.float64]
    """
    The exponential coordinates.
    """

    @property
    def left_jacobians(self) -> npt.NDArray[np.float64]:
        r"""
        The left Jacobian of each transformation, stacked to shape (..., 6, 6):

        :math:`\exp((\xi + \delta)^\wedge) \approx \exp((J \delta)^\wedge)
        \exp(\xi^\wedge)`, with the rotation's :math:`J(\phi)` on the diagonal and
        Barfoot's :math:`Q(\phi, \rho)` below it.
        """
        rotation_vectors = RotationVectorStack(entries=self.entries[..., :3])
        rotation = rotation_vectors.skew_symmetric_matrices
        translation = Vector3Stack(
            entries=self.entries[..., 3:]
        ).skew_symmetric_matrices
        rotation_translation = rotation @ translation
        translation_rotation = translation @ rotation
        sandwich = rotation_translation @ rotation
        coupling = (
            translation / 2
            + rotation_vectors.coefficients(
                closed_form=lambda angle: (angle - np.sin(angle)) / angle**3,
                series=lambda angle: 1 / 6 - angle**2 / 120 + angle**4 / 5040,
            )
            * (rotation_translation + translation_rotation + sandwich)
            + rotation_vectors.coefficients(
                closed_form=lambda angle: (
                    (angle**2 + 2 * np.cos(angle) - 2) / (2 * angle**4)
                ),
                series=lambda angle: 1 / 24 - angle**2 / 720 + angle**4 / 40320,
            )
            * (
                rotation @ rotation_translation
                + translation_rotation @ rotation
                - 3 * sandwich
            )
            + rotation_vectors.coefficients(
                closed_form=lambda angle: (
                    (2 * angle - 3 * np.sin(angle) + angle * np.cos(angle))
                    / (2 * angle**5)
                ),
                series=lambda angle: 1 / 120 - angle**2 / 2520 + angle**4 / 120960,
            )
            * (sandwich @ rotation + rotation @ sandwich)
        )
        rotation_jacobians = rotation_vectors.left_jacobians
        jacobians = np.zeros(self.entries.shape[:-1] + (6, 6))
        jacobians[..., :3, :3] = rotation_jacobians
        jacobians[..., 3:, 3:] = rotation_jacobians
        jacobians[..., 3:, :3] = coupling
        return jacobians


# %% Rigid transformations
@dataclass(kw_only=True)
class TransformationStack:
    """
    Homogeneous transformation matrices, stacked to shape (..., 4, 4).
    """

    matrices: npt.NDArray[np.float64]
    """
    The transformations.
    """

    @property
    def adjoints(self) -> npt.NDArray[np.float64]:
        r"""
        The adjoint of each transformation :math:`T`, stacked to shape (..., 6, 6), for
        twists ordered rotation first: :math:`T \exp(\delta^\wedge) =
        \exp((\mathrm{Ad}_T \delta)^\wedge) T`.
        """
        rotations = self.matrices[..., :3, :3]
        adjoints = np.zeros(self.matrices.shape[:-2] + (6, 6))
        adjoints[..., :3, :3] = rotations
        adjoints[..., 3:, 3:] = rotations
        adjoints[..., 3:, :3] = (
            Vector3Stack(entries=self.matrices[..., :3, 3]).skew_symmetric_matrices
            @ rotations
        )
        return adjoints
