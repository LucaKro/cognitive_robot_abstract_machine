"""
Tests of the derivatives of rotations and rigid transformations by their exponential
coordinates.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import RigidTransform, Rotation

from semantic_digital_twin.articulation_estimation.lie_group_derivatives import (
    ExponentialCoordinateStack,
    RotationVectorStack,
    TransformationStack,
    Vector3Stack,
)


# %% Helpers
def numerical_rotation_left_jacobian(
    *, rotation_vector: npt.NDArray[np.float64], step: float = 1e-6
) -> npt.NDArray[np.float64]:
    """
    :return: The left Jacobian of the rotation, by central differences of the rotation
        vector of :math:`\\exp((\\phi + \\delta)^\\wedge) \\exp(\\phi^\\wedge)^{-1}`.
    """
    rotation = Rotation.from_rotvec(rotation_vector)
    return np.stack(
        [
            (
                (
                    Rotation.from_rotvec(rotation_vector + step * unit) * rotation.inv()
                ).as_rotvec()
                - (
                    Rotation.from_rotvec(rotation_vector - step * unit) * rotation.inv()
                ).as_rotvec()
            )
            / (2 * step)
            for unit in np.eye(3)
        ],
        axis=-1,
    )


def numerical_transformation_left_jacobian(
    *, exponential_coordinates: npt.NDArray[np.float64], step: float = 1e-6
) -> npt.NDArray[np.float64]:
    """
    :return: The left Jacobian of the transformation, by central differences as for
        :func:`numerical_rotation_left_jacobian`.
    """
    transformation = RigidTransform.from_exp_coords(exponential_coordinates)
    return np.stack(
        [
            (
                (
                    RigidTransform.from_exp_coords(
                        exponential_coordinates + step * unit
                    )
                    * transformation.inv()
                ).as_exp_coords()
                - (
                    RigidTransform.from_exp_coords(
                        exponential_coordinates - step * unit
                    )
                    * transformation.inv()
                ).as_exp_coords()
            )
            / (2 * step)
            for unit in np.eye(6)
        ],
        axis=-1,
    )


angles = [1e-8, 1e-3, 0.05, 0.099, 0.101, 1.0, 3.0]
"""
Angles of rotation on both sides of :attr:`RotationVectorStack.series_angle`, up to
nearly half a revolution.
"""


def rotation_vector(*, angle: float) -> npt.NDArray[np.float64]:
    """
    :return: A rotation by ``angle`` about a fixed, oblique axis.
    """
    return angle * np.array([1.0, -2.0, 2.0]) / 3.0


# %% Rotations
def test_skew_symmetric_matrix_takes_the_cross_product():
    """
    :math:`a^\\wedge x = a \\times x`.
    """
    vector, other = np.array([0.3, -1.2, 2.0]), np.array([-0.7, 0.4, 1.1])
    assert_allclose(
        Vector3Stack(entries=vector).skew_symmetric_matrices @ other,
        np.cross(vector, other),
    )


@pytest.mark.parametrize("angle", angles)
def test_rotation_left_jacobian_matches_central_differences(angle: float):
    """
    The closed form and its series both agree with the derivative of the exponential
    map.
    """
    entries = rotation_vector(angle=angle)
    assert_allclose(
        RotationVectorStack(entries=entries).left_jacobians,
        numerical_rotation_left_jacobian(rotation_vector=entries),
        atol=1e-8,
    )


@pytest.mark.parametrize("angle", angles)
def test_inverse_left_jacobian_inverts_the_left_jacobian(angle: float):
    """
    The closed form of the inverse and its series invert the left Jacobian.
    """
    rotation_vectors = RotationVectorStack(entries=rotation_vector(angle=angle))
    assert_allclose(
        rotation_vectors.inverse_left_jacobians @ rotation_vectors.left_jacobians,
        np.eye(3),
        atol=1e-11,
    )


# %% Rigid transformations
@pytest.mark.parametrize("angle", angles)
def test_transformation_left_jacobian_matches_central_differences(angle: float):
    """
    The left Jacobian of a rigid transformation agrees with the derivative of its
    exponential map, for nearly pure translations too.
    """
    entries = np.concatenate([rotation_vector(angle=angle), [0.4, -0.8, 1.5]])
    assert_allclose(
        ExponentialCoordinateStack(entries=entries).left_jacobians,
        numerical_transformation_left_jacobian(exponential_coordinates=entries),
        atol=1e-7,
    )


def test_adjoint_moves_a_twist_from_the_right_to_the_left():
    """
    :math:`T \\exp(\\delta^\\wedge) = \\exp((\\mathrm{Ad}_T \\delta)^\\wedge) T`.
    """
    transformation = RigidTransform.from_exp_coords([0.3, -0.5, 1.1, 0.4, 0.2, -0.9])
    twist = np.array([0.02, -0.01, 0.03, 0.05, -0.04, 0.01])
    adjoint = TransformationStack(matrices=transformation.as_matrix()).adjoints
    assert_allclose(
        (transformation * RigidTransform.from_exp_coords(twist)).as_matrix(),
        (RigidTransform.from_exp_coords(adjoint @ twist) * transformation).as_matrix(),
        atol=1e-12,
    )


def test_stacks_compute_each_entry_independently():
    """
    A stack of exponential coordinates gives the left Jacobian of each entry.
    """
    entries = np.array(
        [
            np.concatenate([rotation_vector(angle=angle), [0.4, -0.8, 1.5]])
            for angle in angles
        ]
    )
    jacobians = ExponentialCoordinateStack(entries=entries).left_jacobians
    for entry, jacobian in zip(entries, jacobians):
        assert_allclose(
            jacobian, ExponentialCoordinateStack(entries=entry).left_jacobians
        )
