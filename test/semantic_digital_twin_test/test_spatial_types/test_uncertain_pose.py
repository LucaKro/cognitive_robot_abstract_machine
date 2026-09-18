"""
Tests for a pose carried together with how uncertain it is, and for the operations that
move both halves at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood.symbolic_math.exceptions import HasFreeVariablesError
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    PoseCovariance,
    UncertainPose,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose


def an_uncertain_pose() -> UncertainPose:
    """
    :return: A pose away from the origin, turned, and uncertain about both where it is
        and which way it faces, so every part of a propagation has something to act on.
    """
    return UncertainPose(
        pose=Pose.from_xyz_rpy(x=2.0, y=1.0, z=0.5, yaw=0.4),
        covariance=PoseCovariance.of(
            {
                (SpatialVariables.x.value,) * 2: 0.09,
                (SpatialVariables.yaw.value,) * 2: 0.04,
                (SpatialVariables.x.value, SpatialVariables.yaw.value): 0.01,
            }
        ),
    )


# %% moving it into another frame


def test_transforming_moves_the_pose_itself():
    uncertain_pose = an_uncertain_pose()
    new_reference_T_reference = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-1.0, z=2.0, pitch=0.3
    )

    moved = uncertain_pose.transformed_by(new_reference_T_reference)

    np.testing.assert_allclose(
        moved.pose.to_np(),
        (new_reference_T_reference @ uncertain_pose.pose).to_np(),
        atol=1e-12,
    )


def test_transforming_moves_the_uncertainty_along_with_the_pose():
    """
    The reason the two are held together at all: neither half may be left behind in the
    frame the other one came from.
    """
    uncertain_pose = an_uncertain_pose()
    new_reference_T_reference = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-1.0, z=2.0, pitch=0.3
    )

    moved = uncertain_pose.transformed_by(new_reference_T_reference)

    np.testing.assert_allclose(
        moved.covariance.values,
        uncertain_pose.covariance.transformed_by(new_reference_T_reference).values,
        atol=1e-12,
    )


# %% turning it around


def test_inverting_moves_the_uncertainty_by_the_pose_inverse():
    uncertain_pose = an_uncertain_pose()

    inverted = uncertain_pose.inverse()

    reference_T_pose_inverse = uncertain_pose.pose.to_homogeneous_matrix().inverse()
    np.testing.assert_allclose(
        inverted.pose.to_np(), reference_T_pose_inverse.to_np(), atol=1e-12
    )
    np.testing.assert_allclose(
        inverted.covariance.values,
        uncertain_pose.covariance.transformed_by(reference_T_pose_inverse).values,
        atol=1e-12,
    )


def test_inverting_twice_gives_the_uncertainty_back():
    uncertain_pose = an_uncertain_pose()

    there_and_back = uncertain_pose.inverse().inverse()

    np.testing.assert_allclose(
        there_and_back.covariance.values, uncertain_pose.covariance.values, atol=1e-12
    )


# %% what it refuses


def test_a_pose_whose_numbers_are_not_known_cannot_be_inverted():
    """
    Most poses in this stack are symbolic forward-kinematics expressions, and there is
    no numeric uncertainty to move through one.
    """
    symbolic = UncertainPose(
        pose=HomogeneousTransformationMatrix.create_with_variables(
            name="joint"
        ).to_pose(),
        covariance=PoseCovariance.of({(SpatialVariables.x.value,) * 2: 1.0}),
    )

    with pytest.raises(HasFreeVariablesError):
        symbolic.inverse()
