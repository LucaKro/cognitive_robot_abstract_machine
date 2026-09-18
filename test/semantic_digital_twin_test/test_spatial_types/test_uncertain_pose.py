"""
Tests for a pose carried together with how uncertain it is, and for the operations that
move both halves at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood.symbolic_math.exceptions import HasFreeVariablesError
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import UncertaintyCorrelationUnknownError
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
        moved.covariance.as_array,
        uncertain_pose.covariance.transformed_by(new_reference_T_reference).as_array,
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
        inverted.covariance.as_array,
        uncertain_pose.covariance.transformed_by(reference_T_pose_inverse).as_array,
        atol=1e-12,
    )


def test_inverting_twice_gives_the_uncertainty_back():
    uncertain_pose = an_uncertain_pose()

    there_and_back = uncertain_pose.inverse().inverse()

    np.testing.assert_allclose(
        there_and_back.covariance.as_array,
        uncertain_pose.covariance.as_array,
        atol=1e-12,
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


# %% extending it further along the chain


def test_extending_by_a_certain_transform_moves_the_pose():
    uncertain_pose = an_uncertain_pose()
    pose_T_further = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, yaw=0.2)

    extended = uncertain_pose @ pose_T_further

    np.testing.assert_allclose(
        extended.pose.to_np(),
        (uncertain_pose.pose.to_homogeneous_matrix() @ pose_T_further).to_np(),
        atol=1e-12,
    )


def test_extending_by_a_certain_transform_leaves_the_uncertainty_as_it_is():
    """
    A bottle held rigidly in an uncertain drawer is exactly as uncertain as the drawer.

    The uncertainty is a displacement of the whole assembly in the frame the drawer is
    reported in, and a rigid offset from the drawer does not change that displacement.
    """
    drawer = an_uncertain_pose()
    drawer_T_bottle = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, z=0.1)

    bottle = drawer @ drawer_T_bottle

    np.testing.assert_allclose(
        bottle.covariance.as_array, drawer.covariance.as_array, atol=1e-12
    )


def test_composing_two_uncertain_poses_is_refused():
    """
    The answer depends on whether the two uncertainties are related, which nothing here
    is told, and a wrong covariance is worse than an absent one.
    """
    uncertain_pose = an_uncertain_pose()

    with pytest.raises(UncertaintyCorrelationUnknownError):
        uncertain_pose @ an_uncertain_pose()
