"""
Tests for how uncertain a pose is, the summaries taken over that uncertainty, and
reading it in another frame.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from random_events.variable import Continuous

from krrood.symbolic_math.exceptions import HasFreeVariablesError
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import (
    PoseCovarianceNotSixBySixError,
    VariableNotInPoseError,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    PoseCovariance,
    PoseDisplacementMap,
)

# %% the degrees of freedom it is indexed by


def test_a_pose_is_the_position_variables_followed_by_the_rotation_ones():
    """
    The order every array over a pose's degrees of freedom is laid out in, so the two
    halves have to cover it exactly once each and in this order.
    """
    assert (
        SpatialVariables.pose == SpatialVariables.position + SpatialVariables.rotation
    )


def test_the_two_halves_of_a_pose_do_not_overlap():
    """
    They are what :attr:`position_variance` and :attr:`rotation_variance` sum over.
    """
    assert set(SpatialVariables.position).isdisjoint(SpatialVariables.rotation)


def test_every_degree_of_freedom_reads_its_own_entry_off_the_diagonal():
    variances = {
        variable: float(row) + 1.0 for row, variable in enumerate(SpatialVariables.pose)
    }

    covariance = PoseCovariance.of(
        {(variable, variable): variance for variable, variance in variances.items()}
    )

    for variable, variance in variances.items():
        assert covariance.variance_of(variable) == variance, variable


def test_a_variable_that_is_not_a_degree_of_freedom_of_a_pose_is_rejected():
    not_a_pose_variable = Continuous(name="gripper_opening")
    covariance = PoseCovariance.of({})

    with pytest.raises(VariableNotInPoseError) as error:
        covariance.variance_of(not_a_pose_variable)

    assert error.value.variable_name == not_a_pose_variable.name


def test_a_matrix_that_does_not_relate_all_six_degrees_of_freedom_is_rejected():
    side = len(SpatialVariables.pose)
    too_small = np.zeros((side - 1, side), dtype=np.float64)

    with pytest.raises(PoseCovarianceNotSixBySixError) as error:
        PoseCovariance(values=too_small)

    assert error.value.given_shape == too_small.shape
    assert error.value.expected_shape == (side, side)


# %% building one from named entries


def test_two_degrees_of_freedom_vary_together_by_one_number_stated_once():
    """
    A covariance is symmetric, so naming a pair fills its mirror as well.
    """
    x, yaw = SpatialVariables.x.value, SpatialVariables.yaw.value

    covariance = PoseCovariance.of({(x, yaw): 0.25})

    assert covariance.covariance_between(x, yaw) == 0.25
    assert covariance.covariance_between(yaw, x) == 0.25


def test_a_pair_that_is_not_named_is_zero():
    covariance = PoseCovariance.of({(SpatialVariables.x.value,) * 2: 1.0})

    assert (
        covariance.covariance_between(
            SpatialVariables.x.value, SpatialVariables.yaw.value
        )
        == 0.0
    )


def test_building_with_a_variable_that_is_not_a_degree_of_freedom_of_a_pose_is_rejected():
    not_a_pose_variable = Continuous(name="gripper_opening")

    with pytest.raises(VariableNotInPoseError) as error:
        PoseCovariance.of({(not_a_pose_variable, not_a_pose_variable): 1.0})

    assert error.value.variable_name == not_a_pose_variable.name


# %% the summaries taken over it


def test_the_position_variance_sums_the_position_degrees_of_freedom_only():
    covariance = PoseCovariance.of(
        {
            (SpatialVariables.x.value,) * 2: 0.5,
            (SpatialVariables.y.value,) * 2: 0.25,
            (SpatialVariables.z.value,) * 2: 0.125,
            (SpatialVariables.yaw.value,) * 2: 9.0,
        }
    )

    assert covariance.position_variance == 0.875


def test_the_rotation_variance_sums_the_rotation_degrees_of_freedom_only():
    covariance = PoseCovariance.of(
        {
            (SpatialVariables.x.value,) * 2: 9.0,
            (SpatialVariables.roll.value,) * 2: 0.5,
            (SpatialVariables.pitch.value,) * 2: 0.25,
            (SpatialVariables.yaw.value,) * 2: 0.125,
        }
    )

    assert covariance.rotation_variance == 0.875


def test_the_total_variance_covers_every_degree_of_freedom():
    """
    The quantity a caller conditions on, so it must not silently ignore one.
    """
    variances = {
        variable: float(row) + 1.0 for row, variable in enumerate(SpatialVariables.pose)
    }
    covariance = PoseCovariance.of(
        {(variable, variable): variance for variable, variance in variances.items()}
    )

    assert covariance.total_variance == sum(variances.values())


def test_the_total_variance_ignores_how_the_degrees_of_freedom_covary():
    """
    Only the diagonal holds each degree of freedom's own variance.
    """
    variances = {(variable,) * 2: 1.0 for variable in SpatialVariables.pose}
    on_diagonal = PoseCovariance.of(variances)
    with_correlations = PoseCovariance.of(
        {
            **variances,
            (SpatialVariables.x.value, SpatialVariables.yaw.value): 5.0,
        }
    )

    assert with_correlations.total_variance == on_diagonal.total_variance


# %% moving it into another frame


def perturbation_matrix(
    perturbation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    The definition of what a pose covariance is about, written out once for the tests
    that check the propagation against it rather than against the formula it is
    implemented in.

    :param perturbation: How far the true pose is from the reported one, along each
        degree of freedom in :attr:`SpatialVariables.pose`.
    :return: The matrix whose exponential turns a reported pose into the perturbed one.
    """
    shift = perturbation[:3]
    turn = perturbation[3:]
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [0.0, -turn[2], turn[1]],
            [turn[2], 0.0, -turn[0]],
            [-turn[1], turn[0], 0.0],
        ]
    )
    matrix[:3, 3] = shift
    return matrix


def perturbation_of(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    The inverse reading of :func:`perturbation_matrix`.

    :param matrix: A matrix of the shape that function builds.
    :return: The perturbation it stands for.
    """
    return np.concatenate(
        (matrix[:3, 3], np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]]))
    )


def test_a_rotation_turns_the_uncertainty_with_it():
    """
    A pose uncertain along one axis, seen from a frame turned a quarter turn about z, is
    uncertain along the next axis instead.
    """
    covariance = PoseCovariance.of({(SpatialVariables.x.value,) * 2: 4.0})
    quarter_turn = HomogeneousTransformationMatrix.from_xyz_rpy(yaw=np.pi / 2)

    moved = covariance.transformed_by(quarter_turn)

    assert moved.variance_of(SpatialVariables.y.value) == pytest.approx(4.0)
    assert moved.variance_of(SpatialVariables.x.value) == pytest.approx(0.0)


def test_a_translation_leaves_the_rotational_uncertainty_alone():
    """
    How far the pose is turned is the same question wherever the origin sits.
    """
    covariance = PoseCovariance.of({(SpatialVariables.yaw.value,) * 2: 0.25})
    shifted = HomogeneousTransformationMatrix.from_xyz_rpy(x=3.0)

    moved = covariance.transformed_by(shifted)

    assert moved.variance_of(SpatialVariables.yaw.value) == pytest.approx(0.25)


def test_turning_uncertainty_becomes_position_uncertainty_at_the_lever_arm():
    """
    The half a plain rotation of the matrix would silently drop: a pose that is
    uncertain about its heading is uncertain about where it is once the frame it is
    reported in sits a distance away, by the square of that distance.
    """
    heading_variance = 0.25
    lever_arm = 3.0
    covariance = PoseCovariance.of(
        {(SpatialVariables.yaw.value,) * 2: heading_variance}
    )
    shifted = HomogeneousTransformationMatrix.from_xyz_rpy(x=lever_arm)

    moved = covariance.transformed_by(shifted)

    assert moved.variance_of(SpatialVariables.y.value) == pytest.approx(
        heading_variance * lever_arm**2
    )
    assert moved.variance_of(SpatialVariables.x.value) == pytest.approx(0.0)


def test_moving_a_covariance_nowhere_leaves_it_as_it_was():
    covariance = PoseCovariance.of(
        {
            (variable, variable): float(row) + 1.0
            for row, variable in enumerate(SpatialVariables.pose)
        }
    )

    moved = covariance.transformed_by(HomogeneousTransformationMatrix())

    np.testing.assert_allclose(moved.values, covariance.values, atol=1e-12)


def test_moving_through_two_frames_is_moving_through_their_composition():
    """
    Re-expressing a covariance twice and re-expressing it once through the composed
    transform describe the same uncertainty in the same frame.
    """
    covariance = PoseCovariance.of(
        {
            (SpatialVariables.x.value,) * 2: 1.0,
            (SpatialVariables.yaw.value,) * 2: 0.5,
            (SpatialVariables.x.value, SpatialVariables.yaw.value): 0.25,
        }
    )
    outer_T_middle = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=-2.0, yaw=0.3
    )
    middle_T_inner = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=0.5, roll=0.2, pitch=-0.4
    )

    step_by_step = covariance.transformed_by(middle_T_inner).transformed_by(
        outer_T_middle
    )
    in_one_go = covariance.transformed_by(outer_T_middle @ middle_T_inner)

    np.testing.assert_allclose(step_by_step.values, in_one_go.values, atol=1e-12)


def test_a_perturbation_of_the_pose_is_carried_into_the_new_frame():
    """
    Checks the propagation against what a pose covariance means - how far the true pose
    is from the reported one - rather than against the formula it is written in.

    A covariance of one perturbation and nothing else stays a covariance of one
    perturbation, and that one is the same displacement read in the new frame.
    """
    perturbation = np.array([0.1, -0.2, 0.05, 0.3, -0.15, 0.25])
    covariance = PoseCovariance(values=np.outer(perturbation, perturbation))
    new_reference_T_reference = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.5, y=-0.5, z=2.0, roll=0.4, pitch=-0.2, yaw=0.7
    )

    moved = covariance.transformed_by(new_reference_T_reference)

    carried = perturbation_of(
        new_reference_T_reference.to_np()
        @ perturbation_matrix(perturbation)
        @ new_reference_T_reference.inverse().to_np()
    )
    np.testing.assert_allclose(moved.values, np.outer(carried, carried), atol=1e-12)


def test_a_transform_whose_numbers_are_not_known_is_rejected():
    """
    Most poses in this stack are symbolic forward-kinematics expressions, and a
    covariance cannot be moved by a transform that is not yet a number.
    """
    covariance = PoseCovariance.of({(SpatialVariables.x.value,) * 2: 1.0})
    symbolic = HomogeneousTransformationMatrix.create_with_variables(name="joint")

    with pytest.raises(HasFreeVariablesError):
        covariance.transformed_by(symbolic)


# %% the row a degree of freedom occupies


def test_each_degree_of_freedom_knows_the_row_it_occupies():
    """
    The ordering every array over a pose is laid out by, read from the one place that
    holds it.
    """
    for row, variable in enumerate(SpatialVariables.pose):
        assert SpatialVariables.row_in_pose(variable) == row, variable


def test_asking_for_the_row_of_a_variable_that_is_not_a_degree_of_freedom_is_rejected():
    not_a_pose_variable = Continuous(name="gripper_opening")

    with pytest.raises(VariableNotInPoseError) as error:
        SpatialVariables.row_in_pose(not_a_pose_variable)

    assert error.value.variable_name == not_a_pose_variable.name


# %% how a displacement is read in another frame


def test_turning_the_frame_reads_each_axis_as_the_one_it_turns_into():
    """
    Seen from a frame turned a quarter turn about z, a displacement along x reads as one
    along y.
    """
    quarter_turn = HomogeneousTransformationMatrix.from_xyz_rpy(yaw=np.pi / 2)

    displacement_map = PoseDisplacementMap.of_transform(quarter_turn)

    assert displacement_map.factor_of(
        SpatialVariables.y.value, SpatialVariables.x.value
    ) == pytest.approx(1.0)
    assert displacement_map.factor_of(
        SpatialVariables.x.value, SpatialVariables.x.value
    ) == pytest.approx(0.0)


def test_moving_the_frame_away_makes_position_depend_on_turning():
    """
    The lever arm, read by name: from a frame a distance along x, a turn about z reads
    as a displacement along y by that distance.
    """
    lever_arm = 3.0
    shifted = HomogeneousTransformationMatrix.from_xyz_rpy(x=lever_arm)

    displacement_map = PoseDisplacementMap.of_transform(shifted)

    assert displacement_map.factor_of(
        SpatialVariables.y.value, SpatialVariables.yaw.value
    ) == pytest.approx(-lever_arm)


def test_turning_never_depends_on_where_the_frame_is():
    """
    How far a pose is turned is the same question wherever the origin sits, so no
    rotational row may depend on a positional one.
    """
    moved_and_turned = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, roll=0.3, yaw=-0.7
    )

    displacement_map = PoseDisplacementMap.of_transform(moved_and_turned)

    for turn in SpatialVariables.rotation:
        for shift in SpatialVariables.position:
            assert displacement_map.factor_of(turn, shift) == 0.0, (turn, shift)


def test_a_map_naming_a_variable_that_is_not_a_degree_of_freedom_is_rejected():
    not_a_pose_variable = Continuous(name="gripper_opening")

    with pytest.raises(VariableNotInPoseError) as error:
        PoseDisplacementMap(factors={(not_a_pose_variable, not_a_pose_variable): 1.0})

    assert error.value.variable_name == not_a_pose_variable.name
