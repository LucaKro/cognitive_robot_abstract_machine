"""
Tests for how uncertain a pose is, and for the summaries taken over that uncertainty.
"""

from __future__ import annotations

import numpy as np
import pytest
from random_events.variable import Continuous

from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import (
    PoseCovarianceNotSixBySixError,
    VariableNotInPoseError,
)
from semantic_digital_twin.spatial_types import PoseCovariance

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
