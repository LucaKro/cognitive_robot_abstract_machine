"""
Tests for how uncertain a pose is, and for the summaries taken over that uncertainty.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_digital_twin.exceptions import PoseCovarianceNotSixBySixError
from semantic_digital_twin.spatial_types import PoseAxis, PoseCovariance

# %% helpers


def covariance_with_variances(
    variance_of_axis: dict[PoseAxis, float],
) -> PoseCovariance:
    """
    A covariance whose diagonal carries the given variances and whose other entries are
    zero.

    :param variance_of_axis: The variance to place on each axis; omitted axes are zero.
    :return: The covariance those variances describe.
    """
    values = np.zeros((len(PoseAxis), len(PoseAxis)), dtype=np.float64)
    for axis, variance in variance_of_axis.items():
        values[axis, axis] = variance
    return PoseCovariance(values=values)


# %% the axes it is indexed by


def test_every_axis_reads_its_own_entry_off_the_diagonal():
    variance_of_axis = {axis: float(axis) + 1.0 for axis in PoseAxis}

    covariance = covariance_with_variances(variance_of_axis)

    for axis, variance in variance_of_axis.items():
        assert covariance.variance_of(axis) == variance, axis


def test_the_axes_split_into_a_translational_and_a_rotational_half():
    """
    The two halves are what :attr:`position_variance` and :attr:`rotation_variance` sum
    over, so between them they have to cover every axis exactly once.
    """
    position_axes = PoseAxis.position_axes()
    rotation_axes = PoseAxis.rotation_axes()

    assert set(position_axes).isdisjoint(rotation_axes)
    assert set(position_axes) | set(rotation_axes) == set(PoseAxis)


def test_a_matrix_that_does_not_relate_all_six_axes_is_rejected():
    too_small = np.zeros((len(PoseAxis) - 1, len(PoseAxis)), dtype=np.float64)

    with pytest.raises(PoseCovarianceNotSixBySixError) as error:
        PoseCovariance(values=too_small)

    assert error.value.given_shape == too_small.shape
    assert error.value.expected_shape == (len(PoseAxis), len(PoseAxis))


# %% the summaries taken over it


def test_the_position_variance_sums_the_translational_axes_only():
    covariance = covariance_with_variances(
        {
            PoseAxis.POSITION_X: 0.5,
            PoseAxis.POSITION_Y: 0.25,
            PoseAxis.POSITION_Z: 0.125,
            PoseAxis.ROTATION_Z: 9.0,
        }
    )

    assert covariance.position_variance == 0.875


def test_the_rotation_variance_sums_the_rotational_axes_only():
    covariance = covariance_with_variances(
        {
            PoseAxis.POSITION_X: 9.0,
            PoseAxis.ROTATION_X: 0.5,
            PoseAxis.ROTATION_Y: 0.25,
            PoseAxis.ROTATION_Z: 0.125,
        }
    )

    assert covariance.rotation_variance == 0.875


def test_the_total_variance_covers_every_axis():
    """
    The quantity a caller conditions on, so it must not silently ignore an axis.
    """
    variance_of_axis = {axis: float(axis) + 1.0 for axis in PoseAxis}
    covariance = covariance_with_variances(variance_of_axis)

    assert covariance.total_variance == sum(variance_of_axis.values())


def test_the_total_variance_ignores_the_off_diagonal_entries():
    """
    Only the diagonal holds each axis's own variance; the rest is how axes covary.
    """
    on_diagonal = covariance_with_variances({axis: 1.0 for axis in PoseAxis})
    with_correlations = PoseCovariance(values=on_diagonal.values + 5.0)
    for axis in PoseAxis:
        with_correlations.values[axis, axis] = 1.0

    assert with_correlations.total_variance == on_diagonal.total_variance
