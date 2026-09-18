"""
Tests for reading a pose covariance out of the flat sequence a pose message stores it
in, and for the summaries taken over it.
"""

from __future__ import annotations

import numpy as np
import pytest

from giskardpy.motion_statechart.exceptions import CovarianceNotSixBySixError
from giskardpy.motion_statechart.pose_covariance import PoseAxis, PoseCovariance

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
    entries = np.zeros((len(PoseAxis), len(PoseAxis)), dtype=np.float64)
    for axis, variance in variance_of_axis.items():
        entries[axis, axis] = variance
    return PoseCovariance.from_row_major(entries.reshape(-1))


# %% reading the flat sequence


def test_the_entries_are_read_row_by_row():
    """
    A pose covariance is stored flat and row-major, so entry ``row * 6 + column`` has to
    land at that row and column rather than transposed.
    """
    entries = list(range(len(PoseAxis) * len(PoseAxis)))

    covariance = PoseCovariance.from_row_major(entries)

    assert covariance.values[PoseAxis.POSITION_X, PoseAxis.ROTATION_Z] == float(
        PoseAxis.POSITION_X * len(PoseAxis) + PoseAxis.ROTATION_Z
    )
    assert covariance.values[PoseAxis.ROTATION_Z, PoseAxis.POSITION_X] == float(
        PoseAxis.ROTATION_Z * len(PoseAxis) + PoseAxis.POSITION_X
    )


def test_every_axis_reads_its_own_entry_off_the_diagonal():
    variance_of_axis = {axis: float(axis) + 1.0 for axis in PoseAxis}

    covariance = covariance_with_variances(variance_of_axis)

    for axis, variance in variance_of_axis.items():
        assert covariance.variance_of(axis) == variance, axis


def test_a_sequence_that_is_not_a_six_by_six_matrix_is_rejected():
    too_few = [0.0] * (len(PoseAxis) * len(PoseAxis) - 1)

    with pytest.raises(CovarianceNotSixBySixError) as error:
        PoseCovariance.from_row_major(too_few)

    assert error.value.given_size == len(too_few)
    assert error.value.expected_size == len(PoseAxis) * len(PoseAxis)


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
    The quantity the statechart conditions on, so it must not silently ignore an axis.
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
