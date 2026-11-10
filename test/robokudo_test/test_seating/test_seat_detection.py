"""
Tests for deciding which detected objects are seats for a single person.
"""

# %% imports

from __future__ import annotations

import open3d
import pytest

from robokudo.annotators.seat_detection import SeatDetection
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Classification
from ._seating_scene import cloud_across_view, human, seat

# %% single seats


def test_narrow_seat_is_single_seat(seat_detection: SeatDetection) -> None:
    """
    A seat class detection not wider than a single seat is a single seat.
    """
    assert seat_detection.is_single_seat(seat())


def test_wide_seat_is_not_single_seat(seat_detection: SeatDetection) -> None:
    """
    A seat class detection wider than a single seat, such as a sofa detected as a chair,
    is not a single seat.
    """
    wide = seat(width=seat_detection.maximum_seat_width * 2)

    assert not seat_detection.is_single_seat(wide)


def test_uncertain_seat_is_not_single_seat(seat_detection: SeatDetection) -> None:
    """
    A seat class detection below the minimum seat confidence, such as a table taken for
    a chair, is not a single seat; one at the minimum is.
    """
    minimum = seat_detection.minimum_seat_confidence

    assert not seat_detection.is_single_seat(seat(confidence=minimum - 0.01))
    assert seat_detection.is_single_seat(seat(confidence=minimum))


@pytest.mark.parametrize("class_name", ["laptop", "couch"])
def test_object_of_other_class_is_not_single_seat(
    seat_detection: SeatDetection, class_name: str
) -> None:
    """
    An object not classified as a seat class, a sofa included, is not a single seat.
    """
    assert not seat_detection.is_single_seat(seat(class_name))


@pytest.mark.parametrize("points", [None, open3d.geometry.PointCloud()])
def test_seat_without_points_is_not_single_seat(
    seat_detection: SeatDetection, points: open3d.geometry.PointCloud | None
) -> None:
    """
    A seat without a point cloud, or with an empty one, is not a single seat.
    """
    chair = seat()
    chair.points = points

    assert not seat_detection.is_single_seat(chair)
    assert seat_detection.width_across_view(chair) is None


# %% width


def test_outlying_points_do_not_widen_seat(seat_detection: SeatDetection) -> None:
    """
    A few points far beside a seat, fewer than the outlier share, do not change its
    width.
    """
    chair = seat()
    width = seat_detection.width_across_view(chair)
    far_beside = seat_detection.maximum_seat_width * 5
    chair.points += cloud_across_view(-far_beside, far_beside)

    assert seat_detection.width_across_view(chair) == pytest.approx(width, rel=0.05)
    assert seat_detection.is_single_seat(chair)


def test_wider_seat_measures_wider(seat_detection: SeatDetection) -> None:
    """
    A seat with points spread twice as wide measures twice as wide.
    """
    narrow = seat_detection.width_across_view(seat(width=0.4))
    wide = seat_detection.width_across_view(seat(width=0.8))

    assert wide == pytest.approx(2 * narrow)


# %% seats in the CAS


def test_single_seats_in_cas_exclude_humans_and_wide_seats(
    seat_detection: SeatDetection,
) -> None:
    """
    Of the hypotheses in a CAS, only the narrow seat class detections are single seats;
    humans classified as a seat class and wide seats are not.
    """
    chair = seat()
    wide = seat(width=seat_detection.maximum_seat_width * 2)
    person = human()
    person.annotations.append(Classification(classname=chair.annotations[0].classname))
    pipeline = Pipeline()
    pipeline.setup()
    pipeline.cas.annotations.extend([chair, wide, person])

    single_seats = seat_detection.single_seats(pipeline.cas)

    assert len(single_seats) == 1
    assert single_seats[0] is chair
