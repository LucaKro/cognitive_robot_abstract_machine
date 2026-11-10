"""
Tests for splitting detected sofas into seat regions.
"""

# %% imports

from __future__ import annotations

import numpy as np
import open3d
import pytest

from robokudo.annotators.outputs import AnnotatorOutputStruct
from robokudo.annotators.seat_detection import SeatDetection
from robokudo.annotators.seat_occupancy import SeatOccupancyAnnotator
from robokudo.annotators.sofa_split import SofaSplitAnnotator
from .._marked_image import MarkedRegionImage
from ._seating_scene import annotate, seat, created_seat_regions, sofa

# %% splitting


def test_sofa_is_split_into_seat_regions(
    sofa_width: float, seat_distance: float
) -> None:
    """
    A sofa is split into the configured number of seat regions, added to the CAS.
    """
    annotator = annotate(SofaSplitAnnotator(), sofa(sofa_width, seat_distance))

    assert (
        len(created_seat_regions(annotator))
        == SofaSplitAnnotator.Descriptor.Parameters().seats_per_sofa
    )


def test_seat_regions_name_their_source(
    sofa_width: float, seat_distance: float
) -> None:
    """
    Every seat region names the annotator that created it.
    """
    annotator = annotate(SofaSplitAnnotator(), sofa(sofa_width, seat_distance))

    assert {region.source for region in created_seat_regions(annotator)} == {
        annotator.name
    }


def test_seat_regions_are_ordered_left_to_right(
    sofa_width: float, seat_distance: float
) -> None:
    """
    The seat regions lie side by side from left to right, at the depth of the sofa.
    """
    annotator = annotate(SofaSplitAnnotator(), sofa(sofa_width, seat_distance))

    left, right = [
        SeatOccupancyAnnotator.position_in_camera_frame(region)
        for region in created_seat_regions(annotator)
    ]
    assert left[0] < 0 < right[0]
    assert left[2] == right[2] == seat_distance


def test_seat_region_images_split_sofa_image(
    sofa_width: float, seat_distance: float
) -> None:
    """
    The image regions of the seat regions split the image region of the sofa.
    """
    annotator = annotate(SofaSplitAnnotator(), sofa(sofa_width, seat_distance))

    left, top, right, bottom = MarkedRegionImage().region.get_corner_points()
    middle = (left + right) // 2
    assert [
        region.roi.roi.get_corner_points() for region in created_seat_regions(annotator)
    ] == [(left, top, middle, bottom), (middle, top, right, bottom)]


def test_sofa_is_split_into_configured_number_of_seats(
    sofa_width: float, seat_distance: float
) -> None:
    """
    A sofa is split into as many seat regions as configured.
    """
    annotator = SofaSplitAnnotator()
    annotator.descriptor.parameters.seats_per_sofa = 3

    annotate(annotator, sofa(sofa_width, seat_distance))

    assert len(created_seat_regions(annotator)) == 3


# %% what counts as a sofa


def test_wide_seat_is_split_as_sofa(
    seat_detection: SeatDetection, sofa_width: float, seat_distance: float
) -> None:
    """
    A seat class detection wider than a single seat is split like a sofa.
    """
    wide = seat(width=sofa_width, depth=seat_distance)

    annotator = annotate(SofaSplitAnnotator(), wide)

    assert len(created_seat_regions(annotator)) == (
        SofaSplitAnnotator.Descriptor.Parameters().seats_per_sofa
    )


def test_narrow_seat_is_not_split(seat_distance: float) -> None:
    """
    A single seat is not split into seat regions.
    """
    annotator = annotate(SofaSplitAnnotator(), seat(depth=seat_distance))

    assert created_seat_regions(annotator) == []


@pytest.mark.parametrize("points", [None, open3d.geometry.PointCloud()])
def test_sofa_without_points_is_not_split(
    points: open3d.geometry.PointCloud | None, sofa_width: float, seat_distance: float
) -> None:
    """
    A sofa without a point cloud, or with an empty one, is not split.
    """
    without_points = sofa(sofa_width, seat_distance)
    without_points.points = points

    annotator = annotate(SofaSplitAnnotator(), without_points)

    assert created_seat_regions(annotator) == []


# %% visualization


def test_seat_regions_are_framed(sofa_width: float, seat_distance: float) -> None:
    """
    The visualization image frames the left seat region in the seat region color.
    """
    annotator = annotate(SofaSplitAnnotator(), sofa(sofa_width, seat_distance))

    image = annotator.get_annotator_output_struct().image
    assert (
        MarkedRegionImage().frame_color(image) == SofaSplitAnnotator.seat_marking.color
    )


def test_no_image_without_visualization(
    sofa_width: float, seat_distance: float
) -> None:
    """
    With visualization turned off, the annotator outputs no image.
    """
    annotator = SofaSplitAnnotator()
    annotator.descriptor.parameters.global_with_visualization = False

    annotate(annotator, sofa(sofa_width, seat_distance))

    np.testing.assert_array_equal(
        annotator.get_annotator_output_struct().image, AnnotatorOutputStruct().image
    )
