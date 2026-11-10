"""
Tests for estimating which seats are occupied by humans.
"""

# %% imports

from __future__ import annotations

import numpy as np

from robokudo.annotators.gpsr.sitting import HumanPosture
from robokudo.annotators.outputs import AnnotatorOutputStruct
from robokudo.annotators.seat_detection import SeatDetection
from robokudo.annotators.seat_occupancy import OccupancyMarking, SeatOccupancyAnnotator
from robokudo.annotators.sofa_split import SofaSplitAnnotator
from robokudo.cas import CAS
from robokudo.types.scene import OccupiedAnnotation
from .._marked_image import MarkedRegionImage
from ._seating_scene import (
    annotate,
    annotate_in_order,
    human,
    located,
    occupancies,
    seat,
    seat_region,
    created_seat_regions,
    sofa,
)

# %% free seats


def test_seat_without_humans_is_free(seat_distance: float) -> None:
    """
    A single seat with no human around is annotated as free.
    """
    chair = located(seat(), x=0.0, z=seat_distance)

    annotate(SeatOccupancyAnnotator(), chair)

    assert occupancies(chair) == [False]


def test_occupancy_names_its_source(seat_distance: float) -> None:
    """
    The occupancy annotation names the annotator that created it.
    """
    chair = located(seat(), x=0.0, z=seat_distance)

    annotator = annotate(SeatOccupancyAnnotator(), chair)

    annotation = CAS.filter_by_type(OccupiedAnnotation, chair.annotations)[0]
    assert annotation.source == annotator.name


def test_standing_human_leaves_seat_free(seat_distance: float) -> None:
    """
    A standing human at a seat leaves it free.
    """
    chair = located(seat(), x=0.0, z=seat_distance)
    person = located(human(HumanPosture.STANDING), x=0.0, z=seat_distance)

    annotate(SeatOccupancyAnnotator(), chair, person)

    assert occupancies(chair) == [False]


def test_distant_human_leaves_seat_free(
    parameters: SeatOccupancyAnnotator.Descriptor.Parameters, seat_distance: float
) -> None:
    """
    A sitting human farther from a seat than the maximum occupant distance leaves it
    free.
    """
    chair = located(seat(), x=0.0, z=seat_distance)
    person = located(
        human(HumanPosture.SITTING),
        x=parameters.maximum_occupant_distance * 2,
        z=seat_distance,
    )

    annotate(SeatOccupancyAnnotator(), chair, person)

    assert occupancies(chair) == [False]


def test_unlocated_human_leaves_seat_free(seat_distance: float) -> None:
    """
    A human without a position is ignored.
    """
    chair = located(seat(), x=0.0, z=seat_distance)

    annotate(SeatOccupancyAnnotator(), chair, human(HumanPosture.SITTING))

    assert occupancies(chair) == [False]


def test_human_with_unclear_posture_leaves_seat_free(seat_distance: float) -> None:
    """
    A human annotated neither as sitting nor as standing leaves a close seat free.
    """
    chair = located(seat(), x=0.0, z=seat_distance)
    person = located(human(), x=0.0, z=seat_distance)

    annotate(SeatOccupancyAnnotator(), chair, person)

    assert occupancies(chair) == [False]


def test_standing_overrules_sitting(seat_distance: float) -> None:
    """
    A human annotated as both standing and sitting leaves a close seat free.
    """
    chair = located(seat(), x=0.0, z=seat_distance)
    person = located(
        human(HumanPosture.STANDING, HumanPosture.SITTING), x=0.0, z=seat_distance
    )

    annotate(SeatOccupancyAnnotator(), chair, person)

    assert occupancies(chair) == [False]


# %% occupied seats


def test_sitting_human_occupies_seat(
    parameters: SeatOccupancyAnnotator.Descriptor.Parameters, seat_distance: float
) -> None:
    """
    A sitting human close to a seat occupies it.
    """
    chair = located(seat(), x=0.0, z=seat_distance)
    person = located(
        human(HumanPosture.SITTING),
        x=parameters.maximum_occupant_distance / 2,
        z=seat_distance,
    )

    annotate(SeatOccupancyAnnotator(), chair, person)

    assert occupancies(chair) == [True]


def test_human_occupies_only_the_close_seat(
    parameters: SeatOccupancyAnnotator.Descriptor.Parameters, seat_distance: float
) -> None:
    """
    Of two seats, a sitting human occupies only the one close to them.
    """
    occupied_chair = located(seat(), x=0.0, z=seat_distance)
    free_chair = located(
        seat(), x=parameters.maximum_occupant_distance * 3, z=seat_distance
    )
    person = located(human(HumanPosture.SITTING), x=0.0, z=seat_distance)

    annotate(SeatOccupancyAnnotator(), occupied_chair, free_chair, person)

    assert occupancies(occupied_chair) == [True]
    assert occupancies(free_chair) == [False]


def test_human_with_unclear_posture_occupies_seat_when_configured(
    seat_distance: float,
) -> None:
    """
    With :attr:`unclear_posture_occupies` set, a human annotated neither as sitting nor
    as standing occupies a close seat.
    """
    occupancy = SeatOccupancyAnnotator()
    occupancy.descriptor.parameters.unclear_posture_occupies = True
    chair = located(seat(), x=0.0, z=seat_distance)
    person = located(human(), x=0.0, z=seat_distance)

    annotate(occupancy, chair, person)

    assert occupancies(chair) == [True]


# %% seats left without occupancy


def test_seat_beyond_reach_is_not_annotated(
    parameters: SeatOccupancyAnnotator.Descriptor.Parameters,
) -> None:
    """
    A seat farther from the camera than the maximum seat distance gets no occupancy.
    """
    chair = located(seat(), x=0.0, z=parameters.maximum_seat_distance * 2)

    annotate(SeatOccupancyAnnotator(), chair)

    assert occupancies(chair) == []


def test_unlocated_seat_is_not_annotated() -> None:
    """
    A seat without a position gets no occupancy.
    """
    chair = seat()

    annotate(SeatOccupancyAnnotator(), chair)

    assert occupancies(chair) == []


def test_wide_seat_is_not_annotated(
    seat_detection: SeatDetection, seat_distance: float
) -> None:
    """
    A seat class detection wider than a single seat gets no occupancy itself.
    """
    wide = located(
        seat(width=seat_detection.maximum_seat_width * 2), x=0.0, z=seat_distance
    )

    annotate(SeatOccupancyAnnotator(), wide)

    assert occupancies(wide) == []


# %% seat regions


def test_seat_region_is_annotated_like_a_seat(seat_distance: float) -> None:
    """
    A seat region with a sitting human on it is occupied, one without is free.
    """
    occupied_region = seat_region(x=-1.0, z=seat_distance)
    free_region = seat_region(x=1.0, z=seat_distance)
    person = located(human(HumanPosture.SITTING), x=-1.0, z=seat_distance)

    annotate(SeatOccupancyAnnotator(), occupied_region, free_region, person)

    assert occupancies(occupied_region) == [True]
    assert occupancies(free_region) == [False]


def test_seat_region_under_single_seat_is_left_out(
    parameters: SeatOccupancyAnnotator.Descriptor.Parameters, seat_distance: float
) -> None:
    """
    A seat region closer to a single seat than the minimum seat spacing gets no
    occupancy, since the single seat stands for the place.
    """
    chair = located(seat(), x=0.0, z=seat_distance)
    region = seat_region(x=parameters.minimum_seat_spacing / 2, z=seat_distance)

    annotate(SeatOccupancyAnnotator(), chair, region)

    assert occupancies(chair) == [False]
    assert occupancies(region) == []


def test_unlocated_seat_region_is_not_annotated() -> None:
    """
    A seat region without a position gets no occupancy.
    """
    region = seat_region(x=0.0, z=0.0)
    region.annotations.clear()

    annotate(SeatOccupancyAnnotator(), region)

    assert occupancies(region) == []


def test_split_sofa_has_one_free_seat_beside_a_sitting_human(
    sofa_width: float, seat_distance: float
) -> None:
    """
    After splitting a sofa, a human sitting on its left half occupies only the left seat
    region.
    """
    occupancy = SeatOccupancyAnnotator()

    annotate_in_order(
        [SofaSplitAnnotator(), occupancy],
        sofa(sofa_width, seat_distance),
        located(human(HumanPosture.SITTING), x=-sofa_width / 4, z=seat_distance),
    )

    assert [occupancies(region) for region in created_seat_regions(occupancy)] == [
        [True],
        [False],
    ]


# %% visualization


def test_free_seat_is_marked_as_free(seat_distance: float) -> None:
    """
    The visualization image frames a free seat in the free color.
    """
    annotator = annotate(
        SeatOccupancyAnnotator(), located(seat(), x=0.0, z=seat_distance)
    )

    image = annotator.get_annotator_output_struct().image
    assert MarkedRegionImage().frame_color(image) == OccupancyMarking.FREE.value.color


def test_occupied_seat_is_marked_as_occupied(seat_distance: float) -> None:
    """
    The visualization image frames an occupied seat in the occupied color.
    """
    annotator = annotate(
        SeatOccupancyAnnotator(),
        located(seat(), x=0.0, z=seat_distance),
        located(human(HumanPosture.SITTING), x=0.0, z=seat_distance),
    )

    image = annotator.get_annotator_output_struct().image
    assert (
        MarkedRegionImage().frame_color(image) == OccupancyMarking.OCCUPIED.value.color
    )


def test_no_image_without_visualization(seat_distance: float) -> None:
    """
    With visualization turned off, the annotator outputs no image.
    """
    annotator = SeatOccupancyAnnotator()
    annotator.descriptor.parameters.global_with_visualization = False

    annotate(annotator, located(seat(), x=0.0, z=seat_distance))

    np.testing.assert_array_equal(
        annotator.get_annotator_output_struct().image, AnnotatorOutputStruct().image
    )
