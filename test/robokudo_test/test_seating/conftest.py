"""
Fixtures shared by the tests of seat detection, sofa splitting and seat occupancy.
"""

# %% imports

from __future__ import annotations

import pytest

from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.annotators.seat_detection import SeatDetection
from robokudo.annotators.seat_occupancy import SeatOccupancyAnnotator

# %% visualization output


@pytest.fixture(autouse=True)
def visualization_output(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> AnnotatorOutputPerPipelineMap:
    """
    Give every annotator under test a buffer to write its visualization output to.
    """
    return annotator_output_buffer


# %% seat placement


@pytest.fixture
def seat_detection() -> SeatDetection:
    """
    The default rules for single seats.
    """
    return SeatDetection()


@pytest.fixture
def parameters() -> SeatOccupancyAnnotator.Descriptor.Parameters:
    """
    The default parameters of the seat occupancy annotator.
    """
    return SeatOccupancyAnnotator.Descriptor.Parameters()


@pytest.fixture
def seat_distance(parameters: SeatOccupancyAnnotator.Descriptor.Parameters) -> float:
    """
    Distance from the camera of a seat within reach.
    """
    return parameters.maximum_seat_distance / 2


@pytest.fixture
def sofa_width(seat_detection: SeatDetection) -> float:
    """
    Width of a sofa, twice the maximum width of a single seat.
    """
    return seat_detection.maximum_seat_width * 2
