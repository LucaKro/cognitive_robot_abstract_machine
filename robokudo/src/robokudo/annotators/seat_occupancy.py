"""
Seat occupancy estimation from detected seats and humans.

Annotates every seat within reach of the camera with whether a sitting human is close to
it, and answers queries with these seats.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum

import numpy as np
from py_trees.common import Status
from robokudo_msgs.action import Query
from robokudo_msgs.msg import ObjectDesignator
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.gpsr.sitting import HumanPosture
from robokudo.annotators.seat_detection import SeatDetection, SeatRegion
from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import PositionAnnotation
from robokudo.types.human import ActivityAnnotation
from robokudo.types.scene import (
    AnalyzableAnnotation,
    HumanHypothesis,
    OccupiedAnnotation,
)
from robokudo.utils.annotation_conversion import Position2ODConverter
from robokudo.utils.query import QueryHandler
from robokudo.vis.box_marking import BoxMarking

if TYPE_CHECKING:
    import numpy.typing as npt

# %% visualization


class OccupancyMarking(Enum):
    """
    How a seat is marked in the visualization image.
    """

    FREE = BoxMarking(label="free seat", color=(0, 200, 0))
    """
    The seat is free.
    """

    OCCUPIED = BoxMarking(label="occupied", color=(0, 0, 255))
    """
    A human occupies the seat.
    """


# %% seats


@dataclass
class Seat:
    """
    A place for one person to sit, located in the camera frame.
    """

    hypothesis: AnalyzableAnnotation
    """
    The hypothesis the occupancy of the seat is annotated on.
    """

    position: npt.NDArray[np.float64]
    """
    Position in meters of the seat in the camera frame.
    """


# %% annotator


class SeatOccupancyAnnotator(BaseAnnotator):
    """
    Annotates every seat within reach of the camera with an
    :class:`~robokudo.types.scene.OccupiedAnnotation`.

    The seats are the single seats, such as chairs, and the
    :class:`~robokudo.annotators.seat_detection.SeatRegion` in the CAS. A seat region
    closer to a single seat than :attr:`Descriptor.Parameters.minimum_seat_spacing` is
    the same place and left out. Seats and humans are located by their
    :class:`~robokudo.types.annotation.PositionAnnotation` in the camera frame; those
    without one are left out. A seat is occupied when an occupant is close to it; see
    :meth:`is_occupant`. The visualization image marks every annotated seat as
    :class:`OccupancyMarking`.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Descriptor for the SeatOccupancyAnnotator.
        """

        class Parameters:
            """
            Parameters for the SeatOccupancyAnnotator.
            """

            def __init__(self) -> None:
                """
                Set every parameter to its default.
                """
                self.seat_detection: SeatDetection = SeatDetection()
                """
                The rules that decide which object hypotheses are single seats.
                """
                self.maximum_seat_distance: float = 3.0
                """
                Distance in meters from the camera beyond which seats are not annotated.
                """
                self.maximum_occupant_distance: float = 0.5
                """
                Distance in meters between the positions of a seat and a human up to
                which the human occupies the seat.
                """
                self.minimum_seat_spacing: float = 0.5
                """
                Distance in meters between the positions of a seat region and a single
                seat below which they are the same place.
                """
                self.unclear_posture_occupies: bool = False
                """
                Whether a human annotated neither as sitting nor as standing occupies
                a seat.
                """

    def __init__(
        self,
        name: str = "SeatOccupancyAnnotator",
        descriptor: SeatOccupancyAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of the annotator in the pipeline.
        :param descriptor: Configuration of the annotator, the default one if None.
        """
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """
        Annotate every seat within reach with whether it is occupied.

        :return: SUCCESS after processing.
        """
        parameters: SeatOccupancyAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        occupant_positions = self.occupant_positions()
        seats_in_reach = [
            seat
            for seat in self.seats()
            if np.linalg.norm(seat.position) <= parameters.maximum_seat_distance
        ]
        occupancies = [
            any(
                np.linalg.norm(seat.position - occupant_position)
                <= parameters.maximum_occupant_distance
                for occupant_position in occupant_positions
            )
            for seat in seats_in_reach
        ]
        for seat, occupied in zip(seats_in_reach, occupancies):
            seat.hypothesis.annotations.append(
                OccupiedAnnotation(source=self.name, occupied=occupied)
            )

        if parameters.global_with_visualization:
            self._visualize(seats_in_reach, occupancies)
        self.feedback_message = f"{occupancies.count(False)} free seats"
        return Status.SUCCESS

    def _visualize(self, seats: list[Seat], occupancies: list[bool]) -> None:
        """
        Mark every seat in a copy of the color image by its occupancy and output the
        image.

        :param seats: The seats annotated in this update.
        :param occupancies: Whether each of the seats is occupied, in the same order.
        """
        image = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        for seat, occupied in zip(seats, occupancies):
            marking = OccupancyMarking.OCCUPIED if occupied else OccupancyMarking.FREE
            marking.value.draw(image, seat.hypothesis.roi.roi)
        self.get_annotator_output_struct().set_image(image)

    def seats(self) -> list[Seat]:
        """
        :return: The located single seats and the located seat regions that no single
            seat lies on.
        """
        parameters: SeatOccupancyAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        single_seats = self.located(
            parameters.seat_detection.single_seats(self.get_cas())
        )
        seat_regions = self.located(
            self.get_cas().filter_annotations_by_type(SeatRegion)
        )
        separate_seat_regions = [
            seat_region
            for seat_region in seat_regions
            if all(
                np.linalg.norm(seat_region.position - single_seat.position)
                >= parameters.minimum_seat_spacing
                for single_seat in single_seats
            )
        ]
        return single_seats + separate_seat_regions

    def located(self, hypotheses: list[AnalyzableAnnotation]) -> list[Seat]:
        """
        :param hypotheses: The hypotheses of seats.
        :return: A seat for every hypothesis with a position, in the same order.
        """
        seats = []
        for hypothesis in hypotheses:
            position = self.position_in_camera_frame(hypothesis)
            if position is not None:
                seats.append(Seat(hypothesis=hypothesis, position=position))
        return seats

    def occupant_positions(self) -> list[npt.NDArray[np.float64]]:
        """
        :return: The positions of all located occupants.
        """
        positions = []
        for human in self.get_cas().filter_annotations_by_type(HumanHypothesis):
            position = self.position_in_camera_frame(human)
            if position is None or not self.is_occupant(human):
                continue
            positions.append(position)
        return positions

    def is_occupant(self, human: HumanHypothesis) -> bool:
        """
        :param human: The human to judge.
        :return: Whether the human occupies the seat they are close to: never when
            annotated as standing, always when annotated as sitting, and as
            :attr:`Descriptor.Parameters.unclear_posture_occupies` says otherwise.
        """
        activities = {
            activity.type
            for activity in CAS.filter_by_type(ActivityAnnotation, human.annotations)
        }
        if HumanPosture.STANDING in activities:
            return False
        if HumanPosture.SITTING in activities:
            return True
        return self.descriptor.parameters.unclear_posture_occupies

    @staticmethod
    def position_in_camera_frame(
        hypothesis: AnalyzableAnnotation,
    ) -> npt.NDArray[np.float64] | None:
        """
        :param hypothesis: The hypothesis to locate.
        :return: The position of the hypothesis in camera coordinates, from its first
            position annotation, or None if it has none.
        """
        positions = CAS.filter_by_type(PositionAnnotation, hypothesis.annotations)
        if not positions:
            return None
        return np.asarray(positions[0].translation, dtype=np.float64)


# %% answer vocabulary


class SeatFurniture(StrEnum):
    """
    The furniture a seat belongs to, as reported in :attr:`ObjectDesignator.type`.
    """

    CHAIR = "chair"
    """
    A seat for a single person.
    """

    SOFA = "sofa"
    """
    One seat region of a piece of furniture for several persons.
    """


class SeatState(StrEnum):
    """
    Whether a seat is free, as reported in :attr:`ObjectDesignator.attribute`.
    """

    FREE = "free"
    """
    No sitting human is close to the seat.
    """

    OCCUPIED = "occupied"
    """
    A sitting human is close to the seat.
    """


# %% query answer


class SeatQueryAnswerAnnotator(BaseAnnotator):
    """
    Answers the current query with one object designator for every seat annotated
    with an :class:`~robokudo.types.scene.OccupiedAnnotation`, holding its
    :class:`SeatFurniture`, its :class:`SeatState` and its world frame position. The text
    result states the number of free seats.
    """

    def __init__(self, name: str = "SeatQueryAnswerAnnotator") -> None:
        """
        :param name: Name of the annotator in the pipeline.
        """
        super().__init__(name)
        self.position_converter: Position2ODConverter = Position2ODConverter()
        """
        Converts the camera frame position of a seat into a world frame pose of its
        object designator.
        """

    def update(self) -> Status:
        """
        Send the seats as the answer to the current query.

        :return: SUCCESS after sending the answer.
        """
        object_designators = [
            self.describe_seat(seat)
            for seat in self.get_cas().filter_annotations_by_type(AnalyzableAnnotation)
            if CAS.filter_by_type(OccupiedAnnotation, seat.annotations)
        ]
        free_seats = sum(
            SeatState.FREE in object_designator.attribute
            for object_designator in object_designators
        )
        result = Query.Result(
            text_result=f"{free_seats} free seats", res=object_designators
        )
        QueryHandler.send_answer(result)
        self.feedback_message = result.text_result
        return Status.SUCCESS

    def describe_seat(self, seat: AnalyzableAnnotation) -> ObjectDesignator:
        """
        :param seat: A seat annotated with its occupancy and its position in the camera
            frame.
        :return: The object designator describing the seat.
        """
        occupied = CAS.filter_by_type(OccupiedAnnotation, seat.annotations)[0].occupied
        object_designator = ObjectDesignator(
            type=SeatFurniture.SOFA
            if isinstance(seat, SeatRegion)
            else SeatFurniture.CHAIR,
            attribute=[SeatState.OCCUPIED if occupied else SeatState.FREE],
        )
        position = CAS.filter_by_type(PositionAnnotation, seat.annotations)[0]
        self.position_converter.convert(position, self.get_cas(), object_designator)
        return object_designator
