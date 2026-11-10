"""
Tests for answering queries with the seats found and whether they are free.
"""

# %% imports

from __future__ import annotations

import pytest
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point2
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from robokudo_msgs.action import Query
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.annotators.gpsr.sitting import HumanPosture
from robokudo.annotators.seat_occupancy import (
    SeatFurniture,
    SeatQueryAnswerAnnotator,
    SeatState,
)
from robokudo.identifier import BBIdentifier
from robokudo.pipeline import Pipeline
from robokudo.types.camera import CameraObservation
from robokudo.types.core import Annotation
from robokudo.types.scene import OccupiedAnnotation
from ._seating_scene import human, located, seat, seat_region

# %% camera


@pytest.fixture
def world_frame_name() -> str:
    """
    Name of the world frame the seats are reported in.
    """
    return "map"


@pytest.fixture
def camera_translation() -> list[float]:
    """
    Position of the camera in the world frame, in x, y, z order.
    """
    return [1.0, 2.0, 0.0]


@pytest.fixture
def camera_observation(
    world_frame_name: str, camera_translation: list[float]
) -> CameraObservation:
    """
    A camera at :func:`camera_translation` with its axes aligned to the world axes.
    """
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=500.0,
        focal_length_y=500.0,
        principal_point=Point2(x=319.5, y=239.5),
    )
    camera_body = Body(name=PrefixedName(name="camera"))
    camera_x, camera_y, camera_z = camera_translation
    world_T_camera = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=camera_x,
        pos_y=camera_y,
        pos_z=camera_z,
        reference_frame=Body(name=PrefixedName(name=world_frame_name)),
        child_frame=camera_body,
    )
    return CameraObservation(
        camera=Camera(
            name=PrefixedName(name="camera"), root=camera_body, camera_model=model
        ),
        effective_camera_model=model,
        world_T_camera=world_T_camera,
        timestamp_nanoseconds=0,
    )


# %% answering


def answer_query(
    camera_observation: CameraObservation, *annotations: Annotation
) -> Query.Result:
    """
    Run a :class:`SeatQueryAnswerAnnotator` on a CAS holding the given annotations.

    :return: The answer it sent.
    """
    annotator = SeatQueryAnswerAnnotator()
    pipeline = Pipeline()
    pipeline.add_child(annotator)
    pipeline.setup()
    pipeline.cas.camera_observation = camera_observation
    pipeline.cas.annotations.extend(annotations)
    assert annotator.update() == Status.SUCCESS
    return Blackboard().get(BBIdentifier.QUERY_ANSWER)


def add_occupancy(seat: Annotation, occupied: bool) -> Annotation:
    """
    :return: The given seat, annotated with the given occupancy.
    """
    seat.annotations.append(OccupiedAnnotation(occupied=occupied))
    return seat


def test_every_seat_is_reported_with_furniture_and_state(
    camera_observation: CameraObservation, seat_distance: float
) -> None:
    """
    A free chair and an occupied sofa seat are reported with their furniture and state,
    and the text result counts the free seat.
    """
    chair = add_occupancy(located(seat(), x=0.5, z=seat_distance), occupied=False)
    sofa_seat = add_occupancy(seat_region(x=-0.5, z=seat_distance), occupied=True)

    answer = answer_query(camera_observation, chair, sofa_seat)

    assert [(designator.type, designator.attribute) for designator in answer.res] == [
        (SeatFurniture.CHAIR, [SeatState.FREE]),
        (SeatFurniture.SOFA, [SeatState.OCCUPIED]),
    ]
    assert answer.text_result == "1 free seats"


def test_seat_is_reported_in_world_frame(
    camera_observation: CameraObservation,
    camera_translation: list[float],
    world_frame_name: str,
    seat_distance: float,
) -> None:
    """
    The position of a seat is reported in the world frame of the camera observation.
    """
    chair = add_occupancy(located(seat(), x=0.5, z=seat_distance), occupied=False)
    camera_x, camera_y, camera_z = camera_translation

    (designator,) = answer_query(camera_observation, chair).res

    (pose,) = designator.pose
    assert pose.header.frame_id == world_frame_name
    assert [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] == (
        pytest.approx([camera_x + 0.5, camera_y, camera_z + seat_distance])
    )


def test_only_annotated_seats_are_reported(
    camera_observation: CameraObservation, seat_distance: float
) -> None:
    """
    Seats without an occupancy, such as those beyond reach, and humans are not reported.
    """
    unannotated_chair = located(seat(), x=0.0, z=seat_distance)
    person = located(human(HumanPosture.SITTING), x=0.0, z=seat_distance)

    answer = answer_query(camera_observation, unannotated_chair, person)

    assert list(answer.res) == []
    assert answer.text_result == "0 free seats"
