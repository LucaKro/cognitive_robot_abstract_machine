"""
Tests for marking every hypothesis that stands in the queried room.

The shared world is replaced by a world with two known rooms, so no world server has to
run.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

import pytest
import robokudo.annotators.gpsr.filter_objects_by_region
from py_trees.common import Status
from robokudo.annotators.gpsr.filter_objects_by_region import FilterObjectsByRegion
from robokudo.types.annotation import PoseAnnotation
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from typing_extensions import TYPE_CHECKING

from ._gpsr_scene import GpsrScene
from ._room_world import TwoRoomWorld

if TYPE_CHECKING:
    from rclpy.node import Node
    from semantic_digital_twin.world import World

# %% shared world stand-in


@dataclass
class CountingWorldSource:
    """
    Hands out a new world with two rooms on every fetch and counts the fetches, in place
    of the world server.
    """

    rooms: TwoRoomWorld = field(default_factory=TwoRoomWorld)
    """
    The rooms of the handed out world.
    """

    fetch_count: int = 0
    """
    How often the world was fetched.
    """

    def fetch(self, node: Node) -> World:
        """
        :param node: The ROS node the real fetch uses.
        :return: A new copy of the world with the two rooms.
        """
        self.fetch_count += 1
        return self.rooms.build()


@pytest.fixture
def world_source(monkeypatch: pytest.MonkeyPatch, node: Node) -> CountingWorldSource:
    """
    Make the region filter fetch its world from the stand-in and keep it unsynchronized.

    :param monkeypatch: Replaces the world fetching.
    :param node: The shared ROS node.
    :return: The stand-in.
    """
    source = CountingWorldSource()
    module = robokudo.annotators.gpsr.filter_objects_by_region
    monkeypatch.setattr(module, "fetch_world_from_service", source.fetch)
    monkeypatch.setattr(module, "WorldSynchronizer", lambda **kwargs: None)
    return source


# %% queried room


def test_object_in_the_queried_room_is_marked(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    Object in the queried room is marked.
    """
    scene.query.obj.location = world_source.rooms.kitchen_name
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    cup = scene.add_object(position=[2.0, 1.0, 0.0])

    status = annotator.update()

    assert status == Status.SUCCESS
    assert scene.locations_of(cup) == [world_source.rooms.kitchen_name]


def test_object_in_another_room_is_not_marked(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    Object in another room is not marked.
    """
    scene.query.obj.location = world_source.rooms.kitchen_name
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    cup = scene.add_object(position=[5.0, 1.0, 0.0])

    annotator.update()

    assert scene.locations_of(cup) == []


def test_human_in_the_queried_room_is_marked(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    Human in the queried room is marked.
    """
    scene.query.obj.location = world_source.rooms.office_name
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    human_hypothesis = scene.add_human(position=[5.0, 1.0, 0.0])

    annotator.update()

    assert scene.locations_of(human_hypothesis) == [world_source.rooms.office_name]


def test_pose_is_used_like_a_position(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    Pose is used like a position.
    """
    scene.query.obj.location = world_source.rooms.kitchen_name
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    cup = scene.add_object(
        PoseAnnotation(translation=[2.0, 1.0, 0.0], rotation=[0.0, 0.0, 0.0, 1.0])
    )

    annotator.update()

    assert scene.locations_of(cup) == [world_source.rooms.kitchen_name]


def test_object_on_the_room_border_is_in_the_room(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    Object on the room border is in the room.
    """
    scene.query.obj.location = world_source.rooms.kitchen_name
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    cup = scene.add_object(position=[0.0, 1.0, 0.0])

    annotator.update()

    assert scene.locations_of(cup) == [world_source.rooms.kitchen_name]


def test_height_does_not_decide_the_room(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    Height does not decide the room.
    """
    scene.query.obj.location = world_source.rooms.kitchen_name
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    cup_on_shelf = scene.add_object(position=[2.0, 1.0, 2.0])

    annotator.update()

    assert scene.locations_of(cup_on_shelf) == [world_source.rooms.kitchen_name]


def test_position_is_taken_from_the_camera_into_the_world(
    world_source: CountingWorldSource,
) -> None:
    """
    Position is taken from the camera into the world.
    """
    scene = GpsrScene(
        world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(x=3.0, y=0.0)
    )
    scene.query.obj.location = world_source.rooms.kitchen_name
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    cup_in_kitchen = scene.add_object(position=[0.5, 1.0, 0.0])
    cup_in_office = scene.add_object(position=[2.0, 1.0, 0.0])

    annotator.update()

    assert scene.locations_of(cup_in_kitchen) == [world_source.rooms.kitchen_name]
    assert scene.locations_of(cup_in_office) == []


def test_hypothesis_without_position_is_not_marked(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    Hypothesis without position is not marked.
    """
    scene.query.obj.location = world_source.rooms.kitchen_name
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    cup = scene.add_object()

    annotator.update()

    assert scene.locations_of(cup) == []


# %% no room to filter by


def test_unknown_room_marks_nothing(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    Unknown room marks nothing.
    """
    scene.query.obj.location = "garden"
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    cup = scene.add_object(position=[2.0, 1.0, 0.0])

    status = annotator.update()

    assert status == Status.SUCCESS
    assert scene.locations_of(cup) == []


def test_query_without_room_neither_marks_nor_fetches(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    Query without room neither marks nor fetches.
    """
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    cup = scene.add_object(position=[2.0, 1.0, 0.0])

    status = annotator.update()

    assert status == Status.SUCCESS
    assert scene.locations_of(cup) == []
    assert world_source.fetch_count == 0


def test_cas_without_query_neither_marks_nor_fetches(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    CAS without query neither marks nor fetches.
    """
    scene.query = None
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)
    cup = scene.add_object(position=[2.0, 1.0, 0.0])

    status = annotator.update()

    assert status == Status.SUCCESS
    assert scene.locations_of(cup) == []
    assert world_source.fetch_count == 0


# %% fetching the world


def test_world_is_fetched_once_for_several_queries(
    scene: GpsrScene, world_source: CountingWorldSource
) -> None:
    """
    World is fetched once for several queries.
    """
    scene.query.obj.location = world_source.rooms.kitchen_name
    annotator = FilterObjectsByRegion()
    scene.place_in_pipeline(annotator)

    annotator.update()
    annotator.update()

    assert world_source.fetch_count == 1
