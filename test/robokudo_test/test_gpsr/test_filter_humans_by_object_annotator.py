"""
Tests for keeping only the persons near the piece of furniture a query names.
"""

from __future__ import annotations

# %% Imports
import math
from dataclasses import dataclass

import numpy as np
import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.filter_humans_by_object_annotator import (
    FilterHumansByObjectAnnotator,
)
from robokudo.types.annotation import Classification
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point3

from ._gpsr_scene import GpsrScene

# %% annotator under test


def furniture_filter(
    target_type: str | None = "chair",
) -> FilterHumansByObjectAnnotator:
    """
    :param target_type: The furniture to look for.
    :return: A furniture filter with its own default parameters, looking for the given
        furniture.
    """
    annotator = FilterHumansByObjectAnnotator(
        descriptor=FilterHumansByObjectAnnotator.Descriptor()
    )
    annotator.descriptor.parameters.target_type = target_type
    return annotator


def box_size(annotator: FilterHumansByObjectAnnotator) -> list[float]:
    """
    :param annotator: The filter with the search box.
    :return: The size of the region searched around a piece of furniture.
    """
    parameters = annotator.descriptor.parameters
    return [parameters.box_x, parameters.box_y, parameters.box_z]


# %% search region


@pytest.mark.parametrize(
    "fraction_of_half_box, inside",
    [(0.5, True), (1.0, True), (1.1, False)],
    ids=["inside", "on the border", "outside"],
)
def test_region_without_direction_is_a_box_around_the_furniture(
    fraction_of_half_box: float, inside: bool
) -> None:
    """
    Region without direction is a box around the furniture.
    """
    annotator = furniture_filter()
    half_box = np.array(box_size(annotator)) / 2.0

    assert (
        bool(
            annotator.is_inside_box(
                Point3.from_iterable(half_box * fraction_of_half_box),
                Point3(0.0, 0.0, 0.0),
                box_size(annotator),
            )
        )
        is inside
    )


@pytest.mark.parametrize(
    "direction, offset, inside",
    [
        ("right", [1.0, 0.0, 0.0], True),
        ("right", [-1.0, 0.0, 0.0], False),
        ("left", [-1.0, 0.0, 0.0], True),
        ("left", [1.0, 0.0, 0.0], False),
        ("back", [0.0, 1.0, 0.0], True),
        ("back", [0.0, -1.0, 0.0], False),
        ("front", [0.0, -1.0, 0.0], True),
        ("front", [0.0, 1.0, 0.0], False),
    ],
)
def test_direction_keeps_one_side_of_the_furniture(
    direction: str, offset: list[float], inside: bool
) -> None:
    """
    Direction keeps one side of the furniture.
    """
    annotator = furniture_filter()
    right = np.array([1.0, 0.0, 0.0])
    forward = np.array([0.0, 1.0, 0.0])

    assert (
        annotator.is_inside_box(
            Point3.from_iterable(offset),
            Point3(0.0, 0.0, 0.0),
            box_size(annotator),
            direction,
            right,
            forward,
        )
        is inside
    )


@dataclass
class FurnitureSide:
    """
    A side of the furniture a query asks for, in a ground frame whose *x* axis is the
    robot's right, *y* axis points behind the furniture and *z* axis points up.
    """

    direction: str
    """
    The direction the query names.
    """

    outward: np.ndarray
    """
    Unit vector from the furniture out to this side.
    """

    def sideways(self) -> np.ndarray:
        """
        :return: Unit vector along the ground across this side.
        """
        return np.abs(self.outward[[1, 0, 2]])

    def reach(self, annotator: FilterHumansByObjectAnnotator) -> float:
        """
        :param annotator: The filter with the search box.
        :return: How far the region of this side reaches out from the furniture.
        """
        parameters = annotator.descriptor.parameters
        return parameters.box_x if self.outward[0] else parameters.box_y

    def half_width(self, annotator: FilterHumansByObjectAnnotator) -> float:
        """
        :param annotator: The filter with the search box.
        :return: How far the region of this side reaches sideways from its middle.
        """
        parameters = annotator.descriptor.parameters
        return (parameters.box_y if self.outward[0] else parameters.box_x) / 2.0

    def contains(
        self, annotator: FilterHumansByObjectAnnotator, offset: np.ndarray
    ) -> bool:
        """
        :param annotator: The filter with the search box.
        :param offset: Position of the person relative to the furniture.
        :return: Whether the annotator finds the person in the region of this side.
        """
        return annotator.is_inside_box(
            Point3.from_iterable(offset),
            Point3(0.0, 0.0, 0.0),
            box_size(annotator),
            self.direction,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )


@pytest.fixture(
    params=[
        FurnitureSide("right", np.array([1.0, 0.0, 0.0])),
        FurnitureSide("left", np.array([-1.0, 0.0, 0.0])),
        FurnitureSide("back", np.array([0.0, 1.0, 0.0])),
        FurnitureSide("front", np.array([0.0, -1.0, 0.0])),
    ],
    ids=["right", "left", "back", "front"],
)
def furniture_side(request: pytest.FixtureRequest) -> FurnitureSide:
    """
    :param request: Holds the side of this run.
    :return: Each side of the furniture in turn.
    """
    return request.param


def test_side_region_reaches_one_box_length_out(
    furniture_side: FurnitureSide,
) -> None:
    """
    Side region reaches one box length out.
    """
    annotator = furniture_filter()
    offset = furniture_side.outward * furniture_side.reach(annotator)

    assert furniture_side.contains(annotator, offset)


def test_side_region_starts_level_with_the_furniture(
    furniture_side: FurnitureSide,
) -> None:
    """
    Side region starts level with the furniture.
    """
    annotator = furniture_filter()
    offset = furniture_side.sideways() * furniture_side.half_width(annotator) / 2.0

    assert furniture_side.contains(annotator, offset)


def test_side_region_ends_one_box_length_out(furniture_side: FurnitureSide) -> None:
    """
    Side region ends one box length out.
    """
    annotator = furniture_filter()
    offset = furniture_side.outward * (furniture_side.reach(annotator) + 0.05)

    assert not furniture_side.contains(annotator, offset)


def test_side_region_reaches_half_a_box_width_sideways(
    furniture_side: FurnitureSide,
) -> None:
    """
    Side region reaches half a box width sideways.
    """
    annotator = furniture_filter()
    offset = furniture_side.outward * furniture_side.reach(
        annotator
    ) / 2.0 + furniture_side.sideways() * furniture_side.half_width(annotator)

    assert furniture_side.contains(annotator, offset)


def test_side_region_ends_half_a_box_width_sideways(
    furniture_side: FurnitureSide,
) -> None:
    """
    Side region ends half a box width sideways.
    """
    annotator = furniture_filter()
    offset = furniture_side.outward * furniture_side.reach(
        annotator
    ) / 2.0 + furniture_side.sideways() * (furniture_side.half_width(annotator) + 0.05)

    assert not furniture_side.contains(annotator, offset)


def test_side_region_reaches_half_a_box_height_up(
    furniture_side: FurnitureSide,
) -> None:
    """
    Side region reaches half a box height up.
    """
    annotator = furniture_filter()
    up = np.array([0.0, 0.0, 1.0])
    offset = (
        furniture_side.outward * furniture_side.reach(annotator) / 2.0
        + up * annotator.descriptor.parameters.box_z / 2.0
    )

    assert furniture_side.contains(annotator, offset)


def test_direction_keeps_the_height_limit() -> None:
    """
    Direction keeps the height limit.
    """
    annotator = furniture_filter()
    parameters = annotator.descriptor.parameters

    assert not annotator.is_inside_box(
        Point3(parameters.box_x / 2.0, 0.0, parameters.box_z / 2.0 + 0.05),
        Point3(0.0, 0.0, 0.0),
        box_size(annotator),
        "right",
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )


def test_unknown_direction_searches_the_whole_box() -> None:
    """
    A direction that is no side searches the whole box.
    """
    annotator = furniture_filter()

    assert annotator.is_inside_box(
        Point3(-0.5, 0.0, 0.0),
        Point3(0.0, 0.0, 0.0),
        box_size(annotator),
        "above",
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )


@pytest.mark.parametrize(
    "fraction_of_half_width, inside",
    [(1.0, True), (1.1, False)],
    ids=["on the border", "outside"],
)
def test_unknown_direction_ends_at_the_border_of_the_whole_box(
    fraction_of_half_width: float, inside: bool
) -> None:
    """
    Unknown direction ends at the border of the whole box.
    """
    annotator = furniture_filter()
    half_box = np.array(box_size(annotator)) / 2.0

    assert (
        bool(
            annotator.is_inside_box(
                Point3.from_iterable(half_box * [fraction_of_half_width, 1.0, 1.0]),
                Point3(0.0, 0.0, 0.0),
                box_size(annotator),
                "above",
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )
        )
        is inside
    )


# %% persons near furniture


def test_person_near_the_furniture_is_kept_and_marked(scene: GpsrScene) -> None:
    """
    Person near the furniture is kept and marked.
    """
    annotator = furniture_filter("chair")
    cas = scene.place_in_pipeline(annotator)
    chair = scene.add_object(
        Classification(classname="chair"), position=[0.0, 0.0, 2.0]
    )
    human_hypothesis = scene.add_human(position=[0.5, 0.0, 2.0])

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == [chair, human_hypothesis]
    assert scene.locations_of(human_hypothesis) == ["chair"]


def test_person_far_from_the_furniture_is_removed(scene: GpsrScene) -> None:
    """
    Person far from the furniture is removed.
    """
    annotator = furniture_filter()
    cas = scene.place_in_pipeline(annotator)
    chair = scene.add_object(position=[0.0, 0.0, 2.0])
    scene.add_human(position=[3.0, 0.0, 2.0])

    annotator.update()

    assert cas.annotations == [chair]


def test_person_without_position_is_removed(scene: GpsrScene) -> None:
    """
    Person without position is removed.
    """
    annotator = furniture_filter()
    cas = scene.place_in_pipeline(annotator)
    chair = scene.add_object(position=[0.0, 0.0, 2.0])
    scene.add_human()

    annotator.update()

    assert cas.annotations == [chair]


def test_person_near_any_piece_of_furniture_is_kept(scene: GpsrScene) -> None:
    """
    Person near any piece of furniture is kept.
    """
    annotator = furniture_filter()
    cas = scene.place_in_pipeline(annotator)
    far_chair = scene.add_object(position=[5.0, 0.0, 2.0])
    near_chair = scene.add_object(position=[0.0, 0.0, 2.0])
    human_hypothesis = scene.add_human(position=[0.5, 0.0, 2.0])

    annotator.update()

    assert cas.annotations == [far_chair, near_chair, human_hypothesis]


# %% no furniture to search around


def test_without_furniture_every_person_is_removed(scene: GpsrScene) -> None:
    """
    Without furniture every person is removed.
    """
    annotator = furniture_filter()
    cas = scene.place_in_pipeline(annotator)
    scene.add_human(position=[0.5, 0.0, 2.0])
    scene.add_human(position=[1.5, 0.0, 2.0])

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == []


def test_furniture_without_position_counts_as_no_furniture(scene: GpsrScene) -> None:
    """
    Furniture without position counts as no furniture.
    """
    annotator = furniture_filter()
    cas = scene.place_in_pipeline(annotator)
    chair = scene.add_object()
    scene.add_human(position=[0.5, 0.0, 2.0])

    annotator.update()

    assert cas.annotations == [chair]


def test_without_furniture_type_nobody_is_removed(scene: GpsrScene) -> None:
    """
    Without furniture type nobody is removed.
    """
    annotator = furniture_filter(target_type=None)
    cas = scene.place_in_pipeline(annotator)
    human_hypothesis = scene.add_human(position=[3.0, 0.0, 2.0])

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == [human_hypothesis]


# %% sides of the furniture


def test_person_on_the_queried_side_of_the_furniture_is_kept() -> None:
    """
    Person on the queried side of the furniture is kept.
    """
    # The camera looks along the world x axis with the image right along the world -y
    # axis, as a robot's head camera does.
    scene = GpsrScene(
        world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(
            roll=-math.pi / 2, yaw=-math.pi / 2
        )
    )
    annotator = furniture_filter()
    annotator.descriptor.parameters.direction = "right"
    cas = scene.place_in_pipeline(annotator)
    chair = scene.add_object(position=[0.0, 0.0, 2.0])
    on_the_right = scene.add_human(position=[1.0, 0.0, 2.0])
    scene.add_human(position=[-1.0, 0.0, 2.0])

    annotator.update()

    assert cas.annotations == [chair, on_the_right]


def test_camera_looking_straight_up_ignores_the_side(scene: GpsrScene) -> None:
    """
    With the camera looking straight up, the queried side is ignored.
    """
    annotator = furniture_filter()
    annotator.descriptor.parameters.direction = "right"
    cas = scene.place_in_pipeline(annotator)
    chair = scene.add_object(position=[0.0, 0.0, 2.0])
    on_the_left = scene.add_human(position=[-0.5, 0.0, 2.0])

    annotator.update()

    assert cas.annotations == [chair, on_the_left]
