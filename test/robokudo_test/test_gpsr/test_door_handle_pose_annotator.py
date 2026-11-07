"""
Tests for estimating a grasp pose for a door handle from the depth image of the door.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.door_handle_pose_annotator import (
    DoorObjectClass,
    DoorHandlePoseAnnotator,
)
from robokudo.cas import CAS
from robokudo.annotators.query import GenerateQueryResult
from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.cv import Point2D, Rect
from robokudo.utils.transform import get_rotation_matrix_from_q
from typing_extensions import TYPE_CHECKING, Callable

from ._door_scene import DoorScene

if TYPE_CHECKING:
    from robokudo.types.scene import ObjectHypothesis

# %% scene and annotator


@pytest.fixture
def door_scene() -> DoorScene:
    """
    :return: A camera facing a door one and a half meters away.
    """
    return DoorScene()


@pytest.fixture
def handle_estimator() -> DoorHandlePoseAnnotator:
    """
    :return: A door handle annotator with its own default parameters.
    """
    return DoorHandlePoseAnnotator(descriptor=DoorHandlePoseAnnotator.Descriptor())


def horizontal_handle() -> Rect:
    """
    :return: The image region of a horizontal handle right of the image center.
    """
    return Rect(pos=Point2D(x=100, y=57), width=30, height=6)


def vertical_handle() -> Rect:
    """
    :return: The image region of a vertical handle right of the image center.
    """
    return Rect(pos=Point2D(x=110, y=40), width=6, height=30)


def handle_box() -> Rect:
    """
    :return: The detection box around the horizontal handle, with some door around it.
    """
    return DoorScene.detection_box_around(horizontal_handle())


def handle_axis(handle: ObjectHypothesis) -> np.ndarray:
    """
    :param handle: The handle with its pose.
    :return: The direction from the handle's pose toward its free end.
    """
    [pose] = CAS.filter_by_type(PoseAnnotation, handle.annotations)
    return get_rotation_matrix_from_q(np.array(pose.rotation))[:, 1]


# %% handles standing out of the door


def test_grasp_point_is_returned_to_the_query(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    The query answer identifies the grasp point and its pose source.
    """
    result_annotator = GenerateQueryResult()
    cas = door_scene.place_in_pipeline(result_annotator)
    cas.annotations.append(
        handle_estimator.create_grasp_hypothesis(np.array([0.0, 0.0, 1.0]), np.eye(3))
    )

    result_annotator.update()

    [object_designator] = door_scene.sent_answer().res
    assert object_designator.type == "grasp_point"
    assert list(object_designator.pose_source) == ["grasp_point"]


def test_handle_standing_out_of_the_door_gets_a_grasp_pose(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    Handle standing out of the door gets a grasp pose.
    """
    door_scene.place_in_pipeline(handle_estimator)
    door_scene.stand_out(horizontal_handle(), 0.05)
    handle = door_scene.add_detection(DoorObjectClass.HANDLE.value, handle_box())

    status = handle_estimator.update()

    assert status == Status.SUCCESS
    [handle_pose] = CAS.filter_by_type(PoseAnnotation, handle.annotations)
    assert handle_pose.source == DoorObjectClass.HANDLE.value
    assert handle_pose.translation[2] == pytest.approx(
        door_scene.door_distance - 0.05, abs=0.005
    )
    [grasp_point] = door_scene.grasp_points()
    assert [
        pose.source
        for pose in CAS.filter_by_type(PoseAnnotation, grasp_point.annotations)
    ] == [DoorObjectClass.GRASP_POINT.value]


def test_grasp_approaches_the_door_from_the_camera(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    Grasp approaches the door from the camera.
    """
    door_scene.place_in_pipeline(handle_estimator)
    door_scene.stand_out(horizontal_handle(), 0.05)
    handle = door_scene.add_detection(DoorObjectClass.HANDLE.value, handle_box())

    handle_estimator.update()

    [pose] = CAS.filter_by_type(PoseAnnotation, handle.annotations)
    approach = get_rotation_matrix_from_q(np.array(pose.rotation))[:, 0]
    np.testing.assert_allclose(approach, [0.0, 0.0, 1.0], atol=1e-3)


@pytest.mark.parametrize(
    "handle_region, axis_index",
    [
        (horizontal_handle(), 0),
        (vertical_handle(), 1),
    ],
    ids=["horizontal", "vertical"],
)
def test_handle_axis_follows_the_handle(
    door_scene: DoorScene,
    handle_estimator: DoorHandlePoseAnnotator,
    handle_region: Rect,
    axis_index: int,
) -> None:
    """
    Handle axis follows the handle.
    """
    door_scene.place_in_pipeline(handle_estimator)
    door_scene.stand_out(handle_region, 0.05)
    handle = door_scene.add_detection(
        DoorObjectClass.HANDLE.value, DoorScene.detection_box_around(handle_region)
    )

    handle_estimator.update()

    assert abs(handle_axis(handle)[axis_index]) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    "handle_region, across_index",
    [
        (Rect(pos=Point2D(x=100, y=20), width=30, height=6), 1),
        (vertical_handle(), 0),
    ],
    ids=["horizontal, high in the image", "vertical, right in the image"],
)
def test_grasp_pose_lies_on_the_middle_line_of_the_handle(
    door_scene: DoorScene,
    handle_estimator: DoorHandlePoseAnnotator,
    handle_region: Rect,
    across_index: int,
) -> None:
    """
    Grasp pose lies on the middle line of the handle.
    """
    door_scene.place_in_pipeline(handle_estimator)
    door_scene.stand_out(handle_region, 0.05)
    handle = door_scene.add_detection(
        DoorObjectClass.HANDLE.value, DoorScene.detection_box_around(handle_region)
    )

    handle_estimator.update()

    [pose] = CAS.filter_by_type(PoseAnnotation, handle.annotations)
    middle = door_scene.camera_position_of(
        handle_region, door_scene.door_distance - 0.05
    )
    assert pose.translation[across_index] == pytest.approx(
        middle[across_index], abs=1e-3
    )


def test_handle_axis_points_to_the_end_standing_out_farthest(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    Handle axis points to the end standing out farthest.
    """
    door_scene.place_in_pipeline(handle_estimator)
    region = horizontal_handle()
    door_scene.stand_out(region, np.linspace(0.04, 0.06, region.width))
    handle = door_scene.add_detection(DoorObjectClass.HANDLE.value, handle_box())

    handle_estimator.update()

    assert handle_axis(handle)[0] > 0.99


def test_handle_axis_of_an_even_handle_points_to_the_door_center(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    A handle standing out evenly has its axis pointing toward the door center.
    """
    door_scene.place_in_pipeline(handle_estimator)
    door_scene.stand_out(horizontal_handle(), 0.05)
    door_scene.add_detection(
        DoorObjectClass.DOOR.value, Rect(pos=Point2D(x=0, y=0), width=80, height=120)
    )
    handle = door_scene.add_detection(DoorObjectClass.HANDLE.value, handle_box())

    handle_estimator.update()

    assert handle_axis(handle)[0] < -0.99


# %% handles without grasp pose


@pytest.mark.parametrize(
    "standing_out",
    [
        lambda parameters: parameters.minimum_protrusion - 0.01,
        lambda parameters: parameters.maximum_protrusion + 0.01,
    ],
    ids=["too close to the door", "too far out of the door"],
)
def test_handle_outside_the_reach_of_the_gripper_is_dropped(
    door_scene: DoorScene,
    handle_estimator: DoorHandlePoseAnnotator,
    standing_out: Callable[[DoorHandlePoseAnnotator.Descriptor.Parameters], float],
) -> None:
    """
    Handle outside the reach of the gripper is dropped.
    """
    cas = door_scene.place_in_pipeline(handle_estimator)
    door_scene.stand_out(
        horizontal_handle(), standing_out(handle_estimator.descriptor.parameters)
    )
    door_scene.add_detection(DoorObjectClass.HANDLE.value, handle_box())

    handle_estimator.update()

    assert cas.annotations == []


def test_handle_with_too_few_points_is_dropped(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    Handle with too few points is dropped.
    """
    cas = door_scene.place_in_pipeline(handle_estimator)
    door_scene.stand_out(Rect(pos=Point2D(x=110, y=58), width=3, height=3), 0.05)
    door_scene.add_detection(DoorObjectClass.HANDLE.value, handle_box())

    handle_estimator.update()

    assert cas.annotations == []


def test_handle_without_depth_measurements_is_dropped(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    Handle without depth measurements is dropped.
    """
    cas = door_scene.place_in_pipeline(handle_estimator)
    door_scene.clear(handle_box())
    door_scene.add_detection(DoorObjectClass.HANDLE.value, handle_box())

    handle_estimator.update()

    assert cas.annotations == []


def test_dropping_can_be_turned_off(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    With dropping turned off, a handle without a pose stays in the CAS.
    """
    handle_estimator.descriptor.parameters.drop_handles_without_pose = False
    cas = door_scene.place_in_pipeline(handle_estimator)
    handle = door_scene.add_detection(DoorObjectClass.HANDLE.value, handle_box())

    handle_estimator.update()

    assert cas.annotations == [handle]
    assert CAS.filter_by_type(PoseAnnotation, handle.annotations) == []


# %% door plane


def test_door_gets_a_pose_on_its_plane(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    Door gets a pose on its plane.
    """
    door_scene.place_in_pipeline(handle_estimator)
    door = door_scene.add_detection(
        DoorObjectClass.DOOR.value, door_scene.whole_image()
    )

    status = handle_estimator.update()

    assert status == Status.SUCCESS
    [door_pose] = CAS.filter_by_type(PoseAnnotation, door.annotations)
    assert door_pose.source == DoorObjectClass.DOOR.value
    assert door_pose.translation[2] == pytest.approx(door_scene.door_distance)


def test_largest_of_several_doors_gets_the_pose(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    Largest of several doors gets the pose.
    """
    door_scene.place_in_pipeline(handle_estimator)
    small_door = door_scene.add_detection(
        DoorObjectClass.CABINET_DOOR.value,
        Rect(pos=Point2D(x=0, y=0), width=40, height=40),
    )
    large_door = door_scene.add_detection(
        DoorObjectClass.DOOR.value, Rect(pos=Point2D(x=40, y=0), width=120, height=120)
    )

    handle_estimator.update()

    assert CAS.filter_by_type(PoseAnnotation, small_door.annotations) == []
    [door_pose] = CAS.filter_by_type(PoseAnnotation, large_door.annotations)
    assert door_pose.source == DoorObjectClass.DOOR.value


def test_door_too_small_for_a_plane_falls_back_to_the_whole_image(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    A door detection too small for a plane falls back to the whole image for the handle
    pose.
    """
    door_scene.place_in_pipeline(handle_estimator)
    door_scene.stand_out(horizontal_handle(), 0.05)
    door_scene.add_detection(
        DoorObjectClass.DOOR.value, Rect(pos=Point2D(x=0, y=0), width=5, height=5)
    )
    handle = door_scene.add_detection(DoorObjectClass.HANDLE.value, handle_box())

    handle_estimator.update()

    [handle_pose] = CAS.filter_by_type(PoseAnnotation, handle.annotations)
    assert handle_pose.source == DoorObjectClass.HANDLE.value


def test_without_door_plane_nothing_is_estimated(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    Without door plane nothing is estimated.
    """
    cas = door_scene.place_in_pipeline(handle_estimator)
    door_scene.clear(door_scene.whole_image())
    handle = door_scene.add_detection(DoorObjectClass.HANDLE.value, handle_box())

    status = handle_estimator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == [handle]
    assert CAS.filter_by_type(PoseAnnotation, handle.annotations) == []


def test_door_beyond_the_valid_depth_has_no_plane() -> None:
    """
    Door beyond the valid depth has no plane.
    """
    handle_estimator = DoorHandlePoseAnnotator(
        descriptor=DoorHandlePoseAnnotator.Descriptor()
    )
    _, farthest_valid_depth = handle_estimator.descriptor.parameters.valid_depth_range
    door_scene = DoorScene(door_distance=farthest_valid_depth + 1.0)
    cas = door_scene.place_in_pipeline(handle_estimator)
    door = door_scene.add_detection(
        DoorObjectClass.DOOR.value, door_scene.whole_image()
    )

    handle_estimator.update()

    assert cas.annotations == [door]
    assert CAS.filter_by_type(PoseAnnotation, door.annotations) == []


def test_without_handle_only_the_door_is_posed(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    Without handle only the door is posed.
    """
    cas = door_scene.place_in_pipeline(handle_estimator)
    door = door_scene.add_detection(
        DoorObjectClass.DOOR.value, door_scene.whole_image()
    )

    handle_estimator.update()

    assert cas.annotations == [door]
    assert door_scene.grasp_points() == []


def test_detections_of_other_classes_are_left_alone(
    door_scene: DoorScene, handle_estimator: DoorHandlePoseAnnotator
) -> None:
    """
    Detections of other classes are left alone.
    """
    cas = door_scene.place_in_pipeline(handle_estimator)
    cup = door_scene.add_detection("cup", handle_box())

    handle_estimator.update()

    assert cas.annotations == [cup]
    assert cup.annotations == [Classification(classname="cup")]
