"""
Tests for answering a room query with everything known about the hypotheses in the room.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
from py_trees.common import Status
from robokudo.annotators.clip_annotator import CLIPAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefEntry, EntityType
from robokudo.annotators.gpsr.generate_all_annotations import GenerateAllQueryResult
from robokudo.annotators.gpsr.type_mapping_annotator import TypeMappingAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.types.annotation import (
    Classification,
    LocationAnnotation,
    PoseAnnotation,
    SemanticColor,
)
from robokudo.types.cv import BoundingBox3D
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from typing_extensions import TYPE_CHECKING

from ._gpsr_scene import GpsrScene

if TYPE_CHECKING:
    from geometry_msgs.msg import Quaternion

# %% rotated camera


def tilted_camera_scene() -> GpsrScene:
    """
    :return: A scene whose camera is turned about all three axes, so that every
        component of its orientation differs from the others.
    """
    return GpsrScene(
        world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(
            roll=0.3, pitch=0.5, yaw=0.9
        )
    )


def is_same_rotation(orientation: Quaternion, expected: np.ndarray) -> bool:
    """
    :param orientation: An answered orientation.
    :param expected: The expected orientation as *x*, *y*, *z*, *w*.
    :return: Whether both quaternions describe the same rotation.
    """
    answered = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
    return abs(np.dot(answered, expected)) == pytest.approx(1.0)


# %% which hypotheses are answered


def test_without_room_every_object_is_answered(scene: GpsrScene) -> None:
    """
    Without room every object is object_designator.
    """
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)
    scene.add_object(Classification(classname="cup"))
    scene.add_object(Classification(classname="bowl"))

    status = annotator.update()

    assert status == Status.SUCCESS
    assert [
        object_designator.type for object_designator in scene.sent_answer().res
    ] == ["cup", "bowl"]


def test_with_room_only_objects_marked_in_the_room_are_answered(
    scene: GpsrScene,
) -> None:
    """
    With room only objects marked in the room are object_designator.
    """
    scene.query.obj.location = "iai_kitchen"
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)
    scene.add_object(
        Classification(classname="cup"), LocationAnnotation(name="iai_kitchen")
    )
    scene.add_object(Classification(classname="bowl"))
    scene.add_object(
        Classification(classname="plate"), LocationAnnotation(name="office")
    )

    annotator.update()

    assert [
        object_designator.type for object_designator in scene.sent_answer().res
    ] == ["cup"]


def test_empty_scene_answers_with_nothing(scene: GpsrScene) -> None:
    """
    Empty scene answers with nothing.
    """
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)

    status = annotator.update()

    assert status == Status.SUCCESS
    assert list(scene.sent_answer().res) == []


# %% what is answered about a hypothesis


def test_clip_classes_are_attributes_and_the_last_other_class_is_the_type(
    scene: GpsrScene,
) -> None:
    """
    CLIP classes are attributes and the last other class is the type.
    """
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)
    scene.add_object(
        Classification(classname="cup", source=YoloAnnotator.__name__),
        Classification(classname="blue", source=CLIPAnnotator.__name__),
        Classification(classname="Mug", source=TypeMappingAnnotator.__name__),
    )

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert object_designator.type == "Mug"
    assert list(object_designator.attribute) == ["blue"]


def test_semantic_colors_are_answered(scene: GpsrScene) -> None:
    """
    Semantic colors are object_designator.
    """
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)
    scene.add_object(SemanticColor(color="red"), SemanticColor(color="white"))

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert list(object_designator.color) == ["red", "white"]


def test_bounding_box_gives_size_and_shape(scene: GpsrScene) -> None:
    """
    Bounding box gives size and shape.
    """
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)
    scene.add_object(BoundingBox3D(x_length=0.1, y_length=0.2, z_length=0.3))

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert object_designator.size == "x: 0.1, y: 0.2, z: 0.3"
    [shape] = object_designator.shape_size
    dimensions = shape.dimensions
    assert (dimensions.x, dimensions.y, dimensions.z) == (0.1, 0.2, 0.3)
    assert shape.radius == 0


def test_belief_id_is_answered(scene: GpsrScene) -> None:
    """
    Belief ID is object_designator.
    """
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)
    scene.add_object(BeliefEntry(4, EntityType.OBJECT, "cup"))

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert object_designator.uid == "4"


def test_position_is_answered_in_the_map() -> None:
    """
    Position is answered in the map.
    """
    scene = GpsrScene(
        world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(x=3.0, y=1.0)
    )
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)
    scene.add_object(position=[1.0, 0.0, 2.0])

    annotator.update()

    [object_designator] = scene.sent_answer().res
    [pose] = object_designator.pose
    position = pose.pose.position
    assert (position.x, position.y, position.z) == pytest.approx((4.0, 1.0, 2.0))
    orientation = pose.pose.orientation
    assert (orientation.x, orientation.y, orientation.z, orientation.w) == (
        pytest.approx((0.0, 0.0, 0.0, 1.0))
    )
    assert pose.header.frame_id == "map"


def test_pose_is_answered_in_the_map() -> None:
    """
    Pose is answered in the map.
    """
    scene = GpsrScene(
        world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(x=3.0, y=1.0)
    )
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)
    scene.add_object(
        PoseAnnotation(
            translation=[1.0, 0.0, 2.0],
            rotation=[0.0, 0.0, 1.0, 0.0],
            source=ClusterPoseBBAnnotator.__name__,
        )
    )

    annotator.update()

    [object_designator] = scene.sent_answer().res
    [pose] = object_designator.pose
    position = pose.pose.position
    assert (position.x, position.y, position.z) == pytest.approx((4.0, 1.0, 2.0))
    assert pose.header.frame_id == "map"
    assert list(object_designator.pose_source) == [ClusterPoseBBAnnotator.__name__]


def test_position_is_answered_with_the_orientation_of_the_camera() -> None:
    """
    A position without orientation is answered with the orientation of the camera.
    """
    scene = tilted_camera_scene()
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)
    scene.add_object(position=[1.0, 0.0, 2.0])

    annotator.update()

    [object_designator] = scene.sent_answer().res
    [pose] = object_designator.pose
    assert is_same_rotation(
        pose.pose.orientation, scene.world_T_camera.quaternion.to_np()
    )


def test_pose_orientation_is_answered_in_the_map() -> None:
    """
    Pose orientation is answered in the map.
    """
    scene = tilted_camera_scene()
    annotator = GenerateAllQueryResult()
    scene.place_in_pipeline(annotator)
    camera_T_object = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=1.0, pos_z=2.0, quat_z=1.0, quat_w=0.0
    )
    scene.add_object(
        PoseAnnotation(
            translation=[1.0, 0.0, 2.0],
            rotation=[0.0, 0.0, 1.0, 0.0],
            source=ClusterPoseBBAnnotator.__name__,
        )
    )

    annotator.update()

    [object_designator] = scene.sent_answer().res
    [pose] = object_designator.pose
    assert is_same_rotation(
        pose.pose.orientation,
        (scene.world_T_camera @ camera_T_object).quaternion.to_np(),
    )
