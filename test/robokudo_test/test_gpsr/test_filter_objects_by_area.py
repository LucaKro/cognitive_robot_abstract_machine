"""
Tests for filtering detections by their world position.
"""

from __future__ import annotations

# %% Imports
import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.exceptions import InvalidAreaPose
from robokudo.annotators.gpsr.filter_objects_by_area import FilterObjectsByArea
from robokudo.types.annotation import PositionAnnotation
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix

from ._gpsr_scene import GpsrScene


# %% world area
def test_area_filter_rejects_an_invalid_position_annotation() -> None:
    """
    Only position and pose annotations can be converted for area filtering.
    """
    annotator = FilterObjectsByArea()

    with pytest.raises(InvalidAreaPose):
        annotator._as_pose(object())


@pytest.mark.parametrize("camera_x, is_inside", [(3.0, True), (12.0, False)])
def test_filter_uses_world_position(camera_x: float, is_inside: bool) -> None:
    """
    A detection is retained according to its position in the world frame.
    """
    scene = GpsrScene(
        world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(x=camera_x)
    )
    annotator = FilterObjectsByArea()
    cas = scene.place_in_pipeline(annotator)
    human_hypothesis = scene.add_human(position=[0.0, 0.0, 1.0])

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == ([human_hypothesis] if is_inside else [])


def test_dynamically_configured_map_frame_uses_world_position() -> None:
    """
    Transform positions when the configured frame name is constructed at runtime.
    """
    scene = GpsrScene(
        world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(x=12.0)
    )
    annotator = FilterObjectsByArea()
    annotator.descriptor.parameters.frame = "".join(("m", "a", "p"))
    cas = scene.place_in_pipeline(annotator)
    scene.add_human(position=[0.0, 0.0, 1.0])

    assert annotator.update() == Status.SUCCESS
    assert cas.annotations == []


def test_person_with_two_positions_outside_area_is_removed_once() -> None:
    """
    Remove a person once when both position annotations are outside the area.
    """
    scene = GpsrScene(
        world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(x=12.0)
    )
    annotator = FilterObjectsByArea()
    cas = scene.place_in_pipeline(annotator)
    human_hypothesis = scene.add_human(position=[0.0, 0.0, 1.0])
    human_hypothesis.annotations.append(PositionAnnotation(translation=[0.0, 0.0, 1.0]))

    assert annotator.update() == Status.SUCCESS
    assert cas.annotations == []
