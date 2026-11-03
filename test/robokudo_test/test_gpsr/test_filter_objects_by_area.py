"""
Tests for filtering detections by their world position.
"""

from __future__ import annotations

# %% imports
import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.exceptions import InvalidAreaPose
from robokudo.annotators.gpsr.filter_objects_by_area import FilterObjectsByArea
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from ._gpsr_scene import GpsrScene


# %% world area
def test_area_filter_rejects_an_invalid_position_annotation() -> None:
    """
    Only position and pose annotations can be converted for area filtering.
    """
    annotator = FilterObjectsByArea()

    with pytest.raises(InvalidAreaPose):
        annotator._FilterObjectsByArea__transform(object())


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
    person = scene.add_human(position=[0.0, 0.0, 1.0])

    status = annotator.update()

    assert status == Status.SUCCESS
    assert cas.annotations == ([person] if is_inside else [])
