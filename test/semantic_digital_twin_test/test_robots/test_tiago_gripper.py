"""
Tests for the named joint states of the TIAGo's grippers.
"""

from copy import deepcopy

import numpy as np
import pytest

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.tiago import TiagoLeftGripper, TiagoRightGripper
from semantic_digital_twin.world import World

# %% opening and closing


@pytest.fixture
def tiago_world(_tiago_world_setup) -> World:
    return deepcopy(_tiago_world_setup)


def pad_distance_in_state(
    world: World, gripper: EndEffector, state: GripperState
) -> float:
    """
    :return: The distance between the gripper's two finger pads once the gripper is in
        the named state.
    """
    gripper.get_joint_state_by_type(state).apply_to(world)
    thumb, finger = (
        world.compute_forward_kinematics_np(world.root, tip)[:3, 3]
        for tip in (gripper.thumb.tip, gripper.finger.tip)
    )
    return float(np.linalg.norm(thumb - finger))


@pytest.mark.parametrize("gripper_type", [TiagoLeftGripper, TiagoRightGripper])
def test_closing_a_tiago_gripper_brings_its_pads_together(tiago_world, gripper_type):
    gripper = tiago_world.get_semantic_annotations_by_type(gripper_type)[0]

    open_distance = pad_distance_in_state(tiago_world, gripper, GripperState.OPEN)
    closed_distance = pad_distance_in_state(tiago_world, gripper, GripperState.CLOSE)

    assert closed_distance < open_distance
