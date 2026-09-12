import numpy as np
import pytest

from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


def _pr2_behind_an_odom(
    world_T_odom: HomogeneousTransformationMatrix,
) -> tuple[World, AbstractRobot]:
    """
    A world holding nothing but a PR2, reached through an odom frame at
    ``world_T_odom``.

    The specification path is what puts the odom between the world root and the robot,
    so the placement is exercised against the same chain a specification builds.
    """
    world = World.create_with_root_body("root")
    try:
        robot = RobotSpecification(
            semantic_annotation_type=PR2, world_T_odom=world_T_odom
        ).spawn(world)
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")
    return world, robot


# %% placing the root through a displaced odom

# The drive is an OmniDrive, which represents x, y and yaw only, so the odom is
# displaced within that plane. A z or roll offset would make these tests assert the
# drive's limits instead of the frame conversion.
_DISPLACED_ODOM = HomogeneousTransformationMatrix.from_xyz_rpy(
    0.5, 0.5, 0, yaw=np.pi / 2
)


def test_root_reaches_a_world_pose_through_a_displaced_odom():
    world, robot = _pr2_behind_an_odom(_DISPLACED_ODOM)
    target = Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)

    robot.set_root_pose(target)

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=1e-9
    )


def test_root_reaches_a_world_pose_through_an_undisplaced_odom():
    world, robot = _pr2_behind_an_odom(HomogeneousTransformationMatrix())
    target = Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)

    robot.set_root_pose(target)

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=1e-9
    )


# %% pose already expressed in the root connection's parent frame


def test_pose_in_the_root_connection_parent_frame_is_applied_unchanged():
    world, robot = _pr2_behind_an_odom(_DISPLACED_ODOM)
    connection = robot.root.parent_connection
    target = Pose.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=0.25, reference_frame=connection.parent
    )

    robot.set_root_pose(target)

    np.testing.assert_allclose(connection.origin.to_np(), target.to_np(), atol=1e-9)


# %% poses at the height the root drives at

_RAISED_ODOM = HomogeneousTransformationMatrix.from_xyz_rpy(0.5, 0.5, 0.034)
"""
An odom lifted onto a floor surface, which is the height the root then drives at.
"""


def test_pose_at_root_height_is_reached_exactly():
    """
    A pose raised onto the root's height is one the drive reproduces without loss,
    unlike the pose it was derived from.
    """
    world, robot = _pr2_behind_an_odom(_RAISED_ODOM)
    target = Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)

    raised = robot.pose_at_root_height(target)
    robot.set_root_pose(raised)

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), raised.to_np(), atol=1e-9
    )


def test_pose_at_root_height_changes_nothing_but_the_height():
    world, robot = _pr2_behind_an_odom(_RAISED_ODOM)
    target = Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)

    raised = robot.pose_at_root_height(target)

    np.testing.assert_allclose(raised.to_np()[:2, :], target.to_np()[:2, :], atol=1e-9)
    np.testing.assert_allclose(
        raised.to_position().z.to_np(), robot.root.global_pose.to_position().z.to_np()
    )


def test_pose_at_root_height_answers_a_tilted_frame_at_the_world_height():
    """
    A pose given in a frame that lies on its side is still moved onto the height the
    root stands at: height is measured against the world, not along whatever axis that
    frame happens to call z.
    """
    world, robot = _pr2_behind_an_odom(_RAISED_ODOM)
    lying_on_its_side = Body(name=PrefixedName("lying_on_its_side"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=lying_on_its_side,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.0, 2.0, 0.9, roll=np.pi / 2
                ),
            )
        )
    target = Pose.from_xyz_rpy(0.0, 0.0, 0.0, reference_frame=lying_on_its_side)

    raised = robot.pose_at_root_height(target)

    np.testing.assert_allclose(
        world.transform(raised, world.root).to_position().z.to_np(),
        robot.root.global_pose.to_position().z.to_np(),
        atol=1e-9,
    )


def test_pose_at_root_height_leaves_the_pose_it_was_given_alone():
    """
    The answer is a pose of its own: a caller's target must not be moved onto the base's
    height behind its back, least of all a motion's own goal.
    """
    world, robot = _pr2_behind_an_odom(_RAISED_ODOM)
    target = Pose.from_xyz_rpy(1.3, 2.0, 0.81, yaw=0.25, reference_frame=world.root)
    as_given = target.to_np().copy()

    robot.pose_at_root_height(target)

    np.testing.assert_allclose(target.to_np(), as_given, atol=1e-9)
