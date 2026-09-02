import numpy as np
import pytest

from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures

BOX_SCALE = Scale(0.1, 0.2, 0.3)
"""
Extents of the box the approach sequence has to clear.
"""


@pytest.fixture
def boxed_pr2_world(mutable_simple_pr2_world):
    """
    A PR2 next to a box of known extents, standing a meter above the world root.
    """
    world, robot, context = mutable_simple_pr2_world
    with world.modify_world():
        box = Body(
            name=PrefixedName("approach_box"),
            collision=ShapeCollection([Box(scale=BOX_SCALE)]),
        )
        connection = Connection6DoF.create_with_dofs(world, world.root, box)
        world.add_connection(connection)
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            1, 0, 1, reference_frame=world.root
        )
    return world, robot, box


def grasp_at_origin(body) -> Pose:
    """
    :param body: The body to grasp.
    :return: A grasp frame at the body's origin, approaching along the body's x-axis.
    """
    return Pose(reference_frame=body)


# %% tool frame goals


def test_tool_frame_goal_keeps_the_grasp_position(boxed_pr2_world):
    _, robot, box = boxed_pr2_world
    grasp = grasp_at_origin(box)
    goal = HasApproachesGraspPoses().tool_frame_goal(grasp, robot.left_arm.end_effector)
    np.testing.assert_allclose(goal.to_np()[:3, 3], grasp.to_np()[:3, 3], atol=1e-9)


def test_tool_frame_goal_applies_the_end_effectors_own_orientation(boxed_pr2_world):
    """
    Two grippers pointing different ways must be sent different orientations for one
    and the same grasp.
    """
    _, robot, box = boxed_pr2_world
    end_effector = robot.left_arm.end_effector
    grasp = grasp_at_origin(box)

    goal = HasApproachesGraspPoses().tool_frame_goal(grasp, end_effector)

    expected = grasp.to_rotation_matrix() @ RotationMatrix.from_quaternion(
        end_effector.front_facing_orientation
    )
    np.testing.assert_allclose(
        goal.to_rotation_matrix().to_np(), expected.to_np(), atol=1e-9
    )


# %% approach sequences


def test_pre_grasp_pose_clears_the_body_it_grasps(boxed_pr2_world):
    """
    A grasp at the body's own origin has to be approached from outside the body, so the
    pre-grasp pose stands off by half of it plus the clearance.
    """
    _, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()

    origin_grasp = grasp_at_origin(box)
    pre_grasp, grasp, _ = action.grasp_pose_sequence(
        origin_grasp,
        robot.left_arm.end_effector,
        action.grasp_in_body_frame(origin_grasp, box),
    )

    expected_standoff = BOX_SCALE.x / 2 + action.approach_clearance
    np.testing.assert_allclose(
        pre_grasp.to_np()[:3, 3], [-expected_standoff, 0, 0], atol=1e-9
    )


def test_pre_grasp_pose_of_a_surface_grasp_only_adds_the_clearance(boxed_pr2_world):
    """
    A grasp already on the body's surface, approached from outside it, needs nothing
    beyond the clearance -- this is what lets a bowl be grasped at its rim.
    """
    _, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()
    surface_grasp = Pose(
        position=Vector3(0, 0, BOX_SCALE.z / 2).to_point3(),
        orientation=RotationMatrix.from_vectors(
            x=Vector3.NEGATIVE_Z(), y=Vector3.X()
        ).to_quaternion(),
        reference_frame=box,
    )

    pre_grasp, _, _ = action.grasp_pose_sequence(
        surface_grasp,
        robot.left_arm.end_effector,
        action.grasp_in_body_frame(surface_grasp, box),
    )

    np.testing.assert_allclose(
        pre_grasp.to_np()[:3, 3],
        [0, 0, BOX_SCALE.z / 2 + action.approach_clearance],
        atol=1e-9,
    )


def test_grasp_pose_is_the_middle_of_the_sequence(boxed_pr2_world):
    _, robot, box = boxed_pr2_world
    end_effector = robot.left_arm.end_effector
    grasp = grasp_at_origin(box)

    approach = HasApproachesGraspPoses()
    _, middle, _ = approach.grasp_pose_sequence(
        grasp, end_effector, approach.grasp_in_body_frame(grasp, box)
    )

    np.testing.assert_allclose(
        middle.to_np(),
        HasApproachesGraspPoses().tool_frame_goal(grasp, end_effector).to_np(),
        atol=1e-9,
    )


def test_retreat_pose_rises_along_the_world_z_axis(boxed_pr2_world):
    world, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()
    grasp = grasp_at_origin(box)

    _, _, retreat = action.grasp_pose_sequence(
        grasp,
        robot.left_arm.end_effector,
        action.grasp_in_body_frame(grasp, box),
    )

    world_P_grasp = world.transform(grasp.to_homogeneous_matrix(), world.root).to_np()
    world_P_retreat = world.transform(
        retreat.to_homogeneous_matrix(), world.root
    ).to_np()
    np.testing.assert_allclose(
        world_P_retreat[:3, 3] - world_P_grasp[:3, 3],
        [0, 0, action.retreat_distance],
        atol=1e-9,
    )


def test_retreat_pose_keeps_the_grasp_orientation(boxed_pr2_world):
    _, robot, box = boxed_pr2_world

    approach = HasApproachesGraspPoses()
    origin_grasp = grasp_at_origin(box)
    _, grasp, retreat = approach.grasp_pose_sequence(
        origin_grasp,
        robot.left_arm.end_effector,
        approach.grasp_in_body_frame(origin_grasp, box),
    )

    np.testing.assert_allclose(
        retreat.to_rotation_matrix().to_np(),
        grasp.to_rotation_matrix().to_np(),
        atol=1e-9,
    )


def test_reversing_turns_the_grasp_into_a_release(boxed_pr2_world):
    _, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()
    grasp = grasp_at_origin(box)

    body_T_grasp = action.grasp_in_body_frame(grasp, box)
    forward = action.grasp_pose_sequence(
        grasp, robot.left_arm.end_effector, body_T_grasp
    )
    backward = action.grasp_pose_sequence(
        grasp, robot.left_arm.end_effector, body_T_grasp, reverse=True
    )

    for expected, actual in zip(reversed(forward), backward):
        np.testing.assert_allclose(expected.to_np(), actual.to_np(), atol=1e-9)


def test_sequence_without_a_body_stands_off_by_the_clearance_alone(boxed_pr2_world):
    _, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()

    pre_grasp, _, _ = action.grasp_pose_sequence(
        grasp_at_origin(box), robot.left_arm.end_effector
    )

    np.testing.assert_allclose(
        pre_grasp.to_np()[:3, 3], [-action.approach_clearance, 0, 0], atol=1e-9
    )


# %% gripper conventions


def test_approach_axis_follows_the_grippers_own_convention(
    boxed_pr2_world, tracy_world
):
    """
    A PR2 gripper points along its tool frame's x-axis and Tracy's along its z-axis, so
    the very same grasp frame has to be approached along a different local axis.
    """
    _, pr2, _ = boxed_pr2_world
    tracy = tracy_world.get_semantic_annotations_by_type(Tracy)[0]

    np.testing.assert_allclose(
        HasApproachesGraspPoses._approach_axis_in_tool_frame(pr2.left_arm.end_effector),
        [1, 0, 0],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        HasApproachesGraspPoses._approach_axis_in_tool_frame(tracy.left_arm.end_effector),
        [0, 0, 1],
        atol=1e-9,
    )
