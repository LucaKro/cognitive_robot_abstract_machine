import numpy as np
import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import MoreThanOneBodyHeld, NothingHeld
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPoses
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import (
    Point3,
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures

GRASP_ROTATION_LEVER_ARM = 0.1
"""
How many meters of reach one radian of wrist rotation is worth in these tests.
"""

HELD_BODY_OFFSET = HomogeneousTransformationMatrix.from_xyz_rpy(0.01, 0.02, 0.03)
"""
Where the body a gripper holds sits relative to the tool frame.
"""


@pytest.fixture
def pr2_gripper(pr2_world_copy) -> EndEffector:
    """
    The left gripper of a PR2 standing in an otherwise empty world.
    """
    return pr2_world_copy.get_semantic_annotations_by_type(PR2)[0].left_arm.end_effector


@pytest.fixture
def graspable_box(pr2_world_copy) -> HasGraspPoses:
    """
    A box within the PR2's reach that offers the default ring of grasps.
    """
    body = Body(
        name=PrefixedName("graspable_box"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.2))]),
    )
    annotation = HasGraspPoses(root=body)
    with pr2_world_copy.modify_world():
        pr2_world_copy.add_connection(
            FixedConnection(
                parent=pr2_world_copy.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.6, z=0.9
                ),
            )
        )
        pr2_world_copy.add_semantic_annotation(annotation)
    return annotation


def hold_body(end_effector: EndEffector, name: str = "held_body") -> Body:
    """
    Attach a body below ``end_effector``'s tool frame, as grasping one does.

    :param end_effector: The gripper that should hold the body.
    :param name: The name of the body it should hold.
    :return: The body it now holds.
    """
    world = end_effector._world
    body = Body(name=PrefixedName(name))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=end_effector.tool_frame,
                child=body,
                parent_T_connection_expression=HELD_BODY_OFFSET,
            )
        )
    return body


# %% the direction a gripper approaches from


def test_front_facing_axis_is_the_grasp_frames_approach_direction(pr2_gripper):
    """
    Rotating the axis into the grasp frame has to yield the direction a grasp frame is
    approached along, which is its x-axis.
    """
    grasp_R_tool = RotationMatrix.from_quaternion(pr2_gripper.front_facing_orientation)

    approach_in_grasp_frame = grasp_R_tool @ pr2_gripper.front_facing_axis

    np.testing.assert_allclose(
        approach_in_grasp_frame.to_np()[:3], Vector3.X().to_np()[:3], atol=1e-9
    )


def test_front_facing_axis_follows_the_grippers_own_convention(pr2_gripper, tracy_world):
    """
    A PR2 gripper points along its tool frame's x-axis and Tracy's along its z-axis, so
    the very same grasp frame is approached along a different local axis.
    """
    tracy_gripper = tracy_world.get_semantic_annotations_by_type(Tracy)[
        0
    ].left_arm.end_effector

    np.testing.assert_allclose(
        pr2_gripper.front_facing_axis.to_np()[:3], [1, 0, 0], atol=1e-9
    )
    np.testing.assert_allclose(
        tracy_gripper.front_facing_axis.to_np()[:3], [0, 0, 1], atol=1e-9
    )


# %% tool frame goals


def test_tool_frame_goal_keeps_the_grasp_position(pr2_gripper, graspable_box):
    grasp = next(iter(graspable_box.grasp_poses()))

    goal = pr2_gripper.tool_frame_goal(grasp)

    np.testing.assert_allclose(goal.to_np()[:3, 3], grasp.to_np()[:3, 3], atol=1e-9)


def test_tool_frame_goal_applies_the_end_effectors_own_orientation(
    pr2_gripper, graspable_box
):
    """
    Two grippers pointing different ways must be sent different orientations for one
    and the same grasp.
    """
    grasp = next(iter(graspable_box.grasp_poses()))

    goal = pr2_gripper.tool_frame_goal(grasp)

    expected = grasp.to_rotation_matrix() @ RotationMatrix.from_quaternion(
        pr2_gripper.front_facing_orientation
    )
    np.testing.assert_allclose(
        goal.to_rotation_matrix().to_np(), expected.to_np(), atol=1e-9
    )


# %% the body a gripper holds


def test_an_empty_gripper_holds_nothing(pr2_gripper):
    assert pr2_gripper.held_body is None


def test_the_held_body_is_the_one_below_the_tool_frame(pr2_gripper):
    body = hold_body(pr2_gripper)

    assert pr2_gripper.held_body is body


def test_a_gripper_with_two_bodies_attached_holds_no_single_one(pr2_gripper):
    """
    Which of them the grasp is on cannot be answered, so it is not guessed at.
    """
    hold_body(pr2_gripper, name="first_body")
    hold_body(pr2_gripper, name="second_body")

    with pytest.raises(MoreThanOneBodyHeld):
        pr2_gripper.held_body


def test_the_grasp_of_an_empty_gripper_cannot_be_read(pr2_gripper):
    with pytest.raises(NothingHeld):
        pr2_gripper.held_body_T_grasp


def test_the_held_grasp_is_turned_the_way_the_gripper_faces(pr2_gripper):
    """
    The grasp is what the tool frame reached, so applying the gripper's own orientation
    to it has to lead back to the way the tool frame points.
    """
    body = hold_body(pr2_gripper)

    body_R_grasp = pr2_gripper.held_body_T_grasp.to_rotation_matrix()

    grasp_R_tool = RotationMatrix.from_quaternion(pr2_gripper.front_facing_orientation)
    body_T_tool = pr2_gripper._world.transform(
        pr2_gripper.tool_frame.global_transform, body
    )
    np.testing.assert_allclose(
        (body_R_grasp @ grasp_R_tool).to_np(),
        body_T_tool.to_rotation_matrix().to_np(),
        atol=1e-9,
    )


def test_the_held_grasp_is_the_offset_the_body_hangs_at(pr2_gripper):
    body = hold_body(pr2_gripper)

    np.testing.assert_allclose(
        pr2_gripper.held_body_T_grasp.to_np()[:3, 3],
        -HELD_BODY_OFFSET.to_np()[:3, 3],
        atol=1e-9,
    )


# %% ranking the grasps an object offers


def grasp_the_gripper_already_faces(
    end_effector: EndEffector, offset: Vector3, yaw: float = 0.0
) -> Pose:
    """
    A grasp frame whose tool frame goal is the gripper's own current orientation, so
    that everything :meth:`distance_to_grasp` reports is the offset it is given.

    :param end_effector: The gripper the grasp is aimed at.
    :param offset: Where the grasp sits relative to the tool frame, in world axes.
    :param yaw: How far the grasp is turned about the world z-axis on top of that.
    :return: The grasp frame, in the world's frame.
    """
    world = end_effector._world
    world_T_tool = end_effector.tool_frame.global_transform
    tool_R_grasp = RotationMatrix.from_quaternion(
        end_effector.front_facing_orientation
    ).inverse()
    world_R_grasp = (
        RotationMatrix.from_rpy(yaw=yaw)
        @ world_T_tool.to_rotation_matrix()
        @ tool_R_grasp
    )
    world_P_grasp = world_T_tool.to_position().to_np()[:3] + offset.to_np()[:3]
    return Pose(
        position=Point3.from_iterable(world_P_grasp),
        orientation=world_R_grasp.to_quaternion(),
        reference_frame=world.root,
    )


def test_a_grasp_the_gripper_already_holds_is_no_distance_away(pr2_gripper):
    """
    Nothing to travel and nothing to turn is a distance of zero, whatever lever arm the
    rotation is priced at.
    """
    grasp = grasp_the_gripper_already_faces(pr2_gripper, Vector3(0, 0, 0))

    assert pr2_gripper.distance_to_grasp(
        grasp, GRASP_ROTATION_LEVER_ARM
    ) == pytest.approx(0.0, abs=1e-9)


def test_distance_to_grasp_counts_the_way_to_the_grasp(pr2_gripper):
    step = 0.05

    distance = pr2_gripper.distance_to_grasp(
        grasp_the_gripper_already_faces(pr2_gripper, Vector3(step, 0, 0)),
        GRASP_ROTATION_LEVER_ARM,
    )

    assert distance == pytest.approx(step, abs=1e-9)


def test_distance_to_grasp_counts_the_wrist_rotation(pr2_gripper):
    """
    A grasp the gripper stands right at, but faces the wrong way round for, still costs
    the turn it has to make.
    """
    yaw = np.pi / 4

    distance = pr2_gripper.distance_to_grasp(
        grasp_the_gripper_already_faces(pr2_gripper, Vector3(0, 0, 0), yaw=yaw),
        GRASP_ROTATION_LEVER_ARM,
    )

    assert distance == pytest.approx(GRASP_ROTATION_LEVER_ARM * yaw, abs=1e-6)


def test_a_lever_arm_of_zero_prices_the_rotation_out(pr2_gripper):
    """
    Without a lever arm the ranking is the plain distance between the two positions.
    """
    turned = grasp_the_gripper_already_faces(
        pr2_gripper, Vector3(0, 0, 0), yaw=np.pi / 4
    )

    assert pr2_gripper.distance_to_grasp(turned, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_the_rotation_is_priced_even_when_no_lever_arm_is_given(pr2_gripper):
    """
    The default has to price the wrist turn, or a grasp behind the gripper's back would
    rank as close as one it faces.
    """
    turned = grasp_the_gripper_already_faces(
        pr2_gripper, Vector3(0, 0, 0), yaw=np.pi / 4
    )

    assert pr2_gripper.distance_to_grasp(turned) > 0


def test_grasp_poses_by_distance_offers_every_grasp_the_object_has(
    pr2_gripper, graspable_box
):
    ranked = pr2_gripper.grasp_poses_by_distance(
        graspable_box, GRASP_ROTATION_LEVER_ARM
    )

    offered = [pose.to_np() for pose in graspable_box.grasp_poses()]
    assert len(ranked) == len(offered)
    for pose in ranked:
        assert any(np.allclose(pose.to_np(), other, atol=1e-9) for other in offered)


def test_grasp_poses_by_distance_puts_the_closest_grasp_first(
    pr2_gripper, graspable_box
):
    ranked = pr2_gripper.grasp_poses_by_distance(
        graspable_box, GRASP_ROTATION_LEVER_ARM
    )

    distances = [
        pr2_gripper.distance_to_grasp(pose, GRASP_ROTATION_LEVER_ARM) for pose in ranked
    ]
    assert distances == sorted(distances)
