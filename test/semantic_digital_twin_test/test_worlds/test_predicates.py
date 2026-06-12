from copy import deepcopy

import numpy as np

from semantic_digital_twin.reasoning.predicates import (
    contact,
    visible,
    Above,
    Below,
    LeftOf,
    RightOf,
    Behind,
    InFrontOf,
    is_body_in_region,
    occluding_bodies,
    is_supported_by,
    reachable,
    is_place_occupied,
)
from semantic_digital_twin.reasoning.robot_predicates import (
    robot_in_collision,
    robot_holds_body,
    blocking,
    is_body_in_gripper,
    bodies_in_gripper,
    is_pose_free_for_robot,
)
from semantic_digital_twin.robots.robot_parts import Camera, EndEffector
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.testing import *
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
    Color,
    BoundingBox,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.specs import BodySpec, RegionSpec
from semantic_digital_twin.world_description.world_entity import Body, Region


@pytest.fixture(scope="function")
def two_block_world():
    def make_body(name: str) -> Body:
        return BodySpec.box(name, Scale(1.0, 1.0, 1.0)).to_body()

    world = World()

    body_1 = make_body("body_1")
    body_2 = make_body("body_2")

    with world.modify_world():
        connection = FixedConnection(
            parent=body_1,
            child=body_2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=3, reference_frame=body_1
            ),
        )
        world.add_connection(connection)
    return body_1, body_2


def test_in_contact():
    w = World()

    b1 = BodySpec.box("b1", Scale(1.0, 1.0, 1.0), color=Color(1.0, 0.0, 0.0)).to_body()
    b2 = BodySpec.box(
        "b2",
        Scale(1.0, 1.0, 1.0),
        color=Color(0.0, 1.0, 0.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(0.9, 0, 0.0, 0, 0, 0),
    ).to_body()
    b3 = BodySpec.box(
        "b3",
        Scale(1.0, 1.0, 1.0),
        color=Color(0.0, 0.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(1.8, 0, 0.0, 0, 0, 0),
    ).to_body()

    with w.modify_world():
        w.add_kinematic_structure_entity(b1)
        w.add_kinematic_structure_entity(b2)
        w.add_kinematic_structure_entity(b3)
        w.add_connection(Connection6DoF.create_with_dofs(parent=b1, child=b2, world=w))
        w.add_connection(Connection6DoF.create_with_dofs(parent=b2, child=b3, world=w))
    assert contact(b1, b2)
    assert not contact(b1, b3)
    assert contact(b2, b3)


def test_robot_in_contact(pr2_world_copy: World):
    pr2 = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    body = BodySpec.box(
        "test_body",
        Scale(1.0, 1.0, 1.0),
        color=Color(1.0, 0.0, 0.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.5),
    ).spawn(pr2_world_copy, connection_type=Connection6DoF)

    # Ensure the call runs without raising
    assert robot_in_collision(pr2)

    body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        4, 0, 0.5, 0, 0, 0, pr2_world_copy.root
    )
    assert not robot_in_collision(pr2)


def test_get_visible_objects(pr2_world_copy: World):
    body = BodySpec.box(
        "test_body",
        Scale(1.0, 1.0, 1.0),
        color=Color(1.0, 0.0, 0.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0, z=1.0),
    ).spawn(pr2_world_copy, connection_type=Connection6DoF)

    camera = pr2_world_copy.get_semantic_annotations_by_type(Camera)[0]

    assert visible(camera, body)


def test_occluding_bodies(pr2_world_state_reset: World):
    world = deepcopy(pr2_world_state_reset)
    world.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0)
    )

    block = BodySpec(name="block", shapes=[Box(scale=Scale(1.0, 1.0, 1.0))])
    with world.modify_world():
        obstacle = block.spawn(
            world,
            name="obstacle",
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=3, z=0.8),
        )
        occluded_body = block.spawn(
            world,
            name="occluded_body",
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=10, z=0.5),
        )

    camera = world.get_semantic_annotations_by_type(Camera)[0]

    bodies = occluding_bodies(camera, occluded_body)
    assert obstacle in bodies
    assert camera not in bodies
    assert occluded_body not in bodies


def test_above_and_below(two_block_world):
    center, top = two_block_world

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=-3)
    assert Above(top.center_of_mass, center.center_of_mass, pov)()
    assert Below(center.center_of_mass, top.center_of_mass, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, yaw=np.pi)
    assert Above(top.center_of_mass, center.center_of_mass, pov)()
    assert Below(center.center_of_mass, top.center_of_mass, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, roll=np.pi)
    assert Above(center.center_of_mass, top.center_of_mass, pov)()
    assert Below(top.center_of_mass, center.center_of_mass, pov)()


def test_left_and_right(two_block_world):
    center, top = two_block_world

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, roll=np.pi / 2)
    assert LeftOf(top.center_of_mass, center.center_of_mass, pov)()
    assert RightOf(center.center_of_mass, top.center_of_mass, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, roll=-np.pi / 2)
    assert RightOf(top.center_of_mass, center.center_of_mass, pov)()
    assert LeftOf(center.center_of_mass, top.center_of_mass, pov)()


def test_behind_and_in_front_of(two_block_world):
    center, top = two_block_world

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(z=-5, pitch=np.pi / 2)
    assert Behind(top.center_of_mass, center.center_of_mass, pov)()
    assert InFrontOf(center.center_of_mass, top.center_of_mass, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(z=5, pitch=-np.pi / 2)
    assert InFrontOf(top.center_of_mass, center.center_of_mass, pov)()
    assert Behind(center.center_of_mass, top.center_of_mass, pov)()


def test_body_in_region(two_block_world):
    center, top = two_block_world
    region = RegionSpec(
        name="test_region", shapes=[Box(scale=Scale(1.0, 1.0, 1.0))]
    ).spawn(
        center._world,
        parent=center,
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.5),
    )
    assert is_body_in_region(center, region) == 0.5
    assert is_body_in_region(top, region) == 0.0


def test_supporting(two_block_world):
    center, top = two_block_world

    with center._world.modify_world():
        top.parent_connection.parent_T_connection_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=center, z=1.0)
        )
    assert is_supported_by(top, center)
    assert not is_supported_by(center, top)


def test_is_body_in_gripper(pr2_world_copy):
    pr2 = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]

    gripper = pr2_world_copy.get_semantic_annotations_by_type(EndEffector)

    left_gripper = (
        gripper[0]
        if LeftOf(
            gripper[0].root.center_of_mass,
            gripper[1].root.center_of_mass,
            pr2.root.global_transform,
        )()
        else gripper[1]
    )

    # Create krrood_test box between fingers
    test_box = BodySpec.box(
        "test_box", Scale(0.05, 0.01, 0.05), color=Color(1.0, 0.0, 0.0)
    ).to_body()

    # Calculate position between fingers
    finger1_pos = (
        left_gripper.finger.tip.collision.center_of_mass_in_world().to_vector3()
    )
    finger2_pos = (
        left_gripper.thumb.tip.collision.center_of_mass_in_world().to_vector3()
    )
    between_fingers = (finger1_pos + finger2_pos) / 2.0

    # Add box to world
    with pr2_world_copy.modify_world():
        root = pr2_world_copy.root
        connection = Connection6DoF.create_with_dofs(
            parent=root,
            child=test_box,
            world=pr2_world_copy,
        )
        pr2_world_copy.add_connection(connection)
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=between_fingers[0],
            y=between_fingers[1],
            z=between_fingers[2],
            reference_frame=root,
        )

    assert is_body_in_gripper(test_box, left_gripper) > 0
    assert robot_holds_body(pr2, test_box)
    connection.origin = HomogeneousTransformationMatrix()
    assert is_body_in_gripper(test_box, left_gripper) == 0


def test_reachable(pr2_world_state_reset, rclpy_node):
    pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

    tool_frame_T_reachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.2,
        y=0.3,
        reference_frame=pr2.left_arm.end_effector.tool_frame,
    )

    assert reachable(
        tool_frame_T_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )
    assert not blocking(
        tool_frame_T_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )
    tool_frame_T_unreachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=10, y=10, reference_frame=pr2.left_arm.end_effector.tool_frame
    )
    assert not reachable(
        tool_frame_T_unreachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )

    tool_frame_T_rotated_reachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.2,
        y=0.3,
        yaw=np.pi / 2,
        reference_frame=pr2.left_arm.end_effector.tool_frame,
    )
    assert reachable(
        tool_frame_T_rotated_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )

    tool_frame_T_rotated_unreachable_goal = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-0.2,
            y=0.3,
            yaw=-np.pi / 2,
            reference_frame=pr2.left_arm.end_effector.tool_frame,
        )
    )
    assert not reachable(
        tool_frame_T_rotated_unreachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )


def test_blocking(pr2_world_copy):
    pr2 = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    obstacle = BodySpec.box(
        "obstacle",
        Scale(3.0, 1.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, z=0.5),
    ).spawn(pr2_world_copy, connection_type=Connection6DoF)

    assert obstacle not in pr2.bodies
    assert robot_in_collision(pr2)

    tool_frame_T_reachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.2,
        y=0.3,
        reference_frame=pr2.left_arm.end_effector.tool_frame,
    )
    assert blocking(
        tool_frame_T_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )


def test_region_is_occupied(pr2_world_state_reset):
    view = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

    target_box = BoundingBox(0, 0, 0, 1, 1, 1, HomogeneousTransformationMatrix())
    assert not is_place_occupied(
        target_box,
        Pose.from_xyz_rpy(2.5, 2, 0, reference_frame=pr2_world_state_reset.root),
        pr2_world_state_reset,
    )

    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        3.5, 2.5, 0
    )
    pr2_world_state_reset.notify_state_change()

    assert is_place_occupied(target_box, view.root.global_pose, pr2_world_state_reset)

    assert not is_place_occupied(
        target_box,
        Pose.from_xyz_rpy(3.5, 2.5, 1, 0, reference_frame=pr2_world_state_reset.root),
        pr2_world_state_reset,
        view.bodies_with_collision,
    )


def test_is_pose_free_for_robot(pr2_apartment_state_reset):
    view = pr2_apartment_state_reset.get_semantic_annotations_by_type(PR2)[0]
    assert is_pose_free_for_robot(
        view,
        Pose.from_xyz_rpy(2, -2, 0, reference_frame=pr2_apartment_state_reset.root),
    )

    assert not is_pose_free_for_robot(
        view,
        Pose.from_xyz_rpy(3, 2, 0, reference_frame=pr2_apartment_state_reset.root),
    )

    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, -2, 0
    )

    assert is_pose_free_for_robot(
        view,
        Pose.from_xyz_rpy(2, -2, 0, reference_frame=pr2_apartment_state_reset.root),
    )

    assert is_pose_free_for_robot(
        view,
        Pose.from_xyz_rpy(2.1, -2.1, 0, reference_frame=pr2_apartment_state_reset.root),
    )


def test_is_pose_free_for_robot_with_robot_pose(pr2_apartment_state_reset):
    view = pr2_apartment_state_reset.get_semantic_annotations_by_type(PR2)[0]
    assert is_pose_free_for_robot(
        view,
        Pose.from_xyz_rpy(2, -2, 0, reference_frame=pr2_apartment_state_reset.root),
    )

    assert is_pose_free_for_robot(
        view,
        view.root.global_pose,
    )


def test_bodies_in_gripper(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    tcp = world.get_body_by_name("l_gripper_tool_frame")
    pr2 = world.get_semantic_annotations_by_type(PR2)[0]

    body = BodySpec.box("mock_milk", Scale(0.05, 0.05, 0.3)).spawn(world, parent=tcp)

    pr2.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, -2, 0
    )

    bodies = bodies_in_gripper(pr2.left_arm.end_effector)

    assert len(bodies) == 1
    assert bodies[0].name.name == "mock_milk"
    assert bodies[0] == body
