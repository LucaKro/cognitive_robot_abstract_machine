"""
A Unitree G1 moves parcels between two pallet stacks of the AWS RoboMaker small
warehouse.

Needs the ``aws_robomaker_small_warehouse_world`` package built in the workspace, since
the world and its meshes are read from its share directory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans import MoveJointsMotion
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.testing import start_visualization
from coraplex.view_manager import ViewManager
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import Body

# %% where everything stands in the warehouse

WORLD_URI = (
    "package://aws_robomaker_small_warehouse_world/worlds/no_roof_small_warehouse/"
    "no_roof_small_warehouse.world"
)
"""
The roofless variant of the warehouse, which can be looked into from above in RViz.
"""

PELVIS_HEIGHT_ABOVE_FLOOR = 0.7923
"""
How far the G1's pelvis stands above the floor with all of its leg joints at zero.

The pelvis is the robot's root, so its ``odom`` has to be lifted by this much for the
robot's feet to rest on the floor rather than sink through it.
"""

ROBOT_START_POSE = Pose.from_xyz_rpy(1.75, 7.41, PELVIS_HEIGHT_ABOVE_FLOOR)
"""
Where the robot starts, in the aisle south of the two pallet stacks.
"""

STANDING_DISTANCE = 0.6
"""
How far the robot stands from a pose, in meters, opposite its FRONT-facing side.

Within the G1's reach, and far enough from a pallet stack to leave its footprint free.
"""

CROSSING_POINT = Point3(1.75, 7.41, PELVIS_HEIGHT_ABOVE_FLOOR)
"""
Where the robot drives through on its way from one pallet stack to the other.

It navigates in a straight line, so it goes over this point in the open aisle instead of
through what stands between the two stacks.
"""

CARRYING_ARM = Arms.LEFT
"""
The arm the robot carries the parcels with.
"""

ROUND_TRIPS = 10
"""
How often the robot brings every parcel over to its place pose and back again.
"""

DELIVERY_TOLERANCE = 0.05
"""
How far a delivered parcel may sit from its place pose, in meters.
"""

FLOOR_CONTACT_TOLERANCE = 1e-3
"""
How far the robot's lowest collision geometry may sit off the floor, in meters.
"""

# %% the parcels the robot moves


@dataclass
class Parcel:
    """
    A box the robot transports, together with the two poses it travels between.
    """

    name: str
    """
    The name the parcel's body carries in the world.
    """

    scale: Scale
    """
    The extents of the box.
    """

    color: Color
    """
    The color the box is drawn in.
    """

    pick_pose: Pose
    """
    Where the parcel starts, and the direction the robot faces when it approaches it.
    """

    place_pose: Pose
    """
    Where the parcel is delivered to, and the direction the robot faces when it
    approaches it.
    """

    def to_specification(self) -> BodySpecification:
        """
        :return: The specification spawning the parcel at its pick pose.
        """
        return BodySpecification.box(
            self.name,
            self.scale,
            color=self.color,
            parent_T_self=self.pick_pose.to_homogeneous_matrix(),
        )

    def body_in(self, world: World) -> Body:
        """
        :param world: The world the parcel was spawned in.
        :return: The body of the parcel.
        """
        return world.get_body_by_name(self.name)


PARCELS = (
    Parcel(
        name="parcel1",
        scale=Scale(0.08, 0.08, 0.14),
        color=Color(0.85, 0.45, 0.1),
        pick_pose=Pose.from_xyz_rpy(-0.77, 5.61, 0.795, yaw=np.pi),
        place_pose=Pose.from_xyz_rpy(2.6, 7.7, 0.8),
    ),
    Parcel(
        name="parcel2",
        scale=Scale(0.08, 0.08, 0.1),
        color=Color(0.6, 0.45, 0.1),
        pick_pose=Pose.from_xyz_rpy(-0.77, 5.91, 0.775, yaw=np.pi),
        place_pose=Pose.from_xyz_rpy(2.6, 8.0, 0.78),
    ),
    Parcel(
        name="parcel3",
        scale=Scale(0.04, 0.04, 0.04),
        color=Color(0.75, 0.45, 0.1),
        pick_pose=Pose.from_xyz_rpy(-0.637, 4.38, 0.74, yaw=np.pi),
        place_pose=Pose.from_xyz_rpy(2.75, 9.0, 0.743),
    ),
    Parcel(
        name="parcel4",
        scale=Scale(0.06, 0.06, 0.14),
        color=Color(0.45, 0.45, 0.1),
        pick_pose=Pose.from_xyz_rpy(-0.637, 4.7, 0.788, yaw=np.pi),
        place_pose=Pose.from_xyz_rpy(2.75, 9.3, 0.793),
    ),
)
"""
The parcels standing on the pick stack when the demo starts.
"""


# %% building the world and the plan


def build_world() -> World:
    """
    :return: The warehouse with the G1 and the parcels in it.
    """
    return WorldSpecification.from_gazebo(
        WORLD_URI,
        robots=[
            RobotSpecification(
                semantic_annotation_type=UnitreeG1,
                world_T_odom=ROBOT_START_POSE.to_homogeneous_matrix(),
            )
        ],
        objects=[parcel.to_specification() for parcel in PARCELS],
    ).to_domain_object()


def in_world(pose: Pose, world: World) -> Pose:
    """
    :param pose: The pose to anchor, given in world coordinates.
    :param world: The world to express the pose in.
    :return: The same pose, referenced against the world's root.
    """
    return Pose(pose.to_position(), pose.to_quaternion(), reference_frame=world.root)


def standing_pose_in_front_of(pose: Pose, world: World) -> Pose:
    """
    :param pose: The pose the robot should approach from its FRONT-facing side.
    :param world: The world the pose is expressed in.
    :return: The pose the robot stands in to reach that pose with a FRONT grasp.
    """
    yaw = float(pose.yaw)
    return Pose.from_xyz_rpy(
        pose.x - STANDING_DISTANCE * np.cos(yaw),
        pose.y - STANDING_DISTANCE * np.sin(yaw),
        PELVIS_HEIGHT_ABOVE_FLOOR,
        yaw=yaw,
        reference_frame=world.root,
    )


def standing_pose_at(position: Point3, facing: Point3, world: World) -> Pose:
    """
    :param position: Where the robot stands.
    :param facing: The point the robot looks towards.
    :param world: The world the points are expressed in.
    :return: The pose the robot stands in at that position, turned towards the other
        point.
    """
    x, y = float(position.x), float(position.y)
    return Pose.from_xyz_rpy(
        x,
        y,
        float(position.z),
        yaw=float(np.arctan2(float(facing.y) - y, float(facing.x) - x)),
        reference_frame=world.root,
    )


def turned_around(pose: Pose, world: World) -> Pose:
    """
    :param pose: Where the robot stands.
    :param world: The world the pose is expressed in.
    :return: The same standing position, facing the opposite way.
    """
    return Pose.from_xyz_rpy(
        float(pose.x),
        float(pose.y),
        float(pose.z),
        yaw=float(pose.yaw) + np.pi,
        reference_frame=world.root,
    )


def straighten_torso(robot: UnitreeG1) -> MoveJointsMotion:
    """
    :param robot: The robot to straighten up.
    :return: The motion moving every torso joint back to zero.
    """
    return MoveJointsMotion(
        names=[connection.name for connection in robot.torso.active_connections],
        positions=[0.0] * len(robot.torso.active_connections),
    )


def build_transport_plan(
    world: World, robot: UnitreeG1, parcel: Parcel, source: Pose, target: Pose
) -> Plan:
    """
    :param world: The world the plan acts in.
    :param robot: The robot carrying out the plan.
    :param parcel: The parcel to carry.
    :param source: Where the parcel stands, in world coordinates.
    :param target: Where the parcel should end up, in world coordinates.
    :return: The plan carrying the parcel from one pallet stack to the other.
    """
    body = parcel.body_in(world)
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(CARRYING_ARM, robot),
    )
    context = Context(world=world, robot=robot, evaluate_conditions=False)
    pose_at_source = standing_pose_in_front_of(source, world)
    pose_at_target = standing_pose_in_front_of(target, world)

    return sequential(
        [
            ParkArmsAction(Arms.BOTH),
            NavigateAction(pose_at_source),
            PickUpAction(body, CARRYING_ARM, grasp),
            ParkArmsAction(Arms.BOTH),
            straighten_torso(robot),
            NavigateAction(turned_around(pose_at_source, world)),
            NavigateAction(
                standing_pose_at(CROSSING_POINT, pose_at_target.to_position(), world)
            ),
            NavigateAction(pose_at_target),
            PlaceAction(body, in_world(target, world), CARRYING_ARM),
            ParkArmsAction(Arms.BOTH),
            straighten_torso(robot),
            NavigateAction(turned_around(pose_at_target, world)),
        ],
        context=context,
    ).plan


def deliver_every_parcel(world: World, robot: UnitreeG1) -> None:
    """
    Brings every parcel from its pick pose over to its place pose.

    :param world: The world the parcels stand in.
    :param robot: The robot carrying the parcels.
    """
    for parcel in PARCELS:
        build_transport_plan(
            world, robot, parcel, parcel.pick_pose, parcel.place_pose
        ).perform()


def return_every_parcel(world: World, robot: UnitreeG1) -> None:
    """
    Brings every parcel from its place pose back to its pick pose.

    :param world: The world the parcels stand in.
    :param robot: The robot carrying the parcels.
    """
    for parcel in PARCELS:
        build_transport_plan(
            world, robot, parcel, parcel.place_pose, parcel.pick_pose
        ).perform()


def lowest_collision_point_of(robot: UnitreeG1, world: World) -> float:
    """
    :param robot: The robot to measure.
    :param world: The world the height is expressed in.
    :return: The height of the robot's lowest collision geometry above the world's floor.
    """
    return min(
        body.collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_box()
        .min_z
        for body in world.get_kinematic_structure_entities_of_branch(robot.root)
        if body.collision
    )


# %% running the demo

world = build_world()
robot = world.get_semantic_annotations_by_type(UnitreeG1)[0]

# Keeps PELVIS_HEIGHT_ABOVE_FLOOR honest: the robot has to stand on the floor rather than
# sink into it or hover above it.
assert abs(lowest_collision_point_of(robot, world)) < FLOOR_CONTACT_TOLERANCE

start_visualization(world)

with simulated_robot:
    for _ in range(ROUND_TRIPS):
        deliver_every_parcel(world, robot)
        return_every_parcel(world, robot)
    deliver_every_parcel(world, robot)

for parcel in PARCELS:
    delivered_pose = parcel.body_in(world).global_pose
    print(f"{parcel.name} delivered to {np.round(delivered_pose.to_position(), 3)}")
    print(f"Expected it at {np.round(parcel.place_pose.to_position(), 3)}")
    assert np.allclose(delivered_pose, parcel.place_pose, atol=DELIVERY_TOLERANCE)
