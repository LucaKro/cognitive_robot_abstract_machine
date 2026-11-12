"""
The HSR picks a milk carton up off the floor of the kitchen environment and drops it into
the trash can, which the demo moves into the living room next to the sofa. The run is a
kinematic simulation that can be followed in RViz2 on ``/semworld/viz_marker``.

Run it from this folder, with ROS 2 and the ROS workspace sourced::

    python throw_trash_rviz_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from coraplex.datastructures.enums import DetectionTechnique
from coraplex.demonstrations import RobotDemonstration
from coraplex.locations.locations import ReachabilityLocation
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    PlanExecutor,
)
from coraplex.plans.plan_transformation import PlanTransformation
from coraplex.querying.grasp_closeness import GraspsByCloseness
from coraplex.robot_plans.actions.composite.carrying import CarryAction
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
)
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import (
    FaceAtAction,
    LookAtAction,
    NavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from cramph.composites import Parallel, Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.executor import SimulationPacer
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match
from krrood.patterns.path_enumeration import PathEnumeration
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Pose2D,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Milk,
    TrashCan,
)
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% the scene


class SceneFile(PathEnumeration):
    """
    The files the scene is built from, found relative to this file in the coraplex
    resources, so the demo runs wherever the repository is checked out.
    """

    MILK = Path(__file__).resolve().parents[2] / "resources" / "objects" / "milk.stl"
    """
    The mesh of the milk carton.
    """


class SceneBody(StrEnum):
    """
    The names of the bodies the demo adds to the kitchen.
    """

    MILK = "milk"


# %% driving the base in a world without a drive


@dataclass
class DrivingKinematicallySimulatedPlanExecutor(KinematicallySimulatedPlanExecutor):
    """
    Simulates the plan kinematically, like
    :class:`~coraplex.plans.executors.KinematicallySimulatedPlanExecutor`, but drives the
    base to every pose it navigates to instead of putting it there, so the drive can be
    watched.
    """

    plan_adaptations: list[PlanTransformation] = field(
        default_factory=list, kw_only=True
    )
    """
    No plan transformations, so that no drive is replaced by setting the base's pose.
    """


# %% the demonstration


@dataclass
class FloorToTrashCanDemonstration(RobotDemonstration):
    """
    The HSR takes the milk off the floor and drops it into the trash can. The robot only
    stands where each step can be done from; the standing pose for the pick-up is chosen
    while the plan runs. The plan, step by step::

        park the arm
        turn to and look at search_spot, then detect the milk
        drive to where the milk can be reached, take it from the side facing the
        robot and lift it
        park the arm, so the milk is held close beside the robot
        turn towards passage_spot and drive there, keeping the milk upright
        drive on to the approach spot, keeping the milk upright
        drive on to drop_distance before the trash can and face it
        bring the milk over the opening, lower it, let go of it and pull the hand back
        park the arm and lower the torso
    """

    used_robot: type[AbstractRobot] = HSRB
    """
    The robot of this demo: Toyota's HSR.
    """

    ros_node_name: str = "hsr_floor_to_trash_can_demo"
    """
    Name of the ROS node the demo registers.
    """

    executor_type: type[PlanExecutor] = DrivingKinematicallySimulatedPlanExecutor
    """
    The executor running the plan; by default a kinematic simulation whose base drives.
    """

    collision_avoidance: bool = True
    """
    Whether the robot keeps away from the furniture, from itself and from what it holds.
    """

    robot_start: Pose2D = field(
        default_factory=lambda: Pose2D.from_xy_yaw(x=0.9, y=4.2)
    )
    """
    Where the HSR stands before the plan begins, facing along the world's x-axis.
    """

    milk_start: Pose2D = field(
        default_factory=lambda: Pose2D.from_xy_yaw(x=1.7, y=4.6, yaw=0.3)
    )
    """
    Where the milk stands on the floor before the plan begins.
    """

    trash_can_spot: Pose2D = field(
        default_factory=lambda: Pose2D.from_xy_yaw(x=2.45, y=1.6)
    )
    """
    Where the trash can stands: in the living room, next to the sofa, with room around
    it for the robot.
    """

    search_spot: Pose2D = field(
        default_factory=lambda: Pose2D.from_xy_yaw(x=1.7, y=4.6)
    )
    """
    The spot on the floor the robot looks at to find the milk.
    """

    passage_spot: Pose2D = field(
        default_factory=lambda: Pose2D.from_xy_yaw(x=1.7, y=3.3)
    )
    """
    Where the robot passes with the milk from the bedroom into the living room, south of
    the wall between the bedroom and the office, which a drive straight to the trash can
    would run along.
    """

    approach_distance: float = 1.0
    """
    How far before the trash can, on the side of the milk, the robot first stops with
    the milk, in meters, so the drive on to where it drops the milk is short and clear.
    """

    drop_distance: float = 0.75
    """
    How far before the trash can, on the same side, the robot's base stands while it
    drops the milk, in meters.
    """

    drop_clearance: float = 0.08
    """
    How far above the trash can's rim the bottom of the milk is when the hand lets go of
    it, in meters.
    """

    real_time_factor: float | None = 1.0
    """
    How much faster than real time a kinematic simulation moves the robot, so the run can
    be followed in RViz; ``None`` runs it as fast as the computer allows.
    """

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification(world_source=KitchenEnvironment())

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            localization_pose=self.robot_start.homogeneous_matrix,
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(SceneBody.MILK)

    def populate_scene(self, world: World) -> None:
        """
        Unlike :meth:`RobotDemonstration.populate_scene`, moves the trash can to
        :attr:`trash_can_spot` and stands the milk on the floor at :attr:`milk_start`.
        """
        [trash_can] = world.get_semantic_annotations_by_type(TrashCan)
        with world.modify_world():
            world.remove_connection(trash_can.root.parent_connection)
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=trash_can.root,
                    parent_T_connection_expression=self._compute_floor_transform(
                        self.trash_can_spot, trash_can.root
                    ),
                )
            )
        with world.modify_world():
            milk = Milk.get_annotation_specification(
                name=SceneBody.MILK,
                root_specification=BodySpecification.mesh(
                    SceneBody.MILK, str(SceneFile.MILK)
                ),
                parent_connection_specification=Connection6DoFSpecification(),
            ).spawn(world)
        milk.root.parent_connection.origin = self._compute_floor_transform(
            self.milk_start, milk.root
        )

    def create_executor(self, world: World) -> PlanExecutor:
        """
        Does what :meth:`RobotDemonstration.create_executor` does, and lets a kinematic
        simulation run at :attr:`real_time_factor`.
        """
        executor = super().create_executor(world)
        if self.real_time_factor is None or not isinstance(
            executor, KinematicallySimulatedPlanExecutor
        ):
            return executor
        executor.pacer = SimulationPacer(real_time_factor=self.real_time_factor)
        executor.pacer.pace_ticks_of(executor.context)
        return executor

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        world = context.world
        hsrb = context.require_extension(RobotAccess).robot
        [arm] = hsrb.all_arms
        [milk] = world.get_semantic_annotations_by_type(Milk)
        [trash_can] = world.get_semantic_annotations_by_type(TrashCan)
        [grasp] = GraspsByCloseness(
            milk, Pose(reference_frame=hsrb.root), number_of_grasps=1
        )()
        drop_pose = self._compute_drop_pose(trash_can.root, grasp)
        approach_pose = self._compute_standing_pose_before(
            drop_pose, self.approach_distance
        )
        passage_pose = self._compute_floor_pose_facing(
            np.array([float(self.passage_spot.x), float(self.passage_spot.y)]),
            approach_pose,
        )
        # the floor rule comes last: once it is in place, every change of the collision
        # rules takes about a second to record
        self._let_wrist_touch_the_object(arm, milk.root)
        self._let_robot_touch_the_floor(hsrb)

        milk_pose = Pose(reference_frame=milk.root)
        search_pose = Pose.from_xyz_rpy(
            x=self.search_spot.x, y=self.search_spot.y, reference_frame=world.root
        )
        steps: list[StatechartNode] = [
            ParkArmsAction(arms=hsrb.all_arms),
            FaceAndLookAtAction(
                face_at=FaceAtAction(target=search_pose),
                look_at=LookAtAction(target=search_pose),
            ),
            DetectAction(
                technique=DetectionTechnique.TYPES, object_sem_annotation=Milk
            ),
            a(MoveAndPickUpAction)(
                navigate=a(NavigateAction)(
                    target_location=self._leave_standing_pose_open(
                        milk_pose, arm, context
                    )
                ),
                face_and_look_at=self._face_and_look_at(milk_pose),
                pick_up=a(PickUpAction)(grasp=grasp, arm=arm),
            ),
            ParkArmsAction(arms=hsrb.all_arms),
            FaceAtAction(target=passage_pose),
            CarryAction(
                navigate=NavigateAction(target_location=passage_pose),
                holding_arms=[arm],
            ),
            CarryAction(
                navigate=NavigateAction(target_location=approach_pose),
                holding_arms=[arm],
            ),
            CarryAction(
                navigate=NavigateAction(
                    target_location=self._compute_standing_pose_before(
                        drop_pose, self.drop_distance
                    )
                ),
                holding_arms=[arm],
            ),
            FaceAtAction(target=drop_pose),
            a(PlaceAction)(object_designator=milk, target_location=drop_pose),
            Parallel(
                nodes=[
                    ParkArmsAction(arms=hsrb.all_arms),
                    MoveTorsoAction(torso_state=TorsoState.LOW),
                ]
            ),
        ]

        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=steps))
        return statechart

    @staticmethod
    def _compute_floor_transform(
        spot: Pose2D, body: Body
    ) -> HomogeneousTransformationMatrix:
        """
        :param spot: Where on the floor `body` stands, and which way it faces.
        :param body: A body with collision geometry, standing upright.
        :return: The pose of `body` in the world frame that puts the bottom of its
            collision geometry onto the floor at `spot`.
        """
        world = body._world
        body_box = body.collision.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=spot.x,
            y=spot.y,
            z=-body_box.min_z,
            yaw=spot.yaw,
            reference_frame=world.root,
        )

    @staticmethod
    def _let_robot_touch_the_floor(robot: AbstractRobot) -> None:
        """
        Lets the robot touch every floor of the world, which collision avoidance would
        otherwise keep its base away from.

        :param robot: The robot that may touch the floors.
        """
        world = robot._world
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=list(robot.bodies_with_collision),
                    body_group_b=[
                        floor.root
                        for floor in world.get_semantic_annotations_by_type(Floor)
                    ],
                )
            )

    @staticmethod
    def _let_wrist_touch_the_object(arm: Arm, item: Body) -> None:
        """
        Lets the wrist the hand is mounted on touch `item`, which collision avoidance
        otherwise only allows the hand that grasps it.

        :param arm: The arm whose wrist may touch `item`.
        :param item: The object grasped.
        """
        world = item._world
        wrist = arm.end_effector.root.parent_connection.parent
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(body_group_a=[wrist], body_group_b=[item])
            )

    @staticmethod
    def _leave_standing_pose_open(
        target: Pose, arm: Arm, context: StatechartContext
    ) -> Variable[Pose]:
        """
        :param target: What the arm is to reach once the robot stands.
        :param arm: The arm that reaches.
        :param context: The context of the plan, whose robot drives.
        :return: The standing poses `target` may be reached from. While the plan runs,
            each is tried in a copy of the world, and the first that works is taken.
        """
        return variable(
            Pose,
            domain=ReachabilityLocation(target_pose=target, arm=arm, context=context),
        )

    @staticmethod
    def _face_and_look_at(target: Pose) -> Match:
        """
        :param target: What the robot turns to and looks at.
        :return: The turn of the base towards `target`, then the look at it.
        """
        return a(FaceAndLookAtAction)(
            face_at=a(FaceAtAction)(target=target),
            look_at=a(LookAtAction)(target=target),
        )

    def _compute_standing_pose_before(self, drop_pose: Pose, distance: float) -> Pose:
        """
        :param drop_pose: Where the milk is let go of, above the trash can.
        :param distance: How far from the trash can the base stands, in meters.
        :return: A pose on the floor `distance` from the trash can, on the line towards
            :attr:`milk_start`, facing the trash can.
        """
        drop_xy = np.array([float(drop_pose.x), float(drop_pose.y)])
        standing_xy = drop_xy - distance * self._compute_carrying_direction(drop_xy)
        return self._compute_floor_pose_facing(standing_xy, drop_pose)

    def _compute_carrying_direction(self, drop_xy: np.ndarray) -> np.ndarray:
        """
        :param drop_xy: Where on the floor, in the world frame, the milk is let go of.
        :return: The unit vector along the floor from :attr:`milk_start` towards
            `drop_xy`, which is the way the robot faces when it stands before the trash
            can.
        """
        towards_drop = drop_xy - np.array(
            [float(self.milk_start.x), float(self.milk_start.y)]
        )
        return towards_drop / np.linalg.norm(towards_drop)

    @staticmethod
    def _compute_floor_pose_facing(standing_xy: np.ndarray, target: Pose) -> Pose:
        """
        :param standing_xy: Where on the floor the pose lies, as x and y in the frame of
            `target`.
        :param target: What the pose faces; only its horizontal position matters.
        :return: A pose on the floor at `standing_xy` whose x-axis points at `target`.
        """
        facing = np.array([float(target.x), float(target.y)]) - standing_xy
        return Pose.from_xyz_rpy(
            x=standing_xy[0],
            y=standing_xy[1],
            yaw=np.arctan2(facing[1], facing[0]),
            reference_frame=target.reference_frame,
        )

    def _compute_drop_pose(self, trash_can: Body, grasp: GraspCandidate) -> Pose:
        """
        :param trash_can: The body of the trash can.
        :param grasp: The grasp the robot holds the milk by.
        :return: Where the milk is when the hand lets go of it: upright, centred over the
            trash can's opening, its bottom :attr:`drop_clearance` above the rim, and
            turned so that the hand reaches over the opening from the robot's side.
        """
        world = trash_can._world
        milk = grasp.graspable.root
        trash_can_box = trash_can.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        milk_box = milk.collision.as_bounding_box_collection_in_frame(
            milk
        ).bounding_box()
        drop_xy = np.array(
            [
                (trash_can_box.min_x + trash_can_box.max_x) / 2,
                (trash_can_box.min_y + trash_can_box.max_y) / 2,
            ]
        )
        facing = self._compute_carrying_direction(drop_xy)
        _, _, grasp_yaw = grasp.grasp_pose.rotation_matrix.rpy
        return Pose.from_xyz_rpy(
            x=drop_xy[0],
            y=drop_xy[1],
            z=trash_can_box.max_z + self.drop_clearance - milk_box.min_z,
            yaw=np.arctan2(facing[1], facing[0]) - float(grasp_yaw),
            reference_frame=world.root,
        )


def main(
    executor_type: type[PlanExecutor] = DrivingKinematicallySimulatedPlanExecutor,
) -> World:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it drives
        the real robot or simulates it.
    :return: The world the milk was carried in.
    """
    return FloorToTrashCanDemonstration(executor_type=executor_type).run()


if __name__ == "__main__":
    main()
