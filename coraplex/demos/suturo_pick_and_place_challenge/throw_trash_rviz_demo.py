"""
The HSR picks a milk carton up off the floor of the kitchen environment and drops it into
the trash can, which the demo moves into the living room next to the sofa. The run is a
kinematic simulation that can be followed in RViz2 on ``/semworld/viz_marker``.

Run it from this folder, with ROS 2 and the ROS workspace sourced::

    python throw_trash_rviz_demo.py
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import krrood.symbolic_math.symbolic_math as sm
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
from coraplex.robot_plans.actions.base import SequentialAction
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
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    MoveTorsoAction,
    ParkArmsAction,
)
from coraplex.robot_plans.mixins import (
    MovesToolCenterPoint,
    OperatesGripper,
    ReattachesBranches,
)
from cramph.composites import Parallel, Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.executor import SimulationPacer
from cramph.node import NodeArtifacts, StatechartNode, SucceedsOnObservingTrue
from cramph.statechart import Statechart
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match
from krrood.patterns.path_enumeration import PathEnumeration
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
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
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

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


@dataclass
class FloorSpot:
    """
    A spot on the floor: where something stands, in the world frame, and which way it
    faces.
    """

    x: float
    """
    The spot's x-coordinate, in meters.
    """

    y: float
    """
    The spot's y-coordinate, in meters.
    """

    yaw: float = 0.0
    """
    Which way the thing standing there faces, as a turn about the world's z-axis, in rad.
    """

    def create_transform(
        self,
        reference_frame: KinematicStructureEntity | None = None,
        height: float = 0.0,
    ) -> HomogeneousTransformationMatrix:
        """
        :param reference_frame: The world root the spot is given in; ``None`` for a
            transform that is not tied to a world yet.
        :param height: How far above the floor the transform lies, in meters.
        :return: The position and orientation of this spot, `height` above the floor.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x,
            y=self.y,
            z=height,
            yaw=self.yaw,
            reference_frame=reference_frame,
        )


# %% letting go of the milk in a world without gravity


@dataclass(eq=False, repr=False, kw_only=True)
class SettleOnContainerBottom(SucceedsOnObservingTrue, StatechartNode):
    """
    Puts an object upright onto the inner bottom of a container, under the container in
    the kinematic structure, right below where it was let go of.

    Stands in for gravity in a kinematic simulation, where a released object stays in the
    air.
    """

    item: Body
    """
    The object that falls into the container.
    """

    container: Body
    """
    The container the object falls into.
    """

    wall_thickness: float
    """
    How thick the bottom of the container is, in meters.
    """

    def build_artifacts(self) -> NodeArtifacts:
        """
        Unlike :meth:`StatechartNode.build_artifacts`, observes ``TRUE`` from the start,
        so the node succeeds as soon as it has moved the object.
        """
        return NodeArtifacts(observation=sm.Scalar.const_true())

    def on_start(self) -> None:
        """
        Unlike :meth:`StatechartNode.on_start`, moves :attr:`item` onto the bottom of
        :attr:`container`.
        """
        world = self.context.world
        container_box = self.container.collision.as_bounding_box_collection_in_frame(
            self.container
        ).bounding_box()
        item_box = self.item.collision.as_bounding_box_collection_in_frame(
            self.item
        ).bounding_box()
        container_P_item = world.transform(
            self.item.global_pose, self.container
        ).position
        world.move_branch(self.item, self.container)
        self.item.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=float(container_P_item.x),
                y=float(container_P_item.y),
                z=container_box.min_z + self.wall_thickness - item_box.min_z,
                reference_frame=self.container,
            )
        )


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


# %% standing to reach down onto an object


@dataclass
class ReachabilityFromAboveLocation(ReachabilityLocation):
    """
    Where the robot can stand to reach down onto a target on the floor: like
    :class:`~coraplex.locations.locations.ReachabilityLocation`, but never closer to
    the target than :attr:`minimum_distance`, from where the hand cannot turn to point
    down onto it.
    """

    minimum_distance: float = field(default=0.4, kw_only=True)
    """
    How close to the target, along the floor, the robot may stand at the least, in
    meters.
    """

    def candidates(self) -> Iterator[Pose]:
        """
        Unlike :meth:`ReachabilityLocation.candidates`, leaves out the poses closer to
        the target than :attr:`minimum_distance`.
        """
        target = self._in_world(self.target_pose).position.to_np()[:2]
        for candidate in super().candidates():
            offset = candidate.position.to_np()[:2] - target
            if np.linalg.norm(offset) >= self.minimum_distance:
                yield candidate


# %% dropping into a container


@dataclass(eq=False, repr=False, kw_only=True)
class DropAction(
    MovesToolCenterPoint, OperatesGripper, ReattachesBranches, SequentialAction
):
    """
    Brings a held object over a container and lets go of it, then pulls the hand up.
    Unlike :class:`~coraplex.robot_plans.actions.core.placing.PlaceAction`, the object
    never has to touch anything, so collision avoidance keeps it away from everything,
    the container's rim included.
    """

    grasp: GraspCandidate
    """
    The grasp the object is held by, which also names the object.
    """

    arm: Arm
    """
    The arm that holds the object.
    """

    release_pose: Pose
    """
    Where the object is when the hand lets go of it.
    """

    retreat_distance: float = 0.15
    """
    How far the hand rises after letting go, in meters.
    """

    def create_steps(self) -> list[StatechartNode]:
        end_effector = self.arm.end_effector
        world_T_release = self.world.transform(self.release_pose, self.world.root)
        world_T_retreat = Pose.from_xyz_rpy(
            x=float(world_T_release.x),
            y=float(world_T_release.y),
            z=float(world_T_release.z) + self.retreat_distance,
            reference_frame=self.world.root,
        )
        return [
            self.tool_center_point_goal(
                end_effector.tool_frame_goal_for_grasp(
                    self.grasp.moved_to(self.release_pose)
                ),
                self.arm,
            ),
            self.gripper_goal_type.moving_gripper_to(
                end_effector.default_configuration(GripperState.OPEN)
            ),
            *self.reattach(self.grasp.graspable.root, self.world.root),
            self.tool_center_point_goal(
                end_effector.tool_frame_goal_for_grasp(
                    self.grasp.moved_to(world_T_retreat)
                ),
                self.arm,
            ),
        ]


# %% the demonstration


@dataclass
class FloorToTrashCanDemonstration(RobotDemonstration):
    """
    The HSR takes the milk off the floor and drops it into the trash can. The robot only
    stands where each step can be done from; the standing pose for the pick-up is chosen
    while the plan runs. The plan, step by step::

        park the arm
        turn to and look at search_spot, then detect the milk
        drive to where the milk can be reached, take it from above and lift it
        bring the milk upright in front of the base, as high as it is dropped from
        turn towards passage_spot and drive there, keeping the milk upright
        drive on to the approach spot, keeping the milk upright
        drive on to drop_distance before the trash can and face it
        bring the milk over the opening, let go of it and pull the hand up
        park the arm and lower the torso
        put the milk onto the trash can's bottom (kinematic simulation only)
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

    robot_start: FloorSpot = field(default_factory=lambda: FloorSpot(x=0.9, y=4.2))
    """
    Where the HSR stands before the plan begins, facing along the world's x-axis.
    """

    milk_start: FloorSpot = field(
        default_factory=lambda: FloorSpot(x=1.7, y=4.6, yaw=0.3)
    )
    """
    Where the milk stands on the floor before the plan begins.
    """

    trash_can_spot: FloorSpot = field(default_factory=lambda: FloorSpot(x=2.45, y=1.6))
    """
    Where the trash can stands: in the living room, next to the sofa, with room around
    it for the robot.
    """

    search_spot: FloorSpot = field(default_factory=lambda: FloorSpot(x=1.7, y=4.6))
    """
    The spot on the floor the robot looks at to find the milk.
    """

    passage_spot: FloorSpot = field(default_factory=lambda: FloorSpot(x=1.7, y=3.3))
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

    carrying_reach: float = 0.45
    """
    How far in front of the base's origin the milk is carried, in meters.
    """

    grasp_height: float = 0.04
    """
    How far above the milk's origin the hand takes hold of it, in meters.
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

    trash_can_wall_thickness: float = 0.02
    """
    How thick the walls and the bottom of the trash can are, in meters, as
    :class:`KitchenEnvironment` builds it.
    """

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification(world_source=KitchenEnvironment())

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            localization_pose=self.robot_start.create_transform(),
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(SceneBody.MILK)

    def populate_scene(self, world: World) -> None:
        """
        Unlike :meth:`RobotDemonstration.populate_scene`, moves the trash can to
        :attr:`trash_can_spot` and stands the milk on the floor at :attr:`milk_start`.
        """
        [trash_can] = world.get_semantic_annotations_by_type(TrashCan)
        trash_can_box = trash_can.root.collision.as_bounding_box_collection_in_frame(
            trash_can.root
        ).bounding_box()
        with world.modify_world():
            world.remove_connection(trash_can.root.parent_connection)
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=trash_can.root,
                    parent_T_connection_expression=self.trash_can_spot.create_transform(
                        world.root, height=-trash_can_box.min_z
                    ),
                )
            )
        with world.modify_world():
            milk = SemanticAnnotationWithRootSpecification(
                name=SceneBody.MILK,
                semantic_annotation_type=Milk,
                root_specification=BodySpecification.mesh(
                    SceneBody.MILK,
                    str(SceneFile.MILK),
                    connection_specification=Connection6DoFSpecification(),
                ),
            ).spawn(world)
        milk_box = milk.root.collision.as_bounding_box_collection_in_frame(
            milk.root
        ).bounding_box()
        milk.root.parent_connection.origin = self.milk_start.create_transform(
            world.root, height=-milk_box.min_z
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
        drop_pose = self._compute_drop_pose(trash_can.root, milk.root)
        approach_pose = self._compute_standing_pose_before(
            drop_pose, self.approach_distance
        )
        passage_pose = self._compute_passage_pose(approach_pose)
        # the floor rule comes last: once it is in place, every change of the collision
        # rules takes about a second to record
        self._let_wrist_touch_the_object(arm, milk.root)
        self._let_robot_touch_the_floor(hsrb)

        grasp = self._create_grasp_from_above(milk)
        milk_pose = Pose(reference_frame=milk.root)
        search_pose = self.search_spot.create_transform(world.root).pose
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
            MoveManipulatorAction(
                target_pose=arm.end_effector.tool_frame_goal_for_grasp(
                    grasp.moved_to(self._compute_carrying_pose(hsrb, drop_pose))
                ),
                end_effector=arm.end_effector,
                allow_gripper_collision=False,
            ),
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
            DropAction(grasp=grasp, arm=arm, release_pose=drop_pose),
            Parallel(
                nodes=[
                    ParkArmsAction(arms=hsrb.all_arms),
                    MoveTorsoAction(torso_state=TorsoState.LOW),
                ]
            ),
        ]
        if issubclass(self.executor_type, KinematicallySimulatedPlanExecutor):
            steps.append(
                SettleOnContainerBottom(
                    item=milk.root,
                    container=trash_can.root,
                    wall_thickness=self.trash_can_wall_thickness,
                )
            )

        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=steps))
        return statechart

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
        Lets the wrist the hand is mounted on touch `item`. Coming down onto a short
        object from above brings the wrist as close to it as the hand, but only the hand
        may touch what it grasps.

        :param arm: The arm whose wrist may touch `item`.
        :param item: The object grasped from above.
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
            domain=ReachabilityFromAboveLocation(
                target_pose=target, arm=arm, context=context
            ),
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
        towards_milk = np.array([self.milk_start.x, self.milk_start.y]) - drop_xy
        standing_xy = drop_xy + distance * towards_milk / np.linalg.norm(towards_milk)
        facing = drop_xy - standing_xy
        return Pose.from_xyz_rpy(
            x=standing_xy[0],
            y=standing_xy[1],
            yaw=np.arctan2(facing[1], facing[0]),
            reference_frame=drop_pose.reference_frame,
        )

    def _compute_passage_pose(self, approach_pose: Pose) -> Pose:
        """
        :param approach_pose: Where the robot stops with the milk before the trash can.
        :return: A pose on the floor at :attr:`passage_spot`, facing `approach_pose`, so
            the drive on to it needs no turn first.
        """
        towards_approach = np.array(
            [
                float(approach_pose.x) - self.passage_spot.x,
                float(approach_pose.y) - self.passage_spot.y,
            ]
        )
        return Pose.from_xyz_rpy(
            x=self.passage_spot.x,
            y=self.passage_spot.y,
            yaw=np.arctan2(towards_approach[1], towards_approach[0]),
            reference_frame=approach_pose.reference_frame,
        )

    def _create_grasp_from_above(self, milk: Milk) -> GraspCandidate:
        """
        :param milk: The milk standing upright on the floor.
        :return: A grasp that comes down onto the milk, :attr:`grasp_height` above its
            origin so the fingers stay clear of the floor. The pitch of a quarter turn
            points the grasp's approach axis, its x-axis, downwards.
        """
        return GraspCandidate(
            milk,
            Pose.from_xyz_rpy(
                z=self.grasp_height, pitch=np.pi / 2, reference_frame=milk.root
            ),
        )

    def _compute_drop_pose(self, trash_can: Body, milk: Body) -> Pose:
        """
        :param trash_can: The body of the trash can.
        :param milk: The body of the milk.
        :return: Where the milk is when the hand lets go of it: upright, centred over the
            trash can's opening, its bottom :attr:`drop_clearance` above the rim.
        """
        world = trash_can._world
        trash_can_box = trash_can.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        milk_box = milk.collision.as_bounding_box_collection_in_frame(
            milk
        ).bounding_box()
        return Pose.from_xyz_rpy(
            x=(trash_can_box.min_x + trash_can_box.max_x) / 2,
            y=(trash_can_box.min_y + trash_can_box.max_y) / 2,
            z=trash_can_box.max_z + self.drop_clearance - milk_box.min_z,
            reference_frame=world.root,
        )

    def _compute_carrying_pose(self, robot: AbstractRobot, drop_pose: Pose) -> Pose:
        """
        :param robot: The robot carrying the milk.
        :param drop_pose: Where the milk is let go of, above the trash can.
        :return: Where the milk is while the robot drives: upright, :attr:`carrying_reach`
            in front of the base and as high above the floor as `drop_pose`.
        """
        world = robot._world
        base_height = world.compute_forward_kinematics_np(world.root, robot.root)[2, 3]
        return Pose.from_xyz_rpy(
            x=self.carrying_reach,
            z=float(drop_pose.z) - base_height,
            reference_frame=robot.root,
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
