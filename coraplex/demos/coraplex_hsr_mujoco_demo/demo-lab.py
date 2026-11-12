"""
The HSR carries a milk carton from the counter to the kitchen island in the lab, in MuJoCo.

Run it from this folder::

    python demo-lab.py
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import mujoco
import numpy as np
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor
from coraplex.querying.grasp_closeness import GraspsByCloseness
from coraplex.robot_plans.actions.core.navigation import (
    NavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from cramph.composites import Sequence
from cramph.statechart import Statechart
from krrood.patterns.path_enumeration import PathEnumeration
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose, Point3, RotationMatrix, Vector3
from segmind.detectors.agent_event_detector_nodes import GraspDetector
from segmind.detectors.coarse_event_detector_nodes import (
    PickUpDetector,
    PlacingDetector,
)
from segmind.event_segmentation import Segmind

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.predefined_maps.kitchen_environment import KitchenEnvironment
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
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
from semantic_digital_twin.world_description.contact import ContactParameters
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from coraplex.plans.executors import PlanExecutor
    from cramph.context import ContextExtension, StatechartContext
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body

# %% the scene


class SceneFile(PathEnumeration):
    """
    The files the scene is built from, under the coraplex resources.
    """

    MILK = Path(__file__).resolve().parents[2] / "resources" / "objects" / "milk.stl"
    """
    The mesh of the milk carton.
    """


class SceneBody(StrEnum):
    """
    The names of the scene's bodies the demonstration acts on.
    """

    COUNTER = "counter_top"
    """
    The worktop along the wall, which holds the sink.
    """

    KITCHEN_SURFACE = "sideboard"
    """
    The worktop of the kitchen island.
    """

    LIVING_ROOM_FLOOR = "living_room_floor"
    """
    The floor of the living room.
    """

    MILK = "milk"
    """
    The milk carton the plan carries, which marks whether the scene was already populated.
    """


class HSRBBody(StrEnum):
    """
    The HSR's own bodies the demonstration refers to.
    """

    WRIST = "wrist_roll_link"
    """
    The wrist the hand is mounted on, which is not part of the end effector.
    """


@dataclass
class SurfaceSpot:
    """
    A spot on a horizontal surface of the lab that the milk can stand on.
    """

    surface: SceneBody
    """
    The body the milk stands on.
    """

    x: float
    """
    Where the milk stands along the world's x axis, in meters.
    """

    y: float
    """
    Where the milk stands along the world's y axis, in meters.
    """

    yaw: float
    """
    Which way the milk faces in the world, which is the way the robot faces it there.
    """

    def compute_milk_pose(self, milk: Body) -> Pose:
        """
        :param milk: The body of the milk.
        :return: The pose of ``milk`` in the world root at which its lowest point rests
            on the surface.
        """
        world = milk._world
        surface_box = (
            world.get_body_by_name(self.surface)
            .collision.as_bounding_box_collection_in_frame(world.root)
            .bounding_box()
        )
        milk_box = milk.collision.as_bounding_box_collection_in_frame(
            milk
        ).bounding_box()
        return Pose.from_xyz_rpy(
            x=self.x,
            y=self.y,
            z=surface_box.max_z - milk_box.min_z,
            yaw=self.yaw,
            reference_frame=world.root,
        )


# %% the demonstration


@dataclass
class LabMilkTransportDemonstration(RobotDemonstration):
    """
    The HSR takes the milk from the counter in the lab, carries it to the kitchen island
    with its arm parked and its torso lowered, and sets it down there.
    """

    used_robot: type[AbstractRobot] = HSRB
    """
    The robot that carries the milk.
    """

    ros_node_name: str = "hsr_lab_milk_transport_demo"
    """
    Name of the node this demonstration registers.
    """

    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor
    """
    The executor running the plan, which performs it in the physics of
    :meth:`build_simulation`.
    """

    collision_avoidance: bool = True
    """
    Whether collision avoidance is added to every motion state chart of this run.
    """

    robot_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.1, y=2.5
        )
    )
    """
    Where the HSR stands before the plan begins: on the floor between the counter and the
    kitchen island.
    """

    milk_start: SurfaceSpot = field(
        default_factory=lambda: SurfaceSpot(
            SceneBody.LIVING_ROOM_FLOOR, x=3.5, y=3.3, yaw=-np.pi / 2
        )
    )
    """
    Where the milk stands before the plan begins: on the counter, beside the sink and
    near the edge the robot approaches.
    """

    milk_target: SurfaceSpot = field(
        default_factory=lambda: SurfaceSpot(
            SceneBody.KITCHEN_SURFACE, x=3.0, y=0.22, yaw=np.pi / 2
        )
    )
    """
    Where the milk is set down: on the kitchen island, near the edge facing the counter.
    """

    standing_distance: float = 0.7
    """
    How far from the milk the robot's base stands to take it and to set it down, in
    meters.
    """

    friction_cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_ELLIPTIC
    """
    The friction cone the contacts use: the elliptic cone is the exact one, where the
    default pyramid only approximates it and lets a held carton creep between the
    fingers.
    """

    impedance_ratio: float = 10.0
    """
    How much harder sliding is than pressing in: MuJoCo's own guidance for grasping with
    an elliptic cone.
    """

    no_slip_iterations: int = 10
    """
    Passes spent removing the slip soft contacts leave, so a held carton stays where the
    fingers hold it.
    """

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification(world_source=KitchenEnvironment())

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(
            semantic_annotation_type=self.used_robot, odometry_T_robot_start=self.robot_start
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(SceneBody.MILK)

    def populate_scene(self, world: World) -> None:
        """
        Stand the milk on the counter, free to be picked up, see
        :meth:`RobotDemonstration.populate_scene`.
        """
        milk = Milk.get_annotation_specification(
            name=SceneBody.MILK,
            root_specification=BodySpecification.mesh(
                SceneBody.MILK,
                str(SceneFile.MILK),
                connection_specification=Connection6DoFSpecification(),
            ),
        ).spawn(world)
        ContactParameters.create_for_grasped_object().apply_to([milk.root])
        milk.root.parent_connection.origin = self.milk_start.compute_milk_pose(
            milk.root
        ).homogeneous_matrix

    def acquire_world(self) -> World:
        """
        Obtain the world as :meth:`RobotDemonstration.acquire_world` does, in which the
        HSR's hand and wrist may touch the milk throughout the run, as physics never
        counts it as held, and the whole HSR may touch the lab's floors.
        """
        world = super().acquire_world()
        [robot] = world.get_semantic_annotations_by_type(self.used_robot)
        [arm] = robot.all_arms
        [milk] = world.get_semantic_annotations_by_type(Milk)
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=[
                        world.get_body_by_name(HSRBBody.WRIST),
                        *arm.end_effector.bodies_with_collision,
                    ],
                    body_group_b=[milk.root],
                )
            )
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=robot.bodies_with_collision,
                    body_group_b=[
                        floor.root
                        for floor in world.get_semantic_annotations_by_type(Floor)
                    ],
                )
            )
        return world

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def segment_events(self, world: World) -> AbstractContextManager:
        """
        Detect the milk being grasped, picked up and placed, see
        :meth:`RobotDemonstration.segment_events`.
        """
        return Segmind.create_for_semantic_annotation_types(
            world,
            (Milk,),
            detector_types=(PickUpDetector, PlacingDetector, GraspDetector),
        )

    def build_statechart(self, context: StatechartContext) -> Statechart:
        robot = context.require_extension(RobotAccess).robot

        [milk] = context.world.get_semantic_annotations_by_type(Milk)
        [trash_can] = context.world.get_semantic_annotations_by_type(TrashCan)
        [arm] = robot.all_arms
        taking_pose = self._standing_pose_facing(
            milk.root.global_pose, robot.root.global_pose
        )
        middle_pose = self._standing_pose_facing(
            Pose.from_xyz_rpy(1, 3, reference_frame=context.world.root),
            taking_pose
        )

        target = self._standing_pose_facing(trash_can.root.global_pose, middle_pose)
        drop_pose = self._get_pose_dropping(trash_can, context.world)
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    MoveTorsoAction(torso_state=TorsoState.LOW),
                    ParkArmsAction(arms=robot.all_arms),
                    NavigateAction(target_location=taking_pose),
                    PickUpAction(grasp=self.find_grasp(milk, taking_pose), arm=arm),
                    ParkArmsAction(arms=robot.all_arms),
                    NavigateAction(target_location=middle_pose),
                    NavigateAction(target_location=target),
                    PlaceAction(object_designator=milk, target_location=drop_pose),
                    ParkArmsAction(arms=robot.all_arms)
                ]
            )
        )
        return statechart

    def build_simulation(self, world: World) -> MujocoSim:
        """
        Build the physics of :meth:`RobotDemonstration.build_simulation` with the contacts
        of :attr:`friction_cone`, :attr:`impedance_ratio` and :attr:`no_slip_iterations`.
        """
        return MujocoSim(
            world=world,
            headless=self.headless,
            cone=self.friction_cone,
            impratio=self.impedance_ratio,
            noslip_iterations=self.no_slip_iterations,
        )

    def find_grasp(
            self,
            graspable: HasGraspCandidates,
            standing_pose: Pose,
            fraction_of_height: float = 4 / 6,
    ) -> GraspCandidate:
        """
        :param graspable: The object to take, standing upright.
        :param standing_pose: Where the robot stands while it takes the milk.
        :param fraction_of_height: How far up the milk's bounding box the grasp is, from 0
            at its bottom to 1 at its top; by default where its upper third begins.
        :return: The body's grasp closest to `standing_pose`, moved up to that height.
        """
        box = graspable.root.collision.as_bounding_box_collection_in_frame(
            graspable.root
        ).bounding_box()
        [closest] = GraspsByCloseness(graspable, standing_pose, number_of_grasps=1)()
        grasp_pose = deepcopy(closest.grasp_pose)
        grasp_pose.z = box.min_z + fraction_of_height * (box.max_z - box.min_z)
        return GraspCandidate(graspable=graspable, grasp_pose=grasp_pose)

    def _standing_pose_facing(self, target: Pose, approach_from: Pose) -> Pose:
        """
        :param target: What the robot goes to reach, in the world frame.
        :param approach_from: Where the robot stands before it goes to `target`, in the
            world frame.
        :return: Where the robot's base stands on the floor, :attr:`standing_distance`
            off `target` towards `approach_from` and facing `target`.
        """
        offset = target.position.to_np()[:2] - approach_from.position.to_np()[:2]
        standing_position = (target.position.to_np()[:2]
                             - self.standing_distance
                             * offset / np.linalg.norm(offset))
        return Pose.from_xyz_rpy(
            x=float(standing_position[0]),
            y=float(standing_position[1]),
            yaw=float(np.arctan2(offset[1], offset[0])),
            reference_frame=target.reference_frame,
        )

    def _get_pose_dropping(self, trash_can: TrashCan, world: World) -> Pose:
        """
        Calculate a drop pose above the trash can based on its bounding box.
        :param trash_can: The trash can into which the object will be dropped.
        :param world: The simulation world used as the reference frame.
        :return: The calculated drop pose above the trash can.
        """
        trash_can_box = trash_can.root.collision.as_bounding_box_collection_in_frame(
            world.root
            ).bounding_box()
        z_offset = 0.1

        x = (trash_can_box.min_x + trash_can_box.max_x) / 2
        y = (trash_can_box.min_y + trash_can_box.max_y) / 2
        z = trash_can_box.max_z + z_offset

        return Pose.from_xyz_rpy(x, y, z, reference_frame=trash_can.root.global_pose.reference_frame)

def main(
    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor,
    collision_avoidance: bool = True,
    event_segmentation: bool = False,
    debug: bool = False,
    headless: bool = False,
    scene_joints_movable: bool = False,
) -> World:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it drives
        the real robot or simulates it.
    :param collision_avoidance: Whether the statechart avoids collisions.
    :param event_segmentation: Whether SegMind segments the run into events.
    :param debug: Whether to run in debug mode, publishing every copy of the world a
        candidate is tried in.
    :param headless: Whether the physics runs without a viewer.
    :param scene_joints_movable: Whether the joints of the scene, such as those of its doors
        and drawers, can move, instead of being replaced with fixed connections.
    :return: The world the milk was carried in.
    """
    return LabMilkTransportDemonstration(
        executor_type=executor_type,
        collision_avoidance=collision_avoidance,
        event_segmentation=event_segmentation,
        debug=debug,
        headless=headless,
        scene_joints_movable=scene_joints_movable,
    ).run()


if __name__ == "__main__":
    main()
