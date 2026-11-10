"""
The HSR carries a milk carton from the counter to the kitchen island in the lab, in MuJoCo.

Run it from this folder::

    python demo-lab.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import mujoco
import numpy as np
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from cramph.composites import Sequence
from cramph.statechart import Statechart
from krrood.patterns.path_enumeration import PathEnumeration
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
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
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor, Milk
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
            x=2.6, y=-0.9
        )
    )
    """
    Where the HSR stands before the plan begins: on the floor between the counter and the
    kitchen island.
    """

    milk_start: SurfaceSpot = field(
        default_factory=lambda: SurfaceSpot(
            SceneBody.COUNTER, x=2.6, y=-2.05, yaw=-np.pi / 2
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
            semantic_annotation_type=self.used_robot,
            odometry_T_robot_start=self.robot_start,
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
        [arm] = robot.all_arms
        taking_pose = self._standing_pose_facing(
            self.milk_start.compute_milk_pose(milk.root)
        )
        target = self.milk_target.compute_milk_pose(milk.root)
        grasp_target = self.find_grasp_pose(milk.root)
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    MoveTorsoAction(torso_state=TorsoState.HIGH),
                    ParkArmsAction(arms=robot.all_arms),
                    NavigateAction(target_location=taking_pose),
                    PickUpAction(grasp=GraspCandidate(milk, grasp_target), arm=arm),
                    ParkArmsAction(arms=robot.all_arms),
                    MoveTorsoAction(torso_state=TorsoState.LOW),
                    NavigateAction(target_location=self._standing_pose_facing(target)),
                    MoveTorsoAction(torso_state=TorsoState.HIGH),
                    PlaceAction(object_designator=milk, target_location=target),
                    ParkArmsAction(arms=robot.all_arms),
                    MoveTorsoAction(torso_state=TorsoState.LOW),
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

    def find_grasp_pose(self, body: Body, fraction_of_height: float = 4 / 6) -> Pose:
        """
        :param body: The object to grasp, standing upright in its own frame.
        :param fraction_of_height: How far up the bounding box of ``body`` the grasp is,
            from 0 at its bottom to 1 at its top; by default where its upper third begins.
        :return: The grasp frame at that height on the z axis of ``body``, in its frame.
        """
        box = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()
        return Pose.from_xyz_rpy(
            z=box.min_z + fraction_of_height * (box.max_z - box.min_z),
            reference_frame=body,
        )

    def _standing_pose_facing(self, milk_pose: Pose) -> Pose:
        """
        :param milk_pose: Where the milk stands or is set down.
        :return: Where the robot's base stands on the floor to reach the milk there,
            :attr:`standing_distance` before it and facing it.
        """
        return (
            milk_pose.homogeneous_matrix
            @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-self.standing_distance, z=-float(milk_pose.z)
            )
        ).pose


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
