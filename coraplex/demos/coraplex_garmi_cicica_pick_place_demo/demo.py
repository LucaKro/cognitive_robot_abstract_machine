"""
GARMI carries a milk carton between two desks of the CICICA lab.

GARMI starts in the aisle behind the desks, picks the milk up from one desk with its
right arm, drives along the aisle and sets it down on the other desk. The plan runs in
kinematic simulation and is published to RViz, with ``start/Root`` as the fixed frame.

.. note:: The lab's USD scene is read from :attr:`SceneFile.LAB`, which exists only on
    the machine this demonstration was written on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from pathlib import Path

import trimesh

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor, PlanExecutor
from coraplex.plans.plan_transformation import (
    ConfigurationTransformation,
    PlanTransformation,
)
from coraplex.robot_plans.actions.core.navigation import DrivesBase, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.drive_tasks import DifferentialDriveBaseGoal
from krrood.patterns.path_enumeration import PathEnumeration
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Pose2D,
)
from semantic_digital_twin.adapters.usd.parser import USDComponentParser
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% what the scene is built from


class SceneFile(PathEnumeration):
    """
    The files the demonstration builds its scene from.
    """

    LAB = Path(
        "/home/itsme/work/CostaRica CICICA USD Model/CICICA USD Model/scene/start.usda"
    )
    """
    The USD scene of the lab.
    """

    MILK = Path(__file__).resolve().parents[2] / "resources" / "objects" / "milk.stl"
    """
    The milk's mesh, under the coraplex resources.
    """


class SceneBody(StrEnum):
    """
    The bodies of the scene the demonstration refers to by name.
    """

    MILK = "milk"
    """
    The milk carried between the desks.
    """

    FLOOR = "Meshes/Base/Floor/Actor_0000"
    """
    The floor of the lab, which GARMI's base rests on.
    """


# %% where things stand


@dataclass(frozen=True, kw_only=True)
class ScenePlacement:
    """
    A position with a heading about the vertical, in the lab's root frame.
    """

    x: float
    """
    The position along the lab's x-axis, in meters.
    """

    y: float
    """
    The position along the lab's y-axis, in meters.
    """

    z: float = 0.0
    """
    The height above the lab's floor, in meters.
    """

    yaw: float = 0.0
    """
    The heading about the vertical, in radian.
    """

    @property
    def transform(self) -> HomogeneousTransformationMatrix:
        """
        :return: The placement as a transform.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            self.x, self.y, self.z, yaw=self.yaw
        )

    def raised(self, *, height: float) -> ScenePlacement:
        """
        :param height: How far above this placement the result is, in meters.
        :return: This placement, `height` higher.
        """
        return replace(self, z=self.z + height)

    def pose(self, *, reference_frame: KinematicStructureEntity) -> Pose:
        """
        :param reference_frame: The frame this placement is given in.
        :return: This placement as a pose in `reference_frame`.
        """
        return Pose.from_xyz_rpy(
            self.x, self.y, self.z, yaw=self.yaw, reference_frame=reference_frame
        )


class ScenePose(Enum):
    """
    Where GARMI stands and where the bottom of the milk rests.
    """

    GARMI_START = ScenePlacement(x=9.0, y=-1.0, yaw=math.pi)
    """
    Where GARMI starts, in the aisle, facing the desks.
    """

    GARMI_AT_PICK = ScenePlacement(x=8.85, y=-2.6, yaw=math.pi)
    """
    Where GARMI stands to pick up the milk, its right arm in front of it.
    """

    GARMI_AT_PLACE = ScenePlacement(x=8.85, y=-0.2, yaw=math.pi)
    """
    Where GARMI stands to set down the milk, its right arm in front of it.
    """

    MILK_PICK = ScenePlacement(x=7.9, y=-2.4, z=0.75)
    """
    Where the bottom of the milk starts, on the surface of the first desk.
    """

    MILK_PLACE = ScenePlacement(x=7.9, y=0.0, z=0.75)
    """
    Where the bottom of the milk ends up, on the surface of the second desk.
    """


# %% how the plan is run


@dataclass
class FaceDrivingDirection(ConfigurationTransformation[DrivesBase]):
    """
    Makes the base turn towards its goal, drive there facing it, and only then turn to
    the goal's heading.
    """

    def is_applicable(self, plan_node: DrivesBase) -> bool:
        return True

    def apply(self, plan_node: DrivesBase) -> None:
        plan_node.base_goal_type = DifferentialDriveBaseGoal


@dataclass(kw_only=True)
class DrivingKinematicExecutor(KinematicallySimulatedPlanExecutor):
    """
    A kinematic simulation whose base drives to its goals under the controller, so
    collision avoidance covers the whole way instead of only where the base arrives.
    """

    plan_adaptations: list[PlanTransformation] = field(
        default_factory=lambda: [FaceDrivingDirection()]
    )
    """
    The plan transformations adapting every plan to this simulation.
    """


# %% the demonstration


@dataclass(kw_only=True)
class GarmiLabPickAndPlaceDemonstration(RobotDemonstration):
    """
    GARMI carries a milk carton from one desk of the CICICA lab to another.
    """

    ros_node_name: str = "garmi_cicica_pick_place_demo_node"

    executor_type: type[PlanExecutor] = DrivingKinematicExecutor
    """
    The executor running the plan.
    """

    @property
    def milk_origin_height(self) -> float:
        """
        :return: How far the origin of the milk's mesh is above its bottom, in meters.
        """
        return -float(trimesh.load(str(SceneFile.MILK)).bounds[0][2])

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification(
            world_source=USDComponentParser.from_file(str(SceneFile.LAB))
        )

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            odometry_T_robot_start=ScenePose.GARMI_START.value.transform,
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(
            SceneBody.MILK.value
        )

    def populate_scene(self, world: World) -> None:
        """
        Add the milk, standing on the first desk, on a connection that can carry it.
        """
        milk_start = ScenePose.MILK_PICK.value.raised(height=self.milk_origin_height)
        Milk.get_annotation_specification(
            SceneBody.MILK.value,
            BodySpecification.mesh(
                SceneBody.MILK.value,
                str(SceneFile.MILK),
                parent_T_self=milk_start.transform,
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=world.root)

    def build_simulated_world(self) -> World:
        """
        Differs from :meth:`~coraplex.demonstrations.RobotDemonstration.build_simulated_world`
        in allowing GARMI to touch the floor it stands on.
        """
        world = super().build_simulated_world()
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=list(robot.bodies_with_collision),
                    body_group_b=[world.get_body_by_name(SceneBody.FLOOR.value)],
                )
            )
        return world

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [
            RobotAccess(
                robot=world.get_semantic_annotations_by_type(self.used_robot)[0]
            ),
        ]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Pick up the milk with the right arm from the aisle, carry it along the aisle
        with the arms parked, and set it down on the second desk.
        """
        world = context.world
        robot = context.require_extension(extension_type=RobotAccess).robot
        right_arm = robot.get_right_arm_if_specified()
        [milk] = world.get_semantic_annotations_by_type(Milk)
        milk_grasp = GraspCandidate(
            graspable=milk,
            grasp_pose=Pose.from_xyz_quaternion(
                quat_z=1, quat_w=0, reference_frame=milk.root
            ),
        )
        statechart = Statechart(context=context)
        statechart.add_node(
            node=Sequence(
                nodes=[
                    ParkArmsAction(arms=robot.all_arms),
                    NavigateAction(
                        planar_goal_pose=Pose2D.from_pose(
                            ScenePose.GARMI_AT_PICK.value.pose(
                                reference_frame=world.root
                            )
                        )
                    ),
                    PickUpAction(grasp_candidate=milk_grasp, arm=right_arm),
                    ParkArmsAction(arms=robot.all_arms),
                    NavigateAction(
                        planar_goal_pose=Pose2D.from_pose(
                            ScenePose.GARMI_AT_PLACE.value.pose(
                                reference_frame=world.root
                            )
                        )
                    ),
                    PlaceAction(
                        arm=right_arm,
                        graspable=milk,
                        goal_pose=ScenePose.MILK_PLACE.value.raised(
                            height=self.milk_origin_height
                        ).pose(reference_frame=world.root),
                    ),
                    ParkArmsAction(arms=robot.all_arms),
                ]
            )
        )
        return statechart


# %% entry point


def main() -> None:
    """
    Run the demonstration in kinematic simulation, published to RViz.
    """
    GarmiLabPickAndPlaceDemonstration(used_robot=Garmi, collision_avoidance=True).run()


if __name__ == "__main__":
    main()
