"""
GARMI puts a milk carton from the kitchen counter into a drawer.

GARMI picks up the milk from the front with its right arm, opens the drawer with its
left, lays the milk into the drawer from above, withdraws the right hand straight up out
of it and closes the drawer again with its left arm, one step after the other. Running with :class:`~coraplex.plans.executors.RealRobotPlanExecutor` drives
the actual robot and takes the world from the running world server, merging the apartment
into it unless the server holds it already. The default runs the whole plan in simulation
against a world built from the apartment's MuJoCo scene and GARMI's URDF, so nothing on
the network is needed.

Needs the ``iai_garmi_apartment`` and ``garmi_description`` packages built in the
workspace, since the scene and the robot description are read from their share
directories.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum

from ament_index_python.packages import get_package_share_directory

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
)
from coraplex.plans.plan_transformation import PlanRewriting
from coraplex.robot_plans.actions.core.container import (
    GraspCloseAndRetractAction,
    GraspOpenAndRetractAction,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    ParkArmsAction,
)
from coraplex.robot_plans.plan_transformations import OpenDrawerBeforeMoveAndPickUp
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from cramph.world_modification_nodes import MoveBranch
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Milk,
    Spoon,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% what the scene is built from


class SceneFile(StrEnum):
    """
    The files the demonstration builds its scene from.
    """

    APARTMENT = os.path.join("mjcf", "scene-bodies.xml")
    """
    The apartment scene GARMI acts in, in the ``iai_garmi_apartment`` package.
    """

    MILK = os.path.join("objects", "milk.stl")
    """
    The milk's mesh, under the coraplex resources.
    """

    BOWL = os.path.join("objects", "bowl.stl")
    """
    The bowl's mesh, under the coraplex resources.
    """

    SPOON = os.path.join("objects", "spoon.stl")
    """
    The spoon's mesh, under the coraplex resources.
    """

    @property
    def path(self) -> str:
        """
        :return: Where the file is read from.
        """
        if self is SceneFile.APARTMENT:
            return os.path.join(
                get_package_share_directory("iai_garmi_apartment"), self.value
            )
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "resources",
            self.value,
        )


class SceneBody(StrEnum):
    """
    The bodies of the scene the demonstration acts on.
    """

    MILK = "milk"
    """
    The milk put into the drawer.
    """

    BOWL = "bowl"
    """
    The bowl put into the drawer.
    """

    SPOON = "spoon"
    """
    The transported spoon.
    """

    SPOON_DRAWER = "drawer_1"
    """
    The drawer the spoon starts in.
    """

    STORAGE_DRAWER = "drawer_2"
    """
    The drawer the milk and the bowl are put into, the one below the spoon's.
    """

    APARTMENT = "apartment"
    """
    The apartment's origin, posed in the occupancy map frame the robot localizes in,
    which also marks whether the scene was already populated.
    """


# %% where things stand

HANDLE_RELEASE_CLEARANCE = 0.1
"""
How far the left hand backs off the drawer's handle once it has let go of it, in meters.
"""

MILK_HEIGHT_ABOVE_DRAWER_BEFORE_PLACING = 0.25
"""
How far above where it is laid down the milk is carried first, in meters, so it comes
into the drawer from above its walls rather than across them.
"""

RETRACT_HEIGHT_ABOVE_PLACE = 0.3
"""
How far straight up from where it let go of the milk the right hand withdraws, in
meters, so it is clear of the drawer before the arms park.
"""


@dataclass(frozen=True)
class ScenePlacement:
    """
    A position and a heading about the vertical.
    """

    x: float
    y: float
    z: float
    yaw: float = 0.0
    """
    The heading about the vertical, in radian.
    """

    pitch: float = 0.0
    """
    The tilt about the y-axis, in radian.
    """

    @property
    def transform(self) -> HomogeneousTransformationMatrix:
        """
        :return: The placement as a transform.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            self.x, self.y, self.z, pitch=self.pitch, yaw=self.yaw
        )

    @property
    def position(self) -> Point3:
        """
        :return: Where the placement is, without its orientation.
        """
        return Point3.from_iterable([self.x, self.y, self.z])

    def pose(self, reference_frame: KinematicStructureEntity) -> Pose:
        """
        :param reference_frame: The frame the placement is given in.
        :return: The placement as a pose in `reference_frame`.
        """
        return Pose.from_xyz_rpy(
            self.x,
            self.y,
            self.z,
            pitch=self.pitch,
            yaw=self.yaw,
            reference_frame=reference_frame,
        )


class ScenePose(Enum):
    """
    Where GARMI and the transported objects start, and where the objects are carried to.
    """

    GARMI_START = ScenePlacement(-4.8, 1.5, 0, yaw=math.pi)
    """
    Where GARMI starts, in its ``odom`` frame: facing the kitchen from the room.

    The simulated world's ``odom`` frame is the occupancy map frame the apartment is posed
    in, not the apartment's own frame the other placements are relative to.
    """

    GARMI_AT_COUNTER = ScenePlacement(-4.5, 1.75, 0, yaw=math.pi)
    """
    Where GARMI stands to pick up the milk with its right arm and to open and close the
    drawer with its left: between the drawer and the milk, facing the counter.
    """

    MILK_START = ScenePlacement(-6, 1.87, 1.028)
    """
    Where the milk starts, on the kitchen counter, relative to the apartment.
    """

    BOWL_START = ScenePlacement(0.2, 7.2, 0.973)
    """
    Where the bowl starts, on the kitchen counter to the right of the milk, relative to
    the apartment.
    """

    MILK_IN_DRAWER = ScenePlacement(
        -0.545, 6.965, 0.516, yaw=math.pi / 2, pitch=-math.pi / 2
    )
    """
    Where the milk is put into the drawer once that is open, relative to the apartment:
    lying on its side just above the drawer's bottom plate, so the fingertips, which
    reach below the carton, stay clear of it, along the drawer's depth and in the middle
    of the part that stands out of the cabinet.

    The milk is grasped from the front, along its x-axis, so tipping that axis up turns
    the grasp into one from above, and the hand lays the milk down from above. Turned so
    the hand is broad across the drawer rather than along it, which keeps it clear of the
    handle of the drawer above and the counter top while the wrist reaches down past the
    drawer's front.
    """

    BOWL_IN_DRAWER = ScenePlacement(-0.52, 6.925, 0.517)
    """
    Where the bowl is put into the drawer, relative to the apartment: right of the milk,
    so the hand putting it in does not reach over the milk.
    """

    MILK_TARGET = ScenePlacement(1.6, 5.2, 0.8)
    """
    Where the milk is carried to, relative to the apartment.
    """

    SPOON_START = ScenePlacement(-0.265, 0.0, -0.033)
    """
    Where the spoon starts, relative to its drawer: lying on the drawer's bottom plate,
    clear of its walls.

    The spoon only becomes a body collisions are checked against once it is grasped, so a
    placement that reaches through a wall goes unnoticed until the pick-up aborts on it.
    """

    SPOON_TARGET = ScenePlacement(1.6, 5.4, 0.8)
    """
    Where the spoon is carried to, relative to the apartment.
    """


# %% how the run is set up


@dataclass(frozen=True)
class DriveVelocityLimits:
    """
    How fast a base may move.
    """

    translation: float
    """
    In meter per second.
    """

    rotation: float
    """
    In radian per second.
    """


class GarmiDrive(Enum):
    """
    How fast GARMI's base moves in this demonstration.
    """

    DEMONSTRATION = DriveVelocityLimits(translation=0.1, rotation=0.1)


class SamplingSeed(IntEnum):
    """
    The seeds the plan's locations draw from.
    """

    REPEATABLE = 0
    """
    Fixes the poses the plan's locations draw, so this run repeats the one before it.

    The locations draw from their costmaps rather than ranking them, so an unpinned run
    stands somewhere new every time and reaches the drawer only on the attempts whose base
    pose happens to allow it.
    """


# %% the demonstration


@dataclass
class GarmiApartmentDemonstration(RobotDemonstration):
    """
    GARMI puts a milk carton from the kitchen counter into a drawer.
    """

    ros_node_name: str = "garmi_demo_node"

    def scene_specification(self) -> WorldSpecification:
        """
        The apartment's MuJoCo scene.

        The scene keeps its collision geometry in a file that is not loaded, so every
        geom has to stand in for it.
        """
        return WorldSpecification.from_mjcf(
            SceneFile.APARTMENT.path, use_visual_as_collision_backup=True
        )

    def robot_specification(self) -> RobotSpecification:
        """
        GARMI at its start, driving at the demonstration's speed.
        """
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            odom_T_robot_start=ScenePose.GARMI_START.value.transform,
            drive_translation_velocity_limits=GarmiDrive.DEMONSTRATION.value.translation,
            drive_rotation_velocity_limits=GarmiDrive.DEMONSTRATION.value.rotation,
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(
            SceneBody.APARTMENT.value
        )

    def populate_scene(self, world: World) -> None:
        """
        Annotate the apartment's furniture, then add the milk, the bowl and the spoon.

        The furniture is annotated first, so the reasoner describes the apartment the
        plan navigates rather than the objects the plan already knows.
        """
        world_reasoner = WorldReasoner(world)
        inferred = world_reasoner.infer_semantic_annotations()
        with world.modify_world():
            world.add_semantic_annotations(inferred)

        # %% milk

        # The objects are picked up and carried, and a pose can only be written to a
        # connection that has the degrees of freedom to carry it, so they hang off 6DoF
        # connections rather than the default fixed one.
        Milk.get_annotation_specification(
            SceneBody.MILK.value,
            BodySpecification.mesh(
                SceneBody.MILK.value,
                SceneFile.MILK.path,
                parent_T_self=ScenePose.MILK_START.value.transform,
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=world.root)

        # %% bowl

        # Bowl.get_annotation_specification(
        #     SceneBody.BOWL.value,
        #     BodySpecification.mesh(
        #         SceneBody.BOWL.value,
        #         SceneFile.BOWL.path,
        #         parent_T_self=ScenePose.BOWL_START.value.transform,
        #     ),
        #     parent_connection_specification=Connection6DoFSpecification(),
        # ).spawn(world, parent=world.get_body_by_name(SceneBody.APARTMENT.value))

        # %% spoon

        # Hanging off the drawer rather than the world root, so it travels with the drawer
        # when the plan opens it.
        Spoon.get_annotation_specification(
            SceneBody.SPOON.value,
            BodySpecification.mesh(
                SceneBody.SPOON.value,
                SceneFile.SPOON.path,
                parent_T_self=ScenePose.SPOON_START.value.transform,
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=world.get_body_by_name(SceneBody.SPOON_DRAWER.value))

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [
            RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0]),
            StatementGrounding(sampling_seed=SamplingSeed.REPEATABLE),
            PlanRewriting(transformations=[OpenDrawerBeforeMoveAndPickUp()]),
        ]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Pick up the milk from the front with the right arm, open the drawer with the
        left, lay the milk into it from above and close the drawer again, one step after
        the other.
        """
        world = context.world
        robot = context.require_extension(RobotAccess).robot
        left_arm = robot.get_left_arm_if_specified()
        right_arm = robot.get_right_arm_if_specified()
        storage_drawer = world.get_body_by_name(SceneBody.STORAGE_DRAWER.value)
        apartment = world.get_body_by_name(SceneBody.APARTMENT.value)
        [drawer] = [
            drawer
            for drawer in world.get_semantic_annotations_by_type(Drawer)
            if drawer.root is storage_drawer
        ]
        [milk] = world.get_semantic_annotations_by_type(Milk)
        milk_grasp = GraspCandidate(
            graspable=milk,
            grasp_pose=Pose.from_xyz_quaternion(
                quat_z=1, quat_w=0, reference_frame=milk.root
            ),
        )
        handle_grasp = GraspCandidate(
            graspable=drawer.handle,
            grasp_pose=Pose.from_xyz_quaternion(
                quat_y=0.707107,
                quat_z=0.707107,
                quat_w=0,
                reference_frame=drawer.handle.root,
            ),
        )
        milk_in_drawer = ScenePose.MILK_IN_DRAWER.value.pose(apartment)
        at_counter = ScenePose.GARMI_AT_COUNTER.value.pose(world.root)

        # Where the right hand's tool frame is once it has laid the milk down, lifted
        # along the apartment's z-axis, which is the vertical.
        tool_at_place = right_arm.end_effector.tool_frame_goal_for_grasp(
            milk_grasp.moved_to(milk_in_drawer)
        )
        tool_above_place = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                z=RETRACT_HEIGHT_ABOVE_PLACE, reference_frame=apartment
            )
            @ tool_at_place.homogeneous_matrix
        ).pose

        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    ParkArmsAction(arms=robot.all_arms),
                    # NavigateAction(target_location=at_counter),
                    # PickUpAction(grasp=milk_grasp, arm=right_arm),
                    # ParkArmsAction(arms=robot.all_arms),
                    # NavigateAction(target_location=at_counter),
                    # GraspOpenAndRetractAction(
                    #     grasp=handle_grasp,
                    #     arm=left_arm,
                    #     release_clearance=HANDLE_RELEASE_CLEARANCE,
                    # ),
                    # PlaceAction(
                    #     object_designator=milk,
                    #     target_location=milk_in_drawer,
                    #     retreat_distance=MILK_HEIGHT_ABOVE_DRAWER_BEFORE_PLACING,
                    # ),
                    # # A place leaves the milk under the world; under the drawer it moves
                    # # along when the drawer closes.
                    # MoveBranch(branch_root=milk.root, new_parent=storage_drawer),
                    # MoveManipulatorAction(
                    #     target_pose=tool_above_place,
                    #     end_effector=right_arm.end_effector,
                    #     allow_gripper_collision=False,
                    # ),
                    # ParkArmsAction(arms=robot.all_arms),
                    # NavigateAction(target_location=at_counter),
                    # GraspCloseAndRetractAction(
                    #     grasp=handle_grasp,
                    #     arm=left_arm,
                    #     release_clearance=HANDLE_RELEASE_CLEARANCE,
                    # ),
                ]
            )
        )
        return statechart


def main(
    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it
        drives the real robot or simulates it.
    """
    GarmiApartmentDemonstration(
        used_robot=Garmi,
        executor_type=RealRobotPlanExecutor,
        collision_avoidance=True,
        debug=True,
    ).run()


if __name__ == "__main__":
    main()
