"""
GARMI puts a milk carton and a bowl from the kitchen counter into a drawer.

GARMI opens the drawer with its left arm while it picks up the milk with its right, then
holds the drawer open while it puts the milk in, picks up the bowl and puts that in too,
and closes the drawer again. Running with :class:`~coraplex.plans.executors.RealRobotPlanExecutor` drives
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
from coraplex.querying.grasp_closeness import GraspsByCloseness
from coraplex.robot_plans.actions.core.container import (
    CloseAction,
    GraspsAndMovesContainerByItsHandle,
    OpenAction,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import (
    FixateObjectAction,
    GraspingAction,
    PickUpAction,
)
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.plan_transformations import OpenDrawerBeforeMoveAndPickUp
from cramph.composites import Parallel, Sequence, StartedWhenTrue
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from cramph.world_modification_nodes import MoveBranch
from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from giskardpy.motion_statechart.monitors.feature_monitors import DistanceMonitor
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from krrood.entity_query_language.factories import a, flat_variable, variable
from semantic_digital_twin.datastructures.definitions import GripperState
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
    Bowl,
    Drawer,
    Milk,
    Spoon,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.world_entity import (
    Body,
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

GRIPPER_CONTACT_DISTANCE = 0.2
"""
How close a gripper has to be to the handle it is headed for, in meters, before it may
touch its surroundings; further away it avoids them.
"""

RIGHT_HAND_WITHDRAWAL_BEFORE_CLOSING = 0.3
"""
How far the right hand has to be from where it put the bowl, in meters and ignoring
height, before the drawer starts closing, so the closing drawer does not push into it.
"""

MILK_TOP_GRASP_HEIGHT = 0.076
"""
How high above its origin the milk is grasped from above, in meters: 3 cm below the top
of the carton, so the fingers close around its top rather than the palm landing on it.
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

    @property
    def transform(self) -> HomogeneousTransformationMatrix:
        """
        :return: The placement as a transform.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            self.x, self.y, self.z, yaw=self.yaw
        )

    @property
    def position(self) -> Point3:
        """
        :return: Where the placement is, without its heading.
        """
        return Point3.from_iterable([self.x, self.y, self.z])

    def pose(self, reference_frame: KinematicStructureEntity) -> Pose:
        """
        :param reference_frame: The frame the placement is given in.
        :return: The placement as a pose in `reference_frame`.
        """
        return Pose.from_xyz_rpy(
            self.x, self.y, self.z, yaw=self.yaw, reference_frame=reference_frame
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
    Where GARMI stands to open the drawer with its left arm while it picks up the milk
    and the bowl with its right, relative to the apartment: between the drawer and the
    two, facing the counter.
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

    MILK_IN_DRAWER = ScenePlacement(-0.67, 6.925, 0.572)
    """
    Where the milk is put into the drawer once that is open, relative to the apartment:
    on the drawer's bottom plate, left of its middle, and turned as it stood on the
    counter, so the hand holding it does not have to turn.
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


# %% letting a gripper touch its surroundings


def gripper_contact_near(end_effector: EndEffector, handle: Body) -> StartedWhenTrue:
    """
    :param end_effector: The gripper that may touch its surroundings.
    :param handle: The handle the gripper is headed for or holds.
    :return: A node letting `end_effector` touch its surroundings from when it is within
        :data:`GRIPPER_CONTACT_DISTANCE` of `handle`. The distance is measured in the
        handle's frame, so it follows the handle as the drawer moves, and ignores that
        frame's z-axis, which runs along the handle, so it is the distance at right
        angles to the handle wherever along it the gripper is.
    """
    return StartedWhenTrue(
        monitor=DistanceMonitor(
            name=f"{end_effector.name} near {handle.name}",
            root_link=handle,
            tip_link=end_effector.tool_frame,
            reference_point=Point3(reference_frame=handle),
            tip_point=Point3(reference_frame=end_effector.tool_frame),
            lower_limit=0.0,
            upper_limit=GRIPPER_CONTACT_DISTANCE,
        ),
        monitored_node=UpdateTemporaryCollisionRules.for_end_effector(end_effector),
    )


# %% the demonstration


@dataclass
class GarmiApartmentDemonstration(RobotDemonstration):
    """
    GARMI carries a milk carton off the kitchen counter and a spoon out of a drawer,
    and places both on the table.
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
        Open the drawer below the spoon's with the left arm while picking up the milk
        with the right, then, while the left arm holds the drawer open, place the milk
        into it, pick up the bowl and place that into it too. Then close the drawer with
        the left arm, still holding its handle, while the right one parks, and let go of
        the handle and park the left arm.
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
        # [bowl] = world.get_semantic_annotations_by_type(Bowl)
        left_hand = left_arm.end_effector
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    ParkArmsAction(arms=robot.all_arms),
                    NavigateAction(
                        target_location=ScenePose.GARMI_AT_COUNTER.value.pose(
                            world.root
                        )
                    ),
                    # Both grasps are chosen once GARMI stands at the counter, as the
                    # ones closest to it there.
                    Parallel(
                        nodes=[
                            Sequence(
                                nodes=[
                                    a(GraspingAction)(
                                        grasp=flat_variable(
                                            GraspsByCloseness(
                                                drawer.handle,
                                                variable(
                                                    Pose,
                                                    domain=[
                                                        Pose(reference_frame=robot.root)
                                                    ],
                                                ),
                                                number_of_grasps=None,
                                            )
                                        ),
                                        arm=left_arm,
                                    ),
                                    OpenAction(arm=left_arm, handle=drawer.handle),
                                ]
                            ),
                            # Taken from above, so the hand stays over the milk
                            # rather than beside it when it goes into the drawer.
                            a(PickUpAction)(
                                grasp=flat_variable(
                                    GraspsByCloseness(
                                        milk,
                                        variable(
                                            Pose,
                                            domain=[Pose(reference_frame=robot.root)],
                                        ),
                                        number_of_grasps=None,
                                    )
                                ),
                                arm=right_arm,
                            ),
                            # Lets the left hand stay on the handle once the drawer is
                            # open, while the right one may still be picking up, but
                            # only once it is near the handle.
                            gripper_contact_near(left_hand, drawer.handle.root),
                        ]
                    ),
                    # The left hand holds the drawer open where it is while the right
                    # one puts the milk in, then the bowl.
                    #         Parallel(
                    #             nodes=[
                    #                 FixateObjectAction(arm=left_arm),
                    #                 Sequence(
                    #                     nodes=[
                    #                         # Grounded once it starts, when the milk is in
                    #                         # hand, since a place reads the grasp off the
                    #                         # gripper holding it.
                    #                         a(PlaceAction)(
                    #                             object_designator=milk,
                    #                             target_location=ScenePose.MILK_IN_DRAWER.value.pose(
                    #                                 apartment
                    #                             ),
                    #                         ),
                    #                         # A place leaves the milk under the world; under
                    #                         # the drawer it moves along when the drawer closes.
                    #                         MoveBranch(
                    #                             branch_root=milk.root, new_parent=storage_drawer
                    #                         ),
                    #                         a(PickUpAction)(
                    #                             grasp=flat_variable(
                    #                                 GraspsByCloseness(
                    #                                     bowl,
                    #                                     variable(
                    #                                         Pose,
                    #                                         domain=[
                    #                                             Pose(reference_frame=robot.root)
                    #                                         ],
                    #                                     ),
                    #                                     number_of_grasps=None,
                    #                                 )
                    #                             ),
                    #                             arm=right_arm,
                    #                         ),
                    #                         a(PlaceAction)(
                    #                             object_designator=bowl,
                    #                             target_location=ScenePose.BOWL_IN_DRAWER.value.pose(
                    #                                 apartment
                    #                             ),
                    #                         ),
                    #                         MoveBranch(
                    #                             branch_root=bowl.root, new_parent=storage_drawer
                    #                         ),
                    #                     ]
                    #                 ),
                    #             ]
                    #         ),
                    #         # The left hand never lets go of the handle, so it closes the
                    #         # drawer from where it holds it, while the right arm parks. The
                    #         # drawer only starts closing once the right hand has withdrawn
                    #         # from above it, so it does not push into the hand.
                    #         Parallel(
                    #             nodes=[
                    #                 ParkArmsAction(arms=[right_arm]),
                    #                 Sequence(
                    #                     nodes=[
                    #                         # Until then the left hand keeps the drawer where
                    #                         # it is, or GARMI's body, following the right arm,
                    #                         # would drag it off. The fixation is at its goal
                    #                         # all along, so the monitor decides when this ends.
                    #                         Parallel(
                    #                             nodes=[
                    #                                 FixateObjectAction(arm=left_arm),
                    #                                 DistanceMonitor(
                    #                                     name="Right hand withdrawn from the drawer",
                    #                                     root_link=world.root,
                    #                                     tip_link=right_arm.end_effector.tool_frame,
                    #                                     reference_point=ScenePose.BOWL_IN_DRAWER.value.pose(
                    #                                         apartment
                    #                                     ).position,
                    #                                     tip_point=Point3(
                    #                                         reference_frame=right_arm.end_effector.tool_frame
                    #                                     ),
                    #                                     lower_limit=RIGHT_HAND_WITHDRAWAL_BEFORE_CLOSING,
                    #                                     upper_limit=math.inf,
                    #                                 ),
                    #                             ],
                    #                         ),
                    #                         CloseAction(arm=left_arm, handle=drawer.handle),
                    #                     ]
                    #                 ),
                    #                 # Lets the left hand stay on the handle once the drawer is
                    #                 # closed, while the right arm, and with it GARMI's body,
                    #                 # may still be moving. The hand holds the handle from the
                    #                 # start, so the allowance is not left to wait a tick for
                    #                 # a distance monitor, with the fingers around the handle.
                    #                 UpdateTemporaryCollisionRules.for_end_effector(left_hand),
                    #             ]
                    #         ),
                    #         # Let go of the handle and back off it, along GARMI's approach
                    #         # axis, the tool frame's z-axis.
                    #         Parallel(
                    #             nodes=[
                    #                 UpdateTemporaryCollisionRules.for_end_effector(left_hand),
                    #                 Sequence(
                    #                     nodes=[
                    #                         MoveGripper(
                    #                             end_effector=left_hand, state=GripperState.OPEN
                    #                         ),
                    #                         CartesianPose(
                    #                             name="Back off handle",
                    #                             root_link=world.root,
                    #                             tip_link=left_hand.tool_frame,
                    #                             goal_pose=Pose.from_xyz_rpy(
                    #                                 z=-GraspsAndMovesContainerByItsHandle.release_clearance,
                    #                                 reference_frame=left_hand.tool_frame,
                    #                             ),
                    #                         ),
                    #                     ]
                    #                 ),
                    #             ]
                    #         ),
                    #         ParkArmsAction(arms=[left_arm]),
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
