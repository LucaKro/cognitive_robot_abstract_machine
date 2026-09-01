"""
GARMI transports a bowl and a spoon across the apartment.

The bowl starts on the kitchen counter and the spoon inside a drawer, and both are carried
to the table. Running with :attr:`~coraplex.datastructures.enums.ExecutionType.REAL` drives
the actual robot and takes the world from the running world server. The default runs the
whole plan in simulation against a world built from the apartment's MuJoCo scene and
GARMI's URDF, so nothing on the network is needed.

Needs the ``iai_garmi_apartment`` and ``garmi_description`` packages built in the
workspace, since the scene and the robot description are read from their share
directories.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from ament_index_python.packages import get_package_share_directory
from typing_extensions import ClassVar

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    ExecutionType,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World

# %% the apartment and the robot in it

GARMI_ENV_XML = os.path.join(
    get_package_share_directory("iai_garmi_apartment"), "mjcf", "scene-bodies.xml"
)
"""
The apartment scene GARMI acts in.
"""

ODOM_T_GARMI_START = HomogeneousTransformationMatrix.from_xyz_rpy(
    0, 6, 0, yaw=math.pi / 2
)
"""
Where GARMI starts, in its ``odom`` frame.
"""

DRIVE_TRANSLATION_VELOCITY_LIMITS = 0.1
"""
How fast the base drives, in meter per second.
"""

DRIVE_ROTATION_VELOCITY_LIMITS = 0.1
"""
How fast the base turns, in radian per second.
"""

# %% the transported objects

OBJECT_RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../", "resources", "objects"
)
"""
Where the meshes of the transported objects are kept.
"""

BOWL_NAME = "bowl"
"""
Name of the transported bowl, which also marks whether the scene was already populated.
"""

BOWL_STL = os.path.join(OBJECT_RESOURCES, "bowl.stl")
"""
The bowl's mesh.
"""

BOWL_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0.0, 7.2, 1.0)
"""
Where the bowl starts, on the kitchen counter.
"""

BOWL_TARGET_POINT = Point3.from_iterable([1.6, 5.2, 0.8])
"""
Where the bowl is carried to.
"""

SPOON_NAME = "spoon"
"""
Name of the transported spoon.
"""

SPOON_STL = os.path.join(OBJECT_RESOURCES, "spoon.stl")
"""
The spoon's mesh.
"""

SPOON_DRAWER_NAME = "drawer_1"
"""
Name of the drawer the spoon starts in.
"""

SPOON_IN_DRAWER_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(-0.09, 0.0, -0.069)
"""
Where the spoon starts, relative to its drawer: lying on the drawer's bottom plate,
clear of its walls.

The spoon only becomes a body collisions are checked against once it is grasped, so a
placement that reaches through a wall goes unnoticed until the pick-up aborts on it.
"""

SPOON_TARGET_POINT = Point3.from_iterable([1.6, 5.3, 0.8])
"""
Where the spoon is carried to.
"""

# %% the demonstration


@dataclass
class GarmiApartmentDemonstration(RobotDemonstration):
    """
    GARMI carries a bowl off the kitchen counter and a spoon out of a drawer, and places
    both on the table.
    """

    ros_node_name: ClassVar[str] = "garmi_demo_node"

    def build_simulated_world(self) -> World:
        """
        Put GARMI into the apartment's MuJoCo scene.

        The scene keeps its collision geometry in a file that is not loaded, so every
        geom has to stand in for it.
        """
        return WorldSpecification.from_mjcf(
            GARMI_ENV_XML,
            use_visual_as_collision_backup=True,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    odom_T_robot_start=ODOM_T_GARMI_START,
                    drive_translation_velocity_limits=DRIVE_TRANSLATION_VELOCITY_LIMITS,
                    drive_rotation_velocity_limits=DRIVE_ROTATION_VELOCITY_LIMITS,
                )
            ],
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(BOWL_NAME)

    def populate_scene(self, world: World) -> None:
        """
        Annotate the apartment's furniture, then add the bowl and the spoon.

        The furniture is annotated first, so the reasoner describes the apartment the
        plan navigates rather than the two objects the plan already knows.
        """
        world_reasoner = WorldReasoner(world)
        inferred = world_reasoner.infer_semantic_annotations()
        with world.modify_world():
            world.add_semantic_annotations(inferred)

        # %% bowl

        # Both objects are picked up and carried, and a pose can only be written to a
        # connection that has the degrees of freedom to carry it, so they hang off 6DoF
        # connections rather than the default fixed one.
        Bowl.get_annotation_specification(
            BOWL_NAME,
            BodySpecification.mesh(BOWL_NAME, BOWL_STL, parent_T_self=BOWL_POSE),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world)

        # %% spoon

        # Hanging off the drawer rather than the world root, so it travels with the drawer
        # when the plan opens it.
        Spoon.get_annotation_specification(
            SPOON_NAME,
            BodySpecification.mesh(
                SPOON_NAME, SPOON_STL, parent_T_self=SPOON_IN_DRAWER_POSE
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=world.get_body_by_name(SPOON_DRAWER_NAME))

    def build_context(self, world: World) -> Context:
        """
        Build the plan context around the GARMI in ``world``.

        ..note:: The ROS node has to be in the context for a real robot.
        """
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            evaluate_conditions=True,
            alternative_motion_mappings=self.alternative_motion_mappings,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Carry the bowl and then the spoon to the table.
        """
        world = context.world
        end_effector = context.robot.get_right_arm_if_specified().end_effector

        return sequential(
            [
                ParkArmsAction(arm=Arms.BOTH),
                # Note: always need TorsoState.HIGH or next(iter(self)) of CostmapLocation fails
                TransportAction(
                    object_designator=world.get_semantic_annotations_by_type(Bowl)[0],
                    arm=Arms.RIGHT,
                    grasp_description=GraspDescription(
                        ApproachDirection.RIGHT,
                        VerticalAlignment.TOP,
                        end_effector,
                        rotate_gripper=True,
                    ),
                    target_location=Pose(
                        position=BOWL_TARGET_POINT, reference_frame=world.root
                    ),
                ),
                TransportAction(
                    object_designator=world.get_semantic_annotations_by_type(Spoon)[0],
                    arm=Arms.RIGHT,
                    grasp_description=GraspDescription(
                        ApproachDirection.RIGHT,
                        VerticalAlignment.TOP,
                        rotate_gripper=True,
                        end_effector=end_effector,
                    ),
                    target_location=Pose(
                        position=SPOON_TARGET_POINT, reference_frame=world.root
                    ),
                ),
            ],
            context,
        )


def main(execution_type: ExecutionType = ExecutionType.SIMULATED) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    """
    GarmiApartmentDemonstration(
        used_robot=Garmi, execution_type=execution_type, collision_avoidance=True
    ).run()


if __name__ == "__main__":
    main()
