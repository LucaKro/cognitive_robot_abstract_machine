"""
The PR2 lays a place setting: it carries the milk, a bowl and a spoon onto the table.

Runs in simulation against the apartment, so nothing on the network is needed. The
scaffolding in :mod:`coraplex.demonstrations` owns the ROS session and publishes the
world to Rviz, so the run can be watched while it happens.

The bowl is the interesting one. It offers a grasp all around its rim and only some of
them can be reached from anywhere the robot may stand, so the plan names its grasp as a
variable rather than letting the action take the first one the bowl generates. The domain
is worked out when that transport grounds -- by which time the milk has been put down and
the robot has moved -- rather than when the plan is built.
"""

import os
from dataclasses import dataclass

from typing_extensions import ClassVar

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.locations.factories import ReachableGrasps
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.testing import setup_world
from krrood.entity_query_language.factories import a, an, entity, variable
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Drawer,
    Handle,
    Milk,
    Spoon,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection

# %% the scene

OBJECT_MESH_DIRECTORY = os.path.join(
    os.path.dirname(__file__), "..", "..", "resources", "objects"
)
"""
Where the meshes this demonstration spawns are read from.
"""

ROBOT_START = HomogeneousTransformationMatrix.from_xyz_rpy(1.1, 2.5, 0)
"""
Where the PR2 stands before the plan begins.

Far enough back that its parked grippers clear the counter: at the 1.5 m
:func:`~coraplex.testing.setup_world` places it at, they sit inside cabinet9 and
cabinet10, which a run with collision avoidance refuses to start from. Behind 1.0 m the
torso meets the cabinet doors on the other side instead.
"""


@dataclass
class BulletWorldDemonstration(RobotDemonstration):
    """
    The PR2 transports the milk, a bowl and a spoon onto the table in the apartment.
    """

    ros_node_name: ClassVar[str] = "bullet_world_demo_node"

    def build_simulated_world(self) -> World:
        """
        The apartment with the PR2 in it, stood back from the counter.

        :func:`~coraplex.testing.setup_world` merges the robot's bodies but names no
        robot among the world's annotations, so it is registered here -- once, rather
        than per repetition, since registering twice leaves two of it.
        """
        world = setup_world()
        with world.modify_world():
            world.get_body_by_name("base_footprint").parent_connection.origin = (
                ROBOT_START
            )
        self.used_robot.from_world(world)
        return world

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name("bowl.stl")

    def populate_scene(self, world: World) -> None:
        """
        Put the bowl on the counter and the spoon in the drawer, and name what the plan
        acts on.
        """
        spoon = STLParser(os.path.join(OBJECT_MESH_DIRECTORY, "spoon.stl")).parse()
        bowl = STLParser(os.path.join(OBJECT_MESH_DIRECTORY, "bowl.stl")).parse()

        with world.modify_world():
            world.merge_world_at_pose(
                bowl,
                HomogeneousTransformationMatrix.from_xyz_quaternion(
                    2.4, 2.2, 1, reference_frame=world.root
                ),
            )
            world.merge_world(
                spoon,
                FixedConnection(
                    parent=world.get_body_by_name("cabinet10_drawer_top"),
                    child=spoon.root,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        -0.05, -0.05, 0
                    ),
                ),
            )

        with world.modify_world():
            WorldReasoner(world).reason()
            world.add_semantic_annotations(
                [
                    Bowl(root=world.get_body_by_name("bowl.stl")),
                    Spoon(root=world.get_body_by_name("spoon.stl")),
                ]
            )
            world.add_semantic_annotation_recursively(
                Drawer(
                    root=world.get_body_by_name("cabinet10_drawer_top"),
                    handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
                )
            )

    def build_context(self, world: World) -> Context:
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            _debug=True,
            alternative_motion_mappings=self.alternative_motion_mappings,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Carry each object to its place on the table.
        """
        world = context.world
        bowl = next(
            an(entity(variable(Bowl, domain=world.semantic_annotations))).evaluate()
        )
        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                TransportAction(
                    next(
                        an(
                            entity(variable(Milk, domain=world.semantic_annotations))
                        ).evaluate()
                    ),
                    Pose.from_xyz_rpy(
                        4.9, 3.3, 0.8, yaw=1.57, reference_frame=world.root
                    ),
                    Arms.LEFT,
                ),
                a(TransportAction)(
                    object_designator=bowl,
                    target_location=Pose.from_xyz_rpy(
                        5, 3.3, 0.75, yaw=1.57, reference_frame=world.root
                    ),
                    arm=Arms.LEFT,
                    grasp_pose=variable(
                        Pose, domain=ReachableGrasps(bowl, context, Arms.LEFT)
                    ),
                ),
                TransportAction(
                    next(
                        an(
                            entity(variable(Spoon, domain=world.semantic_annotations))
                        ).evaluate()
                    ),
                    Pose.from_xyz_rpy(
                        5.1, 3.3, 0.75, yaw=1.57, reference_frame=world.root
                    ),
                    Arms.LEFT,
                ),
            ],
            context=context,
        ).plan


def main(
    execution_type: ExecutionType = ExecutionType.SIMULATED,
    collision_avoidance: bool = True,
) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    :param collision_avoidance: Whether every motion state chart avoids collisions.
    """
    BulletWorldDemonstration(
        used_robot=PR2,
        execution_type=execution_type,
        collision_avoidance=collision_avoidance,
    ).run()


if __name__ == "__main__":
    main()
