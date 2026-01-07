import os

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.process_module import simulated_robot
from pycram.robot_plans import ParkArmsActionDescription
from pycram.robot_plans import (
    PlaceActionDescription,
    NavigateActionDescription,
    PickUpActionDescription,
)
from pycram.testing import setup_world_mmp
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.mmp_dresden import MMPDresden
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)

world = setup_world_mmp()  # setup_world()

try:
    import rclpy

    rclpy.init()
    from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher

    v = VizMarkerPublisher(world, rclpy.create_node("viz_marker"))
except ImportError:
    pass

pr2 = MMPDresden.from_world(world)
context = Context.from_world(world)


with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()

with simulated_robot:
    SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        NavigateActionDescription(
            PoseStamped.from_list(
                position=[3.7, 2.0, 0], orientation=[0, 0, 1, 0], frame=world.root
            ),
            True,
        ),
        PickUpActionDescription(
            world.get_body_by_name("milk.stl"),
            Arms.LEFT,
            grasp_description=GraspDescription(
                ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
            ),
        ),
        ParkArmsActionDescription(Arms.BOTH),
        NavigateActionDescription(
            PoseStamped.from_spatial_type(
                HomogeneousTransformationMatrix.from_xyz_quaternion(
                    4.7, 2.3, 0, 0, 0, 1, 1, reference_frame=world.root
                ),
            ),
            True,
        ),
        PlaceActionDescription(
            world.get_body_by_name("milk.stl"),
            PoseStamped.from_list([4.9, 3.3, 0.8], [0, 0, 1, 1], frame=world.root),
            Arms.LEFT,
        ),
        ParkArmsActionDescription(Arms.BOTH),
        NavigateActionDescription(
            PoseStamped.from_list(
                position=[3.7, 2.0, 0], orientation=[0, 0, 1, 0], frame=world.root
            ),
            True,
        ),
        PickUpActionDescription(
            world.get_body_by_name("breakfast_cereal.stl"),
            Arms.LEFT,
            grasp_description=GraspDescription(
                ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
            ),
        ),
        ParkArmsActionDescription(Arms.BOTH),
        NavigateActionDescription(
            PoseStamped.from_spatial_type(
                HomogeneousTransformationMatrix.from_xyz_quaternion(
                    4.9, 2.4, 0, 0, 0, 1, 1, reference_frame=world.root
                ),
            ),
            True,
        ),
        PlaceActionDescription(
            world.get_body_by_name("breakfast_cereal.stl"),
            PoseStamped.from_list([5.2, 3.3, 0.8], [0, 0, 1, 1], frame=world.root),
            Arms.LEFT,
        ),
        ParkArmsActionDescription(Arms.BOTH),
    ).perform()
