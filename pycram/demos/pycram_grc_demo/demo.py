### PyCram Tutorial: Writing Your Own Robot Plans

import rclpy

from pycram.datastructures.dataclasses import Context
from pycram.designators.location_designator import CostmapLocation
from pycram.testing import setup_world
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.pr2 import PR2

# Initialize the simulator and world
world = setup_world()

# Initialize ROS 2 for visualization
rclpy.init()
node = rclpy.create_node("pycram_visualizer")

# Setup TF and Marker publishers to see the world in RViz
tf_publisher = TFPublisher(world=world, node=node)
viz_publisher = VizMarkerPublisher(world=world, node=node)

# Define the robot and the execution context
robot = PR2.from_world(world)
context = Context.from_world(world)

#### 2. Components and Actions Summary
"""
DATA STRUCTURES:
- PoseStamped: Represents a 6D pose (position + orientation) relative to a frame.
    Example: PoseStamped.from_list([x, y, z], [qx, qy, qz, qw], frame=world.root)
- Arms: Enum for selecting robot arms (Arms.LEFT, Arms.RIGHT, Arms.BOTH).
- GripperState: Enum for gripper commands (GripperState.OPEN, GripperState.CLOSE).
- TorsoState: Enum for torso height (TorsoState.HIGH, TorsoState.LOW).
- Body: Represents any object or robot in the world. Retrieve via `world.get_body_by_name("name")`.

CORE ACTIONS (Atomic operations):
- NavigateActionDescription(target_location): Moves the robot base to a target pose.
- PickUpActionDescription(object_designator: Body, arm: Arms, grasp_description: GraspDescription): Grasps a specific object.
- PlaceActionDescription(object_designator: Body, target_location: PoseStamped, arm: Arms): Places the held object.
- LookAtActionDescription(target: PoseStamped): Points the robot's head/camera at a specific pose.
- ParkArmsActionDescription(arm: Arms): Moves arm(s) to a predefined 'parked' configuration.
- MoveTorsoActionDescription(torso_state: TorsoState): Adjusts the robot's torso height.
- SetGripperActionDescription(gripper: Manipulator, motion: GripperStateEnum): Manually opens or closes a gripper.

COMPOSITE ACTIONS (High-level behaviors):
- TransportActionDescription(object_designator, target_location, arm): 
    A complete sequence: Navigate to object -> Pick up -> Navigate to target -> Place.
- SearchActionDescription(object_type, search_region): 
    Commands the robot to find an object within a specific area.
"""

#### 3. Building and Executing a Plan
"""
Plans are constructed using `SequentialPlan` and executed within a `simulated_robot` context.
"""

from pycram.language import SequentialPlan
from pycram.process_module import simulated_robot
from pycram.robot_plans import (
    ParkArmsActionDescription,
    NavigateActionDescription,
    PickUpActionDescription,
    PlaceActionDescription,
)
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped

milk = world.get_body_by_name("milk.stl")

plan = SequentialPlan(
    context,
    ParkArmsActionDescription(Arms.BOTH),
    NavigateActionDescription(
        pickup_costmap := CostmapLocation(target=milk, reachable_for=robot)
    ),
    PickUpActionDescription(
        milk,
        arm=pickup_costmap.last_arm,
        grasp_description=pickup_costmap.last_grasp_description,
    ),
    ParkArmsActionDescription(Arms.BOTH),
    NavigateActionDescription(
        place_costmap := CostmapLocation(
            target=(
                place_pose := PoseStamped.from_list([4.9, 3.3, 0.8], frame=world.root)
            ),
            reachable_for=robot,
            reachable_arm=pickup_costmap.last_arm,
            grasp_descriptions=pickup_costmap.last_grasp_description,
        )
    ),
    PlaceActionDescription(
        object_designator=milk, target_location=place_pose, arm=place_costmap.last_arm
    ),
    ParkArmsActionDescription(Arms.BOTH),
)

# 2. Perform the plan
with simulated_robot:
    plan.perform()
