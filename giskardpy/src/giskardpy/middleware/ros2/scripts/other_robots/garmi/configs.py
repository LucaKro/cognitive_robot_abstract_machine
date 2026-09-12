from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.middleware.ros2.command_publishing import MultiDOFCommandFormat
from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    RootTipPair,
)
from giskardpy.model.world_config import WorldWithOmniDriveRobot
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.world_description.connections import OmniDrive

GARMI_LEFT_ARM_JOINTS = [
    "left_fr3_joint1",
    "left_fr3_joint2",
    "left_fr3_joint3",
    "left_fr3_joint4",
    "left_fr3_joint5",
    "left_fr3_joint6",
    "left_fr3_joint7",
]
"""
Names of the seven left FR3 arm joints, ordered from base to tip.
"""

GARMI_RIGHT_ARM_JOINTS = [
    "right_fr3_joint1",
    "right_fr3_joint2",
    "right_fr3_joint3",
    "right_fr3_joint4",
    "right_fr3_joint5",
    "right_fr3_joint6",
    "right_fr3_joint7",
]
"""
Names of the seven right FR3 arm joints, ordered from base to tip.
"""

GARMI_FINGER_JOINTS = [
    "left_fr3_finger_joint1",
    "left_fr3_finger_joint2",
    "right_fr3_finger_joint1",
    "right_fr3_finger_joint2",
]
"""
Names of the two finger joints of each FR3 gripper.
"""

GARMI_WHEEL_JOINTS = [
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
]
"""
Names of the four mecanum wheel joints of the base.
"""

GARMI_HEAD_JOINTS = ["o1_motor_1", "o1_motor_2"]
"""
Names of the head pan and tilt joints.
"""

GARMI_LIFT_JOINTS = ["lift_0_lower_joint", "lift_0_upper_joint"]
"""
Names of the two prismatic torso lift joints.
"""

GARMI_INTERACTIVE_MARKER_CHAINS = [
    RootTipPair(root_link="arm_mount_left_link", tip_link="left_fr3_hand_tcp"),
    RootTipPair(root_link="map", tip_link="right_fr3_hand_tcp"),
]
"""
The kinematic chains controllable via interactive markers.
"""


@dataclass
class WorldWithGarmiConfig(WorldWithOmniDriveRobot):
    """
    World configuration for the GARMI robot.

    Builds a map -> odom_combined -> GARMI kinematic tree using an omni-drive base.
    """

    odom_body_name: PrefixedName = field(
        default_factory=lambda: PrefixedName("odom_combined")
    )
    urdf_view: Garmi = field(kw_only=True, default=Garmi)


@dataclass
class GarmiStandaloneInterface(RobotInterfaceConfig):
    """
    Simulates the mecanum wheels, lift, head, both FR3 arms, grippers and drive of GARMI
    without talking to hardware.
    """

    def setup(self) -> None:
        self.register_controlled_joints(
            [
                *GARMI_WHEEL_JOINTS,
                *GARMI_LIFT_JOINTS,
                *GARMI_HEAD_JOINTS,
                *GARMI_LEFT_ARM_JOINTS,
                *GARMI_RIGHT_ARM_JOINTS,
                *GARMI_FINGER_JOINTS,
                self.world.get_connections_by_type(OmniDrive)[0].name,
            ]
        )


@dataclass
class GarmiVelocityInterface(RobotInterfaceConfig):
    """
    Closed-loop velocity interface for the real GARMI robot.

    Synchronizes the world state from the arm joint-state topic and sends joint
    velocities to the two arm group controllers. The base, head and lift are not wired
    up yet.
    """

    def setup(self) -> None:
        self.sync_joint_state_topic("/garmi/arms/joint_states")

        self.add_joint_velocity_group_controller(
            cmd_topic="/garmi/arms/left_arm_joint_velocity_controller/reference",
            connections=GARMI_LEFT_ARM_JOINTS,
            command_format=MultiDOFCommandFormat(),
        )
        self.add_joint_velocity_group_controller(
            cmd_topic="/garmi/arms/right_arm_joint_velocity_controller/reference",
            connections=GARMI_RIGHT_ARM_JOINTS,
            command_format=MultiDOFCommandFormat(),
        )
