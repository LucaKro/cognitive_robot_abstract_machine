"""
Tests for reading each of Tracy's signals out of the message its driver publishes.
"""

from __future__ import annotations

import numpy as np
import pytest
from control_msgs.msg import DynamicJointState, InterfaceValue
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState

from experiments.articulated_manipulation.tracy_signals import (
    ArmJointEffort,
    DriverInterface,
    DriverNamespace,
    FingerPosition,
    GripperMotorCurrent,
    ObjectDetection,
    ObjectDetectionStatus,
    TopicName,
    TracySide,
    TracySignalInventory,
    Unit,
    WristWrench,
)

# %% wrist wrench


def test_wrist_wrench_reads_force_then_torque():
    message = WrenchStamped()
    message.wrench.force.x, message.wrench.force.y, message.wrench.force.z = (
        1.0,
        2.0,
        3.0,
    )
    message.wrench.torque.x, message.wrench.torque.y, message.wrench.torque.z = (
        4.0,
        5.0,
        6.0,
    )

    values = WristWrench(side=TracySide.LEFT).read(message)

    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_wrist_wrench_channels_are_forces_then_torques():
    units = [channel.unit for channel in WristWrench(side=TracySide.LEFT).channels]

    assert units == [Unit.NEWTON] * 3 + [Unit.NEWTON_METRE] * 3


def test_wrist_wrench_is_published_by_the_arm_driver():
    signal = WristWrench(side=TracySide.RIGHT)

    assert signal.topic == TracySide.RIGHT.topic(DriverNamespace.ARM, TopicName.WRENCH)


def test_message_type_is_the_bound_generic_parameter():
    assert WristWrench.message_type() is WrenchStamped


def test_stamp_is_the_header_time_in_seconds():
    message = WrenchStamped()
    message.header.stamp.sec = 3
    message.header.stamp.nanosec = 500_000_000

    assert WristWrench(side=TracySide.LEFT).stamp(message) == pytest.approx(3.5)


# %% gripper joint states


def gripper_joint_state(side: TracySide) -> JointState:
    """
    A gripper joint state carrying an unrelated joint before the knuckle joint.
    """
    message = JointState()
    message.name = ["unrelated_joint", side.gripper_knuckle_joint]
    message.position = [0.1, 0.5]
    message.velocity = [0.0, 0.2]
    message.effort = [0.0, 40.0]
    return message


def test_finger_position_reads_the_knuckle_joint_position():
    message = gripper_joint_state(TracySide.LEFT)

    values = FingerPosition(side=TracySide.LEFT).read(message)

    np.testing.assert_array_equal(values, [message.position[1]])


def test_gripper_motor_current_reads_the_knuckle_joint_effort():
    message = gripper_joint_state(TracySide.RIGHT)

    values = GripperMotorCurrent(side=TracySide.RIGHT).read(message)

    np.testing.assert_array_equal(values, [message.effort[1]])


def test_joint_state_without_the_joint_is_not_read():
    message = gripper_joint_state(TracySide.LEFT)

    assert not FingerPosition(side=TracySide.RIGHT).provides(message)


def test_joint_state_without_efforts_is_not_read():
    message = gripper_joint_state(TracySide.LEFT)
    message.effort = []

    assert not GripperMotorCurrent(side=TracySide.LEFT).provides(message)


# %% arm joint states


def test_arm_joint_effort_is_read_in_joint_order_whatever_the_message_order():
    signal = ArmJointEffort(side=TracySide.LEFT)
    joints = TracySide.LEFT.arm_joints
    message = JointState()
    message.name = list(reversed(joints))
    message.position = [0.0] * len(joints)
    message.effort = [float(index) for index in range(len(joints))]

    values = signal.read(message)

    np.testing.assert_array_equal(values, list(reversed(message.effort)))


# %% object detection


def gripper_dynamic_joint_state(side: TracySide) -> DynamicJointState:
    """
    A gripper dynamic joint state reporting the object detection status after the
    position.
    """
    message = DynamicJointState()
    message.joint_names = [side.gripper_knuckle_joint]
    message.interface_values = [
        InterfaceValue(
            interface_names=[
                DriverInterface.POSITION,
                DriverInterface.OBJECT_DETECTION_STATUS,
            ],
            values=[0.3, float(ObjectDetectionStatus.OBJECT_DETECTED_CLOSING)],
        )
    ]
    return message


def test_object_detection_reads_the_status_interface():
    message = gripper_dynamic_joint_state(TracySide.LEFT)

    values = ObjectDetection(side=TracySide.LEFT).read(message)

    np.testing.assert_array_equal(
        values, [float(ObjectDetectionStatus.OBJECT_DETECTED_CLOSING)]
    )


def test_dynamic_joint_state_without_the_status_interface_is_not_read():
    message = gripper_dynamic_joint_state(TracySide.LEFT)
    message.interface_values[0].interface_names = [DriverInterface.POSITION]
    message.interface_values[0].values = [0.3]

    assert not ObjectDetection(side=TracySide.LEFT).provides(message)


# %% inventory


def test_inventory_holds_every_signal_for_both_sides():
    inventory = TracySignalInventory.of_tracy()

    signal_types = {type(signal) for signal in inventory.signals}
    for signal_type in signal_types:
        sides = {
            signal.side
            for signal in inventory.signals
            if isinstance(signal, signal_type)
        }
        assert sides == set(TracySide)
    assert signal_types == {
        WristWrench,
        FingerPosition,
        GripperMotorCurrent,
        ObjectDetection,
        ArmJointEffort,
    }
