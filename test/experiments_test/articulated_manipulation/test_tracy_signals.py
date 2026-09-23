"""
Tests for reading each of Tracy's signals out of the message its driver publishes.

Skipped where Tracy's description is not installed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pint
import pytest
from control_msgs.msg import DynamicJointState, InterfaceValue
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState

from krrood.adapters.json_serializer import from_json, to_json

from experiments.articulated_manipulation.tracy_signals import (
    ArmJointEffort,
    DriverInterface,
    FingerPosition,
    GripperMotorCurrent,
    ObjectDetection,
    PartWithoutDriverError,
    SignalChannel,
    TopicName,
    TracyDriverNamespace,
    TracySignalInventory,
    WrenchComponent,
    WristWrench,
)
from semantic_digital_twin.robots.robotiq_85_gripper import ObjectDetectionStatus
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.utils import tracy_installed

pytestmark = pytest.mark.skipif(
    not tracy_installed(), reason="iai_tracy_description is not installed"
)

# %% helpers


@pytest.fixture
def tracy(tracy_world) -> Tracy:
    return tracy_world.get_semantic_annotations_by_type(Tracy)[0]


def gripper_joint_state(knuckle_joint_name: str) -> JointState:
    """
    A gripper joint state carrying an unrelated joint before the knuckle joint.
    """
    message = JointState()
    message.name = ["unrelated_joint", knuckle_joint_name]
    message.position = [0.1, 0.5]
    message.velocity = [0.0, 0.2]
    message.effort = [0.0, 40.0]
    return message


def gripper_dynamic_joint_state(knuckle_joint_name: str) -> DynamicJointState:
    """
    A gripper dynamic joint state reporting the object detection status after the
    position.
    """
    message = DynamicJointState()
    message.joint_names = [knuckle_joint_name]
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


# %% wrist wrench


def test_wrist_wrench_reads_force_then_torque(tracy):
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

    values = WristWrench(sensor=tracy.left_arm.sensors[0]).read(message)

    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_wrist_wrench_channels_are_forces_then_torques(tracy):
    units = pint.get_application_registry()

    channels = WristWrench(sensor=tracy.left_arm.sensors[0]).channels

    assert [channel.unit for channel in channels] == [units.newton] * 3 + [
        units.newton * units.meter
    ] * 3


def test_wrist_wrench_is_published_by_its_arm_driver(tracy):
    signal = WristWrench(sensor=tracy.right_arm.sensors[0])

    assert signal.topic == f"{TracyDriverNamespace.RIGHT_ARM}/{TopicName.WRENCH}"


def test_message_type_is_the_bound_generic_parameter():
    assert WristWrench.message_type() is WrenchStamped


def test_stamp_is_the_header_time_in_seconds(tracy):
    message = WrenchStamped()
    message.header.stamp.sec = 3
    message.header.stamp.nanosec = 500_000_000

    assert WristWrench(sensor=tracy.left_arm.sensors[0]).stamp(
        message
    ) == pytest.approx(3.5)


# %% gripper joint states


def test_finger_position_reads_the_knuckle_joint_position(tracy):
    gripper = tracy.left_arm.end_effector
    message = gripper_joint_state(gripper.knuckle_joint.name.name)

    values = FingerPosition(gripper=gripper).read(message)

    np.testing.assert_array_equal(values, [message.position[1]])


def test_gripper_motor_current_is_read_back_as_register_counts(tracy):
    gripper = tracy.right_arm.end_effector
    message = gripper_joint_state(gripper.knuckle_joint.name.name)
    message.effort = [0.0, GripperMotorCurrent.driver_maximum_force]

    values = GripperMotorCurrent(gripper=gripper).read(message)

    np.testing.assert_allclose(values, [GripperMotorCurrent.register_maximum])


def test_joint_state_without_the_joint_is_not_read(tracy):
    message = gripper_joint_state(tracy.left_arm.end_effector.knuckle_joint.name.name)

    assert not FingerPosition(gripper=tracy.right_arm.end_effector).provides(message)


def test_joint_state_without_efforts_is_not_read(tracy):
    gripper = tracy.left_arm.end_effector
    message = gripper_joint_state(gripper.knuckle_joint.name.name)
    message.effort = []

    assert not GripperMotorCurrent(gripper=gripper).provides(message)


# %% arm joint states


def test_arm_joint_effort_is_read_in_joint_order_whatever_the_message_order(tracy):
    signal = ArmJointEffort(arm=tracy.left_arm)
    joint_names = [channel.name for channel in signal.channels]
    message = JointState()
    message.name = list(reversed(joint_names))
    message.position = [0.0] * len(joint_names)
    message.effort = [float(index) for index in range(len(joint_names))]

    values = signal.read(message)

    np.testing.assert_array_equal(values, list(reversed(message.effort)))


# %% object detection


def test_object_detection_reads_the_status_interface(tracy):
    gripper = tracy.left_arm.end_effector
    message = gripper_dynamic_joint_state(gripper.knuckle_joint.name.name)

    values = ObjectDetection(gripper=gripper).read(message)

    np.testing.assert_array_equal(
        values, [float(ObjectDetectionStatus.OBJECT_DETECTED_CLOSING)]
    )


def test_dynamic_joint_state_without_the_status_interface_is_not_read(tracy):
    gripper = tracy.left_arm.end_effector
    message = gripper_dynamic_joint_state(gripper.knuckle_joint.name.name)
    message.interface_values[0].interface_names = [DriverInterface.POSITION]
    message.interface_values[0].values = [0.3]

    assert not ObjectDetection(gripper=gripper).provides(message)


# %% driver namespaces


def test_part_without_a_driver_has_no_namespace(tracy):
    with pytest.raises(PartWithoutDriverError):
        TracyDriverNamespace.of_part(tracy.get_default_camera())


# %% units


def test_channel_survives_a_round_trip_through_json():
    units = pint.get_application_registry()
    channel = SignalChannel(
        name=WrenchComponent.TORQUE_X, unit=units.newton * units.meter
    )

    assert from_json(to_json(channel)) == channel


# %% inventory


def test_inventory_holds_every_signal_once_per_arm(tracy):
    inventory = TracySignalInventory.of_tracy(tracy)

    assert Counter(type(signal) for signal in inventory.signals) == {
        signal_type: len(tracy.arms)
        for signal_type in [
            WristWrench,
            FingerPosition,
            GripperMotorCurrent,
            ObjectDetection,
            ArmJointEffort,
        ]
    }
