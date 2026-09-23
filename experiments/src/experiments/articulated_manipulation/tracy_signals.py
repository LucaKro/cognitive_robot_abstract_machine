"""
The signals the real Tracy's drivers publish that tell whether the hand holds the
handle and whether the mechanism follows it: the wrist wrench of each UR10e arm, the
finger position, motor current and object detection status of each Robotiq 2F-85
gripper, and the arms' joint efforts.

Each signal is read for a part of Tracy's semantic model. What it is published on -
namespaces, topics, message fields and nominal rates - comes from the driver
configuration Tracy is brought up with (``iai_tracy``'s ``tracy_ros2.launch.py``), not
measured; how often the signals really arrive and how noisy they are is measured with
:mod:`experiments.articulated_manipulation.tracy_signal_measurement`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

import numpy as np
import numpy.typing as npt
import pint
from control_msgs.msg import DynamicJointState, InterfaceValue
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from typing_extensions import Dict, Generic, List, Type, TypeVar

from krrood.exceptions import DataclassException
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobotPart,
    Arm,
    ForceTorqueSensor,
)
from semantic_digital_twin.robots.robotiq_85_gripper import Robotiq85Gripper
from semantic_digital_twin.robots.tracy import (
    Tracy,
    TracyLeftArm,
    TracyLeftGripper,
    TracyLeftWristForceTorqueSensor,
    TracyRightArm,
    TracyRightGripper,
    TracyRightWristForceTorqueSensor,
)

MessageType = TypeVar("MessageType")

# %% exceptions


@dataclass
class PartWithoutDriverError(DataclassException):
    """
    Raised when a signal is asked for a part of Tracy that none of its drivers
    reports.
    """

    part_type: Type[AbstractRobotPart]
    """
    The type of the part.
    """

    def error_message(self) -> str:
        return f"No driver of Tracy reports on a {self.part_type.__name__}."

    def suggest_correction(self) -> str:
        return "Read signals only of Tracy's arms, their grippers and wrist sensors."


# %% names in the drivers' vocabulary


class TracyDriverNamespace(StrEnum):
    """
    The ROS namespace each of Tracy's drivers runs in.
    """

    LEFT_ARM = "/left_arm"
    RIGHT_ARM = "/right_arm"
    LEFT_GRIPPER = "/left_gripper"
    RIGHT_GRIPPER = "/right_gripper"

    @classmethod
    def of_part(cls, part: AbstractRobotPart) -> TracyDriverNamespace:
        """
        :param part: A part of Tracy.
        :return: The namespace of the driver that reports on the part.
        :raises PartWithoutDriverError: If none of Tracy's drivers reports on it.
        """
        namespaces: Dict[Type[AbstractRobotPart], TracyDriverNamespace] = {
            TracyLeftArm: cls.LEFT_ARM,
            TracyLeftWristForceTorqueSensor: cls.LEFT_ARM,
            TracyLeftGripper: cls.LEFT_GRIPPER,
            TracyRightArm: cls.RIGHT_ARM,
            TracyRightWristForceTorqueSensor: cls.RIGHT_ARM,
            TracyRightGripper: cls.RIGHT_GRIPPER,
        }
        if type(part) not in namespaces:
            raise PartWithoutDriverError(part_type=type(part))
        return namespaces[type(part)]


class TopicName(StrEnum):
    """
    Topics the drivers publish the signals on, within their namespace.
    """

    WRENCH = "force_torque_sensor_broadcaster/wrench"
    """
    The force/torque sensor broadcaster's topic; its ``topic_name`` parameter in the
    driver configuration is not read by the Jazzy broadcaster.
    """

    JOINT_STATES = "joint_states"
    """
    The joint state broadcaster's position, velocity and effort of every joint.
    """

    DYNAMIC_JOINT_STATES = "dynamic_joint_states"
    """
    The joint state broadcaster's every state interface of every joint, including
    those that are not a position, velocity or effort.
    """


class WrenchComponent(StrEnum):
    """
    The components of a wrench, named as the force/torque sensor's state interfaces.
    """

    FORCE_X = "force.x"
    FORCE_Y = "force.y"
    FORCE_Z = "force.z"
    TORQUE_X = "torque.x"
    TORQUE_Y = "torque.y"
    TORQUE_Z = "torque.z"


class DriverInterface(StrEnum):
    """
    Names of the state interfaces a driver reports in a dynamic joint state.
    """

    POSITION = "position"
    OBJECT_DETECTION_STATUS = "object_detection_status"


class ObjectDetectionStatus(IntEnum):
    """
    What a Robotiq gripper reports about its fingers' motion, as its ``gOBJ`` register
    holds it.
    """

    MOVING = 0
    """
    The fingers are moving towards the requested position.
    """

    OBJECT_DETECTED_OPENING = 1
    """
    The fingers stopped while opening because they touched something.
    """

    OBJECT_DETECTED_CLOSING = 2
    """
    The fingers stopped while closing because they touched something.
    """

    AT_REQUESTED_POSITION = 3
    """
    The fingers reached the requested position without touching anything.
    """


# %% channels


@dataclass(frozen=True)
class SignalChannel:
    """
    One value a signal carries in every sample.
    """

    name: str
    """
    What the value is, as the driver names it.
    """

    unit: pint.Unit
    """
    The unit of the value, from pint's application registry.
    """


# %% signals


@dataclass
class TracySignal(Generic[MessageType], SubClassSafeGeneric, ABC):
    """
    A signal about one part of Tracy, published by the part's driver as messages of
    the bound type.
    """

    @classmethod
    def message_type(cls) -> Type[MessageType]:
        """
        :return: The type of the messages the signal is published in.
        """
        return cls.get_generic_type_parameters()[0]

    @property
    @abstractmethod
    def part(self) -> AbstractRobotPart:
        """
        The part of Tracy the signal is about.
        """

    @property
    @abstractmethod
    def topic_name(self) -> TopicName:
        """
        The topic the signal is published on, within its driver's namespace.
        """

    @property
    @abstractmethod
    def channels(self) -> List[SignalChannel]:
        """
        The values the signal carries, in the order :meth:`read` returns them.
        """

    @property
    def topic(self) -> str:
        """
        The full name of the topic the signal is published on.
        """
        return f"{TracyDriverNamespace.of_part(self.part)}/{self.topic_name}"

    def provides(self, message: MessageType) -> bool:
        """
        :param message: A message published on :attr:`topic`.
        :return: Whether the message carries the signal.
        """
        return True

    @abstractmethod
    def read(self, message: MessageType) -> npt.NDArray[np.float64]:
        """
        :param message: A message that carries the signal.
        :return: The value of every channel, in the order of :attr:`channels`.
        """

    def stamp(self, message: MessageType) -> float:
        """
        :param message: A message that carries the signal.
        :return: When the driver took the sample, in seconds.
        """
        units = pint.get_application_registry()
        stamp = (
            message.header.stamp.sec * units.second
            + message.header.stamp.nanosec * units.nanosecond
        )
        return stamp.to(units.second).magnitude


@dataclass
class WristWrench(TracySignal[WrenchStamped]):
    """
    The force and torque a UR10e's built-in sensor measures at its flange, expressed
    in the tool frame configured on the robot's controller and compensated for the
    configured payload. It is published at the rate the UR10e's controller manager runs
    at, 500 Hz (``ur10e_update_rate.yaml`` of the UR driver).

    ..note:: The header names ``<side>_tool0`` as the frame, which is only right while
        the tool frame configured on the controller is the flange itself.
    """

    sensor: ForceTorqueSensor
    """
    The wrist sensor the wrench is measured by.
    """

    @property
    def part(self) -> AbstractRobotPart:
        return self.sensor

    @property
    def topic_name(self) -> TopicName:
        return TopicName.WRENCH

    @property
    def channels(self) -> List[SignalChannel]:
        units = pint.get_application_registry()
        forces = [
            WrenchComponent.FORCE_X,
            WrenchComponent.FORCE_Y,
            WrenchComponent.FORCE_Z,
        ]
        torques = [
            WrenchComponent.TORQUE_X,
            WrenchComponent.TORQUE_Y,
            WrenchComponent.TORQUE_Z,
        ]
        return [SignalChannel(name=force, unit=units.newton) for force in forces] + [
            SignalChannel(name=torque, unit=units.newton * units.meter)
            for torque in torques
        ]

    def read(self, message: WrenchStamped) -> npt.NDArray[np.float64]:
        force, torque = message.wrench.force, message.wrench.torque
        return np.array([force.x, force.y, force.z, torque.x, torque.y, torque.z])


@dataclass
class JointStateSignal(TracySignal[JointState], ABC):
    """
    One field of some joints in the joint states a driver publishes.
    """

    @property
    @abstractmethod
    def joint_names(self) -> List[str]:
        """
        The joints whose field the signal carries, in the order of :attr:`channels`.
        """

    @property
    @abstractmethod
    def unit(self) -> pint.Unit:
        """
        The unit the signal reports the field in.
        """

    @abstractmethod
    def field_values(self, message: JointState) -> List[float]:
        """
        :param message: A joint state.
        :return: The field the signal carries, for every joint in the message.
        """

    @property
    def topic_name(self) -> TopicName:
        return TopicName.JOINT_STATES

    @property
    def channels(self) -> List[SignalChannel]:
        return [SignalChannel(name=name, unit=self.unit) for name in self.joint_names]

    def provides(self, message: JointState) -> bool:
        return len(self.field_values(message)) == len(message.name) and all(
            name in message.name for name in self.joint_names
        )

    def read(self, message: JointState) -> npt.NDArray[np.float64]:
        values = self.field_values(message)
        return np.array([values[message.name.index(name)] for name in self.joint_names])


@dataclass
class GripperJointStateSignal(JointStateSignal, ABC):
    """
    One field of a gripper's driven knuckle joint, the only gripper joint its driver
    reports.

    It is published at the rate the gripper's controller manager runs at, 100 Hz
    (``robotiq_controllers_85.yaml`` of ``iai_tracy``). The driver reads the gripper over
    its serial connection in a separate loop, so new values may arrive less often.
    """

    gripper: Robotiq85Gripper
    """
    The gripper the signal is about.
    """

    @property
    def part(self) -> AbstractRobotPart:
        return self.gripper

    @property
    def joint_names(self) -> List[str]:
        return [self.gripper.knuckle_joint.name.name]


@dataclass
class FingerPosition(GripperJointStateSignal):
    """
    How far the gripper is closed, as the angle of its driven knuckle, from 0 when
    open. The driver reads it from the gripper's position register (0 to 255), so it
    is quantised; the reported velocity is only its difference between two samples.
    """

    @property
    def unit(self) -> pint.Unit:
        return pint.get_application_registry().radian

    def field_values(self, message: JointState) -> List[float]:
        return list(message.position)


@dataclass
class GripperMotorCurrent(GripperJointStateSignal):
    """
    The current drawn by the gripper's motor, which rises when the fingers press on
    something, as the count its current register holds.

    The driver publishes the register as the knuckle joint's effort, mapped linearly
    onto 0 to the gripper's maximum force; it is read back as counts, since it is a
    current and not a force.
    """

    driver_maximum_force: float = field(default=235.0, kw_only=True)
    """
    The driver's ``gripper_max_force``, which the register's largest count is mapped
    onto; 235 is its default for the 2F-85 (``2f_85.ros2_control.xacro`` of
    ``ros2_robotiq_gripper``).
    """

    @property
    def register_maximum(self) -> int:
        """
        The largest count the gripper's current register holds, which is one byte.
        """
        return int(np.iinfo(np.uint8).max)

    @property
    def unit(self) -> pint.Unit:
        return pint.get_application_registry().count

    def field_values(self, message: JointState) -> List[float]:
        return list(message.effort)

    def read(self, message: JointState) -> npt.NDArray[np.float64]:
        return super().read(message) * self.register_maximum / self.driver_maximum_force


@dataclass
class ArmJointEffort(JointStateSignal):
    """
    The motor current of every joint of the arm, which rises when the arm pulls or
    pushes against something.

    It is published with the arm's joint states at the rate the UR10e's controller
    manager runs at, 500 Hz (``ur10e_update_rate.yaml`` of the UR driver).

    ..note:: The UR driver reports the current as the joints' effort unless its
        ``use_currents_as_efforts`` is turned off, in which case it is converted to a
        torque.
    """

    arm: Arm
    """
    The arm the signal is about.
    """

    @property
    def part(self) -> AbstractRobotPart:
        return self.arm

    @property
    def joint_names(self) -> List[str]:
        return [connection.name.name for connection in self.arm.active_connections]

    @property
    def unit(self) -> pint.Unit:
        return pint.get_application_registry().ampere

    def field_values(self, message: JointState) -> List[float]:
        return list(message.effort)


@dataclass
class ObjectDetection(TracySignal[DynamicJointState]):
    """
    The gripper's own verdict on whether its fingers stopped on an object, as an
    :class:`ObjectDetectionStatus`. It is published with the gripper's joint states, at
    100 Hz.

    ..note:: It is only published in the dynamic joint states, which the joint state
        broadcaster fills with every interface the driver exports.
    """

    gripper: Robotiq85Gripper
    """
    The gripper the signal is about.
    """

    @property
    def part(self) -> AbstractRobotPart:
        return self.gripper

    @property
    def topic_name(self) -> TopicName:
        return TopicName.DYNAMIC_JOINT_STATES

    @property
    def channels(self) -> List[SignalChannel]:
        return [
            SignalChannel(
                name=DriverInterface.OBJECT_DETECTION_STATUS,
                unit=pint.get_application_registry().dimensionless,
            )
        ]

    def provides(self, message: DynamicJointState) -> bool:
        return (
            self.gripper.knuckle_joint.name.name in message.joint_names
            and DriverInterface.OBJECT_DETECTION_STATUS
            in self.knuckle_interfaces(message).interface_names
        )

    def read(self, message: DynamicJointState) -> npt.NDArray[np.float64]:
        interfaces = self.knuckle_interfaces(message)
        status_index = interfaces.interface_names.index(
            DriverInterface.OBJECT_DETECTION_STATUS
        )
        return np.array([interfaces.values[status_index]])

    def knuckle_interfaces(self, message: DynamicJointState) -> InterfaceValue:
        """
        :param message: A dynamic joint state that reports the driven knuckle joint.
        :return: Every interface the driver reports for that joint.
        """
        return message.interface_values[
            message.joint_names.index(self.gripper.knuckle_joint.name.name)
        ]


# %% inventory


@dataclass
class TracySignalInventory:
    """
    The signals to measure on the real Tracy.
    """

    signals: List[TracySignal]
    """
    Every signal, of every arm.
    """

    @classmethod
    def of_tracy(cls, tracy: Tracy) -> TracySignalInventory:
        """
        :param tracy: Tracy's semantic model.
        :return: Every signal about Tracy's arms, their grippers and wrist sensors.
        """
        signals: List[TracySignal] = []
        for arm in tracy.arms:
            signals.extend(
                WristWrench(sensor=sensor)
                for sensor in arm.sensors
                if isinstance(sensor, ForceTorqueSensor)
            )
            signals.extend(
                [
                    FingerPosition(gripper=arm.end_effector),
                    GripperMotorCurrent(gripper=arm.end_effector),
                    ObjectDetection(gripper=arm.end_effector),
                    ArmJointEffort(arm=arm),
                ]
            )
        return cls(signals=signals)
