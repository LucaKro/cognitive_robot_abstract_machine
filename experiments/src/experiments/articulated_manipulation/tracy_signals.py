"""
The signals the real Tracy's drivers publish that tell whether the hand holds the
handle and whether the mechanism follows it: the wrist wrench of each UR10e arm, and
the finger position, motor current and object detection status of each Robotiq 2F-85
gripper, plus the arms' joint efforts.

Topics, message fields and nominal rates are taken from the driver configuration Tracy
is brought up with (``iai_tracy``'s ``tracy_ros2.launch.py``), not measured; how often
they really arrive and how noisy they are is measured with
:mod:`experiments.articulated_manipulation.tracy_signal_measurement`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np
from control_msgs.msg import DynamicJointState, InterfaceValue
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from typing_extensions import Generic, List, Type, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.robots.tracy import TracyJoint

MessageType = TypeVar("MessageType")

NANOSECONDS_PER_SECOND = 1e9
"""
Nanoseconds in a second, for converting a message stamp to seconds.
"""

ARM_CONTROLLER_RATE = 500.0
"""
How often the UR10e's controller manager runs, in Hz, which is how often its
broadcasters publish (``ur10e_update_rate.yaml`` of the UR driver).
"""

GRIPPER_CONTROLLER_RATE = 100.0
"""
How often each gripper's controller manager runs, in Hz, which is how often its joint
state broadcaster publishes (``robotiq_controllers_85.yaml`` of ``iai_tracy``). The
driver reads the gripper over its serial connection in a separate loop, so new values
may arrive less often than this.
"""

# %% names in the drivers' vocabulary


class TracySide(StrEnum):
    """
    One of Tracy's two arms, each with its own gripper, spelled as the prefix of their
    namespaces and joint names.
    """

    LEFT = "left"
    RIGHT = "right"

    def topic(self, namespace: DriverNamespace, name: TopicName) -> str:
        """
        :param namespace: Which of this side's drivers publishes the topic.
        :param name: The topic's name within that driver's namespace.
        :return: The topic's full name.
        """
        return f"/{self}_{namespace}/{name}"

    @property
    def arm_joints(self) -> List[TracyJoint]:
        """
        The six joints of this side's arm, from the shoulder to the wrist.
        """
        if self == TracySide.LEFT:
            return [
                TracyJoint.LEFT_SHOULDER_PAN,
                TracyJoint.LEFT_SHOULDER_LIFT,
                TracyJoint.LEFT_ELBOW,
                TracyJoint.LEFT_WRIST_1,
                TracyJoint.LEFT_WRIST_2,
                TracyJoint.LEFT_WRIST_3,
            ]
        return [
            TracyJoint.RIGHT_SHOULDER_PAN,
            TracyJoint.RIGHT_SHOULDER_LIFT,
            TracyJoint.RIGHT_ELBOW,
            TracyJoint.RIGHT_WRIST_1,
            TracyJoint.RIGHT_WRIST_2,
            TracyJoint.RIGHT_WRIST_3,
        ]

    @property
    def gripper_knuckle_joint(self) -> TracyJoint:
        """
        The joint that drives this side's gripper, the only gripper joint the driver
        reports on the real robot; the gripper's other joints follow it through its
        linkage.
        """
        if self == TracySide.LEFT:
            return TracyJoint.LEFT_GRIPPER_LEFT_KNUCKLE
        return TracyJoint.RIGHT_GRIPPER_LEFT_KNUCKLE


class DriverNamespace(StrEnum):
    """
    The namespace, after the side's prefix, each of Tracy's drivers runs in.
    """

    ARM = "arm"
    GRIPPER = "gripper"


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


class Unit(StrEnum):
    """
    The unit a channel's values are in.
    """

    NEWTON = "N"
    NEWTON_METRE = "N m"
    RADIAN = "rad"

    GRIPPER_CURRENT_SCALE = "gripper current scale"
    """
    The gripper's motor current register (0 to 255) mapped linearly onto 0 to the
    gripper's maximum force (235 by default) by the driver. It is proportional to the
    current, and not a measured force.
    """

    ARM_MOTOR_EFFORT = "A or N m"
    """
    The arm's motor current, in amperes, or that current converted to a torque, in
    newton metres, depending on the UR driver's ``use_currents_as_efforts``.
    """

    STATUS = "status"
    """
    A member of :class:`ObjectDetectionStatus`.
    """


class ObjectDetectionStatus(IntEnum):
    """
    What the gripper reports about its fingers' motion, as its ``gOBJ`` register holds it.
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


# %% signals


@dataclass(frozen=True)
class SignalChannel:
    """
    One value a signal carries in every sample.
    """

    name: str
    """
    What the value is, as the driver names it.
    """

    unit: Unit
    """
    The unit of the value.
    """


@dataclass
class TracySignal(Generic[MessageType], SubClassSafeGeneric, ABC):
    """
    A signal of one side of Tracy, published by its driver as messages of the bound
    type.
    """

    side: TracySide
    """
    The side whose arm or gripper the signal comes from.
    """

    @classmethod
    def message_type(cls) -> Type[MessageType]:
        """
        :return: The type of the messages the signal is published in.
        """
        return cls.get_generic_type_parameters()[0]

    @property
    @abstractmethod
    def topic(self) -> str:
        """
        The topic the signal is published on.
        """

    @property
    @abstractmethod
    def channels(self) -> List[SignalChannel]:
        """
        The values the signal carries, in the order :meth:`read` returns them.
        """

    @property
    @abstractmethod
    def nominal_rate(self) -> float:
        """
        How often the driver is configured to publish the signal, in Hz.
        """

    def provides(self, message: MessageType) -> bool:
        """
        :param message: A message published on :attr:`topic`.
        :return: Whether the message carries the signal.
        """
        return True

    @abstractmethod
    def read(self, message: MessageType) -> np.ndarray:
        """
        :param message: A message that carries the signal.
        :return: The value of every channel, in the order of :attr:`channels`.
        """

    def stamp(self, message: MessageType) -> float:
        """
        :param message: A message that carries the signal.
        :return: When the driver took the sample, in seconds.
        """
        return (
            message.header.stamp.sec
            + message.header.stamp.nanosec / NANOSECONDS_PER_SECOND
        )


@dataclass
class WristWrench(TracySignal[WrenchStamped]):
    """
    The force and torque the arm's built-in sensor measures at its flange, expressed in
    the tool frame configured on the robot's controller and compensated for the
    configured payload.

    ..note:: The header names ``<side>_tool0`` as the frame, which is only right while
        the tool frame configured on the controller is the flange itself.
    """

    @property
    def topic(self) -> str:
        return self.side.topic(DriverNamespace.ARM, TopicName.WRENCH)

    @property
    def channels(self) -> List[SignalChannel]:
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
        return [SignalChannel(name=force, unit=Unit.NEWTON) for force in forces] + [
            SignalChannel(name=torque, unit=Unit.NEWTON_METRE) for torque in torques
        ]

    @property
    def nominal_rate(self) -> float:
        return ARM_CONTROLLER_RATE

    def read(self, message: WrenchStamped) -> np.ndarray:
        force, torque = message.wrench.force, message.wrench.torque
        return np.array([force.x, force.y, force.z, torque.x, torque.y, torque.z])


@dataclass
class JointStateSignal(TracySignal[JointState], ABC):
    """
    One field of some joints in the joint states a driver publishes.
    """

    @property
    @abstractmethod
    def joints(self) -> List[TracyJoint]:
        """
        The joints whose field the signal carries, in the order of :attr:`channels`.
        """

    @abstractmethod
    def field_values(self, message: JointState) -> List[float]:
        """
        :param message: A joint state.
        :return: The field the signal carries, for every joint in the message.
        """

    @property
    @abstractmethod
    def unit(self) -> Unit:
        """
        The unit of the field.
        """

    @property
    def channels(self) -> List[SignalChannel]:
        return [SignalChannel(name=joint, unit=self.unit) for joint in self.joints]

    def provides(self, message: JointState) -> bool:
        return len(self.field_values(message)) == len(message.name) and all(
            joint in message.name for joint in self.joints
        )

    def read(self, message: JointState) -> np.ndarray:
        values = self.field_values(message)
        return np.array([values[message.name.index(joint)] for joint in self.joints])


@dataclass
class GripperJointStateSignal(JointStateSignal, ABC):
    """
    One field of the gripper's driven knuckle joint, in the gripper driver's joint
    states.
    """

    @property
    def topic(self) -> str:
        return self.side.topic(DriverNamespace.GRIPPER, TopicName.JOINT_STATES)

    @property
    def joints(self) -> List[TracyJoint]:
        return [self.side.gripper_knuckle_joint]

    @property
    def nominal_rate(self) -> float:
        return GRIPPER_CONTROLLER_RATE


@dataclass
class FingerPosition(GripperJointStateSignal):
    """
    How far the gripper is closed, as the angle of its driven knuckle, from 0 when
    open. The driver reads it from the gripper's position register (0 to 255), so it
    is quantised; the reported velocity is only its difference between two samples.
    """

    @property
    def unit(self) -> Unit:
        return Unit.RADIAN

    def field_values(self, message: JointState) -> List[float]:
        return list(message.position)


@dataclass
class GripperMotorCurrent(GripperJointStateSignal):
    """
    The current drawn by the gripper's motor, which rises when the fingers press on
    something. The driver publishes it as the knuckle joint's effort.
    """

    @property
    def unit(self) -> Unit:
        return Unit.GRIPPER_CURRENT_SCALE

    def field_values(self, message: JointState) -> List[float]:
        return list(message.effort)


@dataclass
class ArmJointEffort(JointStateSignal):
    """
    The motor effort of every joint of the arm, which rises when the arm pulls or
    pushes against something.
    """

    @property
    def topic(self) -> str:
        return self.side.topic(DriverNamespace.ARM, TopicName.JOINT_STATES)

    @property
    def joints(self) -> List[TracyJoint]:
        return self.side.arm_joints

    @property
    def unit(self) -> Unit:
        return Unit.ARM_MOTOR_EFFORT

    @property
    def nominal_rate(self) -> float:
        return ARM_CONTROLLER_RATE

    def field_values(self, message: JointState) -> List[float]:
        return list(message.effort)


@dataclass
class ObjectDetection(TracySignal[DynamicJointState]):
    """
    The gripper's own verdict on whether its fingers stopped on an object, as an
    :class:`ObjectDetectionStatus`.

    ..note:: It is only published in the dynamic joint states, which the joint state
        broadcaster fills with every interface the driver exports.
    """

    @property
    def topic(self) -> str:
        return self.side.topic(DriverNamespace.GRIPPER, TopicName.DYNAMIC_JOINT_STATES)

    @property
    def channels(self) -> List[SignalChannel]:
        return [
            SignalChannel(
                name=DriverInterface.OBJECT_DETECTION_STATUS, unit=Unit.STATUS
            )
        ]

    @property
    def nominal_rate(self) -> float:
        return GRIPPER_CONTROLLER_RATE

    def provides(self, message: DynamicJointState) -> bool:
        return (
            self.side.gripper_knuckle_joint in message.joint_names
            and DriverInterface.OBJECT_DETECTION_STATUS
            in self.knuckle_interfaces(message).interface_names
        )

    def read(self, message: DynamicJointState) -> np.ndarray:
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
            message.joint_names.index(self.side.gripper_knuckle_joint)
        ]


# %% inventory


@dataclass
class TracySignalInventory:
    """
    The signals to measure on the real Tracy.
    """

    signals: List[TracySignal]
    """
    Every signal, of every side.
    """

    @classmethod
    def of_tracy(cls) -> TracySignalInventory:
        """
        :return: Every signal of both of Tracy's sides.
        """
        signal_types = [
            WristWrench,
            FingerPosition,
            GripperMotorCurrent,
            ObjectDetection,
            ArmJointEffort,
        ]
        return cls(
            signals=[
                signal_type(side=side)
                for side in TracySide
                for signal_type in signal_types
            ]
        )
