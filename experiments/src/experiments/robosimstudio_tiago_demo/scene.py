"""
The physics scene RoboSimStudio simulates: the demonstration's world exported to MJCF by
:mod:`semantic_digital_twin`, with the physics choices of this demonstration applied on
top.

The robot's links follow their servos and pass through everything but the finger pads,
the apartment does not collide with itself, and the groceries collide with the apartment
and the finger pads.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations
from pathlib import Path

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.multi_sim import MujocoBuilder
from semantic_digital_twin.robots.tiago import TiagoJoint
from semantic_digital_twin.world_description.connection_properties import ServoGains
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.world_entity import GravityCompensation

from experiments.robosimstudio_tiago_demo.world import Grocery, TiagoApartment

# %% MJCF vocabulary


class Element(StrEnum):
    """
    The MJCF elements the scene reads or writes.
    """

    BODY = "body"
    GEOM = "geom"
    JOINT = "joint"
    INERTIAL = "inertial"
    ACTUATOR = "actuator"
    POSITION = "position"
    CONTACT = "contact"
    EXCLUDE = "exclude"
    KEYFRAME = "keyframe"
    WORLDBODY = "worldbody"


class Attribute(StrEnum):
    """
    The MJCF attributes the scene reads or writes.
    """

    NAME = "name"
    TYPE = "type"
    AXIS = "axis"
    ARMATURE = "armature"
    DAMPING = "damping"
    JOINT = "joint"
    STIFFNESS = "kp"
    VELOCITY_GAIN = "kv"
    CONTACT_TYPE = "contype"
    CONTACT_AFFINITY = "conaffinity"
    CONTACT_DIMENSION = "condim"
    FRICTION = "friction"
    BODY_1 = "body1"
    BODY_2 = "body2"
    MASS = "mass"
    DIAGONAL_INERTIA = "diaginertia"
    MESH = "mesh"
    SIZE = "size"
    POSITION = "pos"
    COLOR = "rgba"


class JointType(StrEnum):
    """
    The MJCF joint types the scene uses.
    """

    SLIDE = "slide"
    HINGE = "hinge"
    FREE = "free"


class GeometryType(StrEnum):
    """
    The MJCF geometry types the scene uses.
    """

    BOX = "box"


class ContactSetting(StrEnum):
    """
    Values of a geometry's contact type and affinity.
    """

    NONE = "0"


class NameSuffix(StrEnum):
    """
    Suffixes of the names of elements the scene adds.
    """

    SERVO = "_servo"
    VISUAL = "_visual"


def as_mjcf_vector(values: Iterable[float]) -> str:
    """
    :param values: The numbers of an MJCF vector attribute.
    :return: The attribute's text.
    """
    return " ".join(str(value) for value in values)


# %% physics choices


class BaseJoint(StrEnum):
    """
    The joints the exported scene moves the robot's base with, standing in for the
    drive, which the export leaves out.
    """

    X = "base_x"
    Y = "base_y"
    YAW = "base_yaw"

    @property
    def joint_type(self) -> JointType:
        """
        Whether the joint slides or turns.
        """
        return JointType.HINGE if self is BaseJoint.YAW else JointType.SLIDE

    @property
    def axis(self) -> list[float]:
        """
        The axis the joint moves along or turns about, in the base frame.
        """
        return {
            BaseJoint.X: [1.0, 0.0, 0.0],
            BaseJoint.Y: [0.0, 1.0, 0.0],
            BaseJoint.YAW: [0.0, 0.0, 1.0],
        }[self]

    @classproperty
    def merged_label(cls) -> str:
        """
        The label of the one joint Newton merges the base joints of a body into; its
        coordinates follow the order of the members.
        """
        return "_".join(cls)


def tiago_servo_gains() -> dict[str, ServoGains]:
    """
    :return: The gains of the servo on each joint the plan commands, by joint name.
    """
    unlimited = float("inf")
    base = {
        BaseJoint.X: ServoGains(1e5, 6e3, unlimited),
        BaseJoint.Y: ServoGains(1e5, 6e3, unlimited),
        BaseJoint.YAW: ServoGains(3e4, 2e3, unlimited),
    }
    body = {
        TiagoJoint.TORSO_LIFT: ServoGains(2e4, 2e3, unlimited),
        TiagoJoint.HEAD_1: ServoGains(100.0, 10.0, unlimited),
        TiagoJoint.HEAD_2: ServoGains(100.0, 10.0, unlimited),
        TiagoJoint.LEFT_GRIPPER_FINGER: ServoGains(80.0, 3.0, unlimited),
        TiagoJoint.RIGHT_GRIPPER_FINGER: ServoGains(80.0, 3.0, unlimited),
    }
    upper_arms = {
        joint: ServoGains(2000.0, 100.0, unlimited)
        for side in ("LEFT", "RIGHT")
        for joint in (TiagoJoint[f"{side}_ARM_{number}"] for number in range(1, 5))
    }
    forearms = {
        joint: ServoGains(500.0, 25.0, unlimited)
        for side in ("LEFT", "RIGHT")
        for joint in (TiagoJoint[f"{side}_ARM_{number}"] for number in range(5, 8))
    }
    return {
        str(joint): gains
        for joint, gains in {**base, **body, **upper_arms, **forearms}.items()
    }


@dataclass
class PhysicsChoices:
    """
    How this demonstration wants its world simulated, on top of what the export
    describes.
    """

    servo_gains: dict[str, ServoGains] = field(default_factory=tiago_servo_gains)
    """
    The gains of the servo on each joint the plan commands, by joint name.
    """

    base_armature: float = 1.0
    """
    The armature of the base joints, which steadies the base under its stiff servos.
    """

    gripper_joint_armature: float = 0.001
    """
    The armature of the gripper's linkage joints, which keeps the coupled linkage from
    chattering.
    """

    gripper_joint_damping: float = 0.05
    """
    The damping of the gripper's linkage joints.
    """

    grip_friction: list[float] = field(default_factory=lambda: [2.0, 0.05, 0.001])
    """
    The sliding, torsional and rolling friction of the finger pads, as rubber pads have.
    """

    grip_contact_dimension: int = 4
    """
    The contact dimension of the pads and the groceries, which adds torsional friction
    so a held object does not spin about the line between the pads.
    """

    frame_only_mass: float = 1e-4
    """
    The mass of a robot link that only marks a frame, such as a grasping frame.
    """

    frame_only_moment: float = 1e-8
    """
    Each principal moment of a robot link that only marks a frame.
    """

    single_precision_margin: float = 1.01
    """
    How far the principal moments must clear the triangle inequality to still satisfy
    it once Newton rounds them to single precision.
    """


# %% export


@dataclass
class PhysicsScene:
    """
    An MJCF scene of the demonstration's world, with this demonstration's physics choices
    applied.
    """

    tree: ElementTree.ElementTree
    """
    The scene's document.
    """

    robot_root_name: str
    """
    The name of the robot's root body, whose subtree is the robot.
    """

    grip_body_names: list[str]
    """
    The names of the robot bodies that touch what the robot grasps.
    """

    gripper_body_names: list[str]
    """
    The names of the bodies of the robot's grippers.
    """

    choices: PhysicsChoices
    """
    The physics choices applied to the scene.
    """

    @classmethod
    def export(
        cls,
        apartment: TiagoApartment,
        file_path: Path,
        choices: PhysicsChoices | None = None,
    ) -> PhysicsScene:
        """
        Export ``apartment`` and apply the physics choices of this demonstration.

        :param apartment: The world to export.
        :param file_path: Where to write the scene to.
        :param choices: The physics choices to apply, or ``None`` for the defaults.
        :return: The scene, written to ``file_path``.
        """
        for body in apartment.robot.bodies:
            if body.get_simulator_property_of_type(GravityCompensation) is None:
                body.add_simulator_property(GravityCompensation(fraction=1.0))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        MujocoBuilder().build_world(world=apartment.world, file_path=str(file_path))
        grippers = [arm.end_effector for arm in apartment.robot.all_arms]
        scene = cls(
            tree=ElementTree.parse(file_path),
            robot_root_name=apartment.robot.root.name.name,
            grip_body_names=[
                body.name.name
                for gripper in grippers
                for finger in gripper.fingers
                for body in (finger.tip, finger.tip.parent_connection.parent)
            ],
            gripper_body_names=[
                body.name.name for gripper in grippers for body in gripper.bodies
            ],
            choices=choices or PhysicsChoices(),
        )
        scene.remove_keyframes()
        scene.add_base_joints()
        scene.steady_gripper_linkage()
        scene.add_servos()
        scene.restrict_collisions()
        scene.collide_groceries_as_boxes()
        scene.lighten_frame_only_links()
        scene.balance_inertias_for_single_precision()
        scene.tree.write(file_path)
        return scene

    @property
    def root(self) -> ElementTree.Element:
        """
        The scene's root element.
        """
        return self.tree.getroot()

    @property
    def robot_bodies(self) -> list[ElementTree.Element]:
        """
        The bodies of the robot, its root first.
        """
        [robot_root] = [
            body
            for body in self.root.iter(Element.BODY)
            if body.get(Attribute.NAME) == self.robot_root_name
        ]
        return list(robot_root.iter(Element.BODY))

    def robot_bodies_named(self, names: list[str]) -> list[ElementTree.Element]:
        """
        :param names: The names of the bodies to look up.
        :return: The robot bodies with one of ``names``.
        """
        return [body for body in self.robot_bodies if body.get(Attribute.NAME) in names]

    @property
    def grocery_bodies(self) -> list[ElementTree.Element]:
        """
        The bodies free to move, which are the groceries.
        """
        return [
            body
            for body in self.root.iter(Element.BODY)
            if any(
                joint.get(Attribute.TYPE) == JointType.FREE
                for joint in body.findall(Element.JOINT)
            )
        ]

    @property
    def apartment_bodies(self) -> list[ElementTree.Element]:
        """
        The bodies with geometry that are neither the robot's nor free to move.
        """
        excluded = set(map(id, self.robot_bodies + self.grocery_bodies))
        return [
            body
            for body in self.root.find(Element.WORLDBODY).iter(Element.BODY)
            if id(body) not in excluded and body.findall(Element.GEOM)
        ]

    def section(self, element: Element) -> ElementTree.Element:
        """
        :param element: The top-level section to look up, such as the actuators.
        :return: That section, added if the scene has none yet.
        """
        found = self.root.find(element)
        if found is None:
            return ElementTree.SubElement(self.root, element)
        return found

    def remove_keyframes(self) -> None:
        """
        Remove the keyframes, whose joint coordinates do not count the base joints.
        """
        for keyframe in self.root.findall(Element.KEYFRAME):
            self.root.remove(keyframe)

    def add_base_joints(self) -> None:
        """
        Give the robot's root body the joints that move the base.
        """
        [robot_root, *_] = self.robot_bodies
        for index, base_joint in enumerate(BaseJoint):
            robot_root.insert(
                index,
                ElementTree.Element(
                    Element.JOINT,
                    {
                        Attribute.NAME: base_joint,
                        Attribute.TYPE: base_joint.joint_type,
                        Attribute.AXIS: as_mjcf_vector(base_joint.axis),
                        Attribute.ARMATURE: str(self.choices.base_armature),
                    },
                ),
            )

    def steady_gripper_linkage(self) -> None:
        """
        Damp the joints of the grippers' linkages.
        """
        for body in self.robot_bodies_named(self.gripper_body_names):
            for joint in body.findall(Element.JOINT):
                joint.set(Attribute.ARMATURE, str(self.choices.gripper_joint_armature))
                joint.set(Attribute.DAMPING, str(self.choices.gripper_joint_damping))

    def add_servos(self) -> None:
        """
        Drive every joint the plan commands with a position servo.
        """
        actuators = self.section(Element.ACTUATOR)
        for joint, gains in self.choices.servo_gains.items():
            ElementTree.SubElement(
                actuators,
                Element.POSITION,
                {
                    Attribute.NAME: f"{joint}{NameSuffix.SERVO}",
                    Attribute.JOINT: joint,
                    Attribute.STIFFNESS: str(gains.stiffness),
                    Attribute.VELOCITY_GAIN: str(gains.damping),
                },
            )

    def restrict_collisions(self) -> None:
        """
        Let only the finger pads of the robot collide, with the friction of rubber, and
        keep the apartment from colliding with itself and with the pads.
        """
        grip_bodies = self.robot_bodies_named(self.grip_body_names)
        for body in self.robot_bodies:
            if body in grip_bodies:
                continue
            for geometry in body.findall(Element.GEOM):
                geometry.set(Attribute.CONTACT_TYPE, ContactSetting.NONE)
                geometry.set(Attribute.CONTACT_AFFINITY, ContactSetting.NONE)
        for body in grip_bodies:
            for geometry in body.findall(Element.GEOM):
                if geometry.get(Attribute.CONTACT_TYPE) == ContactSetting.NONE:
                    continue
                geometry.set(
                    Attribute.FRICTION, as_mjcf_vector(self.choices.grip_friction)
                )
                geometry.set(
                    Attribute.CONTACT_DIMENSION,
                    str(self.choices.grip_contact_dimension),
                )
        apartment_names = [body.get(Attribute.NAME) for body in self.apartment_bodies]
        contact = self.section(Element.CONTACT)
        excluded_pairs = [
            *combinations(apartment_names, 2),
            *(
                (apartment_name, grip_name)
                for apartment_name in apartment_names
                for grip_name in self.grip_body_names
            ),
        ]
        for first, second in excluded_pairs:
            ElementTree.SubElement(
                contact,
                Element.EXCLUDE,
                {Attribute.BODY_1: first, Attribute.BODY_2: second},
            )

    def collide_groceries_as_boxes(self) -> None:
        """
        Collide each grocery as the box bounding its mesh, so flat pads squeeze flat
        sides, and draw it as its mesh in its own color.
        """
        for body in self.grocery_bodies:
            grocery = Grocery(body.get(Attribute.NAME))
            lower, upper = grocery.mesh_bounds
            for geometry in body.findall(Element.GEOM):
                visual = ElementTree.SubElement(
                    body, Element.GEOM, dict(geometry.attrib)
                )
                visual.set(
                    Attribute.NAME, f"{geometry.get(Attribute.NAME)}{NameSuffix.VISUAL}"
                )
                visual.set(Attribute.CONTACT_TYPE, ContactSetting.NONE)
                visual.set(Attribute.CONTACT_AFFINITY, ContactSetting.NONE)
                visual.set(Attribute.COLOR, as_mjcf_vector(grocery.color.to_rgba()))
                del geometry.attrib[Attribute.MESH]
                geometry.set(Attribute.TYPE, GeometryType.BOX)
                geometry.set(Attribute.SIZE, as_mjcf_vector((upper - lower) / 2))
                geometry.set(Attribute.POSITION, as_mjcf_vector((upper + lower) / 2))
                geometry.set(
                    Attribute.CONTACT_DIMENSION,
                    str(self.choices.grip_contact_dimension),
                )

    def lighten_frame_only_links(self) -> None:
        """
        Give the robot links that only mark a frame a negligible mass instead of the
        default the description leaves them with.
        """
        for body in self.robot_bodies:
            inertial = body.find(Element.INERTIAL)
            if inertial is None or body.findall(Element.GEOM):
                continue
            if float(inertial.get(Attribute.MASS)) != Inertial().mass:
                continue
            inertial.set(Attribute.MASS, str(self.choices.frame_only_mass))
            inertial.set(
                Attribute.DIAGONAL_INERTIA,
                as_mjcf_vector([self.choices.frame_only_moment] * 3),
            )

    def balance_inertias_for_single_precision(self) -> None:
        """
        Raise the smaller principal moments of every inertia that does not clear the
        triangle inequality by :attr:`PhysicsChoices.single_precision_margin`.
        """
        margin = self.choices.single_precision_margin
        for inertial in self.root.iter(Element.INERTIAL):
            moments = [
                float(value)
                for value in inertial.get(Attribute.DIAGONAL_INERTIA).split()
            ]
            largest = max(moments)
            if sum(moments) - largest >= margin * largest:
                continue
            inertial.set(
                Attribute.DIAGONAL_INERTIA,
                as_mjcf_vector(
                    [max(moment, margin * largest / 2) for moment in moments]
                ),
            )
