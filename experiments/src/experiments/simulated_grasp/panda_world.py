"""
A Franka Emika Panda in front of a table with a block on it.

The arm is MuJoCo Menagerie's, vendored under
``semantic_digital_twin/resources/mjcf/franka_emika_panda``, so its kinematics, masses,
joint limits and servo gains are the ones its maintainers tuned rather than any invented
here. What this module does is read that scene into a world the controller can drive, and
repair the three things that stop it being drivable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from typing_extensions import List

import semantic_digital_twin
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import (
    MujocoCamera,
    MujocoLight,
    MujocoTendon,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import ServoGains
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body, PositionServo

# %% what the parts of the scene are called


class PandaJointName(StrEnum):
    """
    The joints of the arm and its gripper, as the vendored description names them.
    """

    SHOULDER_PAN = "joint1"
    SHOULDER_LIFT = "joint2"
    UPPER_ARM_ROLL = "joint3"
    ELBOW = "joint4"
    FOREARM_ROLL = "joint5"
    WRIST_FLEX = "joint6"
    WRIST_ROLL = "joint7"
    LEFT_FINGER = "finger_joint1"
    RIGHT_FINGER = "finger_joint2"

    @classmethod
    def arm(cls) -> List[PandaJointName]:
        """
        :return: The seven joints that move the hand, from the base outwards.
        """
        return [
            cls.SHOULDER_PAN,
            cls.SHOULDER_LIFT,
            cls.UPPER_ARM_ROLL,
            cls.ELBOW,
            cls.FOREARM_ROLL,
            cls.WRIST_FLEX,
            cls.WRIST_ROLL,
        ]

    @classmethod
    def gripper(cls) -> List[PandaJointName]:
        """
        :return: The two joints that close the gripper.
        """
        return [cls.LEFT_FINGER, cls.RIGHT_FINGER]


class PandaPartName(StrEnum):
    """
    The bodies of the scene that something else has to name.

    A finger's body and the joint that slides it are named apart in this description, so
    contact is reported against the names here and never against
    :class:`PandaJointName`.
    """

    BASE = "link0"
    HAND = "hand"
    LEFT_FINGER = "left_finger"
    RIGHT_FINGER = "right_finger"
    TOOL_FRAME = "tool_frame"
    FLOOR = "floor"
    TABLE = "table"
    BLOCK = "block"


class BlockPlacement(StrEnum):
    """
    The degrees of freedom the block is placed through, as the description names them.

    They are read as offsets from where the scene itself puts the block, which is what
    lets a block be placed somewhere other than where a motion aiming at the scene's own
    description expects it.
    """

    ALONG_THE_ARM = "x"
    ACROSS_THE_TABLE = "y"


# %% where everything is, and how hard the gripper squeezes

SCENE_PATH = (
    Path(semantic_digital_twin.__file__).parent.parent.parent
    / "resources"
    / "mjcf"
    / "franka_emika_panda"
    / "pick_scene.xml"
)
"""
The scene this world is read from.

Located through the package rather than from the working directory, because the meshes
the arm is drawn with are resolved relative to it.
"""

READY_POSTURE = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
"""
The joint positions, in radians, the arm starts every attempt from.

The vendored description's own ``home`` keyframe: elbow bent, hand over the table and
already facing down, so reaching the block costs the wrist no half-turn.
"""

OPEN_FINGER_OFFSET = 0.04
"""
How far from the palm's centre a finger sits when the gripper is fully open, in metres,
which is the widest its own joint limit allows.
"""

GRIPPED_FINGER_OFFSET = 0.013
"""
How far from the palm's centre a finger is commanded to while gripping, in metres.

Inside the block's surface, so the servo keeps pushing once the block has stopped the
finger: that remaining error is what the grip is made of.
"""

GRASP_DEPTH = 0.10
"""
How far beyond the hand the frame the arm is commanded to sits, in metres.

Between the fingertips, so commanding that frame over the block puts the fingers around
it rather than beside it.
"""

GRASP_HEIGHT = 0.015
"""
How far above the block's centre it is gripped, in metres.

The hand's own shell reaches lower than the fingertips do, so gripping at the centre
presses the palm onto the block before the fingers ever close on it.
"""

TABLE_TOP = 0.20
"""
How high the table's surface is above the floor, in metres.
"""

BLOCK_SIDE = 0.04
"""
How wide and deep the block is, in metres.

Its diagonal is narrower than the gripper's opening, so the fingers straddle it whatever
way the hand is turned about the vertical.
"""

BLOCK_HEIGHT = 0.06
"""
How tall the block is, in metres.
"""

BLOCK_DISTANCE = 0.55
"""
How far in front of the arm the block starts, in metres.
"""

OVERVIEW_CAMERA = "pick_overview"
"""
The name of the camera a recording is made from.
"""

ARM_VELOCITY_LIMIT = 1.0
"""
How fast an arm joint may travel, in radians per second.

..note:: Registering the arm as a robot tightens every joint it covers to this, and
    tightening only ever lowers a limit, so a looser number stated here would not be the
    one in force.
"""

FINGER_VELOCITY_LIMIT = 0.15
"""
How fast a finger may travel, in metres per second.

Tighter than what the arm's registration would leave it at, which is what keeps this the
limit in force for the gripper.
"""

FINGER_SERVO = ServoGains(stiffness=350.0, damping=10.0, torque_limit=200.0)
"""
What drives a finger once the gripper's tendon has been replaced by a servo per finger.

The gains the same description states for the joint-driven variant of this gripper.
"""


@dataclass
class PandaWorld:
    """
    The world an attempt is made in, together with the frames it is read through.
    """

    world: World
    """
    The world the motion is executed in and the physics mirrors.
    """

    hand: Body
    """
    The body the fingers are mounted on, whose own axis is the direction they point in.
    """

    tool_frame: Body
    """
    The frame between the fingertips, which is what the arm is commanded to.
    """

    block: Body
    """
    The block the arm picks up.
    """

    expected_block_position: Point3
    """
    Where a motion aiming at this scene expects the block to be.

    The block itself may sit somewhere else, which is what a perception error looks like
    to a robot that cannot tell the difference.
    """

    table: Body
    """
    The surface the block rests on.
    """

    arm: List[ActiveConnection1DOF]
    """
    The arm's joints, from the base outwards.
    """

    fingers: List[ActiveConnection1DOF]
    """
    The two joints that close the gripper.
    """

    @classmethod
    def of(
        cls, block_distance: float = BLOCK_DISTANCE, block_offset: float = 0.0
    ) -> PandaWorld:
        """
        Read the scene and make the arm drivable.

        :param block_distance: How far in front of the arm the block is, in metres.
        :param block_offset: How far across the table the block actually sits from where
            a motion aiming at this scene expects it, in metres. Nothing in the world
            says it is there, which is what makes it a perception error rather than a
            different scene.
        :return: The world, ready to execute a motion in.
        """
        world = MJCFParser.from_file(str(SCENE_PATH)).parse()
        bodies = {body.name.name: body for body in world.bodies}
        joints = {
            connection.name.name: connection
            for connection in world.connections
            if isinstance(connection, ActiveConnection1DOF)
        }
        arm = [joints[name] for name in PandaJointName.arm()]
        fingers = [joints[name] for name in PandaJointName.gripper()]

        with world.modify_world():
            cls._servo_the_fingers(world, fingers)
            cls._limit_velocities(arm, fingers)
            cls._colour_the_room(bodies)
            cls._light_the_room(bodies[PandaPartName.FLOOR])
            bodies[PandaPartName.FLOOR].add_simulator_property(
                cls._overview_camera(bodies[PandaPartName.FLOOR])
            )
            MinimalRobot.from_branch_in_world(bodies[PandaPartName.BASE])
            tool_frame = Body(name=PrefixedName(PandaPartName.TOOL_FRAME))
            world.add_connection(
                FixedConnection(
                    parent=bodies[PandaPartName.HAND],
                    child=tool_frame,
                    parent_T_connection_expression=(
                        HomogeneousTransformationMatrix.from_xyz_rpy(z=GRASP_DEPTH)
                    ),
                )
            )

        for connection, position in zip(arm, READY_POSTURE):
            world.state[connection.dof.id].position = position
        for finger in fingers:
            world.state[finger.dof.id].position = OPEN_FINGER_OFFSET
        cls._place_the_block(world, block_distance, block_offset)
        world.notify_state_change()

        return cls(
            world=world,
            hand=bodies[PandaPartName.HAND],
            tool_frame=tool_frame,
            block=bodies[PandaPartName.BLOCK],
            expected_block_position=cls.resting_place(block_distance, 0.0),
            table=bodies[PandaPartName.TABLE],
            arm=arm,
            fingers=fingers,
        )

    @staticmethod
    def _place_the_block(
        world: World, block_distance: float, block_offset: float
    ) -> None:
        """
        Put the block down on the table, as far along the arm and as far across it as
        asked for.

        :param world: The world holding the block.
        :param block_distance: How far in front of the arm it is, in metres.
        :param block_offset: How far across the table it is, in metres.

        ..note:: The block rests on the table on a free joint, whose positions the scene
            reads as offsets from where it puts the block itself.
        """
        placement = [
            connection
            for connection in world.connections
            if isinstance(connection, Connection6DoF)
            and connection.child.name.name == PandaPartName.BLOCK
        ][0]
        wanted = {
            BlockPlacement.ALONG_THE_ARM: block_distance - BLOCK_DISTANCE,
            BlockPlacement.ACROSS_THE_TABLE: block_offset,
        }
        for degree_of_freedom in placement.dofs:
            if degree_of_freedom.name.name in wanted:
                world.state[degree_of_freedom.id].position = wanted[
                    degree_of_freedom.name.name
                ]

    @staticmethod
    def resting_place(x: float, y: float) -> Point3:
        """
        :param x: Where on the table, along the arm's forward axis, in metres.
        :param y: Where on the table, across it, in metres.
        :return: Where the block's centre sits when it rests on the table there.
        """
        return Point3(x, y, TABLE_TOP + BLOCK_HEIGHT / 2)

    @property
    def block_position(self) -> np.ndarray:
        """
        :return: Where the block's centre is, as the world holds it.

        ..note:: The physics is what actually moves the block. Read it from there with
            :meth:`~experiments.simulated_grasp.grasp_attempt.PhysicalGrasp.block_position`
            while an attempt is running.
        """
        return self.world.compute_forward_kinematics_np(self.world.root, self.block)[
            :3, 3
        ]

    @staticmethod
    def _servo_the_fingers(world: World, fingers: List[ActiveConnection1DOF]) -> None:
        """
        Drive each finger with a position servo of its own.

        Two things stop the gripper being drivable as it arrives. The description closes
        it through a tendon, which is a transmission the parser cannot follow to a joint,
        so the fingers come through with nothing driving them at all - and an undriven
        joint is written straight into the physics rather than reached through it, which
        is a gripper that passes through whatever it closes on. The coupling between the
        two fingers also leaves both of them carrying a degree of freedom named after the
        first, and both a servo's transmission and the mimic rule that would restore the
        coupling are resolved by that name alone, so the second finger ends up with two
        servos fighting over the first one's joint and none of its own.

        :param world: The world the servos are added to.
        :param fingers: The two joints they drive.
        """
        for actuator in [
            actuator
            for actuator in world.actuators
            if not any(
                degree_of_freedom.name.name in PandaJointName.arm()
                for degree_of_freedom in actuator.dofs
            )
        ]:
            world.remove_actuator(actuator)
        for tendon in [
            entry
            for entry in world.simulator_additional_properties
            if isinstance(entry, MujocoTendon)
        ]:
            world.simulator_additional_properties.remove(tendon)
        for finger in fingers:
            finger.raw_dof.name = PrefixedName(finger.name.name)
            servo = PositionServo(
                name=PrefixedName(f"{finger.name.name}_servo"), gains=FINGER_SERVO
            )
            servo.add_dof(finger.raw_dof)
            world.add_actuator(servo)

    @staticmethod
    def _limit_velocities(
        arm: List[ActiveConnection1DOF], fingers: List[ActiveConnection1DOF]
    ) -> None:
        """
        State how fast each joint may travel.

        The description carries positions and torques but no velocities, and the
        controller cannot build a task for a joint whose velocity is unbounded.

        :param arm: The arm's joints.
        :param fingers: The gripper's joints.
        """
        for connections, limit in (
            (arm, ARM_VELOCITY_LIMIT),
            (fingers, FINGER_VELOCITY_LIMIT),
        ):
            for connection in connections:
                degree_of_freedom = connection.raw_dof
                degree_of_freedom.limits.lower.velocity = -limit
                degree_of_freedom.limits.upper.velocity = limit

    @staticmethod
    def _colour_the_room(bodies: dict[str, Body]) -> None:
        """
        Colour the floor, the table and the block.

        The arm carries its own appearance in its meshes; the room around it is read
        from materials the world does not keep, so it would otherwise all be white.

        :param bodies: The scene's bodies, by name.
        """
        for name, (red, green, blue) in (
            (PandaPartName.FLOOR, (0.34, 0.37, 0.42)),
            (PandaPartName.TABLE, (0.55, 0.42, 0.30)),
            (PandaPartName.BLOCK, (0.88, 0.35, 0.16)),
        ):
            body = bodies[name]
            for shape in body.visual.shapes + body.collision.shapes:
                shape.color = Color(R=red, G=green, B=blue)

    @staticmethod
    def _light_the_room(floor: Body) -> None:
        """
        Add a key light over the table and a softer one from the opposite side, which is
        what makes a recording of the scene readable.

        :param floor: The body the lights are mounted on, so they stay put.
        """
        for name, position, direction, brightness, casts_shadow in (
            ("key_light", [0.8, -1.0, 2.2], [-0.15, 0.4, -1.0], 0.55, True),
            ("fill_light", [-0.7, 1.0, 1.9], [0.3, -0.45, -1.0], 0.25, False),
        ):
            floor.add_simulator_property(
                MujocoLight(
                    name=name,
                    body=floor,
                    directional=True,
                    position=position,
                    direction=direction,
                    cast_shadow=casts_shadow,
                    diffuse=[brightness] * 3,
                    ambient=[0.12, 0.12, 0.12],
                    specular=[0.08, 0.08, 0.08],
                )
            )

    @staticmethod
    def _overview_camera(floor: Body) -> MujocoCamera:
        """
        :param floor: The body the camera is mounted on, so it stays put.
        :return: A fixed camera framing the arm, the table and the block from the side.
        """
        bounds = np.array([[-0.25, -0.55, 0.0], [BLOCK_DISTANCE + 0.4, 0.55, 1.0]])
        pose = MujocoCamera.overview_pose(bounds, distance_factor=1.1)
        quaternion_xyzw = pose.to_quaternion().to_np().tolist()
        return MujocoCamera(
            name=OVERVIEW_CAMERA,
            body=floor,
            position=pose.to_position().to_np()[:3].tolist(),
            quaternion=[quaternion_xyzw[3]] + quaternion_xyzw[:3],
        )
