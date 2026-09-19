"""
A robot arm, a table and a block, built so that the grasp between them is carried by
physics rather than by the world's own kinematics.

Every robot in this repository is described by a ROS package that is not on PyPI, so the
arm is built here instead. That also keeps one code path between what the tests exercise
and what a recording shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from typing_extensions import List, Optional, Tuple

from semantic_digital_twin.adapters.multi_sim import MujocoCamera, MujocoLight
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    JointDynamics,
    ServoGains,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection,
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    PositionServo,
)

# %% what the parts are called


class ArmJointName(StrEnum):
    """
    The joints of the arm, in the order they follow each other from the base outwards.

    A joint's name is also the name of the degree of freedom it moves and of the MuJoCo
    joint backing it: the adapter reads a joint whose degree of freedom is named
    differently as one that mimics another, and builds an equality constraint to a joint
    of that name.
    """

    SHOULDER_YAW = "shoulder_yaw"
    """
    Turns the whole arm about the vertical.
    """

    SHOULDER_PITCH = "shoulder_pitch"
    """
    Swings the upper arm forwards and back.
    """

    ELBOW_PITCH = "elbow_pitch"
    """
    Bends the forearm against the upper arm.
    """

    WRIST_PITCH = "wrist_pitch"
    """
    Tips the hand towards or away from the forearm.
    """

    WRIST_ROLL = "wrist_roll"
    """
    Rotates the hand about the forearm.
    """

    WRIST_YAW = "wrist_yaw"
    """
    Turns the palm about the direction the fingers point in.
    """


class FingerName(StrEnum):
    """
    The two fingers of the parallel gripper.
    """

    LEFT = "left_finger"
    """
    The finger that slides towards positive y of the palm.
    """

    RIGHT = "right_finger"
    """
    The finger that slides towards negative y of the palm.
    """


class PartName(StrEnum):
    """
    The bodies of the world that something else has to name.
    """

    FLOOR = "floor"
    COLUMN = "column"
    SHOULDER = "shoulder"
    UPPER_ARM = "upper_arm"
    FOREARM = "forearm"
    WRIST_PITCH_LINK = "wrist_pitch_link"
    WRIST_ROLL_LINK = "wrist_roll_link"
    PALM = "palm"
    TOOL_FRAME = "tool_frame"
    TABLE = "table"
    BLOCK = "block"


# %% how big everything is

UPPER_ARM_LENGTH = 0.40
"""
Length of the upper arm, in metres.
"""

FOREARM_LENGTH = 0.36
"""
Length of the forearm, in metres.
"""

SHOULDER_HEIGHT = 1.10
"""
How high the shoulder sits above the floor, in metres.

Well above the table, so the arm reaches the block by pointing down and forwards rather
than by folding its wrist back against the forearm.
"""

TABLE_TOP = 0.52
"""
How high the table's surface is above the floor, in metres.
"""

BLOCK_SIDE = 0.05
"""
How wide and deep the block is, in metres.

Its diagonal is narrower than the gripper's opening, so the fingers straddle it whatever
way it is turned and the grasp needs no particular approach angle about the vertical.
"""

BLOCK_HEIGHT = 0.06
"""
How tall the block is, in metres.
"""

BLOCK_MASS = 0.1
"""
How heavy the block is, in kilograms.
"""

BLOCK_DISTANCE = 0.50
"""
How far in front of the arm the block starts, in metres.
"""

FINGER_LENGTH = 0.09
"""
How long a finger is, in metres.
"""

GRASP_DEPTH = 0.09
"""
How far beyond the palm the frame the arm is commanded to sits, in metres.

Between the fingertips, so commanding that frame onto the block's centre puts the
fingers around the block rather than beside it.
"""

WIDEST_FINGER_OFFSET = 0.05
"""
How far from the palm's centre a finger sits when the gripper is fully open, in metres.
"""


@dataclass(frozen=True)
class JointDrive:
    """
    What drives one joint of the arm in the physics.

    The arm carries no measured drives of its own, so these say how hard a joint pulls
    towards the position it is commanded to and how fast it may travel.
    """

    stiffness: float
    """
    How hard the servo pulls towards the commanded position.
    """

    damping: float
    """
    How hard the servo resists the joint's velocity.
    """

    torque_limit: float
    """
    The largest torque the servo may exert.
    """

    armature: float
    """
    The rotor inertia reflected through the gearbox.
    """

    velocity_limit: float
    """
    How fast the joint may travel, either way.
    """

    lower_limit: float
    """
    The lowest position the joint may take.
    """

    upper_limit: float
    """
    The highest position the joint may take.
    """


ARM_DRIVE = JointDrive(
    stiffness=4000.0,
    damping=250.0,
    torque_limit=500.0,
    armature=0.1,
    velocity_limit=1.5,
    lower_limit=-2.8,
    upper_limit=2.8,
)
"""
What drives the three joints carrying the arm's own weight and reach.
"""

WRIST_DRIVE = JointDrive(
    stiffness=1500.0,
    damping=80.0,
    torque_limit=200.0,
    armature=0.05,
    velocity_limit=2.0,
    lower_limit=-3.3,
    upper_limit=3.3,
)
"""
What drives the three wrist joints, which carry only the hand and may turn a full
half-circle either way so the palm can face down from any approach.
"""

FINGER_DRIVE = JointDrive(
    stiffness=2000.0,
    damping=40.0,
    torque_limit=150.0,
    armature=0.01,
    velocity_limit=0.3,
    lower_limit=0.0,
    upper_limit=WIDEST_FINGER_OFFSET,
)
"""
What drives a finger.

A grip is the servo pushing against a finger the block has stopped, so how hard the
block is held follows from :attr:`~JointDrive.stiffness` and how far past the block's
surface the finger is commanded.
"""

READY_POSTURE = (0.0, 1.0, 1.4, 0.74, 0.0, 0.0)
"""
The joint positions, in radians, the arm starts every attempt from.

Its three pitches already sum to half a turn, so the palm faces the table from the
outset and the arm reaches the block by extending rather than by turning the hand over.
"""


@dataclass
class TabletopWorld:
    """
    The world an attempt is made in, together with the frames it is read through.
    """

    world: World
    """
    The world the motion is executed in and the physics mirrors.
    """

    palm: Body
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
    def of(cls, block_position: Optional[Point3] = None) -> TabletopWorld:
        """
        Build the arm, the table and the block.

        :param block_position: Where the block starts, or nothing for the middle of the
            table.
        :return: The world, ready to execute a motion in.
        """
        world = World()
        arm: List[ActiveConnection1DOF] = []
        fingers: List[ActiveConnection1DOF] = []
        with world.modify_world():
            floor = cls._box(PartName.FLOOR, (3.0, 3.0, 0.05), 0.0, (0.32, 0.35, 0.40))
            world.add_body(floor)

            column = cls._box(
                PartName.COLUMN, (0.16, 0.16, 1.05), 5.0, (0.22, 0.25, 0.30)
            )
            cls._fix(world, floor, column, z=SHOULDER_HEIGHT - 0.575)
            column.add_simulator_property(GravityCompensation(fraction=1.0))

            shoulder = cls._box(PartName.SHOULDER, (0.12, 0.12, 0.12), 2.0)
            arm.append(
                cls._joint(
                    world,
                    RevoluteConnection,
                    ArmJointName.SHOULDER_YAW,
                    column,
                    shoulder,
                    Vector3.Z(),
                    HomogeneousTransformationMatrix.from_xyz_rpy(z=0.575),
                    ARM_DRIVE,
                )
            )

            upper_arm = cls._box(
                PartName.UPPER_ARM, (0.08, 0.08, UPPER_ARM_LENGTH), 2.0
            )
            arm.append(
                cls._joint(
                    world,
                    RevoluteConnection,
                    ArmJointName.SHOULDER_PITCH,
                    shoulder,
                    upper_arm,
                    Vector3.Y(),
                    HomogeneousTransformationMatrix.from_xyz_rpy(),
                    ARM_DRIVE,
                    connection_T_child=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=UPPER_ARM_LENGTH / 2
                    ),
                )
            )

            forearm = cls._box(PartName.FOREARM, (0.07, 0.07, FOREARM_LENGTH), 1.5)
            arm.append(
                cls._joint(
                    world,
                    RevoluteConnection,
                    ArmJointName.ELBOW_PITCH,
                    upper_arm,
                    forearm,
                    Vector3.Y(),
                    HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=UPPER_ARM_LENGTH / 2
                    ),
                    ARM_DRIVE,
                    connection_T_child=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=FOREARM_LENGTH / 2
                    ),
                )
            )

            wrist_pitch_link = cls._box(
                PartName.WRIST_PITCH_LINK, (0.06, 0.06, 0.06), 0.6
            )
            arm.append(
                cls._joint(
                    world,
                    RevoluteConnection,
                    ArmJointName.WRIST_PITCH,
                    forearm,
                    wrist_pitch_link,
                    Vector3.Y(),
                    HomogeneousTransformationMatrix.from_xyz_rpy(z=FOREARM_LENGTH / 2),
                    WRIST_DRIVE,
                )
            )

            wrist_roll_link = cls._box(
                PartName.WRIST_ROLL_LINK, (0.06, 0.06, 0.06), 0.4
            )
            arm.append(
                cls._joint(
                    world,
                    RevoluteConnection,
                    ArmJointName.WRIST_ROLL,
                    wrist_pitch_link,
                    wrist_roll_link,
                    Vector3.X(),
                    HomogeneousTransformationMatrix.from_xyz_rpy(z=0.06),
                    WRIST_DRIVE,
                )
            )

            palm = cls._box(PartName.PALM, (0.10, 0.10, 0.05), 0.5, (0.18, 0.42, 0.68))
            arm.append(
                cls._joint(
                    world,
                    RevoluteConnection,
                    ArmJointName.WRIST_YAW,
                    wrist_roll_link,
                    palm,
                    Vector3.Z(),
                    HomogeneousTransformationMatrix.from_xyz_rpy(z=0.05),
                    WRIST_DRIVE,
                )
            )

            for finger_name, side in (
                (FingerName.LEFT, 1.0),
                (FingerName.RIGHT, -1.0),
            ):
                finger = cls._box(
                    finger_name, (0.03, 0.02, FINGER_LENGTH), 0.08, (0.18, 0.42, 0.68)
                )
                fingers.append(
                    cls._joint(
                        world,
                        PrismaticConnection,
                        finger_name,
                        palm,
                        finger,
                        Vector3.Y() * side,
                        HomogeneousTransformationMatrix.from_xyz_rpy(z=0.07),
                        FINGER_DRIVE,
                    )
                )

            tool_frame = Body(name=PrefixedName(PartName.TOOL_FRAME))
            cls._fix(world, palm, tool_frame, z=GRASP_DEPTH)

            table = cls._box(PartName.TABLE, (0.4, 0.6, 0.04), 0.0, (0.45, 0.36, 0.28))
            cls._fix(world, floor, table, x=BLOCK_DISTANCE, z=TABLE_TOP - 0.02)

            block = cls._box(
                PartName.BLOCK,
                (BLOCK_SIDE, BLOCK_SIDE, BLOCK_HEIGHT),
                BLOCK_MASS,
                (0.86, 0.36, 0.20),
            )
            start = block_position or cls.resting_place(BLOCK_DISTANCE, 0.0)
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world,
                    parent=floor,
                    child=block,
                    parent_T_connection_expression=(
                        HomogeneousTransformationMatrix.from_xyz_rpy(
                            *start.to_np()[:3].tolist()
                        )
                    ),
                )
            )

            floor.add_simulator_property(cls._overview_camera(floor))
            for light in cls._lights(floor):
                floor.add_simulator_property(light)
            MinimalRobot.from_branch_in_world(column)

        for connection, position in zip(arm, READY_POSTURE):
            world.state[connection.dof.id].position = position
        for finger in fingers:
            world.state[finger.dof.id].position = WIDEST_FINGER_OFFSET
        world.notify_state_change()

        return cls(
            world=world,
            palm=palm,
            tool_frame=tool_frame,
            block=block,
            table=table,
            arm=arm,
            fingers=fingers,
        )

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
    def _lights(floor: Body) -> List[MujocoLight]:
        """
        :param floor: The body the lights are mounted on, so they stay put.
        :return: A key light over the table and a softer one from the opposite side,
            which is what makes a recording of the scene readable.
        """
        return [
            MujocoLight(
                name="key_light",
                body=floor,
                directional=True,
                position=[0.6, -0.8, 2.2],
                direction=[-0.2, 0.4, -1.0],
                diffuse=[0.85, 0.85, 0.85],
                ambient=[0.35, 0.35, 0.35],
                specular=[0.2, 0.2, 0.2],
            ),
            MujocoLight(
                name="fill_light",
                body=floor,
                directional=True,
                position=[-0.6, 0.9, 2.0],
                direction=[0.3, -0.5, -1.0],
                diffuse=[0.45, 0.45, 0.45],
                ambient=[0.2, 0.2, 0.2],
                specular=[0.0, 0.0, 0.0],
                cast_shadow=False,
            ),
        ]

    @staticmethod
    def _overview_camera(floor: Body) -> MujocoCamera:
        """
        :param floor: The body the camera is mounted on, so it stays put.
        :return: A fixed camera framing the arm, the table and the block from the side.
        """
        bounds = np.array(
            [
                [-0.1, -0.5, TABLE_TOP - 0.2],
                [BLOCK_DISTANCE + 0.35, 0.5, SHOULDER_HEIGHT + 0.15],
            ]
        )
        pose = MujocoCamera.overview_pose(bounds, distance_factor=1.15)
        quaternion_xyzw = pose.to_quaternion().to_np().tolist()
        return MujocoCamera(
            name="grasp_overview_camera",
            body=floor,
            position=pose.to_position().to_np()[:3].tolist(),
            quaternion=[quaternion_xyzw[3]] + quaternion_xyzw[:3],
        )

    @staticmethod
    def _box(
        name: str,
        scale: Tuple[float, float, float],
        mass: float,
        color: Tuple[float, float, float] = (0.62, 0.64, 0.68),
    ) -> Body:
        """
        :param name: What the body is called in the world.
        :param scale: How large the box is along each axis, in metres.
        :param mass: How heavy the body is, in kilograms.
        :param color: What colour it is rendered in.
        :return: A body whose one box is both what is seen and what collides, so a
            recording shows exactly the geometry the physics resolves contacts against.
        """
        shape = Box(
            scale=Scale(*scale), color=Color(R=color[0], G=color[1], B=color[2])
        )
        body = Body.from_shape_collection(
            PrefixedName(name), ShapeCollection(shapes=[shape])
        )
        width, depth, height = scale
        body.inertial = Inertial(
            mass=mass,
            inertia=InertiaTensor.from_values(
                mass / 12.0 * (depth * depth + height * height),
                mass / 12.0 * (width * width + height * height),
                mass / 12.0 * (width * width + depth * depth),
                0.0,
                0.0,
                0.0,
            ),
        )
        return body

    @staticmethod
    def _fix(world: World, parent: Body, child: Body, **placement: float) -> None:
        """
        Attach a body rigidly to another one.

        :param world: The world the attachment is added to.
        :param parent: The body attached to.
        :param child: The body attached.
        :param placement: Where the child sits on the parent.
        """
        world.add_connection(
            FixedConnection(
                parent=parent,
                child=child,
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_rpy(**placement)
                ),
            )
        )

    @staticmethod
    def _joint(
        world: World,
        connection_type: type,
        name: str,
        parent: Body,
        child: Body,
        axis: Vector3,
        parent_T_connection: HomogeneousTransformationMatrix,
        drive: JointDrive,
        connection_T_child: Optional[HomogeneousTransformationMatrix] = None,
    ) -> ActiveConnection1DOF:
        """
        Add one joint of the robot, together with the servo that drives it in the
        physics and the gravity compensation that spares that servo the arm's own
        weight.

        :param world: The world the joint is added to.
        :param connection_type: Whether the joint turns or slides.
        :param name: What the joint, its degree of freedom and its servo are called.
        :param parent: The body the joint moves relative to.
        :param child: The body the joint moves.
        :param axis: What the joint turns about or slides along.
        :param parent_T_connection: Where the joint sits on its parent.
        :param drive: What drives the joint and how far it may travel.
        :param connection_T_child: Where the child sits on the joint.
        :return: The connection that was added.
        """
        degree_of_freedom = DegreeOfFreedom(
            name=PrefixedName(name),
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(
                    position=drive.lower_limit,
                    velocity=-drive.velocity_limit,
                    acceleration=None,
                    jerk=None,
                ),
                upper=DerivativeMap(
                    position=drive.upper_limit,
                    velocity=drive.velocity_limit,
                    acceleration=None,
                    jerk=None,
                ),
            ),
        )
        world.add_degree_of_freedom(degree_of_freedom)
        connection = connection_type(
            name=PrefixedName(name),
            parent=parent,
            child=child,
            axis=axis,
            raw_dof=degree_of_freedom,
            parent_T_connection_expression=parent_T_connection,
            connection_T_child_expression=connection_T_child,
            dynamics=JointDynamics(armature=drive.armature, damping=1.0),
        )
        world.add_connection(connection)
        servo = PositionServo(
            name=PrefixedName(f"{name}_servo"),
            gains=ServoGains(
                stiffness=drive.stiffness,
                damping=drive.damping,
                torque_limit=drive.torque_limit,
            ),
        )
        servo.add_dof(degree_of_freedom)
        world.add_actuator(servo)
        child.add_simulator_property(GravityCompensation(fraction=1.0))
        return connection
