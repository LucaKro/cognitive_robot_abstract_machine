"""
The world for the Push-T benchmark: a T-shaped block that has to be pushed onto a target
pose by a point-sized end effector.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
from typing_extensions import List, Optional

from experiments.push_t.real_time_simulation import RealTimeSimulation
from semantic_digital_twin.adapters.multi_sim import MujocoActuator
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.types import NpMatrix4x4
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Scale,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Actuator, Body

# %% dimensions

GROUND_SCALE = Scale(2.0, 2.0, 0.1)
"""Extents of the box the T block slides on."""

BLOCK_HEIGHT = 0.04
"""How far the T block is extruded above the ground plane."""

CROSSBAR_SCALE = Scale(0.2, 0.05, BLOCK_HEIGHT)
"""Extents of the T's horizontal bar, which is centred on the block's own frame."""

STEM_SCALE = Scale(0.05, 0.15, BLOCK_HEIGHT)
"""Extents of the T's stem."""

STEM_OFFSET = (CROSSBAR_SCALE.y + STEM_SCALE.y) / 2
"""How far along -y the stem's centre sits from the block's own frame."""

PUSHER_RADIUS = 0.02
"""Radius of the sphere standing in for the end effector."""

PUSHER_TRAVEL = 1.0
"""How far the pusher may slide from the origin along either axis."""

PUSHER_SPEED_LIMIT = 1.0
"""Speed limit of the pusher's slide joints, in metres per second."""

PUSHER_STIFFNESS = 2000.0
"""Restoring force per metre the pusher's servos pull towards their set point with."""

PUSHER_DAMPING = 100.0
"""Opposing force per metre per second the pusher's servos damp their motion with."""

PUSHER_FORCE_LIMIT = 100.0
"""The largest force, in newtons, one of the pusher's servos may exert."""

# %% colors

BLOCK_COLOR = Color(0.15, 0.35, 0.85, 1.0)
"""Colour of the T block being pushed."""

TARGET_COLOR = Color(0.0, 0.8, 0.2, 0.3)
"""Colour of the translucent marker showing where the T block should end up."""

GROUND_COLOR = Color(0.8, 0.8, 0.8, 1.0)
"""Colour of the ground plane."""

PUSHER_COLOR = Color(0.9, 0.3, 0.1, 1.0)
"""Colour of the pusher."""


# %% scene


@dataclass
class PlanarPoint:
    """
    A point on the ground plane.
    """

    x: float = 0.0
    """Position along x, in metres."""

    y: float = 0.0
    """Position along y, in metres."""


@dataclass
class PlanarPose(PlanarPoint):
    """
    A pose on the ground plane, which is all the freedom a body sliding on it has.
    """

    yaw: float = 0.0
    """Heading around z, in radians."""


def build_t_shapes(color: Color) -> List[Shape]:
    """
    Build the two boxes forming a T, expressed in the T's own frame.

    Kept in one place so the block and the target marker can never end up different
    shapes.

    :param color: The colour both boxes are given.
    :return: The crossbar and the stem.
    """
    return [
        Box(scale=CROSSBAR_SCALE, color=color),
        Box(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(y=-STEM_OFFSET),
            scale=STEM_SCALE,
            color=color,
        ),
    ]


@dataclass
class PushTScene:
    """
    A T-shaped block lying on a plane, a marker showing where it should end up, and a
    sphere that can push it around.

    The marker carries no collision geometry, so it neither obstructs the block nor is
    moved by it.
    """

    world: World
    """The world holding the whole scene."""

    ground: Body
    """The plane the block slides on, with its top face at z = 0."""

    block: Body
    """The T-shaped block to be pushed."""

    target: Body
    """The marker showing the pose the block should be pushed onto."""

    pusher: Body
    """The sphere standing in for the end effector."""

    block_connection: Connection6DoF
    """Carries the block, leaving it free to be moved by contact alone."""

    pusher_x_connection: PrismaticConnection
    """Slides the pusher along x."""

    pusher_y_connection: PrismaticConnection
    """Slides the pusher along y."""

    pusher_x_actuator: Actuator
    """The servo driving :attr:`pusher_x_connection` towards a commanded position."""

    pusher_y_actuator: Actuator
    """The servo driving :attr:`pusher_y_connection` towards a commanded position."""

    @classmethod
    def create(cls, block_pose: PlanarPose, pusher_position: PlanarPoint) -> PushTScene:
        """
        Build the scene, with the target marker at the world's origin.

        :param block_pose: Where the block starts out on the plane.
        :param pusher_position: Where the pusher starts out on the plane.
        :return: The newly built scene.
        """
        world = World.create_with_root_body("world")
        root = world.root

        # Each shape is shared between the visual and the collision collection rather
        # than duplicated, so it is built as a single geom that is both drawn and
        # collided with.
        ground_shapes = [Box(scale=GROUND_SCALE, color=GROUND_COLOR)]
        block_shapes = build_t_shapes(BLOCK_COLOR)
        pusher_shapes = [Sphere(radius=PUSHER_RADIUS, color=PUSHER_COLOR)]

        ground = Body(
            name=PrefixedName("ground"),
            visual=ShapeCollection(ground_shapes),
            collision=ShapeCollection(ground_shapes),
        )
        block = Body(
            name=PrefixedName("t_block"),
            visual=ShapeCollection(block_shapes),
            collision=ShapeCollection(block_shapes),
        )
        target = Body(
            name=PrefixedName("t_target"),
            visual=ShapeCollection(build_t_shapes(TARGET_COLOR)),
        )
        pusher_slide = Body(name=PrefixedName("pusher_slide"))
        pusher = Body(
            name=PrefixedName("pusher"),
            visual=ShapeCollection(pusher_shapes),
            collision=ShapeCollection(pusher_shapes),
        )

        with world.modify_world():
            world.add_connection(
                FixedConnection(
                    parent=root,
                    child=ground,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=-GROUND_SCALE.z / 2, reference_frame=root
                    ),
                )
            )
            world.add_connection(
                FixedConnection(
                    parent=root,
                    child=target,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=BLOCK_HEIGHT / 2, reference_frame=root
                    ),
                )
            )
            block_connection = Connection6DoF.create_with_dofs(
                world=world, parent=root, child=block
            )
            world.add_connection(block_connection)

            # The pusher hangs off two slide joints rather than a 6DoF connection: those
            # are actively controllable, so a motion controller can command them.
            pusher_x_connection = cls._add_pusher_slide(
                world=world,
                name="pusher_x",
                parent=root,
                child=pusher_slide,
                axis=Vector3.X(reference_frame=root),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=PUSHER_RADIUS, reference_frame=root
                ),
            )
            pusher_y_connection = cls._add_pusher_slide(
                world=world,
                name="pusher_y",
                parent=pusher_slide,
                child=pusher,
                axis=Vector3.Y(reference_frame=pusher_slide),
            )
            pusher_x_actuator = cls._add_pusher_servo(world, pusher_x_connection)
            pusher_y_actuator = cls._add_pusher_servo(world, pusher_y_connection)

        scene = cls(
            world=world,
            ground=ground,
            block=block,
            target=target,
            pusher=pusher,
            block_connection=block_connection,
            pusher_x_connection=pusher_x_connection,
            pusher_y_connection=pusher_y_connection,
            pusher_x_actuator=pusher_x_actuator,
            pusher_y_actuator=pusher_y_actuator,
        )
        scene.block_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=block_pose.x,
            y=block_pose.y,
            z=BLOCK_HEIGHT / 2,
            yaw=block_pose.yaw,
            reference_frame=root,
        )
        world.state[pusher_x_connection.raw_dof.id].position = pusher_position.x
        world.state[pusher_y_connection.raw_dof.id].position = pusher_position.y
        world.notify_state_change()
        return scene

    @staticmethod
    def _add_pusher_slide(
        world: World,
        name: str,
        parent: Body,
        child: Body,
        axis: Vector3,
        parent_T_connection_expression: Optional[
            HomogeneousTransformationMatrix
        ] = None,
    ) -> PrismaticConnection:
        """
        Add one of the two slide joints carrying the pusher.

        The joint's degree of freedom is named after the joint, so that a servo can
        later be matched to it by name. Its position limits are spelled out because a
        slide joint without them is built with an empty MuJoCo range, which locks it.

        :param world: The world to add the connection to.
        :param name: The name shared by the connection and its degree of freedom.
        :param parent: The entity the joint slides relative to.
        :param child: The entity the joint carries.
        :param axis: The direction the joint slides along.
        :param parent_T_connection_expression: Constant pose of the joint relative to
            ``parent``.
        :return: The newly added connection.
        """
        degree_of_freedom = DegreeOfFreedom(
            name=PrefixedName(name),
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(
                    position=-PUSHER_TRAVEL, velocity=-PUSHER_SPEED_LIMIT
                ),
                upper=DerivativeMap(
                    position=PUSHER_TRAVEL, velocity=PUSHER_SPEED_LIMIT
                ),
            ),
        )
        world.add_degree_of_freedom(degree_of_freedom)
        connection = PrismaticConnection(
            name=degree_of_freedom.name,
            parent=parent,
            child=child,
            axis=axis,
            raw_dof=degree_of_freedom,
            parent_T_connection_expression=parent_T_connection_expression,
        )
        world.add_connection(connection)
        return connection

    @staticmethod
    def _add_pusher_servo(world: World, connection: PrismaticConnection) -> Actuator:
        """
        Add a position servo driving ``connection`` towards a commanded position.

        A slide joint left unactuated is knocked aside by the very contact it is meant
        to create, so the pusher pushes with a bounded force rather than being moved to
        a pose outright.

        :param world: The world to add the actuator to.
        :param connection: The joint the servo drives.
        :return: The newly added actuator.
        """
        actuator = Actuator(name=PrefixedName(f"{connection.name.name}_servo"))
        actuator.add_dof(connection.raw_dof)
        actuator.simulator_additional_properties.append(
            MujocoActuator(
                dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                gain_parameters=[PUSHER_STIFFNESS] + [0.0] * 9,
                bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                bias_parameters=[0.0, -PUSHER_STIFFNESS, -PUSHER_DAMPING] + [0.0] * 7,
                control_range=[-PUSHER_TRAVEL, PUSHER_TRAVEL],
                force_range=[-PUSHER_FORCE_LIMIT, PUSHER_FORCE_LIMIT],
            )
        )
        world.add_actuator(actuator)
        return actuator

    def command_pusher(
        self, simulation: RealTimeSimulation, position: PlanarPoint
    ) -> None:
        """
        Tell the pusher's servos which point on the plane to hold.

        They keep driving towards the last point they were given, so this has to be
        called once before the simulation is first advanced, or the pusher rushes from
        wherever it was placed towards the origin.

        :param simulation: The simulation running this scene.
        :param position: The point the pusher should hold.
        """
        simulation.command(self.pusher_x_actuator, position.x)
        simulation.command(self.pusher_y_actuator, position.y)

    def pose_of(self, body: Body) -> NpMatrix4x4:
        """
        Read a body's current pose back out of the world model.

        :param body: The body to look up.
        :return: Its pose relative to the world's root.
        """
        return self.world.compute_forward_kinematics_np(self.world.root, body)
