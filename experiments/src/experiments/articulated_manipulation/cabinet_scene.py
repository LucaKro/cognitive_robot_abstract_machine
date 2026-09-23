"""
Tracy at its table with a cabinet in front of its left arm, whose drawer or door only
the physics moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Door,
    Drawer,
    Handle,
    Hinge,
    Slider,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    JointDynamics,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% the scene


class ArticulatedPart(Enum):
    """
    The part of the cabinet that moves.
    """

    DRAWER = auto()
    """
    A drawer sliding out of the cabinet's open front.
    """

    DOOR = auto()
    """
    A door hinged at one side of the cabinet's open front, swinging outwards.
    """


@dataclass
class CabinetScene:
    """
    Tracy parked at its table, with a cabinet standing on the table in front of its left
    arm.

    The joint of the cabinet's moving part has no hardware interface, so a stepped
    simulation leaves it to the physics: only contact moves it.
    """

    world: World
    """
    The world holding the robot and the cabinet.
    """

    robot: Tracy
    """
    The robot, with both arms parked.
    """

    cabinet: Cabinet
    """
    The cabinet, fixed in place.
    """

    part: Drawer | Door
    """
    The cabinet's moving part.
    """

    handle: Handle
    """
    The handle on the moving part's front.
    """

    @property
    def mechanism(self) -> ActiveConnection1DOF:
        """
        The joint the moving part slides or turns on; zero is closed.
        """
        return self.part.mechanical_joint.root.parent_connection

    def set_opening(self, position: float) -> None:
        """
        Set how far the moving part is open in the world.

        A stepped simulation takes this over only when it starts, since from then on the
        physics alone moves the part.

        :param position: The opening, in the mechanism's joint coordinate.
        """
        with self.world.modify_world():
            self.world.state[self.mechanism.raw_dof.id].position = position
        self.world.notify_state_change()


# %% building it


@dataclass
class CabinetSceneBuilder:
    """
    Builds a :class:`CabinetScene` whose cabinet has one moving part.

    Lengths are in metres and positions are relative to Tracy's table frame, which lies
    on the table top at the edge the arms are mounted at, with x pointing across the
    table.
    """

    articulated_part: ArticulatedPart
    """
    Which part of the cabinet moves.
    """

    cabinet_scale: Scale = field(default_factory=lambda: Scale(x=0.4, y=0.4, z=0.3))
    """
    The outer size of the cabinet's case; x is its depth.
    """

    front_distance: float = 0.85
    """
    How far from the arms' edge of the table the cabinet's open front stands, which
    leaves the hand room to come down in front of it.
    """

    sideways_offset: float = 0.35
    """
    How far to the left of the table's centre line the cabinet stands, in front of the
    left arm.
    """

    wall_thickness: float = 0.02
    """
    The thickness of the case's walls, and the gap between the moving part and them.
    """

    drawer_depth: float = 0.3
    """
    How deep the drawer reaches into the case.
    """

    drawer_travel: float = 0.25
    """
    How far the drawer slides out when fully open.
    """

    door_thickness: float = 0.02
    """
    The thickness of the door.
    """

    door_swing: float = math.pi / 2
    """
    How far the door turns outwards when fully open, in radians.
    """

    mechanism_velocity_limit: float = 1.0
    """
    The velocity limit of the moving part's joint, in the joint's own unit per second.
    """

    mechanism_dynamics: JointDynamics = field(
        default_factory=lambda: JointDynamics(damping=2.0, dry_friction=1.0)
    )
    """
    The damping and dry friction resisting the moving part, so it stays where it was
    pushed.
    """

    handle_scale: Scale = field(default_factory=lambda: Scale(x=0.03, y=0.15, z=0.02))
    """
    The size of the handle; x is how far it stands off the front.
    """

    def build(self) -> CabinetScene:
        """
        :return: A new world with Tracy parked at its table and the cabinet on it.
        """
        world = World()
        with world.modify_world():
            world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
        robot = RobotSpecification(Tracy).spawn(world)
        for arm in robot.get_arms():
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        world.notify_state_change()

        world_T_cabinet = (
            robot.root.global_transform
            @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=self.front_distance + self.cabinet_scale.x / 2,
                y=self.sideways_offset,
                z=self.cabinet_scale.z / 2,
            )
        )
        with world.modify_world():
            cabinet = Cabinet.get_annotation_specification(
                "cabinet",
                Cabinet.get_default_root_kinematic_structure_entity_specification(
                    scale=self.cabinet_scale, wall_thickness=self.wall_thickness
                ),
            ).spawn(world, parent_T_self=world_T_cabinet)
            match self.articulated_part:
                case ArticulatedPart.DRAWER:
                    part, handle = self._add_drawer(world, world_T_cabinet)
                case ArticulatedPart.DOOR:
                    part, handle = self._add_door(world, world_T_cabinet)
            cabinet.add(part)
        part.mechanical_joint.root.parent_connection.dynamics = self.mechanism_dynamics
        return CabinetScene(
            world=world, robot=robot, cabinet=cabinet, part=part, handle=handle
        )

    def _add_drawer(
        self, world: World, world_T_cabinet: HomogeneousTransformationMatrix
    ) -> tuple[Drawer, Handle]:
        """
        Add a drawer that slides out of the cabinet's front, with a handle on its face.

        :param world: The world to add the drawer to.
        :param world_T_cabinet: The pose of the cabinet's centre.
        :return: The drawer and its handle.
        """
        world_T_drawer = world_T_cabinet @ HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-self.cabinet_scale.x / 2 + self.drawer_depth / 2
        )
        drawer_scale = Scale(
            x=self.drawer_depth,
            y=self.cabinet_scale.y - 2 * self.wall_thickness,
            z=self.cabinet_scale.z - 2 * self.wall_thickness,
        )
        drawer = Drawer.create_with_new_body_in_world(
            name="drawer",
            world=world,
            world_root_T_self=world_T_drawer,
            scale=drawer_scale,
        )
        slider = Slider.create_with_new_body_in_world(
            name="drawer_slider",
            world=world,
            world_root_T_self=world_T_drawer,
            parent_connection_specification=Slider.parent_connection_specification(
                axis=Vector3.NEGATIVE_X(),
                dof_limits=self._mechanism_limits(self.drawer_travel),
            ),
        )
        drawer.add(slider)
        handle = self._add_handle(
            world,
            "drawer_handle",
            world_T_drawer
            @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-self.drawer_depth / 2,
                z=drawer_scale.z / 2 - 2 * self.handle_scale.z,
            ),
        )
        drawer.add(handle)
        return drawer, handle

    def _add_door(
        self, world: World, world_T_cabinet: HomogeneousTransformationMatrix
    ) -> tuple[Door, Handle]:
        """
        Add a door just in front of the cabinet's front, hinged at its right edge as
        seen from the robot, with a handle near its free edge.

        :param world: The world to add the door to.
        :param world_T_cabinet: The pose of the cabinet's centre.
        :return: The door and its handle.
        """
        world_T_hinge = world_T_cabinet @ HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-self.cabinet_scale.x / 2 - self.door_thickness,
            y=-self.cabinet_scale.y / 2,
        )
        hinge = Hinge.create_with_new_body_in_world(
            name="door_hinge",
            world=world,
            world_root_T_self=world_T_hinge,
            parent_connection_specification=Hinge.parent_connection_specification(
                axis=Vector3.Z(),
                dof_limits=self._mechanism_limits(self.door_swing),
            ),
        )
        door = Door.create_with_new_body_in_world(
            name="door",
            world=world,
            world_root_T_self=world_T_hinge
            @ HomogeneousTransformationMatrix.from_xyz_rpy(y=self.cabinet_scale.y / 2),
            scale=Scale(
                x=self.door_thickness, y=self.cabinet_scale.y, z=self.cabinet_scale.z
            ),
        )
        door.add(hinge)
        handle = self._add_handle(
            world,
            "door_handle",
            world_T_hinge
            @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-self.door_thickness / 2,
                y=self.cabinet_scale.y - self.handle_scale.y / 2 - self.wall_thickness,
                z=self.cabinet_scale.z / 2 - 2 * self.handle_scale.z,
            ),
        )
        door.add(handle)
        return door, handle

    def _add_handle(
        self,
        world: World,
        name: str,
        world_T_handle: HomogeneousTransformationMatrix,
    ) -> Handle:
        """
        :param world: The world to add the handle to.
        :param name: The name of the handle.
        :param world_T_handle: Where the handle is mounted on its part's front.
        :return: The new handle.
        """
        return Handle.get_annotation_specification(
            name,
            Handle.get_default_root_kinematic_structure_entity_specification(
                scale=self.handle_scale, thickness=self.handle_scale.z
            ),
        ).spawn(world, parent_T_self=world_T_handle)

    def _mechanism_limits(self, fully_open: float) -> DegreeOfFreedomLimits:
        """
        :param fully_open: The joint position at which the part is fully open.
        :return: The limits of a joint that is closed at zero.
        """
        return DegreeOfFreedomLimits(
            lower=DerivativeMap[float](
                position=0.0, velocity=-self.mechanism_velocity_limit
            ),
            upper=DerivativeMap[float](
                position=fully_open, velocity=self.mechanism_velocity_limit
            ),
        )
