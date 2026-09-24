"""
Tracy at its table with a cabinet in front of its left arm, whose drawer or door only
the physics moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

from semantic_digital_twin.api import (
    RobotSpecification,
    SemanticAnnotationWithRootSpecification,
    WorldSpecification,
)
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
    Pose2D,
    RotationMatrix,
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

    @property
    def mechanism(self) -> ActiveConnection1DOF:
        """
        The joint the moving part slides or turns on; zero is closed.
        """
        return self.part.mechanical_joint.root.parent_connection

    @property
    def handle(self) -> Handle:
        """
        The handle on the moving part's front.
        """
        return self.part.handle

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


# %% describing it


@dataclass
class CabinetSceneSpecification:
    """
    World-independent description of a :class:`CabinetScene` whose cabinet has one
    moving part.

    Lengths are in metres. The cabinet is placed relative to Tracy's table frame, which
    lies on the table top at the edge the arms are mounted at, with x pointing across
    the table.
    """

    articulated_part: ArticulatedPart
    """
    Which part of the cabinet moves.
    """

    world_root_name: str = "floor"
    """
    The name of the world's root body, which must differ from the ``map`` link Tracy's
    description brings along: MuJoCo refuses two bodies of the same name.
    """

    world_T_table: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(z=0.88)
    )
    """
    Where Tracy's description places its table frame in the world (``table_joint`` in
    ``tracy.urdf.xacro``).
    """

    cabinet_scale: Scale = field(default_factory=lambda: Scale(x=0.4, y=0.4, z=0.3))
    """
    The outer size of the cabinet's case; x is its depth.
    """

    table_T_cabinet_front: Pose2D = field(
        default_factory=lambda: Pose2D(x=0.85, y=0.35)
    )
    """
    Where the centre of the cabinet's open front stands on the table top; the cabinet
    extends along the pose's x axis and turns with its yaw about that point.

    The default puts the front in front of the left arm, far enough across the table to
    leave the hand room to come down in front of it.
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

    mechanism_axis_deviation: float = 0.0
    """
    How far the moving part's joint axis is turned away from the cabinet's own axes, in
    radians. A drawer's slider turns about the vertical, so the drawer still slides
    level; a door's hinge tilts within the cabinet's front.

    A believed scene whose deviation differs from the true scene's has a wrong prior on
    the joint's parameters.
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

    def world_specification(self) -> WorldSpecification:
        """
        :return: The world with Tracy and the cabinet, before the arms are parked.
        """
        return WorldSpecification(
            robots=[RobotSpecification(Tracy)],
            objects=[self.cabinet_specification()],
        )

    def to_domain_object(self) -> CabinetScene:
        """
        Build a new world from :meth:`world_specification`, with both arms parked.

        :return: The scene in the new world.
        """
        world = self.world_specification().to_domain_object()
        world.force_root_name(PrefixedName(self.world_root_name))
        [robot] = world.get_semantic_annotations_by_type(Tracy)
        for arm in robot.get_arms():
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        world.notify_state_change()
        [cabinet] = world.get_semantic_annotations_by_type(Cabinet)
        [part] = cabinet.drawers + cabinet.doors
        part.mechanical_joint.root.parent_connection.dynamics = self.mechanism_dynamics
        return CabinetScene(world=world, robot=robot, cabinet=cabinet, part=part)

    def cabinet_specification(
        self,
    ) -> SemanticAnnotationWithRootSpecification[Cabinet]:
        """
        :return: The cabinet standing on the table, with its moving part.
        """
        root_specification = (
            Cabinet.get_default_root_kinematic_structure_entity_specification(
                scale=self.cabinet_scale, wall_thickness=self.wall_thickness
            )
        )
        root_specification.parent_T_self = self.world_T_cabinet()
        match self.articulated_part:
            case ArticulatedPart.DRAWER:
                part_specifications = {"drawers": self._drawer_specification()}
            case ArticulatedPart.DOOR:
                part_specifications = {"doors": self._door_specification()}
        return Cabinet.get_annotation_specification(
            "cabinet", root_specification, part_specifications=part_specifications
        )

    def world_T_cabinet(self) -> HomogeneousTransformationMatrix:
        """
        :return: Where the centre of the cabinet's case stands in the world.
        """
        return (
            self.world_T_table
            @ self.table_T_cabinet_front.to_homogeneous_matrix()
            @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=self.cabinet_scale.x / 2, z=self.cabinet_scale.z / 2
            )
        )

    def _drawer_specification(self) -> SemanticAnnotationWithRootSpecification[Drawer]:
        """
        :return: A drawer that slides out of the cabinet's front, with a handle on its
            face.
        """
        drawer_scale = Scale(
            x=self.drawer_depth,
            y=self.cabinet_scale.y - 2 * self.wall_thickness,
            z=self.cabinet_scale.z - 2 * self.wall_thickness,
        )
        slider = Slider.get_annotation_specification(
            "drawer_slider",
            Slider.get_default_root_kinematic_structure_entity_specification(),
            parent_connection_specification=Slider.parent_connection_specification(
                axis=self._turned_axis(Vector3.NEGATIVE_X(), turned_about=Vector3.Z()),
                dof_limits=self._mechanism_limits(self.drawer_travel),
            ),
        )
        handle = self._handle_specification(
            "drawer_handle",
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-self.drawer_depth / 2,
                z=drawer_scale.z / 2 - 2 * self.handle_scale.z,
            ),
        )
        root_specification = (
            Drawer.get_default_root_kinematic_structure_entity_specification(
                scale=drawer_scale
            )
        )
        root_specification.parent_T_self = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-self.cabinet_scale.x / 2 + self.drawer_depth / 2
        )
        return Drawer.get_annotation_specification(
            "drawer",
            root_specification,
            part_specifications={"mechanical_joint": slider, "handle": handle},
        )

    def _door_specification(self) -> SemanticAnnotationWithRootSpecification[Door]:
        """
        :return: A door just in front of the cabinet's front, hinged at its right edge
            as seen from the robot, with a handle near its free edge.
        """
        hinge_root_specification = (
            Hinge.get_default_root_kinematic_structure_entity_specification()
        )
        hinge_root_specification.parent_T_self = (
            HomogeneousTransformationMatrix.from_xyz_rpy(y=-self.cabinet_scale.y / 2)
        )
        hinge = Hinge.get_annotation_specification(
            "door_hinge",
            hinge_root_specification,
            parent_connection_specification=Hinge.parent_connection_specification(
                axis=self._turned_axis(Vector3.Z(), turned_about=Vector3.X()),
                dof_limits=self._mechanism_limits(self.door_swing),
            ),
        )
        handle = self._handle_specification(
            "door_handle",
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-self.door_thickness / 2,
                y=self.cabinet_scale.y / 2
                - self.handle_scale.y / 2
                - self.wall_thickness,
                z=self.cabinet_scale.z / 2 - 2 * self.handle_scale.z,
            ),
        )
        root_specification = (
            Door.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(
                    x=self.door_thickness,
                    y=self.cabinet_scale.y,
                    z=self.cabinet_scale.z,
                )
            )
        )
        root_specification.parent_T_self = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-self.cabinet_scale.x / 2 - self.door_thickness
        )
        return Door.get_annotation_specification(
            "door",
            root_specification,
            part_specifications={"mechanical_joint": hinge, "handle": handle},
        )

    def _handle_specification(
        self, name: str, part_T_handle: HomogeneousTransformationMatrix
    ) -> SemanticAnnotationWithRootSpecification[Handle]:
        """
        :param name: The name of the handle.
        :param part_T_handle: Where the handle is mounted on its part's front.
        :return: The handle.
        """
        root_specification = (
            Handle.get_default_root_kinematic_structure_entity_specification(
                scale=self.handle_scale, thickness=self.handle_scale.z
            )
        )
        root_specification.parent_T_self = part_T_handle
        return Handle.get_annotation_specification(name, root_specification)

    def _turned_axis(self, axis: Vector3, turned_about: Vector3) -> Vector3:
        """
        :param axis: The joint axis the mechanism has without a deviation.
        :param turned_about: The axis the deviation turns it about.
        :return: The joint axis turned by :attr:`mechanism_axis_deviation`.
        """
        return (
            RotationMatrix.from_axis_angle(turned_about, self.mechanism_axis_deviation)
            @ axis
        )

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
