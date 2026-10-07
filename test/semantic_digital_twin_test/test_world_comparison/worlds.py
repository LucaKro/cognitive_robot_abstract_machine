from dataclasses import dataclass, field

import trimesh

from semantic_digital_twin.api import (
    BodySpecification,
    ConnectionSpecification,
    FixedConnectionSpecification,
    PrismaticConnectionSpecification,
    RevoluteConnectionSpecification,
    WorldSpecification,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection


def box(name: str, length: float, centre_x: float) -> BodySpecification:
    """
    A box 0.3 m deep and high, of the given length along x, centred at the given x.
    """
    return BodySpecification.box(
        name,
        scale=Scale(length, 0.3, 0.3),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=centre_x),
    )


def square_face(name: str, x: float, side: float) -> BodySpecification:
    """
    An open surface: one square of the given side, facing along x at the given x and
    centred on the x axis, like the front of a box seen by a scanner.
    """
    half = side / 2
    face = trimesh.Trimesh(
        vertices=[
            [x, -half, -half],
            [x, half, -half],
            [x, half, half],
            [x, -half, half],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    return BodySpecification(name, ShapeCollection([Mesh.from_trimesh(mesh=face)]))


def world_of(*bodies: BodySpecification) -> World:
    return WorldSpecification(objects=list(bodies)).to_domain_object()


def position_limits(lower: float, upper: float) -> DegreeOfFreedomLimits:
    """
    :return: Limits on the position of a degree of freedom.
    """
    return DegreeOfFreedomLimits(
        lower=DerivativeMap(position=lower), upper=DerivativeMap(position=upper)
    )


def door_hinge() -> RevoluteConnectionSpecification:
    """
    :return: A door hinge turning about the vertical axis by up to 1.5 rad.
    """
    return RevoluteConnectionSpecification(
        axis=Vector3(0, 0, 1), dof_limits=position_limits(0.0, 1.5)
    )


def drawer_slide() -> PrismaticConnectionSpecification:
    """
    :return: A drawer slide pulling out along x by up to 0.4 m.
    """
    return PrismaticConnectionSpecification(
        axis=Vector3(1, 0, 0), dof_limits=position_limits(0.0, 0.4)
    )


@dataclass
class CabinetScene:
    """
    A cabinet with a door on its front, and a second cabinet beside it with a drawer.

    The door and drawer always stand in the same place; only how they are connected
    changes, so two scenes differ in their joints alone.
    """

    door_connection: ConnectionSpecification = field(default_factory=door_hinge)
    """
    How the door is connected to its cabinet.
    """

    cabinet_T_hinge: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.32, y=-0.3
        )
    )
    """
    Where the door's connection sits on the cabinet.
    """

    drawer_connection: ConnectionSpecification = field(default_factory=drawer_slide)
    """
    How the drawer is connected to its cabinet.
    """

    has_drawer: bool = True
    """
    Whether the second cabinet has its drawer.
    """

    root_T_scene: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Where the whole scene stands in the world's root frame.
    """

    def create_world(self) -> World:
        """
        :return: A new world holding the scene.
        """
        cabinet_T_door_centre = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.32)
        door = BodySpecification.box(
            "door",
            scale=Scale(0.02, 0.6, 0.6),
            origin=self.cabinet_T_hinge.inverse() @ cabinet_T_door_centre,
            parent_T_self=self.cabinet_T_hinge,
            connection_specification=self.door_connection,
        )
        drawer = BodySpecification.box(
            "drawer",
            scale=Scale(0.02, 0.5, 0.25),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.32),
            connection_specification=self.drawer_connection,
        )
        return world_of(
            BodySpecification.box(
                "door_cabinet",
                scale=Scale(0.6, 0.6, 0.6),
                parent_T_self=self.root_T_scene,
                child_specifications=[door],
            ),
            BodySpecification.box(
                "drawer_cabinet",
                scale=Scale(0.6, 0.6, 0.6),
                parent_T_self=self.root_T_scene
                @ HomogeneousTransformationMatrix.from_xyz_rpy(y=1.0),
                child_specifications=[drawer] if self.has_drawer else [],
            ),
        )


@dataclass
class AnnotatedCabinetScene:
    """
    A cabinet with a drawer in its front and a handle on the drawer, each body carrying
    a semantic annotation, with the drawer a part of the cabinet and the handle a part
    of the drawer.

    The bodies always stand in the same place; only their annotations and how those are
    related change, so two scenes differ in their semantics alone.
    """

    cabinet_type: type[HasRootBody] = Cabinet
    """
    The class of the cabinet's annotation.
    """

    drawer_type: type[HasRootBody] | None = Drawer
    """
    The class of the drawer's annotation, or ``None`` for a drawer body without one.
    """

    drawer_field: str = "drawers"
    """
    The part-whole field of the cabinet the drawer's annotation fills.
    """

    handle_type: type[HasRootBody] = Handle
    """
    The class of the handle's annotation.
    """

    handle_is_part: bool = True
    """
    Whether the handle's annotation is a part of the drawer's, rather than standing
    alone.
    """

    has_handle: bool = True
    """
    Whether the drawer has its handle at all.
    """

    root_T_scene: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Where the whole scene stands in the world's root frame.
    """

    def create_world(self) -> World:
        """
        :return: A new world holding the scene.
        """
        cabinet_T_drawer = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.32)
        drawer_T_handle = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.03)
        handle_specifications = {}
        objects = []
        if self.has_handle and self.handle_is_part and self.drawer_type is not None:
            handle_specifications = {"handle": self._handle(drawer_T_handle)}
        elif self.has_handle:
            objects.append(
                self._handle(self.root_T_scene @ cabinet_T_drawer @ drawer_T_handle)
            )
        drawer_body = BodySpecification.box(
            "drawer", scale=Scale(0.02, 0.5, 0.25), parent_T_self=cabinet_T_drawer
        )
        cabinet_body = BodySpecification.box(
            "cabinet", scale=Scale(0.6, 0.6, 0.6), parent_T_self=self.root_T_scene
        )
        if self.drawer_type is None:
            cabinet_body.child_specifications = [drawer_body]
            cabinet_parts = {}
        else:
            cabinet_parts = {
                self.drawer_field: [
                    self.drawer_type.get_annotation_specification(
                        "drawer",
                        drawer_body,
                        part_specifications=handle_specifications,
                    )
                ]
            }
        objects.append(
            self.cabinet_type.get_annotation_specification(
                "cabinet", cabinet_body, part_specifications=cabinet_parts
            )
        )
        return WorldSpecification(objects=objects).to_domain_object()

    def _handle(self, parent_T_handle: HomogeneousTransformationMatrix):
        """
        :return: The specification of the handle's annotation, posed relative to its
            parent.
        """
        return self.handle_type.get_annotation_specification(
            "handle",
            BodySpecification.box(
                "handle", scale=Scale(0.02, 0.2, 0.03), parent_T_self=parent_T_handle
            ),
        )
