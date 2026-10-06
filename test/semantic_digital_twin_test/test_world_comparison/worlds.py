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
