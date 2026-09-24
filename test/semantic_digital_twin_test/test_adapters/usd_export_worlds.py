from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    Scale,
    Shape,
    Sphere,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% cabinet


def _box_body(name: str, scale: Scale) -> Body:
    """
    A body whose visual and collision geometry is one box of ``scale``.
    """
    shapes = ShapeCollection([Box(scale=scale)])
    return Body(name=PrefixedName(name), visual=shapes, collision=shapes)


def joint_limits(lower: float, upper: float) -> DegreeOfFreedomLimits:
    return DegreeOfFreedomLimits(
        lower=DerivativeMap(position=lower), upper=DerivativeMap(position=upper)
    )


def build_cabinet_world() -> World:
    """
    A cabinet with one of each connection type the USD exporter writes, each with a
    non-trivial joint frame:

    - ``door``: revolute about the connection frame's z axis, hinged at the carcass's
      front-left edge.
    - ``drawer``: prismatic along the connection frame's -y axis, with the frame itself
      rotated, so neither the axis nor the frame is aligned with USD's joint x axis.
    - ``knob``: fixed on the door.

    :return: The built world, rooted at ``carcass``.
    """
    world = World()
    carcass = _box_body("carcass", Scale(0.5, 0.6, 0.8))
    door = _box_body("door", Scale(0.02, 0.6, 0.4))
    drawer = _box_body("drawer", Scale(0.45, 0.55, 0.3))
    knob = _box_body("knob", Scale(0.03, 0.03, 0.03))
    with world.modify_world():
        world.add_body(carcass)
        world.add_body(door)
        world.add_body(drawer)
        world.add_body(knob)
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=carcass,
                child=door,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.26, y=-0.3, z=0.2
                ),
                connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=0.3
                ),
                axis=Vector3.Z(),
                dof_limits=joint_limits(-1.6, 0.0),
            )
        )
        world.add_connection(
            PrismaticConnection.create_with_dofs(
                world=world,
                parent=carcass,
                child=drawer,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.05, z=-0.2, yaw=math.pi / 2
                ),
                axis=Vector3(0.0, -1.0, 0.0),
                dof_limits=joint_limits(0.0, 0.4),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=door,
                child=knob,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.03, y=0.25
                ),
            )
        )
    return world


# %% single bodies


def build_single_body_world(body: Body) -> World:
    """
    :param body: The body to add.
    :return: A world holding only ``body``, as its root.
    """
    world = World()
    with world.modify_world():
        world.add_body(body)
    return world


def build_primitive_shapes_body() -> Body:
    """
    :return: A body with one box, sphere and cylinder, each offset and rotated
        relative to the body.
    """
    shapes = ShapeCollection(
        [
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.1, roll=0.3),
                scale=Scale(0.1, 0.2, 0.3),
            ),
            Sphere(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.2),
                radius=0.05,
            ),
            Cylinder(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.3, pitch=0.4),
                width=0.08,
                height=0.25,
            ),
        ]
    )
    return Body(name=PrefixedName("primitives"), visual=shapes, collision=shapes)


def build_mesh_body() -> Body:
    """
    :return: A body whose geometry is one offset and rotated box-shaped triangle mesh.
    """
    mesh = Mesh.from_trimesh(
        mesh=trimesh.creation.box(extents=(0.2, 0.3, 0.4)),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5, yaw=0.7),
    )
    shapes = ShapeCollection([mesh])
    return Body(name=PrefixedName("mesh_body"), visual=shapes, collision=shapes)


def build_body_with_separate_collision() -> Body:
    """
    :return: A body whose visual geometry is a sphere and whose collision geometry is a
        box, as in an asset with a detailed visual mesh and a simplified collider.
    """
    return Body(
        name=PrefixedName("separate_collision"),
        visual=ShapeCollection([Sphere(radius=0.1)]),
        collision=ShapeCollection([Box(scale=Scale(0.2, 0.2, 0.2))]),
    )


def build_body_with_inertial() -> Body:
    """
    :return: A body with a mass, an off-centre centre of mass, and a non-diagonal
        inertia tensor.
    """
    body = _box_body("heavy", Scale(0.2, 0.2, 0.2))
    body.inertial = Inertial(
        mass=3.5,
        center_of_mass=Point3(0.01, -0.02, 0.03, reference_frame=body),
        inertia=InertiaTensor.from_values(
            ixx=0.04, iyy=0.05, izz=0.06, ixy=0.001, ixz=0.002, iyz=0.003
        ),
    )
    return body


@dataclass(eq=False)
class UnwritableShape(Shape):
    """
    A shape type no USD geometry writer handles.
    """

    @property
    def volume(self) -> float:
        return 0.0

    @property
    def mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh()

    @property
    def local_frame_bounding_box(self) -> VolumetricBoundingBox:
        return VolumetricBoundingBox(0, 0, 0, 0, 0, 0, self.origin)


def build_body_with_unwritable_shape() -> Body:
    """
    :return: A body whose only shape is an :class:`UnwritableShape`.
    """
    shapes = ShapeCollection([UnwritableShape()])
    return Body(name=PrefixedName("unwritable"), visual=shapes, collision=shapes)


def mesh_vertices_in_body_frame(mesh: Mesh) -> np.ndarray:
    """
    :param mesh: A mesh shape.
    :return: The mesh's vertices, sorted, in the frame of the body it belongs to.
    """
    vertices = trimesh.transform_points(mesh.mesh.vertices, mesh.origin.to_np())
    return vertices[np.lexsort(vertices.T)]
