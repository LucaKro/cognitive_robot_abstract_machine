import trimesh

from semantic_digital_twin.api import BodySpecification, WorldSpecification
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
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
