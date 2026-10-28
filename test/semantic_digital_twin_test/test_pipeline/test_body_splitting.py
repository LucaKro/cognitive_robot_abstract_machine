import numpy as np
import pytest
import trimesh

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    BodyWithoutShapeError,
    PieceOutsideShapeError,
)
from semantic_digital_twin.part_finding.splitting import ShapePiece
from semantic_digital_twin.pipeline.body_splitting import SplitBodyIntoPieces
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures


@pytest.fixture
def room() -> World:
    """
    A world whose root body is one box mesh that looks and collides alike.
    """
    world = World.create_with_root_body(root_body_name="room")
    room_body = world.root
    shapes = ShapeCollection(
        [
            Mesh.from_trimesh(
                mesh=trimesh.creation.box(),
                origin=HomogeneousTransformationMatrix(reference_frame=room_body),
            )
        ],
        reference_frame=room_body,
    )
    with world.modify_world():
        room_body.update_shapes(visual=shapes, collision=shapes)
    return world


def face_count(body: Body) -> int:
    return sum(len(shape.mesh.faces) for shape in body.visual)


# %% splitting


def test_a_piece_gets_the_faces_it_takes(room):
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0, 1]))

    SplitBodyIntoPieces(body_name=room.root.name, pieces=[lid]).apply(room)

    assert face_count(lid.body) == len(lid.faces)


def test_the_body_keeps_the_faces_no_piece_takes(room):
    total = face_count(room.root)
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0, 1]))
    side = ShapePiece(body=Body(name=PrefixedName("side")), faces=np.array([2, 3]))

    SplitBodyIntoPieces(body_name=room.root.name, pieces=[lid, side]).apply(room)

    assert face_count(room.root) == total - len(lid.faces) - len(side.faces)


def test_each_piece_is_fixed_to_the_body_it_is_cut_from(room):
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0, 1]))

    SplitBodyIntoPieces(body_name=room.root.name, pieces=[lid]).apply(room)

    assert isinstance(lid.body.parent_connection, FixedConnection)
    assert lid.body.parent_connection.parent is room.root


def test_a_piece_without_faces_gets_a_body_without_shape(room):
    empty = ShapePiece(body=Body(name=PrefixedName("empty")), faces=np.array([], int))

    SplitBodyIntoPieces(body_name=room.root.name, pieces=[empty]).apply(room)

    assert empty.body in room.bodies
    assert not empty.body.visual


def test_pieces_collide_as_they_look_when_the_body_did(room):
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0, 1]))

    SplitBodyIntoPieces(body_name=room.root.name, pieces=[lid]).apply(room)

    assert lid.body.collision is lid.body.visual


# %% misuse


def test_a_piece_taking_a_face_the_shape_lacks_raises(room):
    beyond = ShapePiece(
        body=Body(name=PrefixedName("beyond")),
        faces=np.array([face_count(room.root)]),
    )

    with pytest.raises(PieceOutsideShapeError):
        SplitBodyIntoPieces(body_name=room.root.name, pieces=[beyond]).apply(room)


def test_cutting_a_body_without_shape_raises():
    world = World.create_with_root_body(root_body_name="room")
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0]))

    with pytest.raises(BodyWithoutShapeError):
        SplitBodyIntoPieces(body_name=world.root.name, pieces=[lid]).apply(world)
