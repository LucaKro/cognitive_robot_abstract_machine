"""
The parts of the Warsaw split that are decided by arithmetic rather than by a model.

Everything here is built from a handful of triangles rather than read from a scan, so it
says the same thing on every machine and every run: which faces several objects claim,
who keeps them once ownership is decided, and -- the one a picture cannot check -- that
the bodies come out centred on themselves without the geometry having moved an inch.
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from experiments.warsaw.scene_split import (
    Ownership,
    Pairing,
    exclusive_faces,
    owner_by_ontology,
    pairings,
    split_world,
)
from experiments.warsaw.segment_relations import claimant_groups
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


@pytest.fixture
def taxonomy():
    """
    :return: The taxonomy's classes by name.
    """
    return annotation_classes(SemanticAnnotation)


# %% which faces are claimed by whom


def test_claimant_groups_gathers_faces_by_who_claims_them():
    """
    A face claimed by the same objects as another belongs in the same question.
    """
    groups = claimant_groups(
        [np.array([0, 1, 2]), np.array([2, 3]), np.array([2])],
        ["one", "other", "third"],
        face_count=4,
    )
    assert {group.names: sorted(group.faces.tolist()) for group in groups} == {
        ("one", "other", "third"): [2],
        ("one", "other"): [],
    } or {group.names: sorted(group.faces.tolist()) for group in groups} == {
        ("one", "other", "third"): [2]
    }


def test_a_face_claimed_once_is_not_contested():
    """
    Only faces several objects claim are gathered; the rest are nobody's question.
    """
    groups = claimant_groups(
        [np.array([0, 1]), np.array([2, 3])], ["one", "other"], face_count=4
    )
    assert groups == []


def test_claimant_groups_are_the_same_every_time():
    """
    The split is applied to the groups a later run computes, so they must not drift.
    """
    segments = [np.array([0, 1, 2, 5]), np.array([2, 3, 5]), np.array([2, 4])]
    names = ["one", "other", "third"]
    first = claimant_groups(segments, names, face_count=6)
    again = claimant_groups(segments, names, face_count=6)
    assert [group.names for group in first] == [group.names for group in again]
    assert all(
        np.array_equal(one.faces, other.faces) for one, other in zip(first, again)
    )


# %% who a contested face belongs to


def test_the_ontology_gives_the_faces_to_the_part(taxonomy):
    """
    A cabinet can hold a drawer, so a face both claim is the drawer's surface.
    """
    assert (
        owner_by_ontology(
            ("cabinet_1", "drawer_1"),
            {"cabinet_1": taxonomy["Cabinet"], "drawer_1": taxonomy["Drawer"]},
        )
        == "drawer_1"
    )


def test_the_ontology_says_nothing_where_neither_can_hold_the_other(taxonomy):
    """
    Two classes with no part-whole relation between them settle nothing.
    """
    assert (
        owner_by_ontology(
            ("cabinet_1", "wall_1"),
            {"cabinet_1": taxonomy["Cabinet"], "wall_1": taxonomy["Wall"]},
        )
        is None
    )


def test_the_owner_keeps_the_faces_and_the_others_lose_them():
    """
    Applying one decision per set of claimants can leave nothing claimed twice.
    """
    segments = {
        "cabinet_1": np.array([0, 1, 2]),
        "drawer_1": np.array([1, 2, 3]),
    }
    split = exclusive_faces(
        segments,
        [Ownership(("cabinet_1", "drawer_1"), "drawer_1", np.array([1, 2]))],
    )
    assert split.faces["cabinet_1"].tolist() == [0]
    assert split.faces["drawer_1"].tolist() == [1, 2, 3]
    assert split.contested == {}
    assert split.lost_to == {"cabinet_1": {"drawer_1": 2}}


def test_an_object_that_loses_everything_is_reported_not_dropped_quietly():
    """
    An ownership answer covers a whole class of objects, so a wrong one empties every
    object of that kind at once and has to be answerable for.
    """
    segments = {"cabinet_1": np.array([1, 2]), "drawer_1": np.array([1, 2])}
    split = exclusive_faces(
        segments,
        [Ownership(("cabinet_1", "drawer_1"), "drawer_1", np.array([1, 2]))],
    )
    assert split.emptied == ["cabinet_1"]
    assert "cabinet_1" not in split.faces
    assert split.lost_to["cabinet_1"] == {"drawer_1": 2}


def test_a_pairing_whose_end_was_emptied_is_dropped():
    """
    A mount needs both ends to be bodies.
    """
    segments = {"cabinet_1": np.array([1]), "drawer_1": np.array([1]), "handle_1": np.array([9])}
    split = exclusive_faces(
        segments, [Ownership(("cabinet_1", "drawer_1"), "drawer_1", np.array([1]))]
    )
    carried = pairings(
        [
            Pairing(whole="cabinet_1", part="drawer_1", field_name="drawers"),
            Pairing(whole="drawer_1", part="handle_1", field_name="handle"),
        ],
        split,
    )
    assert [(one.whole, one.part) for one in carried] == [("drawer_1", "handle_1")]


# %% the geometry the bodies are built with


@pytest.fixture
def two_boxes():
    """
    :return: Two unit boxes far from the origin and far from each other, and the faces
        of each, so that a lost transform or a lost centring cannot go unnoticed.
    """
    one = trimesh.creation.box(extents=(1, 1, 1)).apply_translation([10, 0, 0])
    other = trimesh.creation.box(extents=(1, 1, 1)).apply_translation([-4, 7, 2])
    scene = trimesh.util.concatenate([one, other])
    return scene, {"one": np.arange(0, 12), "other": np.arange(12, 24)}


def where_it_sits(world, body, mesh) -> np.ndarray:
    """
    :param world: The world holding the body.
    :param body: The body to place.
    :param mesh: The geometry to place, in the body's own frame.
    :return: Its vertices in world coordinates.
    """
    root = next(one for one in world.bodies if one.name.name == "root_body")
    pose = world.compute_forward_kinematics_np(root, body)
    padded = np.column_stack([mesh.vertices, np.ones(len(mesh.vertices))])
    return (pose @ padded.T)[:3].T


def test_bodies_are_centred_on_themselves_and_stay_centred_when_read_back(
    two_boxes, tmp_path
):
    """
    A shape backed by a file loads that file again when a world is read back, so a
    centring that only moved the loaded mesh is a centring the world loses.
    """
    scene, faces = two_boxes
    world = split_world(
        scene, faces, HomogeneousTransformationMatrix(), directory=tmp_path
    )

    for body in world.bodies:
        if body.name.name == "root_body":
            continue
        for mesh in (body.collision[0].mesh, Mesh(filename=body.collision[0].filename).mesh):
            middle = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2
            assert np.allclose(middle, 0, atol=1e-9)


def test_the_geometry_does_not_move(two_boxes, tmp_path):
    """
    Cutting a scene into bodies moves nothing: every face ends up where it was, whether
    the world is the one just built or one read back from the bodies' files.
    """
    scene, faces = two_boxes
    to_world = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.0, 2.0, 3.0, np.pi / 2, 0.0, 0.0
    )
    world = split_world(scene, faces, to_world, directory=tmp_path)

    for body in world.bodies:
        if body.name.name == "root_body":
            continue
        corners = scene.vertices[scene.faces[faces[body.name.name]].ravel()]
        padded = np.column_stack([corners, np.ones(len(corners))])
        expected = (to_world.to_np() @ padded.T)[:3].T

        for mesh in (body.collision[0].mesh, Mesh(filename=body.collision[0].filename).mesh):
            placed = where_it_sits(world, body, mesh)
            assert np.allclose(placed.min(axis=0), expected.min(axis=0), atol=1e-6)
            assert np.allclose(placed.max(axis=0), expected.max(axis=0), atol=1e-6)


def test_every_face_lands_in_exactly_one_body(two_boxes, tmp_path):
    """
    A face belongs to one body's geometry, which is the whole reason ownership had to be
    decided before anything was cut.
    """
    scene, faces = two_boxes
    world = split_world(
        scene, faces, HomogeneousTransformationMatrix(), directory=tmp_path
    )
    built = sum(
        len(body.collision[0].mesh.faces)
        for body in world.bodies
        if body.name.name != "root_body"
    )
    assert built == sum(len(kept) for kept in faces.values())
