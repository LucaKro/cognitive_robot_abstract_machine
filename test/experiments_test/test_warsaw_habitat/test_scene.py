"""
Reading an HM3D scene: its triangles, and which annotated object each of them belongs
to.

The scenes here are two squares side by side, each written as its own geometry with its
own vertices, because that is the shape of the real files and it is what makes welding
and per-face colour reading worth checking at all.
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from experiments.warsaw.exceptions import (
    HabitatMeshesDisagreeError,
    HabitatSceneNotFoundError,
)
from experiments.warsaw.habitat.scene import HabitatDataset, HabitatScene, sampled

from .scenes import Palette, write_scene

# %% the scene to read

SCENE_NAME = "00900-TwoSquares"
"""
A scene named the way the release names one: a number, a dash, an identifier.
"""

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)


@pytest.fixture()
def dataset(tmp_path: Path) -> HabitatDataset:
    """
    :return: A release holding one scene of two objects, one per square.
    """
    write_scene(
        directory=tmp_path,
        scene_name=SCENE_NAME,
        instance_colors=[RED, GREEN],
        room_colors=[BLUE, YELLOW],
        rows=['1,FF0000,"floor",1', '2,00FF00,"table",2'],
    )
    return HabitatDataset(root=tmp_path)


@pytest.fixture()
def scene(dataset: HabitatDataset) -> HabitatScene:
    """
    :return: That scene, read.
    """
    return HabitatScene.read(dataset.scene(SCENE_NAME))


# %% reading a texture


def test_a_texture_is_read_without_blending() -> None:
    """
    A place is the colour of the patch it lands in, not a mixture of its neighbours.

    Blending is what makes a scene unreadable: the colour marks which object a face is,
    so a colour between two patches names no object at all.
    """
    palette = Palette(colors=(RED, GREEN, BLUE))
    read = sampled(
        np.asarray(palette.image),
        np.array([palette.place(RED), palette.place(BLUE)]),
    )
    assert read.tolist() == [list(RED), list(BLUE)]


# %% which object each face belongs to


def test_every_face_is_given_the_object_its_color_marks(scene: HabitatScene) -> None:
    """
    Two triangles per square, and the square's colour says which object they are.
    """
    assert scene.face_objects.tolist() == [1, 1, 2, 2]


def test_a_face_no_annotation_claims_belongs_to_nothing(tmp_path: Path) -> None:
    """
    HM3D leaves faces it never annotated, and they belong to no object rather than to
    the first one that happens to be near them.
    """
    write_scene(
        directory=tmp_path,
        scene_name=SCENE_NAME,
        instance_colors=[RED, GREEN],
        room_colors=[BLUE, YELLOW],
        rows=['1,FF0000,"floor",1'],
    )
    scene = HabitatScene.read(HabitatDataset(root=tmp_path).scene(SCENE_NAME))
    assert scene.face_objects.tolist() == [1, 1, 0, 0]


# %% the geometry


def test_welding_leaves_every_face_where_it_was(scene: HabitatScene) -> None:
    """
    The colours were read per face before the vertices were welded, so a weld that
    dropped or reordered a face would leave every label pointing at another face.
    """
    assert len(scene.mesh.faces) == len(scene.face_objects) == 4


def test_welding_makes_the_squares_touch(scene: HabitatScene) -> None:
    """
    Two objects touch when their faces share an edge, and in these files they only share
    it once the vertices written twice have been merged into one.

    Nothing in the pipeline can see a part meeting a whole without this: HM3D gives each
    face to exactly one object, so an edge is the only evidence two objects meet.
    """
    sides = scene.mesh.face_adjacency
    crossing = scene.face_objects[sides]
    assert (crossing[:, 0] != crossing[:, 1]).sum() == 1


def test_the_scene_keeps_the_colors_the_room_was_seen_in(scene: HabitatScene) -> None:
    """
    The pictures the pipeline shows a model are of the room, so the geometry carries the
    textured mesh's colours and not the flat ones the annotation is written in.
    """
    seen = {tuple(color[:3]) for color in scene.mesh.visual.vertex_colors}
    assert seen == {BLUE, YELLOW}


# %% what a room holds


def test_a_room_is_the_faces_of_the_objects_assigned_to_it(
    scene: HabitatScene,
) -> None:
    """
    A room is chosen by the table, and its geometry follows from which objects it names.
    """
    assert scene.faces_in_room(2).tolist() == [2, 3]


# %% files that are not a scene


def test_a_directory_without_a_scene_is_refused(tmp_path: Path) -> None:
    """
    A release names its scenes, so asking for one it does not hold is an error rather
    than an empty scene.
    """
    with pytest.raises(HabitatSceneNotFoundError):
        HabitatDataset(root=tmp_path).scene(SCENE_NAME)


def test_a_scene_without_its_textured_mesh_is_refused(tmp_path: Path) -> None:
    """
    The room's own colours come from the textured mesh, and a scene written without one
    cannot be shown to anybody.
    """
    write_scene(
        directory=tmp_path,
        scene_name=SCENE_NAME,
        instance_colors=[RED, GREEN],
        room_colors=[BLUE, YELLOW],
        rows=['1,FF0000,"floor",1'],
        with_textured_mesh=False,
    )
    with pytest.raises(HabitatSceneNotFoundError):
        HabitatDataset(root=tmp_path).scene(SCENE_NAME)


def test_two_meshes_over_different_triangles_are_refused(
    dataset: HabitatDataset,
) -> None:
    """
    The colours of the room are read off the textured mesh by face, which is only the
    same face if the two files are the same triangles.

    A release whose two meshes have drifted apart would paint every object the colour of
    another one.
    """
    files = dataset.scene(SCENE_NAME)
    moved = trimesh.load(str(files.textured_mesh), process=False)
    moved.apply_translation([10.0, 0.0, 0.0])
    moved.export(files.textured_mesh)
    with pytest.raises(HabitatMeshesDisagreeError):
        HabitatScene.read(files)
