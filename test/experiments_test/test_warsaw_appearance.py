"""
Drawing a scene from the texture and normals its source ships, rather than from the flat
colours a labelled mesh can carry.

The pipeline's mesh has to be welded and has to carry a label per face, and neither is
compatible with a texture: welding merges the vertices a UV seam splits, and a face
property has nowhere to put an image. So the appearance travels beside the labelled mesh
rather than inside it, over the same faces in the same order, and only the pictures read
it.

A scan ships no appearance mesh, so a scan is drawn exactly as it always was.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from .test_warsaw_world_loader import write_scene
from experiments.warsaw.world_loader.appearance import (
    APPEARANCE_FILE,
    AppearanceMesh,
    slice_keeping_appearance,
)

# %% a textured square, split into two geometries


def textured_square(offset: float, colour: tuple) -> trimesh.Trimesh:
    """
    :param offset: How far along x the square starts.
    :param colour: What its texture is painted.
    :return: A unit square of two triangles, textured, with normals of its own.
    """
    from PIL import Image

    vertices = np.array(
        [
            [offset, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset + 1.0, 1.0, 0.0],
            [offset, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    # Leaning normals, so that keeping them can be told from recomputing them: a flat
    # square's own face normal is straight up, and these are not.
    normals = np.tile(np.array([0.0, 0.6, 0.8]), (4, 1))
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        visual=trimesh.visual.TextureVisuals(
            uv=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            image=Image.new("RGB", (4, 4), colour),
        ),
        process=False,
    )


LEANING = np.array([0.0, 0.6, 0.8])
"""
The vertex normal the fixture gives every corner, which no flat square would have.
"""


@pytest.fixture()
def written(tmp_path: Path) -> Path:
    """
    :return: A directory holding an appearance mesh of two squares, four faces in all.
    """
    pieces = [textured_square(0.0, (255, 0, 0)), textured_square(1.0, (0, 0, 255))]
    AppearanceMesh.write(pieces, tmp_path)
    return tmp_path


# %% slicing one geometry


def test_slicing_keeps_the_texture() -> None:
    """
    A slice that dropped the material would leave the picture no better than the flat
    colours it was meant to replace.
    """
    source = textured_square(0.0, (255, 0, 0))
    sliced = slice_keeping_appearance(source, np.array([0]))
    assert sliced.visual.material is source.visual.material


def test_slicing_keeps_the_uv_of_every_vertex_it_keeps() -> None:
    """
    A texture reaches a triangle through its vertices' UVs, so a slice that renumbered
    the vertices without carrying their UVs across would paint the wrong part of it.
    """
    source = textured_square(0.0, (255, 0, 0))
    sliced = slice_keeping_appearance(source, np.array([0]))
    assert np.allclose(sliced.visual.uv, source.visual.uv[np.unique(source.faces[[0]])])


def test_slicing_keeps_the_source_normals() -> None:
    """
    ``submesh`` drops vertex normals and lets trimesh recompute flat ones, which is what
    makes a coarse triangle read as a facet.

    The source's own normals are what smooth shading needs.
    """
    sliced = slice_keeping_appearance(textured_square(0.0, (255, 0, 0)), np.array([0]))
    assert np.allclose(sliced.vertex_normals, LEANING)


def test_slicing_keeps_only_the_faces_asked_for() -> None:
    """
    A piece holds the faces it was given and the vertices those need, and no others.
    """
    sliced = slice_keeping_appearance(textured_square(0.0, (255, 0, 0)), np.array([1]))
    assert len(sliced.faces) == 1
    assert len(sliced.vertices) == 3


# %% the file beside the mesh


def test_a_directory_without_one_has_no_appearance(tmp_path: Path) -> None:
    """
    A Warsaw scan ships no appearance mesh, and reading one must not become a
    requirement for reading a scene.
    """
    assert AppearanceMesh.beside(tmp_path) is None


def test_an_appearance_mesh_is_read_back_from_beside_the_scene(written: Path) -> None:
    """
    Written and read through a file, since that is how a run reaches one.
    """
    assert (written / APPEARANCE_FILE).is_file()
    assert AppearanceMesh.beside(written) is not None


def test_the_faces_come_back_in_the_order_they_were_written(written: Path) -> None:
    """
    The whole arrangement rests on face *i* of the appearance mesh being face *i* of the
    labelled mesh, so the order the pieces are walked in has to survive the file.
    """
    read = AppearanceMesh.beside(written)
    assert read.face_count == 4


def test_the_texture_survives_the_file(written: Path) -> None:
    """
    The point of the file.
    """
    read = AppearanceMesh.beside(written)
    drawn = read.faces_alone(np.array([0, 1, 2, 3]))
    assert all(
        piece.visual.material.baseColorTexture is not None
        for piece in drawn.geometry.values()
    )


def test_the_normals_survive_the_file(written: Path) -> None:
    """
    The other half of the point: a mesh reloaded with recomputed normals is shaded flat.
    """
    read = AppearanceMesh.beside(written)
    drawn = read.faces_alone(np.array([0, 1, 2, 3]))
    for piece in drawn.geometry.values():
        assert np.allclose(piece.vertex_normals, LEANING, atol=1e-3)


# %% taking some faces out of it


def test_asking_for_one_object_draws_only_its_faces(written: Path) -> None:
    """
    What a close-up of one segment needs: that segment's faces and nothing else's.
    """
    read = AppearanceMesh.beside(written)
    drawn = read.faces_alone(np.array([0, 1]))
    assert sum(len(piece.faces) for piece in drawn.geometry.values()) == 2


def test_faces_spanning_two_pieces_are_both_drawn(written: Path) -> None:
    """
    An object's faces need not come from one source geometry, so a drawable is a scene
    of pieces rather than a single mesh.
    """
    read = AppearanceMesh.beside(written)
    drawn = read.faces_alone(np.array([1, 2]))
    assert len(drawn.geometry) == 2
    assert sum(len(piece.faces) for piece in drawn.geometry.values()) == 2


def test_asking_for_no_faces_draws_nothing(written: Path) -> None:
    """
    A segment the appearance mesh has no faces for is not a drawable of everything.
    """
    read = AppearanceMesh.beside(written)
    assert len(read.faces_alone(np.array([], dtype=int)).geometry) == 0


# %% what a scene says about how it looks


def labelled_mesh(directory: Path) -> Path:
    """
    Write the one mesh every Warsaw scene has: two squares, an instance per face.

    Laid out as the appearance fixture lays its pieces out, so that face *i* of the one
    is face *i* of the other, which is the whole arrangement under test.

    :param directory: Where to write it.
    :return: The file written.
    """
    corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    vertices = np.concatenate(
        [
            np.array([[x, y, 0.0], [x + 0.4, y, 0.0], [x, y + 0.4, 0.0]])
            for x, y in corners
        ]
    )
    faces = np.arange(len(vertices)).reshape(-1, 3)
    return write_scene(
        directory / "mesh_all_classes.ply",
        vertices,
        faces,
        {"square": [1, 1, 2, 2]},
    )


def test_a_scan_reads_as_a_scene_that_says_nothing_about_how_it_looks(
    tmp_path: Path,
) -> None:
    """
    The Warsaw scans ship one PLY and nothing else, and must go on being read and drawn
    exactly as they were.
    """
    from experiments.warsaw.world_loader.scene import WarsawScene

    labelled_mesh(tmp_path)
    assert WarsawScene.from_directory(tmp_path).appearance is None


def test_a_scene_with_an_appearance_mesh_beside_it_reads_it(tmp_path: Path) -> None:
    """
    A converted HM3D room ships both, and the scene hands the pictures the second.
    """
    from experiments.warsaw.world_loader.scene import WarsawScene

    labelled_mesh(tmp_path)
    AppearanceMesh.write([textured_square(0.0, (255, 0, 0))], tmp_path)

    read = WarsawScene.from_directory(tmp_path)
    assert read.appearance is not None
    assert read.appearance.face_count == 2


# %% what the loader draws a plain picture from


def loaded(directory: Path):
    """
    :param directory: A scene directory.
    :return: That scene, loaded the way a run loads it.
    """
    from experiments.warsaw.world_loader.loader import WarsawWorldLoader

    return WarsawWorldLoader(input_directory=directory)


def test_a_scan_is_drawn_from_its_own_face_colours(tmp_path: Path) -> None:
    """
    No appearance mesh, so the plain picture is the mesh it always was, and every scan
    render stays what it was.
    """
    labelled_mesh(tmp_path)
    world = loaded(tmp_path)
    segments = list(world.scene.segments())
    drawn = world.segments_as_scanned(segments)
    assert isinstance(drawn, trimesh.Trimesh)


def test_a_scene_with_an_appearance_mesh_is_drawn_from_it(tmp_path: Path) -> None:
    """
    The change this is all for: the plain picture comes from the source's texture rather
    than from a colour per vertex flattened to a colour per triangle.
    """
    labelled_mesh(tmp_path)
    AppearanceMesh.write(
        [textured_square(0.0, (255, 0, 0)), textured_square(1.0, (0, 0, 255))], tmp_path
    )
    world = loaded(tmp_path)
    segments = list(world.scene.segments())
    drawn = world.segments_as_scanned(segments)
    assert isinstance(drawn, trimesh.Scene)
    assert all(
        isinstance(piece.visual, trimesh.visual.TextureVisuals)
        for piece in drawn.geometry.values()
    )
