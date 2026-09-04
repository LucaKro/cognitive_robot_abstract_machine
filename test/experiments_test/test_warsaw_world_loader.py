"""
Reading a Warsaw scene: one mesh whose faces carry, per class, the instance they are.

The scenes here are a few triangles written the way the dataset writes one, so what is
checked is the reading and not the scan: which objects a file's labels describe, what a
directory that holds no scene says, and where the cameras end up standing.
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh
from plyfile import PlyData, PlyElement

from experiments.warsaw.exceptions import (
    AmbiguousWarsawSceneError,
    WarsawLabelsMissingError,
    WarsawSceneNotFoundError,
)
from experiments.warsaw.world_loader import WarsawScene, WarsawWorldLoader

# %% a scene file written the way the dataset writes one


def write_scene(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: dict[str, list[int]],
) -> Path:
    """
    Write a mesh whose faces carry one instance number per class.

    :param path: Where to write it.
    :param vertices: The scene's vertices.
    :param faces: The scene's faces.
    :param labels: Per class, the instance each face belongs to, 0 where the class does
        not cover the face.
    :return: The file written.
    """
    written_vertices = np.array(
        [tuple(vertex) for vertex in vertices],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    written_faces = np.empty(
        len(faces),
        dtype=[("vertex_indices", "i4", (3,))] + [(name, "i4") for name in labels],
    )
    written_faces["vertex_indices"] = faces
    for name, instances in labels.items():
        written_faces[name] = instances

    PlyData(
        [
            PlyElement.describe(written_vertices, "vertex"),
            PlyElement.describe(written_faces, "face"),
        ]
    ).write(str(path))
    return path


@pytest.fixture
def two_class_scene(tmp_path) -> Path:
    """
    :return: A directory holding a scene of four faces: two cabinets of one face each,
        one drawer covering the second of them, and one face no class covers.
    """
    box = trimesh.creation.box(extents=(1, 1, 1))
    faces = box.faces[:4]
    write_scene(
        tmp_path / "scene.ply",
        box.vertices,
        faces,
        {
            "cabinet": [1, 2, 2, 0],
            "drawer": [0, 1, 1, 0],
        },
    )
    return tmp_path


# %% reading a scene


def test_every_labelled_instance_becomes_a_segment(two_class_scene):
    """
    An object is a class and an instance of it, however many faces it is made of.
    """
    scene = WarsawScene.from_directory(two_class_scene)
    assert [str(segment.name) for segment in scene.segments()] == [
        "cabinet_1",
        "cabinet_2",
        "drawer_1",
    ]


def test_the_faces_no_class_covers_become_no_segment(two_class_scene):
    """
    Instance 0 marks a face a class does not cover, and marks no object.
    """
    scene = WarsawScene.from_directory(two_class_scene)
    assert all(3 not in segment.faces for segment in scene.segments())


def test_a_segment_holds_only_the_faces_of_its_instance(two_class_scene):
    """
    :attr:`LabelSegment.faces` is what a body is later cut from, so it must hold the
    faces of that instance and no others.
    """
    scene = WarsawScene.from_directory(two_class_scene)
    segments = {str(segment.name): segment for segment in scene.segments()}
    assert segments["cabinet_1"].faces.tolist() == [0]
    assert segments["cabinet_2"].faces.tolist() == [1, 2]
    assert segments["drawer_1"].faces.tolist() == [1, 2]


def test_a_face_can_belong_to_objects_of_several_classes(two_class_scene):
    """
    The overlap the whole pipeline exists to resolve: a drawer front is the drawer and
    the cabinet holding it, and reading must not hide that.
    """
    scene = WarsawScene.from_directory(two_class_scene)
    segments = {str(segment.name): segment for segment in scene.segments()}
    assert set(segments["cabinet_2"].faces) == set(segments["drawer_1"].faces)


def test_the_classes_are_read_in_the_order_the_file_declares_them(two_class_scene):
    """
    :return: The classes name the scene's labels, and nothing renames or reorders them.
    """
    assert WarsawScene.from_directory(two_class_scene).class_names == [
        "cabinet",
        "drawer",
    ]


# %% a file that is not a scene


def test_a_directory_without_a_scene_is_reported(tmp_path):
    """
    :raises WarsawSceneNotFoundError: Which says where it looked and for what.
    """
    with pytest.raises(WarsawSceneNotFoundError):
        WarsawScene.from_directory(tmp_path)


def test_a_directory_holding_more_than_one_scene_is_reported(tmp_path, two_class_scene):
    """
    Which of two meshes is the scene is not something to guess at.
    """
    box = trimesh.creation.box()
    write_scene(two_class_scene / "another.ply", box.vertices, box.faces[:1], {"wall": [1]})
    with pytest.raises(AmbiguousWarsawSceneError):
        WarsawScene.from_directory(two_class_scene)


def test_a_mesh_carrying_no_labels_is_reported(tmp_path):
    """
    A mesh without labels describes no objects, which is worth saying rather than
    reading as a scene of nothing.
    """
    trimesh.creation.box().export(str(tmp_path / "scene.ply"))
    with pytest.raises(WarsawLabelsMissingError):
        WarsawScene.from_directory(tmp_path)


# %% loading it into a world


def test_the_scene_becomes_one_body(two_class_scene):
    """
    The scene stays one body until its overlapping labels have been resolved: cutting
    it earlier would have to give the faces two objects claim to one of them, which is
    the question the rest of the pipeline answers.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    assert len(loader.world.bodies_with_collision) == 1
    assert len(loader.label_segments) == 3


def test_the_scene_is_turned_into_the_world_s_coordinates(two_class_scene):
    """
    The file's coordinates are not the world's, and the loader turns them once.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    turned = loader.scene.mesh.copy()
    turned.apply_transform(WarsawWorldLoader.SOURCE_TO_WORLD.to_np())
    assert np.allclose(
        np.sort(loader.scene_mesh.extents), np.sort(turned.extents), atol=1e-6
    )


def test_the_segments_index_the_faces_of_the_loaded_mesh(two_class_scene):
    """
    Labels are face indices, so a mesh whose faces were renumbered on the way in would
    leave every one of them pointing at another face than it was written for.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    assert len(loader.scene_mesh.faces) == len(loader.scene.mesh.faces)
    for segment in loader.label_segments:
        assert segment.faces.max() < len(loader.scene_mesh.faces)


# %% looking at it


def test_every_viewpoint_is_named_and_stands_outside_the_scene(two_class_scene):
    """
    A camera inside the geometry photographs the inside of a wall.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    poses = loader.compute_camera_poses()
    assert set(poses) == {"front_left", "front_right", "back_left", "back_right"}

    low, high = loader.scene_mesh.bounds
    for pose in poses.values():
        eye = pose[:3, 3]
        assert np.any(eye < low) or np.any(eye > high)


def test_framing_a_part_of_the_scene_stands_closer_than_framing_all_of_it(
    two_class_scene,
):
    """
    A close-up is close: what the camera is framed on decides how far away it stands.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    whole = loader.compute_camera_poses()
    part = loader.compute_camera_poses(loader.points_of(loader.label_segments[:1]))
    middle = loader.scene_mesh.vertices.mean(axis=0)
    for name in whole:
        assert np.linalg.norm(part[name][:3, 3] - middle) < np.linalg.norm(
            whole[name][:3, 3] - middle
        )
