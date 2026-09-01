from pathlib import Path

import numpy as np
import pytest

from experiments.warsaw.exceptions import (
    WarsawSceneNotFoundError,
    WarsawSegmentationMissingError,
)
from experiments.warsaw.world_loader import (
    SceneMeshField,
    WarsawSceneFile,
    WarsawWorldLoader,
)

# %% a scene directory written the way the dataset writes one


def write_scene(
    directory: Path,
    class_names: list[str],
    face_instances: dict[str, list[int]],
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Path:
    """
    Write a scene directory holding the arrays a Warsaw scene is made of.

    :param directory: Where to write the scene.
    :param class_names: The classes the scene distinguishes.
    :param face_instances: Per class, the instance each face belongs to.
    :param vertices: The scene's vertices.
    :param faces: The scene's faces.
    :return: The directory written to.
    """
    np.savez(
        directory / WarsawSceneFile.SCENE_MESH.value,
        **{
            SceneMeshField.VERTICES.value: vertices.astype(np.float32),
            SceneMeshField.FACES.value: faces.astype(np.int64),
            SceneMeshField.VERTEX_COLORS.value: np.tile(
                np.array([0.5, 0.5, 0.5], dtype=np.float32), (len(vertices), 1)
            ),
            SceneMeshField.CLASSES.value: np.array(class_names),
        },
    )
    for class_name, instances in face_instances.items():
        np.savez(
            directory / WarsawSceneFile.segmentation_of(class_name),
            **{
                SceneMeshField.FACE_INSTANCES.value: np.array(instances, dtype=np.int64)
            },
        )
    return directory


def stacked_triangles(heights: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one triangle per height, each lying flat at that height along the source
    frame's vertical axis.

    :param heights: The vertical position of each triangle.
    :return: The vertices and faces of all triangles.
    """
    vertices = np.concatenate(
        [
            np.array([[0.0, height, 0.0], [1.0, height, 0.0], [0.0, height, 1.0]])
            for height in heights
        ]
    )
    faces = np.arange(3 * len(heights)).reshape(-1, 3)
    return vertices, faces


@pytest.fixture
def two_class_scene(tmp_path) -> Path:
    """
    A scene of three triangles: two mugs and one table.
    """
    vertices, faces = stacked_triangles([0.0, 1.0, 2.0])
    return write_scene(
        tmp_path,
        class_names=["mug", "table"],
        face_instances={"mug": [1, 2, 0], "table": [0, 0, 1]},
        vertices=vertices,
        faces=faces,
    )


# %% reading a scene


def test_every_segmented_instance_becomes_a_body(two_class_scene):
    """
    Each instance a class segments the mesh into is loaded as its own body.
    """
    loader = WarsawWorldLoader(two_class_scene)

    names = sorted(str(body.name) for body in loader.world.bodies_with_collision)

    assert names == ["mug_1", "mug_2", "table_1"]


def test_the_unsegmented_remainder_becomes_no_body(two_class_scene):
    """
    Instance ``0`` marks the faces a class does not cover, so it is not an object.

    It holds nearly the whole mesh, so loading it would bury the scene under one body
    per class.
    """
    loader = WarsawWorldLoader(two_class_scene)

    assert not [
        body for body in loader.world.bodies_with_collision if "_0" in str(body.name)
    ]


def test_a_body_holds_only_the_faces_of_its_instance(two_class_scene):
    """
    A body's geometry is the faces segmented into it, and no others.
    """
    loader = WarsawWorldLoader(two_class_scene)

    table = next(
        body
        for body in loader.world.bodies_with_collision
        if str(body.name) == "table_1"
    )

    assert len(table.collision[0].mesh.faces) == 1


def test_a_class_named_with_spaces_is_read_from_its_underscored_file(tmp_path):
    """
    The scene names a class ``kitchen island`` while its file is ``kitchen_island.npz``.
    """
    vertices, faces = stacked_triangles([0.0])
    scene = write_scene(
        tmp_path,
        class_names=["kitchen island"],
        face_instances={"kitchen island": [1]},
        vertices=vertices,
        faces=faces,
    )

    loader = WarsawWorldLoader(scene)

    assert [str(body.name) for body in loader.world.bodies_with_collision] == [
        "kitchen island_1"
    ]


# %% orienting the scene


def test_the_floor_is_loaded_below_the_ceiling(tmp_path):
    """
    The source frame's vertical axis points down, so the loaded world has to turn it
    over: a floor written above a ceiling has to end up below it.
    """
    # In the source frame the floor sits at a greater height than the ceiling.
    vertices, faces = stacked_triangles([3.0, -3.0])
    scene = write_scene(
        tmp_path,
        class_names=["floor", "ceiling"],
        face_instances={"floor": [1, 0], "ceiling": [0, 1]},
        vertices=vertices,
        faces=faces,
    )

    loader = WarsawWorldLoader(scene)
    heights = {
        str(body.name): body.collision[0]
        .mesh_in_frame(loader.world.root)
        .vertices[:, 2]
        .mean()
        for body in loader.world.bodies_with_collision
    }

    assert heights["floor_1"] < heights["ceiling_1"]


# %% a directory that holds no scene


def test_a_directory_without_a_scene_mesh_is_reported(tmp_path):
    """
    A directory holding no scene mesh names the file it is missing.
    """
    with pytest.raises(WarsawSceneNotFoundError) as raised:
        WarsawWorldLoader(tmp_path)

    assert WarsawSceneFile.SCENE_MESH.value in str(raised.value)


def test_a_class_without_a_segmentation_is_reported(tmp_path):
    """
    A class the scene declares but does not segment names the file it is missing.
    """
    vertices, faces = stacked_triangles([0.0])
    write_scene(
        tmp_path,
        class_names=["mug", "table"],
        face_instances={"mug": [1]},
        vertices=vertices,
        faces=faces,
    )

    with pytest.raises(WarsawSegmentationMissingError) as raised:
        WarsawWorldLoader(tmp_path)

    assert WarsawSceneFile.segmentation_of("table") in str(raised.value)


# %% looking at the scene


def test_the_cameras_stand_outside_the_scene(tmp_path):
    """
    Every camera the loader renders from stands clear of the scene's geometry.

    A room is rendered from outside itself. A camera placed within the walls faces their
    inside and renders black, so a scene larger than the camera's distance would render
    nothing at all.
    """
    # A room far wider than any fixed camera distance would allow for.
    vertices = np.array(
        [[-20.0, 0.0, -20.0], [20.0, 0.0, -20.0], [-20.0, 0.0, 20.0], [0.0, -8.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    scene = write_scene(
        tmp_path,
        class_names=["floor", "ceiling"],
        face_instances={"floor": [1, 0], "ceiling": [0, 1]},
        vertices=vertices,
        faces=faces,
    )

    loader = WarsawWorldLoader(scene)
    corners = np.array(
        [
            bound
            for body in loader.world.bodies_with_collision
            for shape in body.collision
            for bound in shape.mesh_in_frame(loader.world.root).bounds
        ]
    )
    low, high = corners.min(axis=0), corners.max(axis=0)

    for transform in loader._predefined_camera_transforms:
        camera_position = np.asarray(transform)[:3, 3]
        assert np.any(camera_position < low) or np.any(camera_position > high)


def test_every_viewpoint_is_named(tmp_path):
    """
    The scene is framed from one named viewpoint per corner, the names a caller renders
    by.
    """
    vertices, faces = stacked_triangles([0.0, 1.0])
    scene = write_scene(
        tmp_path,
        class_names=["floor"],
        face_instances={"floor": [1, 1]},
        vertices=vertices,
        faces=faces,
    )

    poses = WarsawWorldLoader(scene).compute_camera_poses()

    assert set(poses) == set(WarsawWorldLoader.VIEWPOINT_AZIMUTHS)
    assert all(pose.shape == (4, 4) for pose in poses.values())
