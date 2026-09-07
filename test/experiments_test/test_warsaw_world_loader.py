"""
Reading a Warsaw scene: one mesh whose faces carry, per class, the instance they are.

The scenes here are a few triangles written the way the dataset writes one, so what is
checked is the reading and not the scan: which objects a file's labels describe, what a
directory that holds no scene says, and where the cameras end up standing.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh
from plyfile import PlyData, PlyElement

from experiments.warsaw.exceptions import (
    AmbiguousWarsawSceneError,
    CameraHasNoDirectionError,
    NoSegmentsGivenError,
    SceneBodyNotFoundError,
    WarsawLabelsMissingError,
    WarsawSceneNotFoundError,
)
from experiments.warsaw.world_loader.loader import (
    SHARED_FACES_LABEL,
    WarsawWorldLoader,
)
from experiments.warsaw.world_loader.viewpoints import changed_pixels
from experiments.warsaw.world_loader.scene import (
    LabelSegment,
    WarsawScene,
    segment_label,
)

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
    assert all(3 not in segment.face_indices for segment in scene.segments())


def test_a_segment_holds_only_the_faces_of_its_instance(two_class_scene):
    """
    :attr:`LabelSegment.face_indices` is what a body is later cut from, so it must hold
    the faces of that instance and no others.
    """
    scene = WarsawScene.from_directory(two_class_scene)
    segments = {str(segment.name): segment for segment in scene.segments()}
    assert segments["cabinet_1"].face_indices.tolist() == [0]
    assert segments["cabinet_2"].face_indices.tolist() == [1, 2]
    assert segments["drawer_1"].face_indices.tolist() == [1, 2]


def test_a_face_can_belong_to_objects_of_several_classes(two_class_scene):
    """
    The overlap the whole pipeline exists to resolve: a drawer front is the drawer and
    the cabinet holding it, and reading must not hide that.
    """
    scene = WarsawScene.from_directory(two_class_scene)
    segments = {str(segment.name): segment for segment in scene.segments()}
    assert set(segments["cabinet_2"].face_indices) == set(
        segments["drawer_1"].face_indices
    )


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
    write_scene(
        two_class_scene / "another.ply", box.vertices, box.faces[:1], {"wall": [1]}
    )
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


# %% a payload that is not shaped the way a scan writes one


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({}, id="nothing at all"),
        pytest.param({"other": 1}, id="no raw block"),
        pytest.param({"_ply_raw": {}}, id="no face element"),
        pytest.param({"_ply_raw": {"face": {}}}, id="no rows"),
        pytest.param({"_ply_raw": {"face": {"data": None}}}, id="empty rows"),
    ],
)
def test_a_payload_without_the_labels_is_refused_rather_than_walked(metadata, tmp_path):
    """
    The labels are read out of a path into the file format's own payload, and every step
    of that path can be absent.

    A mesh missing any of them carries no labels, which is reported rather than raised
    from halfway down the walk.
    """

    class Scanned:
        """
        Stands in for a mesh read from a file that is not a labelled scan.
        """

        def __init__(self, kept):
            self.metadata = kept

    with pytest.raises(WarsawLabelsMissingError):
        WarsawScene._read_face_labels(Scanned(metadata), tmp_path / "scene.ply")


def test_a_payload_holding_only_geometry_carries_no_labels(tmp_path):
    """
    A face always carries its vertices; that property is the geometry, not a label.
    """

    class Scanned:
        """
        Stands in for a mesh whose faces carry geometry and nothing else.
        """

        metadata = {
            "_ply_raw": {
                "face": {"data": np.zeros(3, dtype=[("vertex_indices", "i4")])}
            }
        }

    with pytest.raises(WarsawLabelsMissingError):
        WarsawScene._read_face_labels(Scanned(), tmp_path / "scene.ply")


def test_every_face_property_that_is_not_geometry_is_a_class(tmp_path):
    """
    What is left after the geometry is the classes the scan wrote onto the faces.
    """

    class Scanned:
        """
        Stands in for a mesh whose faces carry two labels beside their geometry.
        """

        metadata = {
            "_ply_raw": {
                "face": {
                    "data": np.zeros(
                        3,
                        dtype=[
                            ("vertex_indices", "i4"),
                            ("cabinet", "i4"),
                            ("drawer", "i4"),
                        ],
                    )
                }
            }
        }

    read = WarsawScene._read_face_labels(Scanned(), tmp_path / "scene.ply")
    assert list(read) == ["cabinet", "drawer"]


# %% loading it into a world


def test_the_scene_becomes_one_body(two_class_scene):
    """
    The scene stays one body until its overlapping labels have been resolved: cutting it
    earlier would have to give the faces two objects claim to one of them, which is the
    question the rest of the pipeline answers.
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
    turned.apply_transform(loader.scene.world_T_source.to_np())
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
        assert segment.face_indices.max() < len(loader.scene_mesh.faces)


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


# %% telling two renders apart


def rendered(color, size=(4, 3)) -> bytes:
    """
    :param color: What to paint every pixel.
    :param size: How large the picture is.
    :return: The picture, as PNG bytes.
    """
    kept = io.BytesIO()
    Image.new("RGB", size, color).save(kept, format="PNG")
    return kept.getvalue()


def test_two_identical_renders_differ_nowhere():
    """
    How much of an object is visible is counted as the pixels a highlight changed, so
    two pictures of the same thing must count as no change at all.
    """
    assert changed_pixels(rendered((10, 20, 30)), rendered((10, 20, 30))) == 0


def test_every_pixel_of_a_repainted_render_is_counted():
    """
    Repainting the whole picture changes every one of its pixels.
    """
    assert changed_pixels(rendered((0, 0, 0)), rendered((255, 255, 255))) == 12


def test_only_the_pixels_that_moved_are_counted():
    """
    A highlight covers part of the view, and what is counted is that part.
    """
    painted = Image.new("RGB", (4, 3), (0, 0, 0))
    painted.putpixel((0, 0), (255, 0, 0))
    painted.putpixel((1, 1), (255, 0, 0))
    kept = io.BytesIO()
    painted.save(kept, format="PNG")
    assert changed_pixels(rendered((0, 0, 0)), kept.getvalue()) == 2


def test_renders_of_different_sizes_are_not_compared():
    """
    Two pictures of different sizes are not two views of the same pose, so there is
    nothing to count.
    """
    assert changed_pixels(rendered((0, 0, 0)), rendered((0, 0, 0), size=(8, 6))) == 0


# %% a scene body the world does not carry


def test_a_scene_body_name_matching_nothing_is_reported(two_class_scene):
    """
    Falling back to whatever body came first renders a different body than was asked
    for.
    """
    loaded = WarsawWorldLoader(input_directory=two_class_scene)
    with pytest.raises(SceneBodyNotFoundError):
        WarsawWorldLoader(world=loaded.world, scene_body_name="not_a_body")


# %% a camera with nothing to look at


def test_a_camera_standing_where_it_looks_is_reported(two_class_scene):
    """
    A camera whose eye is its target has no direction, which silently becomes NaN.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    somewhere = np.array([1.0, 2.0, 3.0])
    with pytest.raises(CameraHasNoDirectionError):
        loader._looking_at(somewhere, somewhere)


def test_a_camera_looking_straight_down_is_reported(two_class_scene):
    """
    Looking along the world's own up axis leaves the sideways direction undefined.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    with pytest.raises(CameraHasNoDirectionError):
        loader._looking_at(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))


# %% asking about no segments at all


def test_rendering_no_segments_is_reported(two_class_scene):
    """
    An empty selection reaches numpy as an empty concatenate, which says nothing useful.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    with pytest.raises(NoSegmentsGivenError):
        loader.points_of([])


# %% naming what a render highlights


def a_segment(class_name: str, instance: int = 0) -> LabelSegment:
    """
    :param class_name: The class labelling the object.
    :param instance: Which object of that class it is.
    :return: A segment covering one face, named after the two.
    """
    return LabelSegment(
        class_name=class_name, instance=instance, face_indices=np.array([0])
    )


def test_a_label_short_enough_names_every_segment():
    """
    The filename says what is colored in the render, so every name is kept while it
    fits.
    """
    segments = [a_segment("cabinet"), a_segment("drawer", 1)]
    assert segment_label(segments) == "cabinet_0-drawer_1"


def test_a_label_too_long_is_cut_short_and_counts_what_it_dropped():
    """
    A filename has a length limit, and what did not fit still has to be accounted for.
    """
    segments = [a_segment("cabinet", index) for index in range(4)]
    assert (
        segment_label(segments, maximum_length=21) == "cabinet_0-cabinet_1-and_2_more"
    )


# %% coloring two segments so the faces they disagree about can be seen


def test_each_segment_keeps_the_faces_it_alone_claims(two_class_scene):
    """
    A face claimed by one segment alone is painted in that segment's own color.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    one = LabelSegment("cabinet", 0, np.array([0, 1, 2]))
    other = LabelSegment("drawer", 0, np.array([2, 3]))

    highlights, _ = loader.pair_highlights(one, other)

    (_, only_one), (_, only_other), (_, shared) = highlights
    assert sorted(only_one.tolist()) == [0, 1]
    assert sorted(only_other.tolist()) == [3]
    assert sorted(shared.tolist()) == [2]


def test_the_faces_both_claim_take_a_third_color(two_class_scene):
    """
    Painted in one of the two colors, an overlap would look like it belonged to
    whichever was painted last, and the question being asked would not be in the picture
    at all.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    one = LabelSegment("cabinet", 0, np.array([0, 1, 2]))
    other = LabelSegment("drawer", 0, np.array([2, 3]))

    highlights, legend = loader.pair_highlights(one, other)

    assert set(legend) == {"cabinet_0", "drawer_0", SHARED_FACES_LABEL}
    colors = [color for color, _ in highlights]
    assert legend["cabinet_0"] == colors[0]
    assert legend["drawer_0"] == colors[1]
    assert legend[SHARED_FACES_LABEL] == colors[2]


def test_segments_sharing_nothing_have_no_shared_color_to_explain(two_class_scene):
    """
    The legend says what is in the picture, and nothing is painted the third color.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    one = LabelSegment("cabinet", 0, np.array([0, 1]))
    other = LabelSegment("drawer", 0, np.array([2, 3]))

    _, legend = loader.pair_highlights(one, other)

    assert set(legend) == {"cabinet_0", "drawer_0"}


# %% measuring how much of a segment turns towards a direction


def test_a_segment_presents_more_area_head_on_than_edge_on(two_class_scene):
    """
    How large a face looks is its area foreshortened by how squarely it faces.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    segment = loader.label_segments[0]
    normal = loader.scene_mesh.face_normals[segment.face_indices[0]]

    head_on = loader.presented_area(segment, normal)
    edge_on = loader.presented_area(
        segment,
        np.cross(normal, [0.0, 0.0, 1.0])
        / np.linalg.norm(np.cross(normal, [0.0, 0.0, 1.0])),
    )

    assert head_on > edge_on


def test_the_area_presented_is_never_more_than_the_segment_has(two_class_scene):
    """
    Foreshortening only ever takes area away.
    """
    loader = WarsawWorldLoader(input_directory=two_class_scene)
    segment = loader.label_segments[0]
    total = float(loader.scene_mesh.area_faces[segment.face_indices].sum())

    presented = loader.presented_area(segment, np.array([0.0, 0.0, 1.0]))

    assert 0.0 <= presented <= total + 1e-9
