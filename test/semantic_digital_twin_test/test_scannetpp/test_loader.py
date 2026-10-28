from pathlib import Path

import numpy as np
import pytest

from semantic_digital_twin.adapters.scannetpp_dataset.exceptions import (
    ScanNetPPSceneNotFoundError,
)
from semantic_digital_twin.adapters.scannetpp_dataset.loader import (
    ScanNetPPDatasetLoader,
)
from semantic_digital_twin.adapters.scannetpp_dataset.schema import ScanNetPPScene
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.world_description.world_modification import (
    AttributeUpdateModification,
)

# %% fixtures


@pytest.fixture
def table_and_cup() -> ScanNetPPScene:
    """
    A two-object scene whose cup starts at a vertex that duplicates a table vertex, and
    whose last face spans both objects.
    """
    loader = ScanNetPPDatasetLoader(directory=Path(__file__).parent / "resources")
    return loader.load("table_and_cup")


@pytest.fixture(scope="module")
def kitchen() -> ScanNetPPScene:
    """
    A real ScanNet++ kitchen, read from a local copy of the dataset.
    """
    directory = Path(
        "/media/itsme/93a4dee6-6389-4190-bde7-497ed57c6fc2/home/me/datasets/scannet++/data/data"
    )
    if not directory.is_dir():
        pytest.skip("ScanNet++ is not downloaded on this machine")
    return ScanNetPPDatasetLoader(directory=directory).load("f9f95681fd")


# %% mesh


def test_load_keeps_every_vertex_the_segments_refer_to(table_and_cup):
    assert len(table_and_cup.mesh.vertices) == len(table_and_cup.vertex_segment_ids)


# %% annotated objects


def test_load_reads_every_annotated_object(table_and_cup):
    assert [
        (scannetpp_object.object_id, scannetpp_object.label)
        for scannetpp_object in table_and_cup.objects
    ] == [(1, "table"), (2, "cup")]


def test_oriented_bounding_box_has_one_axis_per_row(table_and_cup):
    cup = table_and_cup.objects[1]

    np.testing.assert_array_equal(
        cup.oriented_bounding_box.normalized_axes,
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    )


def test_vertex_indices_are_the_vertices_of_the_object_segments(table_and_cup):
    cup = table_and_cup.objects[1]

    np.testing.assert_array_equal(table_and_cup.vertex_indices(cup), [4, 5, 6, 7])


# %% world


def test_create_world_has_a_body_per_object_besides_the_scan(table_and_cup):
    world = table_and_cup.create_world()

    assert sorted(str(body.name) for body in world.bodies) == sorted(
        ["scan", "table_1", "cup_2"]
    )


def test_object_body_shows_the_faces_whose_vertices_all_belong_to_the_object(
    table_and_cup,
):
    world = table_and_cup.create_world()
    [cup_surface] = world.get_body_by_name("cup_2").visual.shapes

    np.testing.assert_array_equal(
        np.unique(cup_surface.mesh.vertices, axis=0),
        np.unique(table_and_cup.mesh.vertices[[4, 5, 6, 7]], axis=0),
    )


def test_face_spanning_two_objects_stays_with_the_scan(table_and_cup):
    world = table_and_cup.create_world()
    [scan_surface] = world.root.visual.shapes

    np.testing.assert_array_equal(
        np.unique(scan_surface.mesh.vertices, axis=0),
        np.unique(table_and_cup.mesh.vertices[[2, 3, 5]], axis=0),
    )


def test_create_world_describes_each_object_body_by_its_label(table_and_cup):
    world = table_and_cup.create_world()

    assert {
        (annotation.root.name.name, annotation.description)
        for annotation in world.get_semantic_annotations_by_type(
            NaturalLanguageDescription
        )
    } == {("table_1", "table"), ("cup_2", "cup")}


def test_scan_world_shows_the_whole_scan(table_and_cup):
    world = table_and_cup.create_scan_world()
    [surface] = world.root.visual.shapes

    np.testing.assert_array_equal(surface.mesh.faces, table_and_cup.mesh.faces)


def test_scan_world_adds_the_scan_with_its_shape_instead_of_changing_it(table_and_cup):
    """
    A change of a body's shapes is recorded as the whole body serialized before and after
    it, which for a scan costs several times the scan's memory.
    """
    world = table_and_cup.create_scan_world()

    assert not any(
        isinstance(modification, AttributeUpdateModification)
        for block in world.get_world_model_manager().model_modification_blocks
        for modification in block.modifications
    )


def test_create_world_records_no_change_of_a_body_s_shapes(table_and_cup):
    """
    The scan is cut before it is in the world: a change of a body's shapes is recorded
    as the whole body serialized before and after it.
    """
    world = table_and_cup.create_world()

    assert not any(
        isinstance(modification, AttributeUpdateModification)
        for block in world.get_world_model_manager().model_modification_blocks
        for modification in block.modifications
    )


# %% missing scene


def test_load_of_a_missing_scene_raises(tmp_path):
    with pytest.raises(ScanNetPPSceneNotFoundError):
        ScanNetPPDatasetLoader(directory=tmp_path).load("missing_scene")


# %% real scene


def test_every_kitchen_object_spans_its_bounding_box(kitchen):
    for scannetpp_object in kitchen.objects:
        box = scannetpp_object.oriented_bounding_box
        vertices = kitchen.mesh.vertices[kitchen.vertex_indices(scannetpp_object)]

        extents_along_axes = np.ptp(
            (vertices - box.centroid) @ box.normalized_axes.T, axis=0
        )

        # The mesh stores its vertices as float32.
        np.testing.assert_allclose(
            extents_along_axes,
            box.axes_lengths,
            atol=np.finfo(np.float32).resolution,
        )
