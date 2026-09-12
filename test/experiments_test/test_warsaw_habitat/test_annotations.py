"""
Reading the table an HM3D scene writes its annotations in.

The table here is a handful of rows written the way HM3D writes them, so what is checked
is the reading: which objects a row describes, which rooms the table names, and what a
file that is not one of these tables says.
"""

from pathlib import Path

import pytest

from experiments.warsaw.exceptions import HabitatAnnotationsUnreadableError
from experiments.warsaw.habitat.annotations import (
    UNASSIGNED_ROOM,
    AnnotatedObject,
    SceneAnnotations,
)

# %% the table to read


@pytest.fixture()
def table() -> Path:
    """
    :return: A small annotation table written the way HM3D writes one.
    """
    return (
        Path(__file__).parent.parent
        / "dataset"
        / "warsaw_habitat"
        / ("two_rooms.semantic.txt")
    )


@pytest.fixture()
def annotations(table: Path) -> SceneAnnotations:
    """
    :return: That table, read.
    """
    return SceneAnnotations.from_file(table)


# %% what a row says


def test_a_row_describes_one_object(annotations: SceneAnnotations) -> None:
    """
    A row names an object, the colour its faces carry, what it was called and where.
    """
    assert annotations.objects[0] == AnnotatedObject(
        object_id=1, color=(255, 0, 0), label="floor", room_id=1
    )


def test_a_label_may_hold_spaces(annotations: SceneAnnotations) -> None:
    """
    HM3D's labels are written as people say them, spaces and all.
    """
    assert annotations.objects[4].label == "bath mat"


def test_the_colors_index_the_objects(annotations: SceneAnnotations) -> None:
    """
    The colour is how the mesh says which object a face belongs to, so it is the key.
    """
    assert annotations.by_color()[(0, 0, 255)].object_id == 3


# %% what the table holds


def test_the_rooms_are_named_in_order(annotations: SceneAnnotations) -> None:
    """
    Every room the table assigns an object to, the leftover bucket included.
    """
    assert annotations.room_ids == [UNASSIGNED_ROOM, 1, 2]


def test_a_room_holds_the_objects_assigned_to_it(
    annotations: SceneAnnotations,
) -> None:
    """
    A room is the objects whose rows name it, and no others.
    """
    assert [one.object_id for one in annotations.in_room(1)] == [1, 2, 3]


def test_a_rooms_labels_are_each_named_once(annotations: SceneAnnotations) -> None:
    """
    Two chairs are one label, which is what the vocabulary is asked about.
    """
    assert annotations.labels_in_room(1) == ["chair", "floor"]


# %% a file that is not one of these tables


def test_a_file_without_the_header_is_refused(tmp_path: Path) -> None:
    """
    The first line says what the file is, and a file that does not say it is not one.
    """
    written = tmp_path / "wrong.semantic.txt"
    written.write_text('1,FF0000,"floor",1\n')
    with pytest.raises(HabitatAnnotationsUnreadableError):
        SceneAnnotations.from_file(written)


def test_a_row_missing_a_field_is_refused(tmp_path: Path) -> None:
    """
    A row with fewer fields than the schema has is not an object with a field missing;
    it is a file that cannot be read.
    """
    written = tmp_path / "short.semantic.txt"
    written.write_text('HM3D Semantic Annotations\n1,FF0000,"floor"\n')
    with pytest.raises(HabitatAnnotationsUnreadableError):
        SceneAnnotations.from_file(written)
