"""
Writing one room of an HM3D scene as a scene the Warsaw pipeline already reads.

What is checked is that what comes out is read back as the pipeline reads its own scans:
one object per annotation, made of that annotation's faces, standing the way the file
puts it. Nothing here goes near the pipeline's steps -- the point of converting is that
they do not have to know.
"""

from pathlib import Path

import numpy as np
import pytest

from experiments.warsaw.habitat.convert import (
    ConvertedRoom,
    label_property,
    write_room,
)
from experiments.warsaw.habitat.scene import HabitatDataset, HabitatScene
from experiments.warsaw.world_loader.scene import SourceFrame, WarsawScene

from .scenes import write_scene

# %% a scene of four objects over two rooms

SCENE_NAME = "00900-FourSquares"

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

ROWS = [
    '7,FF0000,"chair",1',
    '8,00FF00,"chair",1',
    '9,0000FF,"bath mat",1',
    '10,FFFFFF,"table",2',
]
"""
Two chairs, something whose label has a space in it, and an object of another room.
"""


@pytest.fixture()
def scene(tmp_path: Path) -> HabitatScene:
    """
    :return: That scene, read.
    """
    write_scene(
        directory=tmp_path / "release",
        scene_name=SCENE_NAME,
        instance_colors=[RED, GREEN, BLUE, WHITE],
        room_colors=[RED, RED, RED, GREEN],
        rows=ROWS,
    )
    return HabitatScene.read(
        HabitatDataset(root=tmp_path / "release").scene(SCENE_NAME)
    )


@pytest.fixture()
def written(scene: HabitatScene, tmp_path: Path) -> Path:
    """
    :return: The directory room 1 was written into.
    """
    write_room(scene=scene, room_id=1, output=tmp_path / "room")
    return tmp_path / "room"


# %% naming a label so a mesh can carry it


def test_a_label_with_a_space_becomes_one_name() -> None:
    """
    A mesh carries a label as the name of a face property, which holds no spaces.
    """
    assert label_property("bath mat") == "bath_mat"


# %% what the pipeline reads back


def test_every_object_of_the_room_becomes_a_segment(written: Path) -> None:
    """
    A room's objects are what the pipeline is given, one segment apiece.
    """
    read = WarsawScene.from_directory(written)
    assert sorted(str(one.name) for one in read.segments()) == [
        "bath_mat_9",
        "chair_7",
        "chair_8",
    ]


def test_a_segment_is_numbered_the_way_the_dataset_numbers_it(written: Path) -> None:
    """
    The instance is HM3D's own object id rather than a number counted here, so what the
    pipeline calls an object is what the dataset calls it and the two need no matching.
    """
    read = WarsawScene.from_directory(written)
    chairs = [one for one in read.segments() if one.class_name == "chair"]
    assert sorted(one.instance for one in chairs) == [7, 8]


def test_a_segment_holds_only_the_faces_of_its_object(written: Path) -> None:
    """
    Each square is two triangles, and a label covers its object's and nothing else's.
    """
    read = WarsawScene.from_directory(written)
    sizes = {str(one.name): len(one.face_indices) for one in read.segments()}
    assert sizes == {"chair_7": 2, "chair_8": 2, "bath_mat_9": 2}


def test_the_objects_of_another_room_are_left_out(written: Path) -> None:
    """
    A room is converted on its own, so nothing another room holds is in the file.
    """
    read = WarsawScene.from_directory(written)
    assert "table" not in read.class_names
    assert len(read.mesh.faces) == 6


def test_the_scene_says_it_is_already_upright(written: Path) -> None:
    """
    HM3D stands the way the world does, so the file says so rather than being written
    upside down to suit a default meant for scans.
    """
    read = WarsawScene.from_directory(written)
    assert np.allclose(read.world_T_source.to_np(), np.eye(4))
    assert np.allclose(
        read.world_T_source.to_np(), SourceFrame.UPRIGHT.world_T_source.to_np()
    )


def test_the_mesh_keeps_the_colors_the_room_was_seen_in(written: Path) -> None:
    """
    The pictures are of the room, so the file carries the textured mesh's colours and
    not the flat ones the annotation is written in.
    """
    read = WarsawScene.from_directory(written)
    assert {tuple(color[:3]) for color in read.mesh.visual.vertex_colors} == {RED}


# %% what was written, for whoever reads the run afterwards


def test_the_record_keeps_the_dataset_s_own_words(written: Path) -> None:
    """
    A label is turned into a name a mesh can carry, so the words the annotator wrote are
    kept beside it or they are gone.
    """
    record = ConvertedRoom.beside(written)
    assert {one.object_id: one.label for one in record.objects} == {
        7: "chair",
        8: "chair",
        9: "bath mat",
    }


def test_the_record_names_the_scene_and_the_room(written: Path) -> None:
    """
    A converted room says where it came from, since nothing in the mesh does.
    """
    record = ConvertedRoom.beside(written)
    assert (record.scene, record.room_id) == (SCENE_NAME, 1)


def test_the_record_names_each_object_as_the_pipeline_will(written: Path) -> None:
    """
    The name in the record is the name every step of the run writes, so a result can be
    read back against the dataset without guessing how one became the other.
    """
    record = ConvertedRoom.beside(written)
    read = WarsawScene.from_directory(written)
    assert sorted(one.segment for one in record.objects) == sorted(
        str(one.name) for one in read.segments()
    )


# %% an object the table names and the mesh does not carry


def test_an_object_the_mesh_has_no_faces_for_is_recorded_as_absent(
    tmp_path: Path,
) -> None:
    """
    HM3D's table names objects the mesh carries nothing for -- 35 of 660 on scene
    00800 -- and a room converted from it is quietly smaller than the table says.

    Which ones they were has to be written down, or a later comparison against the
    dataset counts them as objects the pipeline lost.
    """
    write_scene(
        directory=tmp_path / "release",
        scene_name=SCENE_NAME,
        instance_colors=[RED, GREEN, BLUE, WHITE],
        room_colors=[RED, RED, RED, GREEN],
        rows=ROWS + ['11,123456,"kettle",1'],
    )
    scene = HabitatScene.read(
        HabitatDataset(root=tmp_path / "release").scene(SCENE_NAME)
    )
    written = write_room(scene=scene, room_id=1, output=tmp_path / "room")
    assert written.absent == [11]
    assert 11 not in {one.object_id for one in written.objects}


# %% what the room says about how it looks


def test_a_converted_room_carries_its_source_appearance(written: Path) -> None:
    """
    The labelled mesh cannot hold a texture, so the room is written with a second file
    that can, over the same faces.
    """
    read = WarsawScene.from_directory(written)
    assert read.appearance is not None


def test_the_appearance_covers_exactly_the_faces_the_labelled_mesh_has(
    written: Path,
) -> None:
    """
    Every picture asks for faces by the number the labelled mesh gives them, so the two
    have to be the same faces in the same order. A mismatch in the count is the one way
    that can be caught without a picture.
    """
    read = WarsawScene.from_directory(written)
    assert read.appearance.face_count == len(read.mesh.faces)


def test_a_segments_faces_are_drawable_on_their_own(written: Path) -> None:
    """
    What a close-up needs: the faces of one object, taken out of the appearance mesh by
    the numbers the labelled mesh gave them.
    """
    read = WarsawScene.from_directory(written)
    chair = next(one for one in read.segments() if str(one.name) == "chair_7")
    drawn = read.appearance.faces_alone(chair.face_indices)
    assert sum(len(piece.faces) for piece in drawn.geometry.values()) == len(
        chair.face_indices
    )
