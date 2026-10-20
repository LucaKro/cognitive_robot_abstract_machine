import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from semantic_digital_twin.adapters.exceptions import (
    BodyNotInMJCFFileError,
    UnsupportedMJCFOrientationError,
)
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.mjcf_body_pose_writer import (
    MJCFAttribute,
    MJCFBodyPoseWriter,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# %% fixtures


@pytest.fixture
def scene_file(tmp_path: Path) -> Path:
    """
    A copy of a scene with a cabinet, a drawer with its handle, and a free book in a
    frame, which the tests may overwrite.
    """
    return Path(
        shutil.copy(os.path.join(DATASET_DIR, "body_pose_writing_scene.xml"), tmp_path)
    )


@pytest.fixture
def scene_world(scene_file: Path) -> World:
    """
    The world parsed from :func:`scene_file`.
    """
    return MJCFParser(str(scene_file)).parse()


@pytest.fixture
def pose_writer(scene_file: Path, scene_world: World) -> MJCFBodyPoseWriter:
    """
    A writer of the poses of :func:`scene_world` into :func:`scene_file`.
    """
    return MJCFBodyPoseWriter(file_path=str(scene_file), world=scene_world)


def reparsed_origin(scene_file: Path, body_name: str) -> np.ndarray:
    """
    :return: the origin of the parent connection of a body, parsed anew from the file.
    """
    world = MJCFParser(str(scene_file)).parse()
    return world.get_body_by_name(body_name).parent_connection.origin.to_np()


# %% unchanged poses


def test_unchanged_poses_leave_the_file_untouched(scene_file, scene_world, pose_writer):
    content_before = scene_file.read_text()

    written_bodies = pose_writer.write(
        [body for body in scene_world.bodies if body is not scene_world.root]
    )

    assert written_bodies == []
    assert scene_file.read_text() == content_before


# %% moved bodies


def test_moved_body_in_a_frame_is_parsed_back_at_its_new_pose(
    scene_file, scene_world, pose_writer
):
    book = scene_world.get_body_by_name("book")
    book.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.4, 2.1, 0.9, 0.1, 0.0, 1.2, reference_frame=book.parent_connection.parent
    )
    expected_origin = book.parent_connection.origin.to_np()

    written_bodies = pose_writer.write([book])

    assert written_bodies == [book]
    assert np.allclose(reparsed_origin(scene_file, "book"), expected_origin, atol=1e-5)


def test_writing_a_moved_body_changes_only_its_start_tag(
    scene_file, scene_world, pose_writer
):
    lines_before = scene_file.read_text().splitlines()
    book = scene_world.get_body_by_name("book")
    book.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.4, 2.1, 0.9, reference_frame=book.parent_connection.parent
    )

    pose_writer.write([book])

    lines_after = scene_file.read_text().splitlines()
    changed_lines = [
        (before, after)
        for before, after in zip(lines_before, lines_after)
        if before != after
    ]
    assert len(lines_after) == len(lines_before)
    assert [before for before, _ in changed_lines] == [
        line for line in lines_before if '<body name="book"' in line
    ]


def test_freed_nested_body_is_written_relative_to_its_parent(
    scene_file, scene_world, pose_writer
):
    handle = scene_world.get_body_by_name("handle")
    scene_world.free_branch(handle)
    handle.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.06, 0.01, 0.04, reference_frame=handle.parent_connection.parent
    )
    expected_origin = handle.parent_connection.origin.to_np()
    drawer_slide = scene_world.get_body_by_name("drawer").parent_connection
    drawer_slide.position = 0.3

    pose_writer.write([handle])

    assert np.allclose(
        reparsed_origin(scene_file, "handle"), expected_origin, atol=1e-5
    )


def test_unrotated_body_without_quaternion_gets_none(
    scene_file, scene_world, pose_writer
):
    handle = scene_world.get_body_by_name("handle")
    scene_world.free_branch(handle)
    handle.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.06, 0.01, 0.04, reference_frame=handle.parent_connection.parent
    )

    pose_writer.write([handle])

    handle_line = next(
        line for line in scene_file.read_text().splitlines() if '"handle"' in line
    )
    assert f'{MJCFAttribute.QUATERNION}="' not in handle_line


# %% bodies the writer cannot write


def test_body_missing_from_the_file_is_refused(scene_world, pose_writer):
    with pytest.raises(BodyNotInMJCFFileError):
        pose_writer.write([scene_world.root])


def test_body_oriented_by_euler_angles_is_refused(tmp_path: Path):
    scene_file = shutil.copy(
        os.path.join(DATASET_DIR, "euler_oriented_body_scene.xml"), tmp_path
    )
    world = MJCFParser(scene_file).parse()
    pose_writer = MJCFBodyPoseWriter(file_path=scene_file, world=world)

    with pytest.raises(UnsupportedMJCFOrientationError):
        pose_writer.write([world.get_body_by_name("tilted_box")])
