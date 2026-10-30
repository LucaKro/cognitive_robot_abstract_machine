"""
Keeping the camera pose of every picture a run shows a model.

A run's pictures are its evidence, and a picture nobody can say where it was taken from
cannot be measured against anything afterwards. The poses are worked out while rendering
and were previously dropped, so what is wanted is that they survive beside the pictures,
and that a directory filled by several renders in turn ends up holding all of them
rather than only the last.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.warsaw.pipeline.camera_poses import CAMERA_POSES_FILE, CameraPoses
from experiments.warsaw.pipeline.steps.evidence import MeasureScene
from experiments.warsaw.world_loader.loader import RenderedPictures

# %% a pose to record


def looking_from(x: float) -> np.ndarray:
    """
    A camera pose standing at a given distance along x.
    """
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


# %% what is written


def test_a_picture_can_be_told_where_it_was_taken_from(tmp_path: Path):
    """
    The whole point: a filename in the run maps to the pose that produced it.
    """
    CameraPoses.record(
        directory=tmp_path,
        frame="world",
        field_of_view=(60.0, 45.0),
        poses={"cabinet__closeup_front_left.png": looking_from(2.0)},
    )

    written = CameraPoses.beside(tmp_path)

    assert (
        written.poses["cabinet__closeup_front_left.png"] == looking_from(2.0).tolist()
    )


def test_a_pose_is_kept_with_what_is_needed_to_use_it(tmp_path: Path):
    """
    A pose alone builds no camera: the frame says what it is relative to and the field
    of view is a loader default that no other file in a run records.
    """
    CameraPoses.record(
        directory=tmp_path,
        frame="world",
        field_of_view=(60.0, 45.0),
        poses={"cabinet__closeup_front_left.png": looking_from(2.0)},
    )

    written = CameraPoses.beside(tmp_path)

    assert written.frame == "world"
    assert written.field_of_view == [60.0, 45.0]


def test_pictures_written_one_render_at_a_time_all_keep_their_pose(tmp_path: Path):
    """
    One directory is filled by a render per label, each writing as it goes, so a record
    that replaced what was there would leave every picture but the last unaccounted for.
    """
    CameraPoses.record(
        directory=tmp_path,
        frame="world",
        field_of_view=(60.0, 45.0),
        poses={"cabinet__closeup_front_left.png": looking_from(2.0)},
    )
    CameraPoses.record(
        directory=tmp_path,
        frame="world",
        field_of_view=(60.0, 45.0),
        poses={"drawer__closeup_front_left.png": looking_from(3.0)},
    )

    written = CameraPoses.beside(tmp_path)

    assert sorted(written.poses) == [
        "cabinet__closeup_front_left.png",
        "drawer__closeup_front_left.png",
    ]


def test_a_directory_no_render_has_reached_holds_no_poses(tmp_path: Path):
    """
    Reading before anything is written is the ordinary first case, not an error.
    """
    assert CameraPoses.beside(tmp_path).poses == {}


def test_the_record_is_written_where_a_reader_will_look_for_it(tmp_path: Path):
    """
    Beside the pictures it describes, under one name, so it is found without being
    searched for.
    """
    CameraPoses.record(
        directory=tmp_path,
        frame="world",
        field_of_view=(60.0, 45.0),
        poses={"cabinet__closeup_front_left.png": looking_from(2.0)},
    )

    assert json.loads((tmp_path / CAMERA_POSES_FILE).read_text())["frame"] == "world"


# %% written beside the pictures themselves


def test_a_written_picture_is_recorded_under_the_name_it_was_written_as(tmp_path: Path):
    """
    The pose is recorded against the filename rather than the viewpoint, because the
    filename is what everything downstream -- the request records, the model calls, a
    reader opening the directory -- refers to a picture by.
    """
    pictures = RenderedPictures(
        images={"closeup_front_left": b"a png"},
        camera_poses={"closeup_front_left": looking_from(2.0)},
        field_of_view=(60.0, 45.0),
        frame="world",
    )

    [written] = MeasureScene.write_images(pictures, tmp_path, "cabinet")

    assert written == "cabinet__closeup_front_left.png"
    assert (
        CameraPoses.beside(tmp_path).pose_of(written).tolist()
        == looking_from(2.0).tolist()
    )
