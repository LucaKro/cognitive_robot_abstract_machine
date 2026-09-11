"""Alignment of a reconstruction to ground truth from corresponding landmarks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from experiments.warsaw.evaluation.alignment import (
    DegenerateLandmarksError,
    InsufficientLandmarksError,
    Landmark,
    LandmarkAlignment,
    LandmarkFile,
    main,
)


# %% a landmark file to read back


def landmark_payload(set_aside_last: str | None = None) -> dict:
    """
    Four landmarks on a cube corner, optionally with the last one set aside.
    """
    places = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 2.0]]
    landmarks = [
        {"name": f"point_{index}", "reconstruction": place, "ground_truth": place}
        for index, place in enumerate(places)
    ]
    if set_aside_last is not None:
        landmarks[-1]["set_aside"] = set_aside_last
    return {
        "schema_version": 1,
        "reconstruction_frame": "scan",
        "ground_truth_frame": "model",
        "estimate_scale": False,
        "landmarks": landmarks,
    }


# %% recovering a coordinate-frame transform


def test_landmarks_recover_a_large_rigid_transform() -> None:
    """Rotation and translation size do not depend on an ICP initial estimate."""
    reconstruction_points = np.asarray(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 2.0]]
    )
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([100.0, -40.0, 7.0])
    ground_truth_points = reconstruction_points @ rotation.T + translation
    landmarks = [
        Landmark(
            name=f"point_{index}",
            reconstruction=tuple(reconstruction),
            ground_truth=tuple(ground_truth),
        )
        for index, (reconstruction, ground_truth) in enumerate(
            zip(reconstruction_points, ground_truth_points)
        )
    ]

    alignment = LandmarkAlignment.fit(landmarks)

    np.testing.assert_allclose(alignment.rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(alignment.translation, translation, atol=1e-12)
    assert alignment.scale == pytest.approx(1.0)
    assert alignment.root_mean_square_error == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        alignment.transform_points(reconstruction_points), ground_truth_points
    )


def test_landmarks_can_recover_a_uniform_scale_difference() -> None:
    """A reconstruction written in another uniform unit can still be aligned."""
    reconstruction_points = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    )
    scale = 0.01
    translation = np.asarray([4.0, 5.0, 6.0])
    ground_truth_points = scale * reconstruction_points + translation
    landmarks = [
        Landmark(
            name=f"point_{index}",
            reconstruction=tuple(reconstruction),
            ground_truth=tuple(ground_truth),
        )
        for index, (reconstruction, ground_truth) in enumerate(
            zip(reconstruction_points, ground_truth_points)
        )
    ]

    alignment = LandmarkAlignment.fit(landmarks, estimate_scale=True)

    assert alignment.scale == pytest.approx(scale)
    np.testing.assert_allclose(
        alignment.transform_points(reconstruction_points), ground_truth_points
    )


# %% rejecting landmarks that cannot determine a transform


def test_fewer_than_three_landmarks_are_rejected() -> None:
    """Two points leave rotation about their connecting line undetermined."""
    landmarks = [
        Landmark(name="one", reconstruction=(0, 0, 0), ground_truth=(0, 0, 0)),
        Landmark(name="two", reconstruction=(1, 0, 0), ground_truth=(1, 0, 0)),
    ]

    with pytest.raises(InsufficientLandmarksError):
        LandmarkAlignment.fit(landmarks)


def test_collinear_landmarks_are_rejected() -> None:
    """Points on one line cannot determine rotation around that line."""
    landmarks = [
        Landmark(name="one", reconstruction=(0, 0, 0), ground_truth=(1, 0, 0)),
        Landmark(name="two", reconstruction=(1, 0, 0), ground_truth=(2, 0, 0)),
        Landmark(name="three", reconstruction=(2, 0, 0), ground_truth=(3, 0, 0)),
    ]

    with pytest.raises(DegenerateLandmarksError):
        LandmarkAlignment.fit(landmarks)


# %% exchanging landmarks without camera poses


def test_landmark_file_fits_and_writes_an_auditable_transform(tmp_path) -> None:
    """A small hand-picked JSON file is enough to establish the two frames."""
    input_path = tmp_path / "landmarks.json"
    output_path = tmp_path / "alignment.json"
    input_path.write_text(
        """{
  "schema_version": 1,
  "reconstruction_frame": "pipeline_mesh",
  "ground_truth_frame": "iai_apartment",
  "estimate_scale": false,
  "landmarks": [
    {"name": "corner", "reconstruction": [0, 0, 0], "ground_truth": [10, 20, 3]},
    {"name": "counter_x", "reconstruction": [1, 0, 0], "ground_truth": [10, 21, 3]},
    {"name": "counter_y", "reconstruction": [0, 1, 0], "ground_truth": [9, 20, 3]},
    {"name": "shelf", "reconstruction": [0, 0, 1], "ground_truth": [10, 20, 4]}
  ]
}"""
    )

    assert main([str(input_path), "--output", str(output_path)]) == 0

    written = json.loads(output_path.read_text())
    assert written["schema_version"] == 1
    assert written["source_frame"] == "pipeline_mesh"
    assert written["target_frame"] == "iai_apartment"
    assert written["landmark_count"] == 4
    assert written["root_mean_square_error"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        written["matrix"],
        [[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 3], [0, 0, 0, 1]],
        atol=1e-12,
    )


def test_landmark_file_rejects_an_unknown_schema_version(tmp_path) -> None:
    """Versioning prevents silently misreading a future landmark format."""
    path = tmp_path / "landmarks.json"
    path.write_text('{"schema_version": 2, "landmarks": []}')

    with pytest.raises(ValueError, match="schema version"):
        LandmarkFile.read(path)


# %% a landmark that was picked but is not used


def test_a_landmark_set_aside_is_not_fitted(tmp_path: Path):
    """
    A point can be picked badly where the scan is noisy or incomplete, and re-picking is
    not always possible. Setting it aside keeps the record of what was picked while
    leaving it out of the fit.
    """
    written = tmp_path / "landmarks.json"
    written.write_text(
        json.dumps(
            landmark_payload(
                set_aside_last="picked on a floor " "corner the scan barely covers"
            )
        )
    )

    read = LandmarkFile.read(written)

    assert len(read.landmarks) == 4
    assert len(read.fitted_landmarks) == 3
    assert len(read.fit().residuals) == 3


def test_a_landmark_that_is_used_says_nothing_about_being_set_aside(tmp_path: Path):
    """
    The reason is what records the decision, so its absence is what says a point counts.
    """
    written = tmp_path / "landmarks.json"
    written.write_text(json.dumps(landmark_payload()))

    read = LandmarkFile.read(written)

    assert [one.set_aside for one in read.landmarks] == [None] * 4
    assert read.fitted_landmarks == read.landmarks


def test_setting_aside_so_many_that_none_are_left_is_refused(tmp_path: Path):
    """
    Three points are the fewest a rigid fit can be made from, and silently fitting fewer
    would return a transform nothing could be trusted from.
    """
    payload = landmark_payload()
    for one in payload["landmarks"][1:]:
        one["set_aside"] = "a reason"
    written = tmp_path / "landmarks.json"
    written.write_text(json.dumps(payload))

    with pytest.raises(InsufficientLandmarksError):
        LandmarkFile.read(written).fit()
