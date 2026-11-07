"""
Tests for detecting faces in the color image and turning each face into a person.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr import face_detector
from robokudo.annotators.gpsr.exceptions import (
    FaceImageScalingFailed,
    UnequalColorDepthRatios,
)
from robokudo.annotators.gpsr.face_detector import FaceDetector
from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import Encoding
from robokudo.types.cv import ColorToDepthRatio
from robokudo.types.human import FaceAnnotation
from robokudo.types.scene import HumanHypothesis

from ._face_recognition_mock import FaceLocation, PlacedFaces
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from ._gpsr_scene import GpsrScene

# %% scene and annotator


@pytest.fixture
def placed_faces(monkeypatch: pytest.MonkeyPatch) -> PlacedFaces:
    """
    Make the face detector find the faces the test places.

    :param monkeypatch: Replaces the face recognition.
    :return: The placed faces.
    """
    faces = PlacedFaces(faces=[FaceLocation(top=50, right=150, bottom=120, left=100)])
    monkeypatch.setattr(face_detector, "face_recognition", faces)
    return faces


def face_finder(with_point_generation: bool) -> FaceDetector:
    """
    :param with_point_generation: Whether the detector measures the points of a face in
        the depth image.
    :return: A face detector with its own default parameters.
    """
    detector = FaceDetector(descriptor=FaceDetector.Descriptor())
    detector.descriptor.parameters.with_point_generation = with_point_generation
    return detector


def place_depth(scene: GpsrScene, ratio: float, distance_in_millimeters: int) -> None:
    """
    Give the scene a depth image of the given size relative to the color image, at the
    same distance everywhere.

    :param scene: The scene that receives the depth image.
    :param ratio: The size of the depth image relative to the color image.
    :param distance_in_millimeters: The depth value of every pixel.
    """
    height, width = scene.color_image().shape[:2]
    scene.cas.set(
        CASViews.DEPTH_IMAGE,
        np.full(
            (int(height * ratio), int(width * ratio)),
            distance_in_millimeters,
            dtype=np.uint16,
        ),
    )
    scene.cas.color_to_depth_ratio = ColorToDepthRatio(horizontal=ratio, vertical=ratio)


def detected_human_hypotheses(scene: GpsrScene) -> list[HumanHypothesis]:
    """
    :param scene: The scene after the update.
    :return: The persons the face detector added to the CAS.
    """
    return scene.cas.filter_annotations_by_type(HumanHypothesis)


# %% faces in the color image


def test_face_becomes_a_person_covering_the_face(
    scene: GpsrScene, placed_faces: PlacedFaces
) -> None:
    """
    A found face becomes a person whose region covers the face.
    """
    detector = face_finder(with_point_generation=False)
    scene.place_in_pipeline(detector)

    status = detector.compute()

    assert status == Status.SUCCESS
    [human_hypothesis] = detected_human_hypotheses(scene)
    [face] = placed_faces.faces
    region = human_hypothesis.roi.roi
    assert (region.pos.x, region.pos.y, region.width, region.height) == (
        face.left,
        face.top,
        face.right - face.left,
        face.bottom - face.top,
    )
    [face_annotation] = CAS.filter_by_type(FaceAnnotation, human_hypothesis.annotations)
    assert face_annotation.roi == human_hypothesis.roi


def test_face_gets_its_encoding(scene: GpsrScene, placed_faces: PlacedFaces) -> None:
    """
    A found face carries the encoding of the face recognition library.
    """
    detector = face_finder(with_point_generation=False)
    scene.place_in_pipeline(detector)

    detector.compute()

    [human_hypothesis] = detected_human_hypotheses(scene)
    [encoding] = CAS.filter_by_type(Encoding, human_hypothesis.annotations)
    [face_encoding] = encoding.encoding
    np.testing.assert_array_equal(face_encoding, placed_faces.encoding)


def test_without_faces_nobody_is_added(
    scene: GpsrScene, placed_faces: PlacedFaces
) -> None:
    """
    An image without faces adds no person and still succeeds.
    """
    placed_faces.faces = []
    detector = face_finder(with_point_generation=False)
    scene.place_in_pipeline(detector)

    status = detector.compute()

    assert status == Status.SUCCESS
    assert detected_human_hypotheses(scene) == []


# %% points of a face


def test_face_detection_reports_image_scaling_failure(
    scene: GpsrScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Report an image scaling failure with the face-specific exception.
    """
    detector = face_finder(with_point_generation=True)
    scene.place_in_pipeline(detector)
    place_depth(scene, ratio=1.0, distance_in_millimeters=2000)

    def fail_scaling(cas: CAS, image: np.ndarray) -> np.ndarray:
        """
        Stand in for an image scaling that always fails.

        :param cas: The CAS of the annotator.
        :param image: The image to scale.
        :raises RuntimeError: Always.
        """
        raise RuntimeError("scaling failed")

    monkeypatch.setattr(
        face_detector.robokudo.utils.cv_helper,
        "get_scaled_color_image_for_depth_image",
        fail_scaling,
    )
    with pytest.raises(FaceImageScalingFailed):
        FaceDetector.compute.__wrapped__(detector)


def test_face_detection_rejects_unequal_color_depth_ratios(
    scene: GpsrScene, placed_faces: PlacedFaces
) -> None:
    """
    Reject different horizontal and vertical color-to-depth ratios.
    """
    detector = face_finder(with_point_generation=True)
    scene.place_in_pipeline(detector)
    place_depth(scene, ratio=1.0, distance_in_millimeters=2000)
    scene.cas.color_to_depth_ratio = ColorToDepthRatio(horizontal=1.0, vertical=0.5)
    with pytest.raises(UnequalColorDepthRatios):
        FaceDetector.compute.__wrapped__(detector)


def test_face_gets_its_points_from_the_depth_image(
    scene: GpsrScene, placed_faces: PlacedFaces
) -> None:
    """
    The points of a person come from the depth image at the distance of the face.
    """
    detector = face_finder(with_point_generation=True)
    scene.place_in_pipeline(detector)
    place_depth(scene, ratio=1.0, distance_in_millimeters=2000)

    detector.compute()

    [human_hypothesis] = detected_human_hypotheses(scene)
    depths = np.asarray(human_hypothesis.points.points)[:, 2]
    assert len(depths) > 0
    np.testing.assert_allclose(depths, 2.0)


def test_face_found_in_a_smaller_depth_image_is_placed_in_the_color_image(
    scene: GpsrScene, placed_faces: PlacedFaces
) -> None:
    """
    A face searched in the scaled image is reported in color image coordinates.
    """
    detector = face_finder(with_point_generation=True)
    scene.place_in_pipeline(detector)
    place_depth(scene, ratio=0.5, distance_in_millimeters=2000)

    detector.compute()

    height, width = scene.color_image().shape[:2]
    assert placed_faces.searched_image_sizes == [(width // 2, height // 2)]
    [human_hypothesis] = detected_human_hypotheses(scene)
    [face] = placed_faces.faces
    region = human_hypothesis.roi.roi
    assert (region.pos.x, region.pos.y, region.width, region.height) == (
        face.left * 2,
        face.top * 2,
        (face.right - face.left) * 2,
        (face.bottom - face.top) * 2,
    )
