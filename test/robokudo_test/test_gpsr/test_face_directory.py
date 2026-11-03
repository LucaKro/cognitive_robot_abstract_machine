"""
Tests for using a local directory for GPSR face images.
"""

from __future__ import annotations

# %% imports
from unittest.mock import Mock
from pathlib import Path

import numpy as np
import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr import face_classification
from robokudo.annotators.gpsr.exceptions import (
    FaceImageDirectoryMissing,
    FaceLabelsMismatch,
)
from robokudo.annotators.gpsr.face_classification import FaceClassification
from robokudo.annotators.gpsr.parametrize_human_classifier import (
    ParametrizeHumanClassifyAnnotator,
)
from robokudo.annotators.gpsr.store_unknown_faces import StoreFaces
from robokudo.descriptors.analysis_engines.gpsr.subtree_faces import Subtree
from robokudo.types.annotation import Classification
from robokudo.types.human import FaceAnnotation
from robokudo.utils.file_loader import FileLoader
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from ._gpsr_scene import GpsrScene


# %% local face directory
def test_classifier_requires_a_face_directory() -> None:
    """
    Report a missing face image directory.
    """
    descriptor = FaceClassification.Descriptor()
    with pytest.raises(FaceImageDirectoryMissing):
        FaceClassification(descriptor=descriptor)


def test_classifier_requires_a_label_for_each_face_image(tmp_path: Path) -> None:
    """
    Report mismatching image and label counts.
    """
    descriptor = FaceClassification.Descriptor()
    descriptor.parameters.face_directory = tmp_path
    descriptor.parameters.file_names = ["face.png"]
    with pytest.raises(FaceLabelsMismatch):
        FaceClassification(descriptor=descriptor)


def test_classifier_loads_a_face_from_the_local_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Load known faces directly from the configured directory.
    """
    image_path = tmp_path / "0.png"
    image_path.touch()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    encoding = np.arange(128.0)
    image_loader = Mock(return_value=image)
    monkeypatch.setattr(
        face_classification.face_recognition, "load_image_file", image_loader
    )
    monkeypatch.setattr(
        face_classification.face_recognition,
        "face_encodings",
        Mock(return_value=[encoding]),
    )
    descriptor = FaceClassification.Descriptor()
    descriptor.parameters.face_directory = tmp_path
    descriptor.parameters.file_names = [image_path.name]
    descriptor.parameters.labels = [image_path.stem]

    annotator = FaceClassification(descriptor=descriptor)

    image_loader.assert_called_once_with(str(image_path))
    np.testing.assert_array_equal(annotator.known_face_encodings, [encoding])


def test_new_face_is_saved_to_the_local_directory(
    tmp_path: Path, scene: GpsrScene
) -> None:
    """
    Store a previously unclassified face in the configured directory.
    """
    descriptor = StoreFaces.Descriptor()
    descriptor.parameters.face_directory = tmp_path
    annotator = StoreFaces(descriptor=descriptor)
    scene.place_in_pipeline(annotator)
    face = FaceAnnotation()
    person = scene.add_human(face)
    face.roi = person.roi

    status = annotator.compute()

    assert status == Status.SUCCESS
    assert (tmp_path / "0.png").is_file()
    [classification] = scene.cas.filter_by_type(Classification, person.annotations)
    assert classification.classname == "0"


def test_classifier_configuration_uses_the_same_local_directory(
    tmp_path: Path, scene: GpsrScene
) -> None:
    """
    Refresh known face names from the configured directory.
    """
    image_path = tmp_path / "0.png"
    image_path.touch()
    descriptor = FaceClassification.Descriptor()
    descriptor.parameters.face_directory = tmp_path
    classifier = FaceClassification(descriptor=descriptor)
    configurator = ParametrizeHumanClassifyAnnotator(
        classifier, face_directory=tmp_path
    )
    scene.place_in_pipeline(configurator)

    status = configurator.update()

    assert status == Status.SUCCESS
    assert classifier.descriptor.parameters.face_directory == tmp_path
    assert classifier.descriptor.parameters.file_names == [image_path.name]
    assert classifier.descriptor.parameters.labels == [image_path.stem]


def test_face_subtree_creates_and_shares_the_local_directory(tmp_path: Path) -> None:
    """
    The face subtree uses one directory for reading and writing images.
    """
    face_directory = tmp_path / "faces"

    tree = Subtree(face_directory=face_directory).implementation()

    [classifier] = [
        child for child in tree.children if isinstance(child, FaceClassification)
    ]
    [storage] = [child for child in tree.children if isinstance(child, StoreFaces)]
    [configurator] = [
        child
        for child in tree.children
        if isinstance(child, ParametrizeHumanClassifyAnnotator)
    ]
    assert face_directory.is_dir()
    assert classifier.descriptor.parameters.face_directory == face_directory
    assert storage.descriptor.parameters.face_directory == face_directory
    assert configurator.face_directory == face_directory


def test_face_subtree_uses_local_share_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Store face images in the user's local data directory by default.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    tree = Subtree().implementation()

    [storage] = [child for child in tree.children if isinstance(child, StoreFaces)]
    assert storage.descriptor.parameters.face_directory == (
        tmp_path / ".local" / "share" / "robokudo" / "faces"
    )


def test_face_subtree_accepts_a_ros_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Keep face images in the specified ROS package when one is provided.
    """
    monkeypatch.setattr(
        FileLoader,
        "get_path_to_file_in_ros_package",
        Mock(return_value=tmp_path / "package.xml"),
    )

    tree = Subtree(ros_pkg_path="face_package").implementation()

    [storage] = [child for child in tree.children if isinstance(child, StoreFaces)]
    assert storage.descriptor.parameters.face_directory == tmp_path / "faces"
