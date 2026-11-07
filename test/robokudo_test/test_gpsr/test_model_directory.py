"""
Tests for where the GPSR model files are loaded from.
"""

from __future__ import annotations

# %% Imports
import pytest
from robokudo.annotators.gpsr.filter_object_by_semantic_annotation import (
    FilterObjectBySemanticAnnotation,
)
from robokudo.annotators.gpsr.type_mapping_annotator import TypeMappingAnnotator
from robokudo.annotators.gpsr.vision_language_annotator import VisionLanguageAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.descriptors.analysis_engines.gpsr.subtree_door_pipeline import (
    Subtree as DoorSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_near_object import (
    Subtree as HumanNearObjectSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_posture_pipeline import (
    Subtree as HumanPostureSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_object_detection import (
    Subtree as ObjectDetectionSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_waving_raising_pipeline import (
    Subtree as WavingRaisingSubtree,
)
from robokudo.utils.file_loader import FileLoader
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from rclpy.node import Node


# %% fixtures
@pytest.fixture(autouse=True)
def registered_node(node: Node) -> None:
    """
    Provide the shared ROS node required to build detector subtrees.

    :param node: The shared ROS node.
    """


@pytest.fixture
def model_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """
    Make every file lookup end in a directory named after the requested ROS package.

    :param monkeypatch: Redirects the package file lookup.
    :param tmp_path: Directory of the stand-in packages.
    :return: The directory of the ROS package that holds the GPSR model files.
    """
    monkeypatch.setattr(
        FileLoader,
        "get_path_to_file_in_ros_package",
        lambda ros_pkg_name, relative_path: tmp_path / ros_pkg_name / relative_path,
    )
    return tmp_path / VisionLanguageAnnotator.Descriptor.Parameters().ros_package_name


# %% files of the model package
@pytest.mark.parametrize(
    ("subtree", "model_name"),
    [
        (DoorSubtree, "door.pt"),
        (ObjectDetectionSubtree, "ultimate_model_v11_1.26.4.pt"),
    ],
    ids=["door detector", "object detector"],
)
def test_detector_subtree_loads_its_files_from_the_model_package(
    model_package: Path, subtree: type, model_name: str
) -> None:
    """
    The trained detectors read weights and class names from the model package.
    """
    tree = subtree().implementation()

    [detector] = [child for child in tree.children if isinstance(child, YoloAnnotator)]
    parameters = detector.descriptor.parameters
    assert detector.model_location == str(model_package / "weights" / model_name)
    assert parameters.file_in_ros_package(parameters.class_names_path) == (
        model_package / parameters.class_names_path
    )


def test_furniture_detector_reads_its_class_names_from_the_model_package(
    model_package: Path,
) -> None:
    """
    The furniture detector reads its class names from the model package.
    """
    tree = HumanNearObjectSubtree().implementation()

    [detector] = [child for child in tree.children if isinstance(child, YoloAnnotator)]
    parameters = detector.descriptor.parameters
    assert parameters.file_in_ros_package(parameters.class_names_path) == (
        model_package / parameters.class_names_path
    )


@pytest.mark.parametrize(
    "annotator_class",
    [TypeMappingAnnotator, FilterObjectBySemanticAnnotation],
    ids=["type mapping", "semantic filter"],
)
def test_semantic_mapping_is_read_from_the_model_package(
    semantic_type_mapping: dict[str, str], annotator_class: type
) -> None:
    """
    Both semantic annotators read the mapping the package holds.
    """
    assert annotator_class()._type_mapping == semantic_type_mapping


# %% models of the core
@pytest.mark.parametrize(
    "subtree",
    [HumanNearObjectSubtree, HumanPostureSubtree, WavingRaisingSubtree],
    ids=["person near furniture", "posture", "waving"],
)
def test_pose_subtrees_keep_the_core_model_directory(subtree: type) -> None:
    """
    Use the pose annotator's own model directory in GPSR.
    """
    default_directory = HumanAndPoseAnnotator.Descriptor.Parameters().models_directory

    tree = subtree().implementation()

    [detector] = [
        child for child in tree.children if isinstance(child, HumanAndPoseAnnotator)
    ]
    assert detector.descriptor.parameters.models_directory == default_directory


def test_furniture_detector_uses_the_core_pretrained_model() -> None:
    """
    Load furniture detection weights from the core model directory.
    """
    default_directory = YoloAnnotator.Descriptor.Parameters().models_directory

    tree = HumanNearObjectSubtree().implementation()

    [detector] = [child for child in tree.children if isinstance(child, YoloAnnotator)]
    assert detector.descriptor.parameters.pretrained_model == "yolov8n.pt"
    assert detector.model_location == str(default_directory / "yolov8n.pt")
