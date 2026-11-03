"""
Tests for configuring GPSR model files independently of a ROS package.
"""

from __future__ import annotations

# %% imports
from pathlib import Path
import json
import shutil
from unittest.mock import Mock
from types import ModuleType

import pytest
from robokudo.analysis_engine import SubtreeInterface
from robokudo.descriptors.analysis_engines.gpsr import gpsr_implementation
from robokudo.descriptors.analysis_engines.gpsr import (
    subtree_obj_in_room,
    subtree_object_detection_pipeline,
)
from robokudo.annotators.gpsr.vision_language_annotator import VisionLanguageAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.gpsr.filter_object_by_semantic_annotation import (
    FilterObjectBySemanticAnnotation,
)
from robokudo.annotators.gpsr.type_mapping_annotator import TypeMappingAnnotator
from robokudo.descriptors.analysis_engines.gpsr.gpsr_implementation import (
    AnalysisEngine,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_door_pipeline import (
    Subtree as DoorSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_object_detection import (
    Subtree as ObjectSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_near_object import (
    Subtree as FurnitureSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_attribute_pipeline import (
    Subtree as AttributeSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_posture_pipeline import (
    Subtree as PostureSubtree,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_waving_raising_pipeline import (
    Subtree as WavingSubtree,
)
from ._data_files import GpsrDataFile
from ._gpsr_routing import PlaceholderSubtree


# %% ROS node
@pytest.fixture(autouse=True)
def registered_node(node: object) -> None:
    """
    Provide the shared ROS node required to build detector subtrees.
    """


# %% model paths
def test_default_model_directory_is_in_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The default model repository is independent of the working directory.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("GPSR_MODELS_DIRECTORY", raising=False)

    assert AnalysisEngine().model_directory == tmp_path / "robokudo_suturo_weights"


def test_model_directory_can_be_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Allow a checkout at a different location.
    """
    monkeypatch.setenv("GPSR_MODELS_DIRECTORY", str(tmp_path))

    assert AnalysisEngine().model_directory == tmp_path


def test_analysis_engine_passes_the_model_directory_to_each_relevant_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Configure each route that needs the shared model repository.
    """
    monkeypatch.setenv("GPSR_MODELS_DIRECTORY", str(tmp_path))
    factories = {}
    for name, value in list(vars(gpsr_implementation).items()):
        if isinstance(value, type) and issubclass(value, SubtreeInterface):
            factories[name] = Mock(return_value=PlaceholderSubtree())
            monkeypatch.setattr(gpsr_implementation, name, factories[name])

    AnalysisEngine().implementation()

    for name in (
        "Subtree_DoorPipeline",
        "Subtree_HumanAttribute",
        "Subtree_HumanNearObjectPipeline",
        "Subtree_ObjectDetectionPipeline",
        "Subtree_ObjInRoomPipeline",
    ):
        factories[name].assert_called_once_with(model_directory=tmp_path)


def test_custom_yolo_model_and_class_names_use_model_directory(tmp_path: Path) -> None:
    """
    Resolve custom detector files without an installed ROS package.
    """
    parameters = YoloAnnotator.Descriptor.Parameters()
    parameters.data_directory = tmp_path
    parameters.weights_path = "weights/detector.pt"

    assert parameters.model_location("detector") == str(
        tmp_path / parameters.weights_path
    )
    assert parameters.file_in_ros_package(parameters.class_names_path) == (
        tmp_path / parameters.class_names_path
    )


@pytest.mark.parametrize(
    ("subtree", "model_name"),
    [(DoorSubtree, "door.pt"), (ObjectSubtree, "ultimate_model_v11_1.26.4.pt")],
)
def test_detector_subtree_uses_configured_model_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subtree: type,
    model_name: str,
) -> None:
    """
    Configure the detector from the checkout instead of a ROS package.
    """
    monkeypatch.setenv("GPSR_MODELS_DIRECTORY", str(tmp_path))

    tree = subtree(model_directory=tmp_path).implementation()

    [detector] = [child for child in tree.children if isinstance(child, YoloAnnotator)]
    assert detector.model_location == str(tmp_path / "weights" / model_name)
    assert detector.descriptor.parameters.data_directory == tmp_path


@pytest.mark.parametrize(
    "pipeline", [subtree_obj_in_room, subtree_object_detection_pipeline]
)
def test_object_pipeline_passes_the_model_directory_to_its_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pipeline: ModuleType
) -> None:
    """
    Keep the model directory when entering a nested object subtree.
    """
    factory = Mock(return_value=PlaceholderSubtree())
    monkeypatch.setattr(pipeline, "ObjectDetectionSubtree", factory)

    pipeline.Subtree(model_directory=tmp_path).implementation()

    assert factory.call_args.kwargs["model_directory"] == tmp_path


def test_attribute_subtree_uses_the_configured_model_directory(tmp_path: Path) -> None:
    """
    Give the vision language annotator the shared model directory.
    """
    tree = AttributeSubtree(model_directory=tmp_path).implementation()

    [annotator] = [
        child for child in tree.children if isinstance(child, VisionLanguageAnnotator)
    ]
    assert annotator.descriptor.parameters.model_directory == tmp_path


def test_furniture_subtree_uses_the_configured_class_names_directory(
    tmp_path: Path,
) -> None:
    """
    Read furniture class names from the shared model directory.
    """
    tree = FurnitureSubtree(model_directory=tmp_path).implementation()

    [detector] = [child for child in tree.children if isinstance(child, YoloAnnotator)]
    assert detector.descriptor.parameters.data_directory == tmp_path


@pytest.mark.parametrize(
    "annotator_class", [TypeMappingAnnotator, FilterObjectBySemanticAnnotation]
)
def test_semantic_mapping_is_read_from_the_model_directory(
    tmp_path: Path, annotator_class: type
) -> None:
    """
    Use the configured mapping even without the old ROS package.
    """
    mapping_path = tmp_path / "mapping_perception_semantic-annotations.json"
    shutil.copy(GpsrDataFile.SEMANTIC_TYPE_MAPPING, mapping_path)

    annotator = annotator_class(mapping_directory=tmp_path)

    assert annotator._type_map == json.loads(mapping_path.read_text())


@pytest.mark.parametrize("subtree", [FurnitureSubtree, PostureSubtree, WavingSubtree])
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

    tree = FurnitureSubtree().implementation()

    [detector] = [child for child in tree.children if isinstance(child, YoloAnnotator)]
    assert detector.descriptor.parameters.pretrained_model == "yolov8n.pt"
    assert detector.model_location == str(default_directory / "yolov8n.pt")
