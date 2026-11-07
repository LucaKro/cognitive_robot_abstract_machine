"""
Fixtures shared by the tests of the GPSR perception.
"""

from __future__ import annotations

# %% Imports
import inspect
import json
import shutil

import pytest
from robokudo.analysis_engine import SubtreeInterface
from robokudo.annotators.gpsr import human_activity_annotator
from robokudo.annotators.gpsr.belief_state import BeliefState, BeliefStateAnnotator
from robokudo.annotators.gpsr.filter_object_by_semantic_annotation import (
    FilterObjectBySemanticAnnotation,
)
from robokudo.annotators.gpsr.human_activity_annotator import HumanActivityAnnotator
from robokudo.annotators.gpsr.type_mapping_annotator import TypeMappingAnnotator
from robokudo.descriptors.analysis_engines.gpsr import gpsr_implementation
from robokudo.descriptors.analysis_engines.gpsr.gpsr_implementation import (
    AnalysisEngine,
)
from robokudo.utils.file_loader import FileLoader
from typing_extensions import TYPE_CHECKING

from ._clip_mock import PretrainedLoader, ScoringModel, ScoringModelAndProcessor
from ._data_files import GpsrDataFile
from ._gpsr_routing import PlaceholderSubtree, scheduler_of
from ._gpsr_scene import GpsrScene

if TYPE_CHECKING:
    from pathlib import Path

    from py_trees.behaviour import Behaviour
    from rclpy.node import Node
    from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
    from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler

# %% visualization output


@pytest.fixture(autouse=True)
def visualization_output(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> AnnotatorOutputPerPipelineMap:
    """
    Give every annotator under test a buffer to write its visualization output to.

    :param annotator_output_buffer: The buffer the annotators write to.
    :return: The buffer.
    """
    return annotator_output_buffer


# %% belief state isolation


@pytest.fixture(autouse=True)
def belief_state(monkeypatch: pytest.MonkeyPatch) -> BeliefState:
    """
    :param monkeypatch: Replaces the shared belief state.
    :return: The belief state all belief state annotators share, empty at the start of
        each test.
    """
    shared_belief_state = BeliefState()
    monkeypatch.setattr(BeliefStateAnnotator, "_state", shared_belief_state)
    monkeypatch.setattr(BeliefStateAnnotator, "_next_id", 1)
    return shared_belief_state


# %% semantic type mapping


@pytest.fixture
def semantic_type_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, str]:
    """
    Make the annotators read the mapping from detector classes to semantic types from
    the test data instead of the installed ROS package.

    :param monkeypatch: Redirects the package file lookup.
    :param tmp_path: Directory of the stand-in package.
    :return: The mapping the annotators read.
    """
    package_share = tmp_path / "package_share"
    mapping_directory = package_share / "data" / "json"
    mapping_directory.mkdir(parents=True)
    for annotator_class in (TypeMappingAnnotator, FilterObjectBySemanticAnnotation):
        default_mapping_file_name = (
            inspect.signature(annotator_class).parameters["mapping_file_name"].default
        )
        shutil.copy(
            GpsrDataFile.SEMANTIC_TYPE_MAPPING,
            mapping_directory / default_mapping_file_name,
        )
    monkeypatch.setattr(
        FileLoader,
        "get_path_to_file_in_ros_package",
        lambda ros_pkg_name, relative_path: package_share / relative_path,
    )
    return json.loads(GpsrDataFile.SEMANTIC_TYPE_MAPPING.read_text())


# %% CLIP stand-in


@pytest.fixture
def scoring_clip(monkeypatch: pytest.MonkeyPatch) -> ScoringModelAndProcessor:
    """
    Make the activity annotator load the scoring stand-in instead of pretrained CLIP.

    :param monkeypatch: Replaces the CLIP loaders.
    :return: The stand-in.
    """
    posture_texts = HumanActivityAnnotator.Descriptor.Parameters().posture_texts
    clip = ScoringModelAndProcessor(
        model=ScoringModel(posture_texts=tuple(posture_texts))
    )
    monkeypatch.setattr(
        human_activity_annotator, "CLIPModel", PretrainedLoader(loaded=clip.model)
    )
    monkeypatch.setattr(
        human_activity_annotator,
        "CLIPProcessor",
        PretrainedLoader(loaded=clip.processor),
    )
    return clip


# %% scene


@pytest.fixture
def scene() -> GpsrScene:
    """
    :return: A camera at the world origin, asked an empty query.
    """
    return GpsrScene()


# %% the GPSR tree with placeholder subtrees


@pytest.fixture
def gpsr_tree(node: Node, monkeypatch: pytest.MonkeyPatch) -> Behaviour:
    """
    :param node: The shared ROS node.
    :param monkeypatch: Replaces the subtrees.
    :return: The GPSR tree with its real routing, every subtree it routes to replaced by
        a :class:`PlaceholderSubtree`.
    """
    for module_attribute, value in list(vars(gpsr_implementation).items()):
        if isinstance(value, type) and issubclass(value, SubtreeInterface):
            monkeypatch.setattr(
                gpsr_implementation, module_attribute, PlaceholderSubtree
            )
    return AnalysisEngine().implementation()


@pytest.fixture
def scheduler(gpsr_tree: Behaviour) -> QueryBasedScheduler:
    """
    :param gpsr_tree: The GPSR tree with placeholder subtrees.
    :return: The scheduler of the GPSR tree with placeholder subtrees.
    """
    return scheduler_of(gpsr_tree)
