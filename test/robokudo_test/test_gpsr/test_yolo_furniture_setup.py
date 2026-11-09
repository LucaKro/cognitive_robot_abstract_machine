"""
Tests for setting the furniture detector and the furniture filter up from the query.
"""

from __future__ import annotations

# %% Imports
import json

import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.filter_humans_by_object_annotator import (
    FilterHumansByObjectAnnotator,
)
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_near_object import (
    YoloFurnitureSetup,
)
from typing_extensions import TYPE_CHECKING

from ._data_files import GpsrDataFile
from ._gpsr_scene import GpsrScene

if TYPE_CHECKING:
    from pathlib import Path

# %% setup under test


@pytest.fixture
def class_names_file() -> Path:
    """
    :return: A detector class name file listing a person and four pieces of furniture.
    """
    return GpsrDataFile.FURNITURE_CLASS_NAMES


@pytest.fixture
def detector_parameters(
    monkeypatch: pytest.MonkeyPatch, class_names_file: Path
) -> YoloAnnotator.Descriptor.Parameters:
    """
    :param monkeypatch: Redirects the file lookup of the parameters.
    :param class_names_file: The class name file.
    :return: Parameters of a furniture detector that reads its class names from the test
        data.
    """
    parameters = YoloAnnotator.Descriptor().parameters
    monkeypatch.setattr(
        parameters, "file_in_ros_package", lambda relative_path: class_names_file
    )
    return parameters


@pytest.fixture
def filter_parameters() -> FilterHumansByObjectAnnotator.Descriptor.Parameters:
    """
    :return: Parameters of a furniture filter searching on the robot's left by default.
    """
    parameters = FilterHumansByObjectAnnotator.Descriptor().parameters
    parameters.direction = "left"
    return parameters


@pytest.fixture
def furniture_setup(
    scene: GpsrScene,
    detector_parameters: YoloAnnotator.Descriptor.Parameters,
    filter_parameters: FilterHumansByObjectAnnotator.Descriptor.Parameters,
) -> YoloFurnitureSetup:
    """
    :param scene: The scene of the setup.
    :param detector_parameters: The detector configuration.
    :param filter_parameters: The filter configuration.
    :return: A furniture setup placed in the scene's pipeline.
    """
    setup = YoloFurnitureSetup(detector_parameters, filter_parameters)
    scene.place_in_pipeline(setup)
    return setup


# %% detector classes


def test_known_furniture_makes_the_detector_ignore_every_other_class(
    scene: GpsrScene,
    furniture_setup: YoloFurnitureSetup,
    detector_parameters: YoloAnnotator.Descriptor.Parameters,
    class_names_file: Path,
) -> None:
    """
    Known furniture makes the detector ignore every class except that furniture.
    """
    scene.query.obj.location = "chair"

    status = furniture_setup.update()

    assert status == Status.SUCCESS
    class_names = json.loads(class_names_file.read_text()).values()
    assert detector_parameters.ignored_class_names == [
        name for name in class_names if name != "chair"
    ]


def test_dining_table_makes_the_detector_keep_the_dining_table(
    scene: GpsrScene,
    furniture_setup: YoloFurnitureSetup,
    detector_parameters: YoloAnnotator.Descriptor.Parameters,
    class_names_file: Path,
) -> None:
    """
    A furniture name of two words is kept by the detector as well.
    """
    scene.query.obj.location = "dining table"

    furniture_setup.update()

    class_names = json.loads(class_names_file.read_text()).values()
    assert detector_parameters.ignored_class_names == [
        name for name in class_names if name != scene.query.obj.location
    ]


def test_unknown_furniture_leaves_the_detector_alone(
    scene: GpsrScene,
    furniture_setup: YoloFurnitureSetup,
    detector_parameters: YoloAnnotator.Descriptor.Parameters,
) -> None:
    """
    Unknown furniture leaves the detector alone.
    """
    scene.query.obj.location = "garden"

    furniture_setup.update()

    assert detector_parameters.ignored_class_names == []


# %% filter target and side


@pytest.mark.parametrize("queried_side", ["left", "right", "front", "back"])
def test_each_query_direction_is_passed_to_the_filter(
    scene: GpsrScene,
    furniture_setup: YoloFurnitureSetup,
    filter_parameters: FilterHumansByObjectAnnotator.Descriptor.Parameters,
    queried_side: str,
) -> None:
    """
    Preserve each direction received from the query.
    """
    scene.query.obj.location = "chair"
    scene.query.obj.attribute = [queried_side]

    furniture_setup.update()

    assert filter_parameters.direction == queried_side


def test_queried_furniture_becomes_the_filter_target(
    scene: GpsrScene,
    furniture_setup: YoloFurnitureSetup,
    filter_parameters: FilterHumansByObjectAnnotator.Descriptor.Parameters,
) -> None:
    """
    Queried furniture becomes the filter target.
    """
    scene.query.obj.location = "couch"

    furniture_setup.update()

    assert filter_parameters.target_type == scene.query.obj.location


def test_queried_side_becomes_the_filter_side(
    scene: GpsrScene,
    furniture_setup: YoloFurnitureSetup,
    filter_parameters: FilterHumansByObjectAnnotator.Descriptor.Parameters,
) -> None:
    """
    Queried side becomes the filter side.
    """
    queried_side = "right"
    scene.query.obj.location = "chair"
    scene.query.obj.attribute = ["sitting", queried_side]

    furniture_setup.update()

    assert filter_parameters.direction == queried_side


def test_without_queried_side_the_default_side_is_searched(
    scene: GpsrScene,
    furniture_setup: YoloFurnitureSetup,
    filter_parameters: FilterHumansByObjectAnnotator.Descriptor.Parameters,
) -> None:
    """
    Without queried side the default side is searched.
    """
    default_side = filter_parameters.direction
    scene.query.obj.location = "chair"
    scene.query.obj.attribute = ["back"]
    furniture_setup.update()
    scene.query.obj.attribute = []

    furniture_setup.update()

    assert filter_parameters.direction == default_side


def test_without_query_nothing_is_set_up(
    detector_parameters: YoloAnnotator.Descriptor.Parameters,
    filter_parameters: FilterHumansByObjectAnnotator.Descriptor.Parameters,
) -> None:
    """
    Without query nothing is set up.
    """
    default_side = filter_parameters.direction
    setup = YoloFurnitureSetup(detector_parameters, filter_parameters)
    GpsrScene(query=None).place_in_pipeline(setup)

    status = setup.update()

    assert status == Status.SUCCESS
    assert detector_parameters.ignored_class_names == []
    assert filter_parameters.target_type is None
    assert filter_parameters.direction == default_side
