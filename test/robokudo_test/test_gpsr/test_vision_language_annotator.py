"""
Tests for asking a vision language model about every detected person.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path

import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr import vision_language_annotator
from robokudo.annotators.gpsr.vision_language_annotator import (
    GpuLayers,
    VisionLanguageAnnotator,
)

from ._gpsr_scene import GpsrScene
from ._vision_language_mock import ModelLoader, ProjectorLoader

# %% model stand-in


@pytest.fixture
def model_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModelLoader:
    """
    Make the annotator load the recording model from the files of a stand-in ROS
    package.
    """
    loader = ModelLoader()
    monkeypatch.setattr(vision_language_annotator, "Llama", loader)
    monkeypatch.setattr(
        vision_language_annotator, "Qwen25VLChatHandler", ProjectorLoader()
    )
    monkeypatch.setattr(
        vision_language_annotator.FileLoader,
        "get_path_to_file_in_ros_package",
        lambda ros_pkg_name, relative_path: tmp_path / ros_pkg_name / relative_path,
    )
    return loader


def asking_annotator() -> VisionLanguageAnnotator:
    """
    :return: A vision language annotator with its own default parameters.
    """
    return VisionLanguageAnnotator(descriptor=VisionLanguageAnnotator.Descriptor())


# %% answers about persons


def test_answer_of_the_model_is_annotated_on_the_person(
    scene: GpsrScene, model_loader: ModelLoader
) -> None:
    annotator = asking_annotator()
    scene.place_in_pipeline(annotator)
    person = scene.add_human()

    status = annotator.compute()

    assert status == Status.SUCCESS
    assert scene.activities_of(person, VisionLanguageAnnotator) == [
        model_loader.loaded.answer
    ]


def test_answer_is_stripped_of_surrounding_spaces(
    scene: GpsrScene, model_loader: ModelLoader
) -> None:
    model_loader.loaded.answer = "  red coat \n"
    annotator = asking_annotator()
    scene.place_in_pipeline(annotator)
    person = scene.add_human()

    annotator.compute()

    assert scene.activities_of(person, VisionLanguageAnnotator) == ["red coat"]


def test_empty_answer_annotates_nothing(
    scene: GpsrScene, model_loader: ModelLoader
) -> None:
    model_loader.loaded.answer = " "
    annotator = asking_annotator()
    scene.place_in_pipeline(annotator)
    person = scene.add_human()

    annotator.compute()

    assert scene.activities_of(person, VisionLanguageAnnotator) == []


def test_every_person_is_asked_about(
    scene: GpsrScene, model_loader: ModelLoader
) -> None:
    annotator = asking_annotator()
    scene.place_in_pipeline(annotator)
    scene.add_human()
    scene.add_human()

    annotator.compute()

    assert len(model_loader.loaded.received_messages) == 2


def test_without_persons_nothing_is_asked(
    scene: GpsrScene, model_loader: ModelLoader
) -> None:
    annotator = asking_annotator()
    scene.place_in_pipeline(annotator)

    status = annotator.compute()

    assert status == Status.FAILURE
    assert model_loader.loaded.received_messages == []


# %% conversation with the model


def test_model_is_asked_the_configured_question_with_the_system_prompt(
    scene: GpsrScene, model_loader: ModelLoader
) -> None:
    annotator = asking_annotator()
    parameters = annotator.descriptor.parameters
    parameters.question = "What is this person holding?"
    scene.place_in_pipeline(annotator)
    scene.add_human()

    annotator.compute()

    [[system, user]] = model_loader.loaded.received_messages
    assert system == {"role": "system", "content": parameters.system_prompt}
    [image, question] = user["content"]
    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert question == {"type": "text", "text": parameters.question}


def test_model_answers_with_the_configured_temperature(
    scene: GpsrScene, model_loader: ModelLoader
) -> None:
    annotator = asking_annotator()
    scene.place_in_pipeline(annotator)
    scene.add_human()

    annotator.compute()

    assert model_loader.loaded.received_temperatures == [
        annotator.descriptor.parameters.temperature
    ]


# %% loading the model


def test_model_is_loaded_once(scene: GpsrScene, model_loader: ModelLoader) -> None:
    annotator = asking_annotator()
    scene.place_in_pipeline(annotator)
    scene.add_human()

    annotator.compute()
    annotator.compute()

    assert len(model_loader.load_arguments) == 1


def test_model_is_loaded_from_the_ros_package(
    scene: GpsrScene, model_loader: ModelLoader, tmp_path: Path
) -> None:
    annotator = asking_annotator()
    parameters = annotator.descriptor.parameters
    scene.place_in_pipeline(annotator)
    scene.add_human()

    annotator.compute()

    [arguments] = model_loader.load_arguments
    package = tmp_path / parameters.ros_package_name
    assert arguments["model_path"] == str(package / parameters.model_file)
    assert arguments["chat_handler"] == str(package / parameters.projector_file)
    assert arguments["n_ctx"] == parameters.context_length


def test_model_is_loaded_from_the_configured_directory(
    model_loader: ModelLoader, tmp_path: Path
) -> None:
    """
    Load both Qwen files from the same configured model directory.
    """
    annotator = asking_annotator()
    parameters = annotator.descriptor.parameters
    parameters.model_directory = tmp_path

    annotator.load_model()

    [arguments] = model_loader.load_arguments
    assert arguments["model_path"] == str(tmp_path / parameters.model_file)
    assert arguments["chat_handler"] == str(tmp_path / parameters.projector_file)


@pytest.mark.parametrize(
    "use_gpu, gpu_layers",
    [(True, GpuLayers.ALL), (False, GpuLayers.NONE)],
    ids=["gpu", "cpu"],
)
def test_gpu_setting_decides_the_layers_on_the_gpu(
    scene: GpsrScene, model_loader: ModelLoader, use_gpu: bool, gpu_layers: GpuLayers
) -> None:
    annotator = asking_annotator()
    annotator.descriptor.parameters.use_gpu = use_gpu
    scene.place_in_pipeline(annotator)
    scene.add_human()

    annotator.compute()

    [arguments] = model_loader.load_arguments
    assert arguments["n_gpu_layers"] == gpu_layers
