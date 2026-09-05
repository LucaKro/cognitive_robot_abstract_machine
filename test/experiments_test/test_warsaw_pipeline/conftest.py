"""
The files a finished run left behind, which the readers under test are read against.

Everything here is what one real run of the pipeline wrote, trimmed where a file's bulk
said nothing a smaller one does not. Reading them is what pins the shapes the steps hand
one another: a step writes what the next one reads, so a change to either that the other
does not follow shows up here rather than three steps later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.warsaw.pipeline.records import (
    Adjudications,
    Classifications,
    OpenQuestions,
    Relations,
    SplitRecord,
    Vocabulary,
    VocabularyRequest,
)
from experiments.warsaw.pipeline.run import Run


@pytest.fixture
def dataset() -> Path:
    """
    :return: The directory holding what a finished run wrote.
    """
    return Path(__file__).resolve().parent.parent / "dataset" / "warsaw_pipeline"


@pytest.fixture
def finished_run(dataset) -> Run:
    """
    :return: A finished run, as a directory of files.
    """
    return Run(directory=dataset / "run")


@pytest.fixture
def replies(dataset) -> Path:
    """
    :return: The directory holding replies as a model returned them.
    """
    return dataset / "replies"


@pytest.fixture
def taxonomy(finished_run) -> dict:
    """
    :return: The ontology as the run read it out.
    """
    return json.loads((finished_run.directory / "taxonomy.json").read_text())


@pytest.fixture
def relations(finished_run) -> Relations:
    """
    :return: How the run measured its scene's objects to meet.
    """
    return Relations.from_json(
        json.loads((finished_run.directory / "relations.json").read_text())
    )


@pytest.fixture
def vocabulary_request(finished_run) -> VocabularyRequest:
    """
    :return: The question the run asked about its labels.
    """
    return VocabularyRequest.from_json(
        json.loads((finished_run.directory / "vocabulary_request.json").read_text())
    )


@pytest.fixture
def vocabulary(finished_run) -> Vocabulary:
    """
    :return: What the run answered about each of its labels.
    """
    return Vocabulary.from_json(
        json.loads((finished_run.directory / "vocabulary.json").read_text())
    )


@pytest.fixture
def questions(finished_run) -> OpenQuestions:
    """
    :return: What the run's measurements and ontology left open.
    """
    return OpenQuestions.from_json(
        json.loads((finished_run.directory / "questions.json").read_text())
    )


@pytest.fixture
def adjudications(finished_run) -> Adjudications:
    """
    :return: What the run answered about each open question.
    """
    return Adjudications.from_json(
        json.loads((finished_run.directory / "adjudications.json").read_text())
    )


@pytest.fixture
def split(finished_run) -> SplitRecord:
    """
    :return: What the run's split built and what it cost.
    """
    return SplitRecord.from_json(
        json.loads((finished_run.directory / "split.json").read_text())
    )


@pytest.fixture
def classifications(finished_run) -> Classifications:
    """
    :return: What the run answered each of its bodies to be.
    """
    return Classifications.from_json(
        json.loads((finished_run.directory / "classifications.json").read_text())
    )
