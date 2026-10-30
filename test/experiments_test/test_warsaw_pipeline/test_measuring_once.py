"""
Measuring a scene once and reading it back, rather than measuring it twice.

A run measures its scene in two of its steps, and the measurement is geometry alone: the
same scene measured twice gives the same answer twice. On a scanned room of nearly two
million faces that answer costs most of the run, so the second step reads what the first
one wrote.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from experiments.warsaw.pipeline.records import Relations
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.evidence import MeasureScene
from experiments.warsaw.world_loader.loader import WarsawWorldLoader

from ..test_warsaw_world_loader import write_scene

# %% a scene that is nothing like the one already measured


@pytest.fixture
def other_scene(tmp_path) -> Path:
    """
    :return: A directory holding a scene of one labelled object, which shares not one
        segment name with the measurement the finished run wrote.
    """
    directory = tmp_path / "other_scene"
    directory.mkdir()
    box = trimesh.creation.box(extents=(1, 1, 1))
    faces = box.faces[:4]
    write_scene(directory / "scene.ply", box.vertices, faces, {"mug": [1] * len(faces)})
    return directory


# %% taking the measurement the run already made


def test_a_measured_scene_is_read_back_rather_than_measured_again(
    finished_run, other_scene
):
    """
    The proof that it was read is that it describes the scene the record was written
    for, which is not the scene the loader holds.
    """
    step = MeasureScene(settings=PipelineSettings(), run=finished_run)
    written = finished_run.read_record(RunFile.RELATIONS, Relations)

    measured = step.measured_scene(WarsawWorldLoader(input_directory=other_scene))

    assert list(measured.descriptors) == [one.name for one in written.segments]
    assert measured.pairs == [one.evidence for one in written.pairs]


def test_a_scene_nothing_has_measured_yet_is_measured(other_scene, tmp_path):
    """
    The first step of a run has nothing to read, and must do the work.
    """
    step = MeasureScene(settings=PipelineSettings(), run=Run(directory=tmp_path))

    measured = step.measured_scene(WarsawWorldLoader(input_directory=other_scene))

    assert list(measured.descriptors) == ["mug_1"]
