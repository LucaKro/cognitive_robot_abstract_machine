"""
Keeping what a scene says about itself inside the run that read it.

A run answers everything from the scene it was given, and its directory is meant to hold
every file it read. The mesh is the exception on purpose -- it is hundreds of megabytes
and the provenance records its hash instead -- but a scene may also carry small records
saying what it is: which building and which room of it a converted scene was cut from,
and which way up the mesh is written. Those are what an evaluation reads a run back
against, and a run that only points at them is a run that stops meaning anything once
the scene directory is converted over.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from experiments.warsaw.pipeline.provenance import record_run_provenance
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.split import SplitScene
from experiments.warsaw.scene_split import exclusive_faces
from experiments.warsaw.world_loader.loader import WarsawWorldLoader

from ..test_warsaw_world_loader import write_scan_tilted_by, write_scene

# %% a scene that says something about itself


@pytest.fixture
def scene_with_a_record(tmp_path: Path) -> Path:
    """
    :return: A scene directory holding a mesh and a record of where it came from.
    """
    directory = tmp_path / "scene"
    directory.mkdir()
    box = trimesh.creation.box(extents=(1, 1, 1))
    write_scene(directory / "mesh.ply", box.vertices, box.faces[:4], {"mug": [1] * 4})
    (directory / "room.json").write_text(
        json.dumps({"scene": "a building", "room_id": 5})
    )
    return directory


@pytest.fixture
def scene_of_a_mesh_alone(tmp_path: Path) -> Path:
    """
    :return: A scene directory holding nothing but its mesh, as every scan does.
    """
    directory = tmp_path / "scan"
    directory.mkdir()
    box = trimesh.creation.box(extents=(1, 1, 1))
    write_scene(directory / "mesh.ply", box.vertices, box.faces[:4], {"mug": [1] * 4})
    return directory


def run_against(scene: Path, tmp_path: Path) -> Run:
    """
    :param scene: The scene to record a run of.
    :param tmp_path: Where to make the run's directory.
    :return: The run, its provenance written.
    """
    run = Run.create(tmp_path / "runs")
    record_run_provenance(
        settings=PipelineSettings(scene_directory=scene),
        run=run,
        repository=Path(__file__).resolve().parents[3],
    )
    return run


# %% what the run keeps


def test_what_the_scene_says_about_itself_is_kept_in_the_run(
    scene_with_a_record, tmp_path
):
    """
    The record is copied rather than pointed at, so the run still says which room it
    read once the scene directory holds another one.
    """
    run = run_against(scene_with_a_record, tmp_path)
    kept = run.path(RunFile.SCENE) / "room.json"
    assert json.loads(kept.read_text()) == {"scene": "a building", "room_id": 5}


def test_the_mesh_itself_is_not_copied(scene_with_a_record, tmp_path):
    """
    A scan's mesh is most of what it weighs, and the provenance already records its
    hash.

    Copying it into every run would cost gigabytes to say nothing new.
    """
    run = run_against(scene_with_a_record, tmp_path)
    assert [one.name for one in sorted(run.path(RunFile.SCENE).iterdir())] == [
        "room.json"
    ]


def test_a_scene_that_says_nothing_leaves_nothing_behind(
    scene_of_a_mesh_alone, tmp_path
):
    """
    Every scan is a mesh and nothing else, so those runs are exactly as they were.
    """
    run = run_against(scene_of_a_mesh_alone, tmp_path)
    assert list(run.path(RunFile.SCENE).iterdir()) == []


# %% the frame a run's world is built in


def test_the_split_records_the_frame_its_world_was_built_in(tmp_path):
    """
    A scan is stood on its own floor, so which way a run's bodies are turned depends on
    the scan; whatever reads those bodies back has to be told rather than assume it.
    """
    scene = write_scan_tilted_by(
        tmp_path, trimesh.transformations.rotation_matrix(np.radians(30), [1, 0, 0])
    )
    loader = WarsawWorldLoader(input_directory=scene)
    step = SplitScene(
        settings=PipelineSettings(scene_directory=scene, persist=False),
        run=Run.create(tmp_path / "runs"),
    )
    faces = {str(one.name): one.face_indices for one in loader.label_segments}
    split = exclusive_faces(faces, [])
    labels = {str(one.name): one.class_name for one in loader.label_segments}

    record = step.record_of(loader, split, [], labels, step.build_world(loader, split))

    assert np.allclose(record.world_T_source, loader.scene.world_T_source.to_np())
