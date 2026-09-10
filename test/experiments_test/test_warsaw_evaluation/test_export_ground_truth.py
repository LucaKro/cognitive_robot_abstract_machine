"""
Writing a scene's ground truth from the command line.

A module run with ``-m`` is imported as ``__main__``, so a record class defined in the
command's own module is written into every file under that name and nothing can read it
back. That is not visible in-process, where the records are imported normally, so the
command is exercised the way it is actually run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.warsaw.evaluation.ground_truth import (
    GroundTruthGraph,
    world_from_urdf,
)

# %% the scene the command is run against

SCENE = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "warsaw_evaluation"
    / "two_drawer_cabinet.urdf"
)
"""
A cabinet with one drawer and a handle, small enough to export in a test.
"""

PROVIDER = (
    "semantic_digital_twin.predetermined_maps.kitchen_environment:KitchenEnvironment"
)
"""
A modelled world written as Python rather than as a file.
"""


def exported(tmp_path: Path) -> Path:
    """
    Run the export command the way a person runs it, and return what it wrote.
    """
    written = tmp_path / "ground_truth.json"
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.warsaw.evaluation.export_ground_truth",
            str(SCENE),
            "--output",
            str(written),
        ],
        capture_output=True,
        text=True,
    )
    assert finished.returncode == 0, finished.stderr
    return written


# %% what the command writes


def test_what_the_command_writes_can_be_read_back(tmp_path: Path):
    """
    The graph is committed and read by everything downstream, so it has to name the
    classes it holds by where they really live.
    """
    written = exported(tmp_path)

    graph = GroundTruthGraph.from_json(json.loads(written.read_text()))

    assert {node.name for node in graph.nodes} == {
        "cabinet/base",
        "cabinet/cabinet",
        "cabinet/drawer",
        "cabinet/handle",
        "drawer_slider",
    }


def test_the_command_infers_the_classes_a_urdf_does_not_carry():
    """
    A URDF says what is jointed to what and never what any of it is.
    """
    graph = GroundTruthGraph.from_world(
        world_from_urdf(SCENE), scene="two_drawer_cabinet"
    )

    classes = {node.name: node.classes for node in graph.nodes}
    assert classes["cabinet/handle"] == ["Handle"]
    assert classes["cabinet/drawer"] == ["Drawer"]
    assert classes["cabinet/cabinet"] == ["Cabinet"]


# %% a world written as Python rather than as a file


def test_the_command_reads_a_world_from_what_builds_it(tmp_path: Path):
    """
    One of the two scenes is modelled as a URDF and the other as a class that builds a
    world, so the command that writes a ground truth has to take either.
    """
    written = tmp_path / "ground_truth.json"
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.warsaw.evaluation.export_ground_truth",
            "--world-provider",
            PROVIDER,
            "--output",
            str(written),
        ],
        capture_output=True,
        text=True,
    )

    assert finished.returncode == 0, finished.stderr
    graph = GroundTruthGraph.from_json(json.loads(written.read_text()))
    assert graph.scene == PROVIDER
    assert {"Drawer", "Handle"} <= {
        one for node in graph.nodes for one in node.semantic_classes
    }


def test_naming_neither_a_urdf_nor_a_provider_is_refused(tmp_path: Path):
    """
    A ground truth of nothing would be written as an empty graph, which reads as a
    modelled world that holds nothing rather than as a command given no world.
    """
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.warsaw.evaluation.export_ground_truth",
            "--output",
            str(tmp_path / "ground_truth.json"),
        ],
        capture_output=True,
        text=True,
    )

    assert finished.returncode != 0
