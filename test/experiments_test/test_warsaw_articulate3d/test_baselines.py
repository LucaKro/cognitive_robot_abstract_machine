"""
Model-free baselines of the whole pipeline: a reading of the labels and a choice of
wholes, built into the graph a run writes and judged as a run is judged.
"""

from __future__ import annotations

import json

import pytest

from experiments.warsaw.articulate3d.baselines import Baseline, BaselineName, main
from experiments.warsaw.articulate3d.ground_truth import GroundTruthFile
from experiments.warsaw.articulate3d.label_readings import TrueClasses
from experiments.warsaw.articulate3d.measured_scene import MeasuredSceneFile
from experiments.warsaw.articulate3d.score import ClassLineage, ScoreOverRuns, Share
from experiments.warsaw.articulate3d.whole_choices import (
    FirstCandidateOfTheRule,
    LargerObjectTouchedMost,
)
from experiments.warsaw.pipeline.run import Run
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.evidence import MeasureScene

from .scenes import SceneName, measured_kitchen_corner

# %% one baseline on one scene


@pytest.fixture
def rule(tmp_path, ontology):
    """
    :return: The pipeline's rule, with the pipeline's settings.
    """
    return FirstCandidateOfTheRule(
        step=MeasureScene(settings=PipelineSettings(), run=Run(directory=tmp_path)),
        ontology=ontology,
    )


@pytest.fixture
def lineage(taxonomy):
    """
    :return: What the ontology's classes derive from.
    """
    return ClassLineage.from_taxonomy(taxonomy, {})


def test_each_segment_becomes_a_node_with_its_class_and_whole(rule, ground_truth):
    graph = Baseline(reading=TrueClasses(), choice=rule).graph(
        measured_kitchen_corner(), ground_truth
    )

    [door] = [one for one in graph.nodes if one.name == "door_1"]
    assert (door.predicted_class, door.parent) == ("Door", "cabinet_1")


def test_the_true_classes_give_every_judged_part_its_class(rule, ground_truth, lineage):
    score = Baseline(reading=TrueClasses(), choice=rule).score(
        measured_kitchen_corner(), ground_truth, lineage
    )

    judged = [one for one in ground_truth.parts if one.semantic_class is not None]
    assert score.classes.accuracy == Share(len(judged), len(judged))


def test_the_rule_s_wholes_are_judged_as_a_run_s(rule, ground_truth, lineage):
    """
    Four parts are hung: three from their true whole, and the handle from the cabinet it
    touches rather than its drawer.

    The control, of no known class, is hung from nothing.
    """
    score = Baseline(reading=TrueClasses(), choice=rule).score(
        measured_kitchen_corner(), ground_truth, lineage
    )

    assert score.links.precision == Share(3, 4)
    assert score.links.recall == Share(3, len(ground_truth.relations))


def test_geometry_hangs_the_control_from_its_switch_base(ground_truth, lineage):
    score = Baseline(
        reading=TrueClasses(),
        choice=LargerObjectTouchedMost(settings=PipelineSettings()),
    ).score(measured_kitchen_corner(), ground_truth, lineage)

    assert score.links.precision == Share(4, 5)


# %% every scene, from the command line


def test_the_command_writes_each_baseline_s_summary(tmp_path, ground_truth):
    measured_directory = tmp_path / "measured"
    measured_directory.mkdir()
    measured_kitchen_corner().write(
        MeasuredSceneFile.SUFFIX.path(measured_directory, SceneName.KITCHEN_CORNER)
    )
    truth_directory = tmp_path / "ground_truth"
    truth_directory.mkdir()
    ground_truth.write(
        GroundTruthFile.SUFFIX.path(truth_directory, SceneName.KITCHEN_CORNER)
    )
    output = tmp_path / "baselines"

    main(
        [
            "--measured",
            str(measured_directory),
            "--ground-truth",
            str(truth_directory),
            "--output",
            str(output),
            "--baseline",
            BaselineName.TRUE_CLASSES_AND_RULE,
        ]
    )

    summary = ScoreOverRuns.from_json(
        json.loads((output / f"{BaselineName.TRUE_CLASSES_AND_RULE}.json").read_text())
    )
    assert summary.links.precision == Share(3, 4)
