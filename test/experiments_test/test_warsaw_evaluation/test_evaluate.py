"""
Gathering every number a run can be judged by into one place.

The numbers come from several comparisons with different requirements: some need only
the two graphs, others need the two worlds related to each other. A command that ran
only the ones it could and said nothing about the rest would read as though the rest had
passed, so what was left out is part of what is written.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.warsaw.evaluation.evaluate import Evaluation
from experiments.warsaw.evaluation.ground_truth import (
    GroundTruthEdge,
    GroundTruthGraph,
    GroundTruthNode,
)
from experiments.warsaw.evaluation.graph import (
    EvaluationEdge,
    EvaluationGraph,
    EvaluationNode,
)
from experiments.warsaw.pipeline.records import SplitRecord
from experiments.warsaw.pipeline.run import Run, RunFile

# %% a run and the world it is judged against


def modelled_world() -> GroundTruthGraph:
    """
    A world holding one drawer in one cabinet, and nothing else.
    """
    return GroundTruthGraph(
        scene="a modelled kitchen",
        frame="modelled",
        geometry_source="visual",
        nodes=[
            GroundTruthNode(
                name=name,
                source_id=name,
                parent=None,
                semantic_classes=[semantic_class],
                faces=100,
                world_transform=[],
                bounds=None,
            )
            for name, semantic_class in (("cabinet", "Cabinet"), ("drawer", "Drawer"))
        ],
        edges=[
            GroundTruthEdge(
                whole="cabinet", part="drawer", relation="part", field_name="drawers"
            )
        ],
    )


def a_finished_run(directory: Path) -> Run:
    """
    A run that put one drawer in a cabinet and left a second drawer in nothing.
    """
    run = Run(directory=directory)
    directory.mkdir(parents=True, exist_ok=True)
    run.write_record(
        RunFile.EVALUATION_GRAPH,
        EvaluationGraph(
            nodes=[
                EvaluationNode(
                    name=name,
                    input_label=name,
                    predicted_class=semantic_class,
                    faces=100,
                    body_id=name,
                    annotation_applied=True,
                )
                for name, semantic_class in (
                    ("cabinet_1", "Cabinet"),
                    ("drawer_1", "Drawer"),
                    ("drawer_2", "Drawer"),
                )
            ],
            edges=[
                EvaluationEdge(
                    whole="cabinet_1",
                    part="drawer_1",
                    relation="part",
                    field_name="drawers",
                    accepted=True,
                )
            ],
        ),
    )
    run.write_record(RunFile.SPLIT, SplitRecord(scene="a scanned kitchen"))
    return run


def evaluation_of(tmp_path: Path) -> Evaluation:
    """
    Judge that run against that world.
    """
    return Evaluation.of(
        run=a_finished_run(tmp_path / "run"), ground_truth=modelled_world()
    )


# %% what it says


def test_an_evaluation_names_the_run_and_the_world_it_is_against(tmp_path: Path):
    """
    A number without the run it came from is a number nobody can go back to.
    """
    judged = evaluation_of(tmp_path)

    assert judged.run == "run"
    assert judged.scene == "a modelled kitchen"


def test_the_relations_a_run_asserted_are_judged_on_their_kind(tmp_path: Path):
    """
    One drawer went into a cabinet, which is what the modelled world does.
    """
    judged = evaluation_of(tmp_path)

    assert len(judged.parenthood.relations) == 1
    assert judged.parenthood.relations_of_a_kind_the_modelled_world_holds == 1


def test_the_part_that_was_given_nothing_is_carried_through(tmp_path: Path):
    """
    The second drawer is what the run has to be asked about, so it survives into the
    gathered numbers rather than being summarised away.
    """
    judged = evaluation_of(tmp_path)

    assert [one.part for one in judged.parenthood.missing] == ["drawer_2"]


# %% what could not be answered


def test_a_comparison_that_needs_the_worlds_related_is_named_as_left_out(
    tmp_path: Path,
):
    """
    Silence about a comparison reads as the comparison having passed.

    Naming it, and saying what it waits on, is the difference between a gap and an
    unknown.
    """
    judged = evaluation_of(tmp_path)

    left_out = {one.comparison: one.because for one in judged.left_out}
    assert "object matching" in left_out
    assert left_out["object matching"]


# %% reading it back


def test_the_numbers_are_written_where_they_can_be_read_back(tmp_path: Path):
    """
    The gathered numbers are the thing later work reads, so they round-trip.
    """
    judged = evaluation_of(tmp_path)
    written = tmp_path / "evaluation.json"
    written.write_text(json.dumps(judged.to_json()))

    assert Evaluation.from_json(json.loads(written.read_text())) == judged


def test_the_summary_names_the_run_and_what_was_left_out(tmp_path: Path):
    """
    The summary is what is read instead of the JSON, so it carries the same caveats.
    """
    summary = evaluation_of(tmp_path).markdown()

    assert "run" in summary
    assert "object matching" in summary
