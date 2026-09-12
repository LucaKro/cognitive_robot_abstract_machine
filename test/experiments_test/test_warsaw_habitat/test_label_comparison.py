"""
Laying a run's answers beside the dataset's own labels, object by object.

Counts say how well a run did and never say where it went wrong. What is checked here is
that every object is carried through with all three of the things needed to judge it --
what the dataset called it, what the run answered, and what the matcher made of that --
and that the disagreements are gathered so the largest is the first thing read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.warsaw.evaluation.label_vocabulary import LexicalMatcher
from experiments.warsaw.habitat.convert import ConvertedObject, ConvertedRoom
from experiments.warsaw.pipeline.records import BodyAnswer, Classifications
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.habitat.label_comparison import (
    COMPARISON_FILE,
    LabelComparison,
    compare_a_run,
)

# %% a run to read back


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """
    :return: A run directory holding what a converted room and a classification leave.

    Written through the records the converter and the classification step write, rather
    than as hand-shaped JSON, so the fixture cannot drift from what a run really holds.
    """
    directory = tmp_path / "2026-01-01_000000"
    (directory / "scene").mkdir(parents=True)
    ConvertedRoom(
        scene="00808-y9hTuugGdiq",
        room_id=5,
        objects=[
            ConvertedObject(
                object_id=1,
                label="kitchen cabinet",
                segment="kitchen_cabinet_1",
                faces=10,
            ),
            ConvertedObject(
                object_id=2,
                label="kitchen cabinet",
                segment="kitchen_cabinet_2",
                faces=10,
            ),
            ConvertedObject(
                object_id=3,
                label="light fixture",
                segment="light_fixture_3",
                faces=5,
            ),
            ConvertedObject(object_id=4, label="sink", segment="sink_4", faces=7),
        ],
    ).write_beside(directory / "scene")
    Run(directory=directory).write_record(
        RunFile.CLASSIFICATIONS,
        Classifications(
            scene="",
            model="a model",
            bodies=[
                BodyAnswer(name="kitchen_cabinet_1", class_name="Cabinet"),
                BodyAnswer(name="kitchen_cabinet_2", class_name="Cabinet"),
                BodyAnswer(name="light_fixture_3", class_name="Decor"),
                BodyAnswer(name="sink_4", class_name=None),
            ],
        ),
    )
    return directory


@pytest.fixture
def compared(run: Path) -> LabelComparison:
    """
    :return: That run, compared by wording alone so the test needs no encoder.
    """
    return compare_a_run(run, LexicalMatcher(), matcher_name="wording")


# %% what every object carries


def test_an_object_carries_what_each_side_called_it(compared: LabelComparison):
    """
    The three things a disagreement has to be judged from, kept together.
    """
    one = {one.segment: one for one in compared.objects}["kitchen_cabinet_1"]
    assert (one.truth, one.predicted, one.matched) == (
        "kitchen cabinet",
        "cabinet",
        "kitchen cabinet",
    )
    assert one.agrees


def test_a_prediction_the_matcher_cannot_place_disagrees(compared: LabelComparison):
    """
    The ontology answers Decor for a light fixture, and no wording makes those one.
    """
    one = {one.segment: one for one in compared.objects}["light_fixture_3"]
    assert (one.predicted, one.matched, one.agrees) == ("decor", None, False)


def test_a_body_left_unannotated_is_carried_through_as_a_disagreement(
    compared: LabelComparison,
):
    """
    A body the run could not annotate is not a body the comparison may quietly drop.
    """
    one = {one.segment: one for one in compared.objects}["sink_4"]
    assert (one.predicted, one.agrees) == (None, False)


# %% where the disagreement is


def test_the_run_is_named_by_what_it_was_run_on(compared: LabelComparison):
    """
    A comparison says which building and room it is of, since the counts alone do not.
    """
    assert (compared.scene, compared.room_id) == ("00808-y9hTuugGdiq", 5)


def test_the_agreement_is_counted(compared: LabelComparison):
    """
    One of four agrees: the two cabinets agree, the light fixture and the sink do not.
    """
    assert (compared.agreed, len(compared.objects)) == (2, 4)


def test_disagreements_are_gathered_largest_first(compared: LabelComparison):
    """
    What is wanted is where most of the disagreement came from, so the pairs are counted
    and the biggest is first.
    """
    assert compared.disagreements()[0] == (("light fixture", "decor"), 1)


# %% the file itself


def test_the_file_holds_every_object_and_says_where_it_is(
    run: Path, compared: LabelComparison
):
    """
    Written into the run it is about, so it is found beside everything else that run
    says rather than somewhere a reader has to be told about.
    """
    written = compared.write_beside(run)
    assert written == run / COMPARISON_FILE
    held = written.read_text()
    assert all(one.segment in held for one in compared.objects)
    assert "wording" in held
