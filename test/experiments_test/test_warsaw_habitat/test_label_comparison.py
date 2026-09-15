"""
Laying a run's answers beside the dataset's own labels, object by object.

Counts say how well a run did and never say where it went wrong. What is checked here is
that every object is carried through with all three of the things needed to judge it --
what the dataset called it, what the run answered, and what the matcher made of that --
and that the disagreements are gathered so the largest is the first thing read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.warsaw.evaluation.label_vocabulary import LexicalMatcher
from experiments.warsaw.habitat.convert import ConvertedObject, ConvertedRoom
from experiments.warsaw.pipeline.records import BodyAnswer, Classifications
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.habitat.label_comparison import (
    COMPARISON_FILE,
    COMPARISON_RECORD,
    ClassificationScores,
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
    What a disagreement has to be judged from, kept together.
    """
    one = {one.segment: one for one in compared.objects}["kitchen_cabinet_1"]
    assert (one.truth, one.predicted, one.agrees) == (
        "kitchen cabinet",
        "cabinet",
        True,
    )


def test_a_prediction_the_matcher_cannot_place_disagrees(compared: LabelComparison):
    """
    The ontology answers Decor for a light fixture, and no wording makes those one.
    """
    one = {one.segment: one for one in compared.objects}["light_fixture_3"]
    assert (one.predicted, one.agrees) == ("decor", False)


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


# %% precision, recall and F1


def test_scores_are_the_harmonic_mean_of_precision_and_recall():
    """
    Two predictions right of four made, of eight there were to find.
    """
    scores = ClassificationScores.from_counts(
        true_positives=2, false_positives=2, false_negatives=6
    )
    assert (scores.precision, scores.recall, scores.f1_score) == (0.5, 0.25, 1 / 3)


def test_scores_of_nothing_predicted_and_nothing_to_find_are_zero():
    """
    An empty ratio is defined as zero rather than raised.
    """
    scores = ClassificationScores.from_counts(
        true_positives=0, false_positives=0, false_negatives=0
    )
    assert (scores.precision, scores.recall, scores.f1_score) == (0.0, 0.0, 0.0)


def test_an_answer_is_known_to_mean_the_labels_of_the_room_it_names(
    compared: LabelComparison,
):
    """
    Which of the room's labels an answer names is what a false positive of another label
    is counted from.
    """
    one = {one.segment: one for one in compared.objects}["kitchen_cabinet_1"]
    assert one.means == ["kitchen cabinet"]


def test_per_object_precision_leaves_out_the_objects_given_no_answer(
    compared: LabelComparison,
):
    """
    Two agree of the three objects answered, and of the four objects there are.
    """
    assert compared.per_object == ClassificationScores.from_counts(
        true_positives=2, false_positives=1, false_negatives=2
    )


def test_per_label_scores_average_every_label_of_the_room(compared: LabelComparison):
    """
    ``kitchen cabinet`` is found both times and nothing else is taken for it; the light
    fixture and the sink are found neither time.
    """
    found_every_time = ClassificationScores.from_counts(
        true_positives=2, false_positives=0, false_negatives=0
    )
    found_never = ClassificationScores.from_counts(
        true_positives=0, false_positives=0, false_negatives=1
    )
    assert compared.per_label == {
        "kitchen cabinet": found_every_time,
        "light fixture": found_never,
        "sink": found_never,
    }


def test_a_label_another_objects_answer_names_counts_a_false_positive(run: Path):
    """
    A cabinet answered for a sink is a sink found nowhere, and a cabinet found where
    there was none.
    """
    compared = compare_a_run(run, LexicalMatcher(), matcher_name="wording")
    compared.objects[3].predicted = "cabinet"
    compared.objects[3].means = ["kitchen cabinet"]
    assert compared.per_label["kitchen cabinet"] == ClassificationScores.from_counts(
        true_positives=2, false_positives=1, false_negatives=0
    )


def test_the_macro_average_is_the_mean_over_labels(compared: LabelComparison):
    """
    The mean of each score over the room's labels, so a rare label weighs as much as a
    common one.
    """
    per_label = list(compared.per_label.values())
    assert compared.per_label_average == ClassificationScores(
        precision=sum(one.precision for one in per_label) / len(per_label),
        recall=sum(one.recall for one in per_label) / len(per_label),
        f1_score=sum(one.f1_score for one in per_label) / len(per_label),
    )


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


def test_the_record_is_written_beside_the_page(run: Path, compared: LabelComparison):
    """
    The counts are kept as data too, so several rooms can be added up without reading a
    page back.
    """
    compared.write_beside(run)
    assert (
        LabelComparison.from_json(json.loads((run / COMPARISON_RECORD).read_text()))
        == compared
    )
