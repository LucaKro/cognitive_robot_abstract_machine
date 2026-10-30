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
    ClassificationScores,
    ComparedRooms,
    ComparisonMatcher,
    LabelComparison,
    LabelGroup,
    ScoreSummary,
    compare_a_run,
)

from ..test_warsaw_evaluation.test_label_vocabulary import (
    matching_by_head_noun,
    turned,
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
    assert written == run / ComparisonMatcher.WORDING.page
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
        LabelComparison.from_json(
            json.loads((run / ComparisonMatcher.WORDING.record).read_text())
        )
        == compared
    )


def test_each_matcher_writes_a_comparison_of_its_own(
    run: Path, compared: LabelComparison
):
    """
    Two matchers compared on one run are both kept, so every number of either can be
    read back without scoring the run again.
    """
    by_head_noun = compare_a_run(
        run, meanings(), matcher_name=ComparisonMatcher.HEAD_NOUN
    )
    assert compared.write_beside(run) != by_head_noun.write_beside(run)


# %% answers too far from their label in meaning


def meanings():
    """
    :return: A head-noun matcher placing every name of the fixture room: the cabinets
        together, and the light fixture's answer far from it.
    """
    return matching_by_head_noun(
        cabinet=turned(0),
        kitchen_cabinet=turned(20),
        decor=turned(100),
        light_fixture=turned(180),
        sink=turned(260),
    )


@pytest.fixture
def compared_by_meaning(run: Path) -> LabelComparison:
    """
    :return: The fixture run, compared by the head-noun matcher.
    """
    return compare_a_run(run, meanings(), matcher_name=ComparisonMatcher.HEAD_NOUN)


def test_an_answer_far_from_its_label_in_meaning_is_marked_far_apart(
    compared_by_meaning: LabelComparison,
):
    """
    Decor for a light fixture is beyond anything the matcher reads the wording for.
    """
    one = {one.segment: one for one in compared_by_meaning.objects}["light_fixture_3"]
    assert one.far_apart


def test_an_agreeing_answer_is_not_far_apart(compared_by_meaning: LabelComparison):
    """
    A cabinet answered for a kitchen cabinet is within what the matcher decides.
    """
    one = {one.segment: one for one in compared_by_meaning.objects}["kitchen_cabinet_1"]
    assert not one.far_apart


def test_an_unanswered_object_is_not_far_apart(compared_by_meaning: LabelComparison):
    """
    Nothing was answered, so there is no meaning to be far from the label.
    """
    one = {one.segment: one for one in compared_by_meaning.objects}["sink_4"]
    assert not one.far_apart


def test_the_decided_objects_leave_out_those_far_apart(
    compared_by_meaning: LabelComparison,
):
    """
    Scored only where the matcher can judge: the two cabinets agree and the sink, left
    unanswered, is still missed.
    """
    decided = compared_by_meaning.decided()
    assert [one.segment for one in decided.objects] == [
        "kitchen_cabinet_1",
        "kitchen_cabinet_2",
        "sink_4",
    ]
    assert decided.per_object == ClassificationScores.from_counts(
        true_positives=2, false_positives=0, false_negatives=1
    )


# %% several rooms added up


@pytest.fixture
def two_rooms(compared_by_meaning: LabelComparison) -> ComparedRooms:
    """
    :return: The fixture room twice over, as two rooms of one building.
    """
    return ComparedRooms(comparisons=[compared_by_meaning, compared_by_meaning])


def test_rooms_add_up_their_objects(two_rooms: ComparedRooms):
    """
    Two rooms of four objects, two agreeing in each, and one far apart in each.
    """
    assert (two_rooms.agreed, two_rooms.objects, two_rooms.far_apart) == (4, 8, 2)


def test_a_label_is_counted_across_rooms_before_it_is_scored(
    two_rooms: ComparedRooms,
):
    """
    A label seen in two rooms is one label of the vocabulary, found four times.
    """
    tally = two_rooms.tallies["kitchen cabinet"]
    assert (tally.objects, tally.agreed, tally.false_positives) == (4, 4, 0)


def test_rooms_average_their_scores_over_distinct_labels(
    two_rooms: ComparedRooms, compared_by_meaning: LabelComparison
):
    """
    The same room twice has the same labels in the same proportions, so the average over
    the vocabulary is that room's own.
    """
    assert two_rooms.per_label_average == compared_by_meaning.per_label_average


def test_rooms_score_their_objects_as_one_room_would(
    two_rooms: ComparedRooms, compared_by_meaning: LabelComparison
):
    """
    Per object, twice the counts give the same ratios.
    """
    assert two_rooms.per_object == compared_by_meaning.per_object


def test_decided_rooms_leave_out_every_far_apart_object(two_rooms: ComparedRooms):
    """
    Adding up only what the matcher could judge drops the far-apart light fixtures, and
    with them the label nothing else carried.
    """
    decided = two_rooms.decided()
    assert (decided.objects, decided.far_apart) == (6, 0)
    assert "light fixture" not in decided.tallies


def test_labels_are_grouped_by_how_many_objects_carry_them(two_rooms: ComparedRooms):
    """
    Across both rooms the light fixture and the sink are carried by two objects each,
    the kitchen cabinet by four.
    """
    carried_by_two = two_rooms.labels_carried_by(LabelGroup(fewest=2, most=2))
    assert sorted(one.label for one in carried_by_two) == ["light fixture", "sink"]


def test_the_most_missed_labels_come_first(two_rooms: ComparedRooms):
    """
    The kitchen cabinet is never missed, so it comes last.
    """
    assert two_rooms.most_missed(count=3)[-1].label == "kitchen cabinet"


def test_a_group_with_no_most_holds_every_label_from_its_fewest(
    two_rooms: ComparedRooms,
):
    """
    The last group of a summary has no upper end, and still holds the most common label.
    """
    assert [one.label for one in two_rooms.labels_carried_by(LabelGroup(fewest=3))] == [
        "kitchen cabinet"
    ]


def test_a_group_with_no_most_is_named_as_open():
    """
    A range with no upper end is read as one, not as a large number.
    """
    assert LabelGroup(fewest=21).name == "21 or more"


# %% the page adding every run up


def test_the_summary_scores_each_building_and_all_of_them(two_rooms: ComparedRooms):
    """
    Every building gets a row, and so does everything together.
    """
    page = ScoreSummary(
        matcher=ComparisonMatcher.HEAD_NOUN, rooms=two_rooms
    ).as_markdown()
    assert "| 00808-y9hTuugGdiq |" in page
    assert "| all |" in page


def test_the_summary_names_its_matcher(two_rooms: ComparedRooms):
    """
    Two summaries of one set of runs differ by matcher, so each says which it is.
    """
    page = ScoreSummary(
        matcher=ComparisonMatcher.HEAD_NOUN, rooms=two_rooms
    ).as_markdown()
    assert ComparisonMatcher.HEAD_NOUN.value in page.splitlines()[0]
