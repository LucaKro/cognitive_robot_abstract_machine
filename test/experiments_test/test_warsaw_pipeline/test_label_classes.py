"""
Turning what a model said about a label into the class the rest of the run asks
questions of.

An answer is either a class of the ontology or a class proposed by naming a superclass
and mixins. The second does not exist until the run's last step generates it, and what
it can hold is decided by exactly that composition -- so it is composed here rather than
looked up, and every later question about what may hold what is asked of the composed
class.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.warsaw.pipeline.label_classes import VocabularyClasses
from experiments.warsaw.pipeline.records import (
    LabelAnswer,
    PairRecord,
    Relations,
    Vocabulary,
)
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.segment_relations import PairEvidence
from semantic_digital_twin.semantic_annotations.part_whole import admissible_relations

from .test_open_questions import described


def classes_of(labels, known):
    """
    :param labels: What was answered about each label.
    :param known: The ontology's classes by name.
    :return: Per label, the class it stands for.
    """
    return VocabularyClasses(
        vocabulary=Vocabulary(
            model="",
            scene="",
            labels=[replace(answer, label=label) for label, answer in labels.items()],
        ),
        known=known,
    ).by_label()


# %% a label that names a class the ontology already has


def test_a_named_class_is_looked_up(known):
    """
    An answer naming a class of the ontology stands for that class.
    """
    classes = classes_of({"cabinet": LabelAnswer(class_name="Cabinet")}, known)
    assert classes["cabinet"] is known["Cabinet"]


def test_a_label_the_ontology_holds_nothing_for_stands_for_nothing(known):
    """
    A ceiling is a spatial feature the ontology has no class for, and saying so is an
    answer rather than a failure.
    """
    assert (
        classes_of({"ceiling": LabelAnswer(class_name=None)}, known)["ceiling"] is None
    )


def test_an_answer_with_a_problem_leaves_the_label_unmapped(known):
    """
    An answer naming a mixin was reported as unusable when it was given, and acting on
    it anyway would put that mistake into the split.
    """
    classes = classes_of(
        {
            "faucet": LabelAnswer(
                class_name="HasHandle", problems=["HasHandle is a mixin"]
            )
        },
        known,
    )
    assert classes["faucet"] is None


# %% a label that names a class the run proposes


def test_a_proposed_class_is_composed_from_what_the_answer_names(known):
    """
    A proposal does not exist until the run's last step generates it, so it is built
    here from the superclass and mixins the answer named.
    """
    classes = classes_of(
        {
            "kitchen_island": LabelAnswer(
                class_name="KitchenIsland",
                is_new_class=True,
                superclass="Table",
                mixins=["HasDrawers"],
            )
        },
        known,
    )
    composed = classes["kitchen_island"]
    assert composed.__name__ == "KitchenIsland"
    assert issubclass(composed, known["Table"])
    assert issubclass(composed, known["HasDrawers"])


def test_a_proposed_class_holds_what_its_mixins_let_it_hold(known):
    """
    The mixins are what decide whether the drawers overlapping an island can ever be
    mounted into it, which is the whole reason a proposal is composed rather than named.
    """
    with_drawers = classes_of(
        {
            "kitchen_island": LabelAnswer(
                class_name="KitchenIsland",
                is_new_class=True,
                superclass="Table",
                mixins=["HasDrawers"],
            )
        },
        known,
    )["kitchen_island"]
    without = classes_of(
        {
            "kitchen_island": LabelAnswer(
                class_name="KitchenIsland",
                is_new_class=True,
                superclass="Table",
                mixins=[],
            )
        },
        known,
    )["kitchen_island"]

    assert admissible_relations(with_drawers, known["Drawer"])
    assert not admissible_relations(without, known["Drawer"])


def test_a_proposal_naming_a_superclass_the_ontology_lacks_stands_for_nothing(known):
    """
    A class cannot be derived from one that is not there, and guessing at another would
    give the object a class nobody proposed.
    """
    classes = classes_of(
        {
            "gadget": LabelAnswer(
                class_name="Gadget", is_new_class=True, superclass="NotAClass"
            )
        },
        known,
    )
    assert classes["gadget"] is None


def test_a_mixin_the_ontology_lacks_is_left_out_rather_than_guessed_at(known):
    """
    A proposal naming a mixin that is not there is still a class, built from the bases
    that are: the alternative is no class at all for an object the run has already
    measured, and the missing mixin was reported when the answer was given.
    """
    composed = classes_of(
        {
            "kitchen_island": LabelAnswer(
                class_name="KitchenIsland",
                is_new_class=True,
                superclass="Table",
                mixins=["HasDrawers", "NotAMixin"],
            )
        },
        known,
    )["kitchen_island"]
    assert issubclass(composed, known["HasDrawers"])
    assert [base.__name__ for base in composed.__bases__] == ["Table", "HasDrawers"]


# %% against what a real run answered


def test_every_usable_answer_of_a_real_run_becomes_a_class(vocabulary, known):
    """
    Whatever a run mapped is what the split can ask questions of; anything else silently
    stops being an object.
    """
    classes = VocabularyClasses(vocabulary=vocabulary, known=known).by_label()
    for answer in vocabulary.labels:
        assert (classes[answer.label] is not None) == answer.is_usable


# %% the classes a label might just as well be


def alternatives_of(answer: LabelAnswer, known) -> list:
    """
    :return: The other classes a label answered like this might be.
    """
    return VocabularyClasses(
        vocabulary=Vocabulary(
            model="", scene="", labels=[replace(answer, label="cabinet")]
        ),
        known=known,
    ).alternatives_by_label()["cabinet"]


def test_the_alternatives_of_a_label_are_the_classes_of_the_ontology_it_named(known):
    answer = LabelAnswer(class_name="Door", alternatives=["Cabinet", "Cupboardish"])

    assert alternatives_of(answer, known) == [known["Cabinet"]]


def test_an_answer_with_a_problem_has_no_alternatives(known):
    answer = LabelAnswer(
        class_name="Door", alternatives=["Cabinet"], problems=["unusable"]
    )

    assert alternatives_of(answer, known) == []


# %% the readings a label keeps beside what its objects touch


def narrowed(answers: dict, touching: dict, known) -> dict:
    """
    :param answers: What was answered about each label.
    :param touching: Per label, the labels its objects were measured to touch.
    :return: Per label, every class it may still mean, its class first.
    """
    vocabulary = VocabularyClasses(
        vocabulary=Vocabulary(
            model="",
            scene="",
            labels=[replace(answer, label=label) for label, answer in answers.items()],
        ),
        known=known,
    ).narrowed_by_contact(touching)
    return {answer.label: answer.candidates for answer in vocabulary.labels}


def test_a_reading_the_ontology_relates_to_nothing_touched_is_dropped(known):
    """
    A handle base on a door read as a door attachment could neither hang from the door
    nor hold the handle, so only the backplate is left for it to be.
    """
    candidates = narrowed(
        {
            "handle_base": LabelAnswer(
                class_name="Backplate", alternatives=["DoorAttachment"]
            ),
            "door": LabelAnswer(class_name="Door"),
            "handle": LabelAnswer(class_name="Handle"),
        },
        {"handle_base": {"door", "handle"}},
        known,
    )

    assert candidates["handle_base"] == ["Backplate"]


def test_a_first_reading_relating_to_nothing_touched_gives_way_to_an_alternative(
    known,
):
    """
    The order the model gave its readings in does not keep one the ontology cannot
    place.
    """
    candidates = narrowed(
        {
            "handle_base": LabelAnswer(
                class_name="DoorAttachment", alternatives=["Backplate"]
            ),
            "door": LabelAnswer(class_name="Door"),
        },
        {"handle_base": {"door"}},
        known,
    )

    assert candidates["handle_base"] == ["Backplate"]


def test_readings_that_all_relate_to_something_touched_are_all_kept(known):
    """
    A backplate and a frame can both hang from a door, so the choice stays open.
    """
    answer = LabelAnswer(class_name="Backplate", alternatives=["Frame"])

    candidates = narrowed(
        {"base": answer, "door": LabelAnswer(class_name="Door")},
        {"base": {"door"}},
        known,
    )

    assert candidates["base"] == answer.candidates


def test_readings_none_of_which_relates_to_anything_touched_are_all_kept(known):
    """
    Dropping every reading would leave the object nothing to be, so a misread neighbour
    narrows nothing.
    """
    answer = LabelAnswer(class_name="Backplate", alternatives=["DoorAttachment"])

    candidates = narrowed(
        {"base": answer, "ceiling": LabelAnswer(class_name=None)},
        {"base": {"ceiling"}},
        known,
    )

    assert candidates["base"] == answer.candidates


def test_a_label_touching_nothing_keeps_its_readings(known):
    """
    An object measured to touch nothing gives nothing to narrow its readings by.
    """
    answer = LabelAnswer(class_name="Backplate", alternatives=["DoorAttachment"])

    assert narrowed({"base": answer}, {}, known)["base"] == answer.candidates


def apart(one: str, other: str, distance: float) -> PairRecord:
    """
    :return: Two objects measured *distance* apart, touching along no edge.
    """
    return PairRecord(
        evidence=PairEvidence(
            one=one,
            other=other,
            shared_faces=0,
            share_of_one=0.0,
            share_of_other=0.0,
            touching_edges=0,
            distance=distance,
        )
    )


def test_labels_whose_objects_meet_within_the_gap_touch_each_other():
    """
    The wall lies twice the gap away, so it is not counted as touched.
    """
    gap = PipelineSettings().largest_gap_counted_as_touching
    relations = Relations(
        scene="a door",
        segments=[
            described(name, label)
            for name, label in [
                ("handle_base_1", "handle_base"),
                ("door_1", "door"),
                ("wall_1", "wall"),
            ]
        ],
        pairs=[
            apart("handle_base_1", "door_1", gap),
            apart("handle_base_1", "wall_1", 2 * gap),
        ],
    )

    assert relations.touching_labels(gap) == {
        "handle_base": {"door"},
        "door": {"handle_base"},
    }
