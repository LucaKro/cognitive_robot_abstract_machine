"""
Asking what a proposed class is a kind of, as a question of its own.

A run composes a class by naming it and its superclass in one answer, alongside the
mixins and a one-sentence reason, and the names come out stable while the superclasses
do not: ``Tap`` was composed with ``HasHandle, HasMechanicalJoint`` in five runs and
with ``Aperture`` in a sixth, and ``SoapDispenser`` has been a ``Furniture``, an
``IsStorageSpace`` and a ``Cabinet``. Telling a model not to do that did not work, even
where the prompt named the exact pair it then composed.

So the subsumption is asked on its own, once per class a run wants, after the name is
settled and before the class is written.
"""

from __future__ import annotations

from experiments.warsaw.pipeline.steps.compose.step import SuperclassChoice
from experiments.warsaw.pipeline.settings import PipelineSettings

TAXONOMY = {
    "root_name": "SemanticAnnotation",
    "classes": [
        {"name": "Furniture", "bases": ["HasRootBody"], "abstract": True},
        {"name": "Cabinet", "bases": ["Furniture"]},
        {"name": "Aperture", "bases": ["HasRootRegion"]},
    ],
    "part_whole_mixins": [{"name": "HasHandle", "introduces": []}],
}
"""
The least a question of this kind will take.
"""


def asking(name: str = "Stool", proposed: list | None = None) -> SuperclassChoice:
    """
    :param name: The class a run wants.
    :param proposed: What an earlier step proposed to build it from.
    :return: The question that settles what it is a kind of.
    """
    return SuperclassChoice(
        taxonomy=TAXONOMY,
        class_name=name,
        proposed_bases=["Furniture"] if proposed is None else proposed,
        labels=["stool"],
    )


# %% the run says whether it asks


def test_a_run_does_not_ask_unless_told_to() -> None:
    """
    Off by default, so every run already made stays comparable and none of them pays for
    a call it never made.
    """
    assert PipelineSettings().settle_the_superclass is False


# %% what is put to the model


def test_the_question_is_about_one_class() -> None:
    """
    One question per class a run wants, not per body and not per label, since what is
    being settled is a property of the class.
    """
    assert asking().key == "Stool"


def test_the_message_names_the_class_and_what_was_proposed() -> None:
    """
    The earlier step's proposal is shown rather than hidden: it is usually right, and
    the question is whether it is, not what to invent instead.
    """
    [said] = asking().message()
    assert "Stool" in said.text
    assert "Furniture" in said.text


def test_the_message_carries_the_label_the_class_came_from() -> None:
    """
    What the annotator called the objects is the only evidence of what they are, since
    this question shows no pictures.
    """
    [said] = asking().message()
    assert "stool" in said.text


def test_the_question_puts_no_pictures() -> None:
    """
    Subsumption is a question about two names.

    Rendering for it would undo the saving that makes asking it affordable.
    """
    assert asking().shows_pictures is False
