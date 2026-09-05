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

import pytest

from experiments.warsaw.pipeline.label_classes import VocabularyClasses
from experiments.warsaw.pipeline.records import LabelAnswer, Vocabulary
from semantic_digital_twin.semantic_annotations.part_whole import admissible_relations
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


@pytest.fixture
def known():
    """
    :return: The ontology's classes by name.
    """
    return annotation_classes(SemanticAnnotation)


def classes_of(labels, known):
    """
    :param labels: What was answered about each label.
    :param known: The ontology's classes by name.
    :return: Per label, the class it stands for.
    """
    return VocabularyClasses(
        vocabulary=Vocabulary(model="", scene="", labels=labels), known=known
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
    for label, answer in vocabulary.labels.items():
        assert (classes[label] is not None) == answer.is_usable
