"""
Showing the classification step the classes an earlier step of the same run proposed.

The ontology a run reads out is the one that is committed, and it is the same for every
run. A class the vocabulary step proposed exists only in this run until its last step
generates it, so a body was described as "labelled kitchen_island, which was read as
KitchenIsland" while KitchenIsland appeared nowhere in the ontology the model was being
asked to choose a name from -- and naming an existing class instead was then the only
coherent answer left. Once it was CounterTop, which accepts no drawer, and the island's
eight drawers could not be mounted.

Widening the ontology the model is shown is what makes the proposal a choice. What must
not happen is the widening reaching the file every run reads.
"""

from __future__ import annotations

import json

import pytest

from experiments.warsaw.pipeline.label_classes import VocabularyClasses
from experiments.warsaw.pipeline.records import LabelAnswer, Vocabulary
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


def widening(labels, known, taxonomy):
    """
    :param labels: What was answered about each label.
    :param known: The ontology's classes by name.
    :param taxonomy: The ontology as it was read out.
    :return: The ontology as the classification step is shown it.
    """
    return VocabularyClasses(
        vocabulary=Vocabulary(model="", scene="", labels=labels), known=known
    ).widened(taxonomy)


def named(taxonomy) -> set:
    """
    :param taxonomy: An ontology as a model reads it.
    :return: The names of the classes in it.
    """
    return {node["name"] for node in taxonomy["classes"]}


# %% what the widening adds


def test_a_proposal_is_put_beside_the_classes_the_ontology_already_has(known, taxonomy):
    """
    The class the vocabulary step proposed is one the classification step may name.
    """
    widened = widening(
        {
            "kitchen_island": LabelAnswer(
                class_name="KitchenIsland",
                is_new_class=True,
                superclass="Table",
                mixins=["HasDrawers"],
            )
        },
        known,
        taxonomy,
    )
    assert named(widened) - named(taxonomy) == {"KitchenIsland"}


def test_a_proposal_says_which_label_it_was_proposed_for(known, taxonomy):
    """
    Marked as proposed, disagreeing with it stays available: the split can leave a body
    that is no longer what its label said.
    """
    widened = widening(
        {
            "kitchen_island": LabelAnswer(
                class_name="KitchenIsland",
                is_new_class=True,
                superclass="Table",
                mixins=["HasDrawers"],
            )
        },
        known,
        taxonomy,
    )
    added = next(one for one in widened["classes"] if one["name"] == "KitchenIsland")
    assert added["proposed_for_label"] == "kitchen_island"
    assert added["bases"] == ["Table", "HasDrawers"]


def test_a_proposal_carries_what_it_can_hold(known, taxonomy):
    """
    What a class can hold is the reason the composition was made, so it is what the
    model has to be shown before it decides the class fits.
    """
    widened = widening(
        {
            "kitchen_island": LabelAnswer(
                class_name="KitchenIsland",
                is_new_class=True,
                superclass="Table",
                mixins=["HasDrawers"],
            )
        },
        known,
        taxonomy,
    )
    added = next(one for one in widened["classes"] if one["name"] == "KitchenIsland")
    assert any(relation["field"] == "drawers" for relation in added["relations"])


def test_the_model_is_told_a_proposed_class_is_not_in_the_ontology_yet(known, taxonomy):
    """
    A class marked as proposed reads as one of the ontology's own unless it is
    explained.
    """
    widened = widening(
        {"a": LabelAnswer(class_name="Gizmo", is_new_class=True, superclass="Table")},
        known,
        taxonomy,
    )
    assert widened["note"].startswith(taxonomy["note"])
    assert "proposed_for_label" in widened["note"]


# %% what the widening leaves alone


def test_the_ontology_every_run_reads_is_not_changed(known, taxonomy, finished_run):
    """
    The file is read afresh by every run, so widening it in place would put one run's
    proposal into the next run's ontology.
    """
    before = json.loads((finished_run.directory / "taxonomy.json").read_text())
    widening(
        {"a": LabelAnswer(class_name="Gizmo", is_new_class=True, superclass="Table")},
        known,
        taxonomy,
    )
    assert taxonomy == before
    assert json.loads((finished_run.directory / "taxonomy.json").read_text()) == before


def test_a_class_the_ontology_already_has_is_not_added_again(known, taxonomy):
    """
    A proposal naming a class that is already there says nothing new.
    """
    widened = widening(
        {
            "cabinet": LabelAnswer(
                class_name="Cabinet", is_new_class=True, superclass="Furniture"
            )
        },
        known,
        taxonomy,
    )
    assert named(widened) == named(taxonomy)


def test_a_proposal_that_cannot_be_composed_is_left_out(known, taxonomy):
    """
    A class that cannot be built is one no body could be given, so showing it would only
    invite an answer nothing can act on.
    """
    widened = widening(
        {
            "a": LabelAnswer(
                class_name="Gizmo", is_new_class=True, superclass="NotAClass"
            )
        },
        known,
        taxonomy,
    )
    assert widened is taxonomy


def test_an_answer_naming_an_existing_class_adds_nothing(known, taxonomy):
    """
    Only proposals widen the ontology; an ordinary answer names something already in it.
    """
    widened = widening({"cabinet": LabelAnswer(class_name="Cabinet")}, known, taxonomy)
    assert widened is taxonomy


# %% against what a real run proposed


def test_a_real_run_s_proposals_are_exactly_what_is_added(vocabulary, known, taxonomy):
    """
    The run that fixed this proposed three classes and was shown three more than the
    ontology holds.
    """
    widened = VocabularyClasses(vocabulary=vocabulary, known=known).widened(taxonomy)
    assert named(widened) - named(taxonomy) == {
        answer.class_name for answer in vocabulary.proposals.values()
    }
