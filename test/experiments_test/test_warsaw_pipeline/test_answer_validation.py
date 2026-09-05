"""
Deciding whether an answer can be acted on before anything acts on it.

An answer naming a class that is not in the ontology is worth nothing, and it costs
nothing to say so while the model is still there to be asked again. Left unchecked it
comes back two steps later as an unmapped label or an unmounted part, where it looks
like the scene's fault rather than the answer's.
"""

from __future__ import annotations

import pytest

from experiments.warsaw.pipeline.records import (
    BodyAnswer,
    ClaimantSet,
    LabelAnswer,
    LabelRequest,
    MembershipAnswer,
    MembershipCandidate,
    MembershipQuestion,
    OwnershipAnswer,
    OwnershipQuestion,
)
from experiments.warsaw.pipeline.steps.adjudicate import (
    MembershipDecision,
    OwnershipDecision,
)
from experiments.warsaw.pipeline.steps.classify import BodyGroupQuestion
from experiments.warsaw.pipeline.steps.vocabulary import LabelQuestion
from experiments.warsaw.world_loader import LabelSegment
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

import numpy as np


@pytest.fixture
def known():
    """
    :return: The ontology's classes by name.
    """
    return annotation_classes(SemanticAnnotation)


@pytest.fixture
def label_question(known, taxonomy, tmp_path):
    """
    :return: A question about one label, ready to judge answers to.
    """
    return LabelQuestion(
        label=LabelRequest(label="cabinet", instances=3, exemplar="cabinet_2"),
        every_label=["cabinet", "drawer"],
        taxonomy=taxonomy,
        known=known,
        images=tmp_path,
    )


# %% what a label may be answered with


def test_a_class_of_the_ontology_is_usable(label_question):
    """
    The ordinary answer names something already there.
    """
    assert label_question.problems_with(LabelAnswer(class_name="Cabinet")) == []


def test_a_label_the_ontology_holds_nothing_for_is_usable(label_question):
    """
    Some labels name nothing the ontology should hold, and saying so is an answer.
    """
    assert label_question.problems_with(LabelAnswer(class_name=None)) == []


def test_a_mixin_given_as_a_class_is_refused(label_question):
    """
    A mixin says what a class can hold, not what something is, so an object cannot be
    one.

    This is the answer that left a real run's faucet unmapped.
    """
    problems = label_question.problems_with(LabelAnswer(class_name="HasHandle"))
    assert len(problems) == 1
    assert "HasHandle" in problems[0]


def test_a_class_that_is_neither_in_the_ontology_nor_proposed_is_refused(
    label_question,
):
    """
    A name nobody can look up and nobody proposed reaches no class at all.
    """
    problems = label_question.problems_with(LabelAnswer(class_name="Whatsit"))
    assert len(problems) == 1
    assert "Whatsit" in problems[0]


def test_a_proposal_of_a_class_that_already_exists_is_refused(label_question):
    """
    Proposing a class the ontology has says the model did not find what was in front of
    it.
    """
    problems = label_question.problems_with(
        LabelAnswer(class_name="Cabinet", is_new_class=True, superclass="Furniture")
    )
    assert any("already in the taxonomy" in one for one in problems)


def test_a_proposal_deriving_from_nothing_is_refused(label_question):
    """
    A class cannot be built from a superclass that is not there.
    """
    problems = label_question.problems_with(
        LabelAnswer(class_name="Gizmo", is_new_class=True, superclass="NotAClass")
    )
    assert any("superclass" in one for one in problems)


def test_a_proposal_composed_from_a_name_that_is_no_mixin_is_refused(label_question):
    """
    The mixins decide what the class can hold, so one that does not exist grants
    nothing.
    """
    problems = label_question.problems_with(
        LabelAnswer(
            class_name="Gizmo",
            is_new_class=True,
            superclass="Table",
            mixins=["NotAMixin"],
        )
    )
    assert any("NotAMixin" in one for one in problems)


def test_a_well_formed_proposal_is_usable(label_question):
    """
    The proposal a real run made for its kitchen island.
    """
    assert (
        label_question.problems_with(
            LabelAnswer(
                class_name="KitchenIsland",
                is_new_class=True,
                superclass="Table",
                mixins=["HasDrawers"],
            )
        )
        == []
    )


# %% what an overlap may be answered with


@pytest.fixture
def ownership_decision(tmp_path):
    """
    :return: An ownership question, ready to judge answers to.
    """
    return OwnershipDecision(
        asked=OwnershipQuestion(
            name="cabinet__drawer",
            pattern=["cabinet", "drawer"],
            shown=["cabinet_8", "drawer_5"],
            covers=[ClaimantSet(claimants=("cabinet_8", "drawer_5"), faces=1503)],
        ),
        labels={"cabinet_8": "cabinet", "drawer_5": "drawer"},
        images=tmp_path,
    )


def test_an_owner_from_the_pattern_is_usable(ownership_decision):
    """
    The answer picks one of the labels that meet like this.
    """
    assert (
        ownership_decision.problems_with(
            OwnershipAnswer(name="cabinet__drawer", owner="drawer")
        )
        == []
    )


def test_an_owner_that_was_not_offered_is_refused(ownership_decision):
    """
    An answer naming something else settles nothing, and the split would reach no owner.
    """
    problems = ownership_decision.problems_with(
        OwnershipAnswer(name="cabinet__drawer", owner="kitchen_island")
    )
    assert len(problems) == 1
    assert "kitchen_island" in problems[0]


@pytest.fixture
def membership_decision(tmp_path):
    """
    :return: A membership question, ready to judge answers to.
    """
    return MembershipDecision(
        asked=MembershipQuestion(
            name="door_10",
            part="door_10",
            shown=["door_10", "cabinet_4", "cabinet_20"],
            candidates={
                "cabinet_4": MembershipCandidate("doors", 3789, 5626, 0.0),
                "cabinet_20": MembershipCandidate("doors", 9, 44, 0.0),
            },
        ),
        labels={"door_10": "door", "cabinet_4": "cabinet", "cabinet_20": "cabinet"},
        images=tmp_path,
    )


def test_a_whole_from_the_candidates_is_usable(membership_decision):
    """
    The answer picks one of the objects the part was measured to meet.
    """
    assert (
        membership_decision.problems_with(
            MembershipAnswer(name="door_10", part="door_10", whole="cabinet_4")
        )
        == []
    )


def test_a_whole_that_was_not_offered_is_refused(membership_decision):
    """
    A mount named against an object the part never met has no end to hang from.
    """
    problems = membership_decision.problems_with(
        MembershipAnswer(name="door_10", part="door_10", whole="cabinet_99")
    )
    assert len(problems) == 1
    assert "cabinet_99" in problems[0]


# %% what a group of bodies may be answered with


@pytest.fixture
def body_group(taxonomy):
    """
    :return: A question about two painted bodies, ready to judge answers to.
    """
    from experiments.warsaw.pipeline.records import Vocabulary

    return BodyGroupQuestion(
        index=0,
        group=[
            LabelSegment(class_name="drawer", instance=19, faces=np.array([0, 1])),
            LabelSegment(class_name="handle", instance=23, faces=np.array([2])),
        ],
        colors={},
        images={},
        taxonomy=taxonomy,
        vocabulary=Vocabulary(model="", scene=""),
    )


def test_naming_every_body_in_the_picture_is_usable(body_group):
    """
    The answer names each object once, by the name it was listed under.
    """
    assert (
        body_group.problems_with(
            {
                "drawer_19": BodyAnswer(class_name="Drawer"),
                "handle_23": BodyAnswer(class_name="Handle"),
            }
        )
        == []
    )


def test_a_body_the_answer_says_nothing_about_is_refused(body_group):
    """
    A body left unnamed gets no annotation, and every mount it took part in is dropped.
    """
    problems = body_group.problems_with({"drawer_19": BodyAnswer(class_name="Drawer")})
    assert any("handle_23" in one for one in problems)


def test_a_body_that_was_not_in_the_picture_is_refused(body_group):
    """
    An answer about an object nobody showed cannot reach a body: the names are how a
    mount finds its two ends.
    """
    problems = body_group.problems_with(
        {
            "drawer_19": BodyAnswer(class_name="Drawer"),
            "handle_23": BodyAnswer(class_name="Handle"),
            "sink_1": BodyAnswer(class_name="Sink"),
        }
    )
    assert any("sink_1" in one for one in problems)


def test_a_body_given_no_class_is_refused(body_group):
    """
    A body with no class is one the annotating step will leave alone.
    """
    problems = body_group.problems_with(
        {
            "drawer_19": BodyAnswer(class_name="Drawer"),
            "handle_23": BodyAnswer(class_name=None),
        }
    )
    assert any("handle_23 was given no class" in one for one in problems)
