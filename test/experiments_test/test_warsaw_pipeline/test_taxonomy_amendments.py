"""
Which amendments a scene's measurements raise, and which of them can be written.

An amendment is proposed when two labelled objects were measured to share faces and the
class of the one cannot hold the other. What must not happen is proposing an amendment
for a class that already admits the part, or for a class this run itself proposed --
amending that would mean amending a proposal rather than the ontology.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.semantic_annotations import mixins as ontology_mixins
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from experiments.warsaw.pipeline.records import AmendmentRecord, PairRecord, Relations
from experiments.warsaw.pipeline.steps.amend.step import AmendTaxonomy
from experiments.warsaw.segment_relations import PairEvidence

# %% a measurement two labels share faces in


def sharing(one: str, other: str, shared_faces: int, classes: dict) -> PairRecord:
    """
    :param one: The label of the first object.
    :param other: The label of the second.
    :param shared_faces: How many faces they were measured to share.
    :param classes: Per label, the name of the class it was read as.
    :return: The pair as the measurement step writes one.
    """
    return PairRecord(
        evidence=PairEvidence(
            one=one,
            other=other,
            shared_faces=shared_faces,
            share_of_one=0.5,
            share_of_other=0.5,
            touching_edges=0,
            distance=0.0,
        ),
        classes=classes,
    )


@pytest.fixture
def ontology() -> dict:
    """
    :return: The ontology's classes by name, before anything was composed.
    """
    return annotation_classes(SemanticAnnotation)


@pytest.fixture
def amendable_mixins() -> list:
    """
    :return: The mixins a class can be given.
    """
    return [ontology_mixins.HasDrawers, ontology_mixins.HasDoors]


@pytest.fixture
def step(finished_run) -> AmendTaxonomy:
    """
    :return: The step, which decides what to raise without needing a run to have started.
    """
    return AmendTaxonomy(settings=None, run=finished_run)


# %% what the measurements raise


def test_a_whole_that_cannot_hold_the_part_raises_the_mixin_that_would_let_it(
    step, ontology, amendable_mixins
):
    """
    A countertop measured to share faces with a drawer cannot hold one, and exactly one
    of the mixins would let it.
    """
    relations = Relations(
        scene="",
        pairs=[
            sharing(
                "countertop",
                "drawer",
                30,
                {"countertop": "CounterTop", "drawer": "Drawer"},
            )
        ],
    )
    classes = {"countertop": ontology["CounterTop"], "drawer": ontology["Drawer"]}

    [raised] = step.candidates(relations, classes, ontology, amendable_mixins)

    assert raised.whole == "CounterTop"
    assert raised.mixin == ontology_mixins.HasDrawers.__name__
    assert raised.part == "Drawer"
    assert raised.whole_labels == ["countertop"]
    assert raised.part_labels == ["drawer"]
    assert raised.measured_pairs == 1
    assert raised.shared_faces == 30


def test_a_whole_that_already_admits_the_part_raises_nothing(
    step, ontology, amendable_mixins
):
    """
    A cabinet already holds drawers, so nothing about that pair is in question.
    """
    relations = Relations(
        scene="",
        pairs=[
            sharing("cabinet", "drawer", 30, {"cabinet": "Cabinet", "drawer": "Drawer"})
        ],
    )
    classes = {"cabinet": ontology["Cabinet"], "drawer": ontology["Drawer"]}

    assert step.candidates(relations, classes, ontology, amendable_mixins) == []


def test_a_pair_sharing_no_faces_raises_nothing(step, ontology, amendable_mixins):
    """
    Only an overlap raises an amendment; standing near something is not being part of
    it.
    """
    relations = Relations(
        scene="",
        pairs=[
            sharing(
                "countertop",
                "drawer",
                0,
                {"countertop": "CounterTop", "drawer": "Drawer"},
            )
        ],
    )
    classes = {"countertop": ontology["CounterTop"], "drawer": ontology["Drawer"]}

    assert step.candidates(relations, classes, ontology, amendable_mixins) == []


def test_a_class_this_run_proposed_is_not_amended(step, ontology, amendable_mixins):
    """
    A proposed class was given its mixins when it was proposed, so amending it would be
    amending a proposal rather than the ontology.
    """
    proposed = type("ProposedTop", (ontology["CounterTop"],), {})
    relations = Relations(
        scene="",
        pairs=[
            sharing(
                "island", "drawer", 30, {"island": "ProposedTop", "drawer": "Drawer"}
            )
        ],
    )
    classes = {"island": proposed, "drawer": ontology["Drawer"]}

    assert step.candidates(relations, classes, ontology, amendable_mixins) == []


def test_the_same_amendment_measured_twice_is_gathered_into_one_record(
    step, ontology, amendable_mixins
):
    """
    One record per class, mixin and part, carrying what every measurement contributed.
    """
    relations = Relations(
        scene="",
        pairs=[
            sharing(
                "countertop",
                "drawer",
                30,
                {"countertop": "CounterTop", "drawer": "Drawer"},
            ),
            sharing(
                "counter",
                "drawer_2",
                12,
                {"counter": "CounterTop", "drawer_2": "Drawer"},
            ),
        ],
    )
    classes = {
        "countertop": ontology["CounterTop"],
        "counter": ontology["CounterTop"],
        "drawer": ontology["Drawer"],
        "drawer_2": ontology["Drawer"],
    }

    [raised] = step.candidates(relations, classes, ontology, amendable_mixins)

    assert raised.measured_pairs == 2
    assert raised.shared_faces == 42
    assert raised.whole_labels == ["countertop", "counter"]
    assert raised.part_labels == ["drawer", "drawer_2"]


# %% which of them can be written


def test_an_amendable_class_is_planned_as_the_edit_it_would_make(step, ontology):
    """
    An accepted amendment carries the line it would change, so it can be shown and
    undone.
    """
    judgement = AmendmentRecord(
        whole="CounterTop", mixin="HasDrawers", part="Drawer", amend=True
    )
    accepted = []

    step.plan_edit(judgement, ontology, accepted)

    assert judgement.blocked is None
    assert judgement.edit is not None
    assert judgement.edit.after.endswith(", HasDrawers):")
    assert [record for record, _ in accepted] == [judgement]


def test_a_class_that_already_has_the_mixin_is_blocked_rather_than_planned(
    step, ontology
):
    """
    Nothing to do is recorded as the reason, not as an edit that changes nothing.
    """
    judgement = AmendmentRecord(
        whole="Cabinet", mixin="HasDrawers", part="Drawer", amend=True
    )
    accepted = []

    step.plan_edit(judgement, ontology, accepted)

    assert judgement.blocked == "the class already has that mixin"
    assert judgement.edit is None
    assert accepted == []
