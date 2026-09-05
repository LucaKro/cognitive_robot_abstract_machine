"""
Applying the answers to the geometry: who keeps which faces, and what stays mountable.

Every decision was made once, at the grain it was asked at -- ownership per class
pattern, membership per part -- and this is where those answers meet the particular
objects of one room. A group the answers do not reach is reported rather than guessed
at, because the alternative is a body that quietly stops existing.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.warsaw.pipeline.records import (
    Adjudications,
    ClaimantSet,
    ForcedMembership,
    MembershipAnswer,
    OwnershipAnswer,
)
from experiments.warsaw.pipeline.run import Run
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.split import SplitScene
from experiments.warsaw.segment_relations import ClaimantGroup
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


@pytest.fixture
def step(tmp_path):
    """
    :return: A split step, for the decisions it applies rather than the world it builds.
    """
    return SplitScene(settings=PipelineSettings(), run=Run(directory=tmp_path))


# %% who a set of contested faces is given to


def test_a_set_the_ontology_settles_is_given_to_the_part(step, known):
    """
    A cabinet can hold a drawer, so a face both claim is the drawer's surface, and
    nobody had to be asked.
    """
    group = ClaimantGroup(names=("cabinet_1", "drawer_1"), faces=np.array([1, 2]))
    ownerships, unreached = step.resolve_owners(
        [group],
        Adjudications(
            model="",
            scene="",
            settled=[ClaimantSet(claimants=("cabinet_1", "drawer_1"), faces=2)],
        ),
        {"cabinet_1": "cabinet", "drawer_1": "drawer"},
        {"cabinet": known["Cabinet"], "drawer": known["Drawer"]},
    )
    assert unreached == []
    assert ownerships[0].owner == "drawer_1"
    assert ownerships[0].settled_by_ontology


def test_a_set_the_ontology_leaves_open_is_given_to_the_answered_label(step, known):
    """
    One answer per class pattern is applied everywhere that pattern occurs.
    """
    group = ClaimantGroup(names=("cabinet_8", "drawer_5"), faces=np.array([1]))
    ownerships, unreached = step.resolve_owners(
        [group],
        Adjudications(
            model="",
            scene="",
            ownership=[
                OwnershipAnswer(
                    name="cabinet__drawer",
                    pattern=["cabinet", "drawer"],
                    owner="drawer",
                )
            ],
        ),
        {"cabinet_8": "cabinet", "drawer_5": "drawer"},
        {"cabinet": known["Cabinet"], "drawer": known["Drawer"]},
    )
    assert unreached == []
    assert ownerships[0].owner == "drawer_5"
    assert not ownerships[0].settled_by_ontology


def test_a_set_no_answer_reaches_is_reported_rather_than_guessed_at(step, known):
    """
    Picking one anyway would empty an object nobody decided to empty.
    """
    group = ClaimantGroup(names=("cabinet_8", "wall_2"), faces=np.array([1]))
    ownerships, unreached = step.resolve_owners(
        [group],
        Adjudications(model="", scene=""),
        {"cabinet_8": "cabinet", "wall_2": "wall"},
        {"cabinet": known["Cabinet"], "wall": known["Wall"]},
    )
    assert ownerships == []
    assert unreached == [group]


def test_an_answer_naming_a_label_two_objects_carry_reaches_neither(step, known):
    """
    An answer says *which label* the faces belong to, and where two objects of that
    label both claim them it does not say which object.
    """
    group = ClaimantGroup(
        names=("drawer_5", "drawer_6", "handle_1"), faces=np.array([1])
    )
    ownerships, unreached = step.resolve_owners(
        [group],
        Adjudications(
            model="",
            scene="",
            ownership=[
                OwnershipAnswer(
                    name="drawer__drawer__handle",
                    pattern=["drawer", "drawer", "handle"],
                    owner="drawer",
                )
            ],
        ),
        {"drawer_5": "drawer", "drawer_6": "drawer", "handle_1": "handle"},
        {"drawer": known["Drawer"], "handle": known["Handle"]},
    )
    assert ownerships == []
    assert unreached == [group]


# %% the mounts carried out of the answers


def test_a_part_with_one_candidate_is_mounted_without_being_asked_about(step):
    """
    A drawer that meets exactly one cabinet has nothing to choose between.
    """
    carried = step.named_pairings(
        Adjudications(
            model="",
            scene="",
            forced=[
                ForcedMembership(part="door_11", whole="cabinet_5", field_name="doors")
            ],
        )
    )
    assert [(one.whole, one.part, one.field_name) for one in carried] == [
        ("cabinet_5", "door_11", "doors")
    ]


def test_an_answered_membership_is_mounted_into_the_field_it_was_measured_for(step):
    """
    Which field holds the part was measured when the candidates were gathered; the
    answer only says which of them it is.
    """
    carried = step.named_pairings(
        Adjudications(
            model="",
            scene="",
            forced=[
                ForcedMembership(part="door_10", whole="cabinet_4", field_name="doors")
            ],
            membership=[
                MembershipAnswer(name="door_10", part="door_10", whole="cabinet_4")
            ],
        )
    )
    answered = carried[-1]
    assert (answered.whole, answered.part, answered.field_name) == (
        "cabinet_4",
        "door_10",
        "doors",
    )


def test_a_membership_nobody_answered_is_not_mounted(step):
    """
    An unanswered question leaves the part where the split put it.
    """
    carried = step.named_pairings(
        Adjudications(
            model="",
            scene="",
            membership=[MembershipAnswer(name="door_10", part="door_10", whole=None)],
        )
    )
    assert carried == []


# %% against what a real run decided


def test_a_real_run_s_answers_reach_every_pattern_it_asked_about(adjudications):
    """
    An ownership answer is looked up by its pattern, so a pattern that answers to no
    lookup is an answer the split silently never applies.
    """
    for answer in adjudications.ownership:
        assert adjudications.owner_by_pattern[tuple(answer.pattern)] == answer.owner


def test_a_real_run_s_pairings_are_the_forced_ones_and_the_answered_ones(
    step, adjudications, split
):
    """
    Both come out of the same file and both have to reach the mounting step.
    """
    carried = step.named_pairings(adjudications)
    answered = [one for one in adjudications.membership if one.whole]
    assert len(carried) == len(adjudications.forced) + len(answered)
    assert {(one.whole, one.part) for one in split.pairings} <= {
        (one.whole, one.part) for one in carried
    }
