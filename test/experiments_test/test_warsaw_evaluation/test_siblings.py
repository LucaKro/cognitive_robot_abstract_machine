"""
Whether a part ended up in the right whole.

Every other measure counts structure without saying where any of it went: a run that
puts every drawer in the wrong cabinet counts the same as one that puts them all right.
This is the measure that separates the two, and it does it without ever having to decide
which predicted cabinet is which modelled one -- only whether parts that belong together
ended up together.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.warsaw.evaluation.matching import (
    Correspondence,
    HowToCompare,
    MatchCost,
    ObjectCorrespondences,
)
from experiments.warsaw.evaluation.siblings import SiblingAgreement

# %% two worlds to compare


@dataclass(frozen=True)
class Relation:
    """
    One relation of either graph, named by the objects at its ends.
    """

    whole: str
    """
    The object at the holding end.
    """

    part: str
    """
    The object at the held end.
    """

    relation: str = "part"
    """
    What the relation means.
    """

    field_name: str = "drawers"
    """
    The ontology field it is held in.
    """


def paired(*names: str) -> ObjectCorrespondences:
    """
    Pair each named run body with the modelled object of the same name prefixed.

    The pairing itself is settled elsewhere and on other evidence, so it is handed over
    already decided rather than re-derived from sizes this test does not care about.
    """
    return ObjectCorrespondences(
        how_compared=HowToCompare(),
        matched=[
            Correspondence(
                predicted=name,
                modelled=f"apartment/{name}",
                cost=MatchCost(disagreeing_class=0.0, differing_size=0.0),
                better_than_the_next_by=1.0,
            )
            for name in names
        ],
    )


# %% parts that belong together


def test_parts_the_run_kept_together_and_the_model_holds_together_agree():
    """
    The case the measure exists to credit: two drawers of one cabinet, found as two
    drawers of one cabinet.
    """
    agreement = SiblingAgreement.between(
        predicted_relations=[
            Relation(whole="cabinet_1", part="drawer_1"),
            Relation(whole="cabinet_1", part="drawer_2"),
        ],
        modelled_relations=[
            Relation(whole="apartment/cabinet_1", part="apartment/drawer_1"),
            Relation(whole="apartment/cabinet_1", part="apartment/drawer_2"),
        ],
        correspondences=paired("drawer_1", "drawer_2"),
    )

    assert agreement.pairs.in_both == 1
    assert agreement.pairs.precision == 1.0
    assert agreement.pairs.recall == 1.0


def test_a_part_put_with_something_it_does_not_belong_with_is_a_placement_mistake():
    """
    The run putting two drawers in one cabinet that the model holds in two different
    ones is the failure this measure is for, and no count of structure shows it.
    """
    agreement = SiblingAgreement.between(
        predicted_relations=[
            Relation(whole="cabinet_1", part="drawer_1"),
            Relation(whole="cabinet_1", part="drawer_2"),
        ],
        modelled_relations=[
            Relation(whole="apartment/cabinet_1", part="apartment/drawer_1"),
            Relation(whole="apartment/cabinet_2", part="apartment/drawer_2"),
        ],
        correspondences=paired("drawer_1", "drawer_2"),
    )

    assert agreement.pairs.precision == 0.0
    assert [(one.one, one.other) for one in agreement.put_together] == [
        ("drawer_1", "drawer_2")
    ]


def test_a_whole_that_arrived_in_pieces_costs_recall_rather_than_precision():
    """
    Two drawers of one cabinet, found in two cabinets that are each half of it.

    Nothing was put anywhere it does not belong, so precision stands; what is lost is
    that the model holds the two together and the run does not.
    """
    agreement = SiblingAgreement.between(
        predicted_relations=[
            Relation(whole="cabinet_1_left", part="drawer_1"),
            Relation(whole="cabinet_1_right", part="drawer_2"),
        ],
        modelled_relations=[
            Relation(whole="apartment/cabinet_1", part="apartment/drawer_1"),
            Relation(whole="apartment/cabinet_1", part="apartment/drawer_2"),
        ],
        correspondences=paired("drawer_1", "drawer_2"),
    )

    assert agreement.pairs.precision == 1.0
    assert agreement.pairs.recall == 0.0
    assert [(one.one, one.other) for one in agreement.left_apart] == [
        ("drawer_1", "drawer_2")
    ]


# %% what is not counted


def test_a_part_nothing_modelled_was_paired_with_is_not_counted():
    """
    A part the correspondence left unpaired has no modelled counterpart to ask about, so
    counting it would score the run against a whole nobody chose.
    """
    agreement = SiblingAgreement.between(
        predicted_relations=[
            Relation(whole="cabinet_1", part="drawer_1"),
            Relation(whole="cabinet_1", part="fragment_of_drawer_1"),
        ],
        modelled_relations=[
            Relation(whole="apartment/cabinet_1", part="apartment/drawer_1"),
        ],
        correspondences=paired("drawer_1"),
    )

    assert agreement.considered == ["drawer_1"]
    assert agreement.pairs.predicted == 0


def test_a_part_in_no_whole_in_either_world_is_not_counted():
    """
    A cabinet standing on the floor belongs to nothing in either world, and pairing it
    with another such object would be counted as agreement the run was never asked for.
    """
    agreement = SiblingAgreement.between(
        predicted_relations=[Relation(whole="cabinet_1", part="drawer_1")],
        modelled_relations=[
            Relation(whole="apartment/cabinet_1", part="apartment/drawer_1")
        ],
        correspondences=paired("drawer_1", "cabinet_9"),
    )

    assert agreement.considered == ["drawer_1"]


def test_an_agreement_without_a_single_pair_scores_nothing_rather_than_everything():
    """
    A run that related nothing must not read as perfect placement, which is what a
    precision defined as one over an empty set would say.
    """
    agreement = SiblingAgreement.between(
        predicted_relations=[],
        modelled_relations=[
            Relation(whole="apartment/cabinet_1", part="apartment/drawer_1"),
            Relation(whole="apartment/cabinet_1", part="apartment/drawer_2"),
        ],
        correspondences=paired("drawer_1", "drawer_2"),
    )

    assert agreement.pairs.in_both == 0
    assert agreement.pairs.recall == 0.0
    assert agreement.pairs.f_score == 0.0
