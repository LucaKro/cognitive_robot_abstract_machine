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
from experiments.warsaw.evaluation.siblings import Depth, SiblingAgreement

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


# %% which of a part's wholes it is judged by


def nested_kitchen() -> list[Relation]:
    """
    A modelled kitchen that nests: a side holds cabinets, and a cabinet holds its
    drawer.
    """
    return [
        Relation(whole="apartment/side_B", part="apartment/cabinet_7"),
        Relation(whole="apartment/side_B", part="apartment/countertop_1"),
        Relation(whole="apartment/cabinet_7", part="apartment/drawer_3"),
    ]


def test_a_part_is_in_the_right_furniture_even_where_the_run_did_not_nest_it():
    """
    The question worth answering is whether a drawer ended up in the island, not whether
    the run reproduced the cabinet between them.

    The run builds the island flat while the modelled world puts the drawer inside a
    cabinet inside the side, so judging by the whole that directly holds each part would
    score that flattening as misplacement.
    """
    agreement = SiblingAgreement.between(
        predicted_relations=[
            Relation(whole="kitchen_island_1", part="cabinet_7"),
            Relation(whole="kitchen_island_1", part="drawer_3"),
        ],
        modelled_relations=nested_kitchen(),
        correspondences=paired("cabinet_7", "drawer_3"),
    )

    assert agreement.depth is Depth.OUTERMOST
    assert agreement.pairs.precision == 1.0
    assert agreement.put_together == []


def test_the_whole_that_directly_holds_a_part_can_still_be_asked_about():
    """
    Whether a single whole arrived split into pieces is a different question from where
    its parts ended up, and it is the one the immediate whole answers.
    """
    agreement = SiblingAgreement.between(
        predicted_relations=[
            Relation(whole="kitchen_island_1", part="cabinet_7"),
            Relation(whole="kitchen_island_1", part="drawer_3"),
        ],
        modelled_relations=nested_kitchen(),
        correspondences=paired("cabinet_7", "drawer_3"),
        depth=Depth.IMMEDIATE,
    )

    assert agreement.pairs.precision == 0.0


def test_a_part_of_something_that_is_itself_a_part_is_judged_by_the_outermost_whole():
    """
    Walking has to reach the top rather than stopping one step up, or a drawer two
    levels down would be judged against its cabinet while a countertop beside it is
    judged against the side.
    """
    agreement = SiblingAgreement.between(
        predicted_relations=[
            Relation(whole="kitchen_island_1", part="drawer_3"),
            Relation(whole="kitchen_island_1", part="countertop_1"),
        ],
        modelled_relations=nested_kitchen(),
        correspondences=paired("drawer_3", "countertop_1"),
    )

    assert agreement.pairs.in_both == 1
