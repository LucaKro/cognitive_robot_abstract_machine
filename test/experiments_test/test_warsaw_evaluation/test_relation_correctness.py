"""
Judging each relation a run asserted, one by one.

Counting how many relations of each kind a run built says nothing about whether any
particular one is right, and agreement on kinds saturates as soon as the two
vocabularies match. What the pipeline claims is that *this* part belongs in *that*
whole, so that is what is judged here, against the modelled world and object by object.

A run that flattens a hierarchy is not wrong about where a part is. The modelled world
puts a drawer inside a cabinet inside a side; a run that puts it straight into the
island has still put it somewhere it really is, so a part counts as correctly placed
when the modelled world holds it anywhere inside the matched whole.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.warsaw.evaluation.matching import (
    Correspondence,
    HowToCompare,
    MatchCost,
    ObjectCorrespondences,
)
from experiments.warsaw.evaluation.relation_correctness import (
    RelationCorrectness,
    RelationVerdict,
)

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
    Pair each named run object with the modelled object of the same name prefixed.
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


def nested_kitchen() -> list[Relation]:
    """
    A modelled kitchen that nests: a side holds a cabinet, and the cabinet holds a
    drawer.
    """
    return [
        Relation(
            whole="apartment/side_B", part="apartment/cabinet_7", field_name="units"
        ),
        Relation(whole="apartment/cabinet_7", part="apartment/drawer_3"),
    ]


# %% a relation that holds


def test_a_relation_the_modelled_world_holds_as_well_is_correct():
    """
    The plain case, and the one the pipeline exists to get right.
    """
    judged = RelationCorrectness.between(
        predicted_relations=[Relation(whole="cabinet_7", part="drawer_3")],
        modelled_relations=nested_kitchen(),
        correspondences=paired("cabinet_7", "drawer_3"),
    )

    assert judged.correct == 1
    assert judged.share_correct == 1.0


def test_a_part_put_somewhere_it_really_is_counts_even_where_a_level_was_skipped():
    """
    A run that hangs a drawer straight off the island rather than off the cabinet inside
    it has still put the drawer somewhere it really is.

    Demanding the immediate whole would score a flattened hierarchy as a wrong relation.
    """
    judged = RelationCorrectness.between(
        predicted_relations=[
            Relation(whole="side_B", part="drawer_3", field_name="units")
        ],
        modelled_relations=nested_kitchen(),
        correspondences=paired("side_B", "drawer_3"),
    )

    assert judged.correct == 1


def test_a_part_put_in_a_whole_it_is_not_inside_is_wrong():
    """
    The failure the measure is for: the part exists, the whole exists, and the modelled
    world does not hold one inside the other.
    """
    judged = RelationCorrectness.between(
        predicted_relations=[Relation(whole="drawer_3", part="cabinet_7")],
        modelled_relations=nested_kitchen(),
        correspondences=paired("cabinet_7", "drawer_3"),
    )

    assert judged.correct == 0
    assert [one.verdict for one in judged.judged] == [RelationVerdict.WRONG_WHOLE]


# %% a relation that cannot be judged


def test_a_relation_with_an_end_nothing_was_paired_with_is_not_judged():
    """
    An end the modelled world has no counterpart for cannot be called right or wrong,
    and counting it either way would score the run against something nobody modelled.
    """
    judged = RelationCorrectness.between(
        predicted_relations=[Relation(whole="cabinet_7", part="fragment_2")],
        modelled_relations=nested_kitchen(),
        correspondences=paired("cabinet_7"),
    )

    assert [one.verdict for one in judged.judged] == [RelationVerdict.NOT_JUDGED]
    assert judged.share_correct is None


def test_what_is_judged_is_counted_apart_from_what_is_asserted():
    """
    The share correct is of the relations that could be judged, so the number of
    relations it was taken over has to be readable beside it or it cannot be compared
    between runs.
    """
    judged = RelationCorrectness.between(
        predicted_relations=[
            Relation(whole="cabinet_7", part="drawer_3"),
            Relation(whole="cabinet_7", part="fragment_2"),
        ],
        modelled_relations=nested_kitchen(),
        correspondences=paired("cabinet_7", "drawer_3"),
    )

    assert judged.asserted == 2
    assert judged.judgeable == 1
    assert judged.share_correct == 1.0


# %% the field a relation was held in


def test_a_relation_held_in_the_field_the_modelled_world_uses_is_grounded():
    """
    Being in the right whole is half of it; the ontology field is what the claim is made
    *in*, so holding a drawer through the field the modelled world holds it through is
    what says the two agree rather than happen to overlap.
    """
    judged = RelationCorrectness.between(
        predicted_relations=[Relation(whole="cabinet_7", part="drawer_3")],
        modelled_relations=nested_kitchen(),
        correspondences=paired("cabinet_7", "drawer_3"),
    )

    assert judged.held_directly == 1
    assert judged.same_field == 1


def test_a_relation_held_in_another_field_than_the_modelled_one_is_not_counted_as_agreeing():
    """
    A part in the right whole through the wrong field is a different claim about it.
    """
    judged = RelationCorrectness.between(
        predicted_relations=[
            Relation(whole="cabinet_7", part="drawer_3", field_name="doors")
        ],
        modelled_relations=nested_kitchen(),
        correspondences=paired("cabinet_7", "drawer_3"),
    )

    assert judged.correct == 1
    assert judged.same_field == 0


# %% two ends that are one object


def test_a_relation_between_two_pieces_of_one_modelled_object_is_not_judged():
    """
    Where a run found one cabinet as several pieces, both ends of a relation between two
    of them stand for the same modelled object.

    The modelled world holds no opinion about the inside of one of its own objects, so
    calling that relation wrong would score the run against something nobody modelled.
    """
    both_are_the_cabinet = ObjectCorrespondences(
        how_compared=HowToCompare(),
        matched=[
            Correspondence(
                predicted=name,
                modelled="apartment/cabinet_7",
                cost=MatchCost(disagreeing_class=0.0, differing_size=0.0),
                better_than_the_next_by=0.0,
            )
            for name in ("piece_1", "piece_2")
        ],
    )

    judged = RelationCorrectness.between(
        predicted_relations=[Relation(whole="piece_1", part="piece_2")],
        modelled_relations=nested_kitchen(),
        correspondences=both_are_the_cabinet,
    )

    assert [one.verdict for one in judged.judged] == [RelationVerdict.ONE_OBJECT]
    assert judged.judgeable == 0
