"""
Which whole each part of a run ended up in, and why some ended up in none.

Counting relations by kind says whether a run builds the right sort of world. It cannot
say that this drawer went into the wrong cabinet, nor that a drawer went into no cabinet
at all because the cabinet stopped existing. Both are read here per object, so a number
that is too low can be taken apart into the reasons it is low.
"""

from __future__ import annotations

from experiments.warsaw.evaluation.parenthood import (
    MissingParentReason,
    ParenthoodComparison,
)
from experiments.warsaw.evaluation.structure import RelationPattern
from experiments.warsaw.pipeline.records import (
    EmptiedSegment,
    RefusedMount,
    SplitRecord,
    TakenFaces,
)
from experiments.warsaw.scene_split import Pairing

from .test_structure import Object, Relation

# %% a modelled world that holds drawers in cabinets

MODELLED_OBJECTS = [
    Object("cabinet", "Cabinet"),
    Object("drawer", "Drawer"),
    Object("handle", "Handle"),
]
"""
The smallest world that attests a cabinet holding a drawer and a drawer holding a
handle.
"""

MODELLED_RELATIONS = [
    Relation("cabinet", "drawer", field_name="drawers"),
    Relation("drawer", "handle", field_name="handle"),
]
"""
The two kinds of relation that world holds.
"""


def comparison(objects, relations, split=None) -> ParenthoodComparison:
    """
    Judge a run against the modelled world above.
    """
    return ParenthoodComparison.between(
        predicted_objects=objects,
        predicted_relations=relations,
        modelled_objects=MODELLED_OBJECTS,
        modelled_relations=MODELLED_RELATIONS,
        split=split or SplitRecord(scene="a kitchen"),
    )


# %% the relations a run did assert


def test_a_relation_of_a_kind_the_modelled_world_holds_is_marked_as_modelled():
    """
    The kind is what can be checked without knowing which cabinet is which.
    """
    judged = comparison(
        [Object("cabinet_1", "Cabinet"), Object("drawer_1", "Drawer")],
        [Relation("cabinet_1", "drawer_1", field_name="drawers")],
    )

    assert [one.modelled for one in judged.relations] == [True]


def test_a_relation_the_modelled_world_never_holds_names_the_objects_at_its_ends():
    """
    A pattern the modelled world lacks is where to go and look, so the two objects have
    to be named rather than only the pattern counted.
    """
    judged = comparison(
        [Object("island_1", "KitchenIsland"), Object("drawer_1", "Drawer")],
        [Relation("island_1", "drawer_1", field_name="drawers")],
    )

    only = judged.relations[0]
    assert only.modelled is False
    assert (only.whole, only.part) == ("island_1", "drawer_1")
    assert only.pattern == RelationPattern(
        whole_class="KitchenIsland",
        field_name="drawers",
        part_class="Drawer",
        relation="part",
    )


# %% the parts that were given no whole


def test_a_part_that_emptied_the_whole_it_belonged_to_is_attributed_to_the_split():
    """
    A drawer that took every face of its own cabinet leaves the cabinet with nothing, so
    the cabinet is dropped and the drawer has nothing left to go into.

    The relation was never wrong: it stopped being expressible.
    """
    split = SplitRecord(
        scene="a kitchen",
        emptied=[
            EmptiedSegment(
                name="cabinet_16", taken_by=[TakenFaces(name="drawer_10", faces=193)]
            )
        ],
    )

    judged = comparison([Object("drawer_10", "Drawer")], [], split)

    only = judged.missing[0]
    assert only.reason is MissingParentReason.LOST_IN_THE_SPLIT
    assert only.whole == "cabinet_16"


def test_the_whole_a_part_is_said_to_have_lost_is_the_one_it_took_most_of():
    """
    A part may have taken faces from more than one object that vanished.

    The one it consumed is the one it stood in for; taking thirty faces of something is
    not that.
    """
    split = SplitRecord(
        scene="a kitchen",
        emptied=[
            EmptiedSegment(
                name="cabinet_26",
                taken_by=[
                    TakenFaces(name="drawer_17", faces=363),
                    TakenFaces(name="drawer_6", faces=32),
                ],
            ),
            EmptiedSegment(
                name="cabinet_9", taken_by=[TakenFaces(name="drawer_6", faces=403)]
            ),
        ],
    )

    judged = comparison([Object("drawer_6", "Drawer")], [], split)

    assert judged.missing[0].whole == "cabinet_9"


def test_a_part_a_mount_was_refused_for_is_told_apart_from_one_never_offered():
    """
    A refusal is the world saying no to a relation the run had decided on, which is a
    different thing to fix from nothing ever proposing one.
    """
    split = SplitRecord(
        scene="a kitchen",
        refused=[
            RefusedMount(
                pairing=Pairing(
                    whole="cabinet_1", part="drawer_1", field_name="drawers"
                ),
                reason="a cabinet does not hold a drawer that way",
            )
        ],
    )

    judged = comparison([Object("drawer_1", "Drawer")], [], split)

    assert judged.missing[0].reason is MissingParentReason.MOUNT_REFUSED
    assert judged.missing[0].whole == "cabinet_1"


def test_a_part_offered_a_whole_that_nothing_mounted_is_told_apart_from_one_never_offered():
    """
    A pairing carried past the split that no relation came of is a step that dropped it,
    which is not the same as the split never proposing one.
    """
    split = SplitRecord(
        scene="a kitchen",
        pairings=[Pairing(whole="cabinet_1", part="drawer_1", field_name="drawers")],
    )

    judged = comparison([Object("drawer_1", "Drawer")], [], split)

    assert judged.missing[0].reason is MissingParentReason.NOT_CHOSEN


def test_a_part_nothing_ever_proposed_a_whole_for_says_exactly_that():
    """
    The commonest miss, and the one that says to look at the measuring rather than at
    anything the model answered.
    """
    judged = comparison([Object("drawer_1", "Drawer")], [])

    assert judged.missing[0].reason is MissingParentReason.NEVER_OFFERED


# %% what is not a part at all


def test_an_object_the_modelled_world_never_holds_as_a_part_is_not_missing_a_whole():
    """
    A mug on a worktop belongs to nothing, and counting it as a missing relation would
    charge the run for a relation that does not exist.
    """
    judged = comparison([Object("mug_1", "Mug")], [])

    assert judged.missing == []


def test_a_part_that_has_a_whole_is_not_also_counted_as_missing_one():
    """
    Every part is counted once, so the two halves add up to the parts there are.
    """
    judged = comparison(
        [Object("cabinet_1", "Cabinet"), Object("drawer_1", "Drawer")],
        [Relation("cabinet_1", "drawer_1", field_name="drawers")],
    )

    assert judged.missing == []
    assert len(judged.relations) == 1


# %% reading it back


def test_a_comparison_reads_back_as_what_was_written():
    """
    The numbers are written beside the run and read by whatever reports on it.
    """
    judged = comparison(
        [Object("cabinet_1", "Cabinet"), Object("drawer_1", "Drawer")],
        [Relation("cabinet_1", "drawer_1", field_name="drawers")],
    )

    assert ParenthoodComparison.from_json(judged.to_json()) == judged
