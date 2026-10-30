"""
Comparing what two graphs are made of without deciding which object is which.

What matters here is that the comparison is said in classes rather than in objects, so
two scenes can be set beside each other without pairing anything up, and that what it
cannot see stays visible: relations in the wrong places count the same as relations in
the right ones.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, Optional

from experiments.warsaw.evaluation.structure import (
    UNCLASSIFIED,
    UNNAMED_FIELD,
    CountedPattern,
    RelationPattern,
    StructuralComparison,
    Tally,
)

# %% two small graphs


@dataclass(frozen=True)
class Object:
    """
    One object of either graph.
    """

    name: str
    """
    What identifies it within its own graph.
    """

    semantic_class: Optional[str] = None
    """
    What it stands as, or nothing where its graph names it nothing.
    """

    @property
    def classes(self) -> List[str]:
        """
        :return: The classes it stands as.
        """
        return [] if self.semantic_class is None else [self.semantic_class]


@dataclass(frozen=True)
class Relation:
    """
    One relation of either graph.
    """

    whole: str
    """
    The object at the holding end.
    """

    part: str
    """
    The object at the held end.
    """

    field_name: str = "drawers"
    """
    The ontology field it is held in.
    """

    relation: str = "part"
    """
    What the relation means.
    """


def cabinet_with_two_drawers(prefix: str) -> tuple:
    """
    A cabinet holding two drawers, as either graph would hold it.
    """
    objects = [
        Object(f"{prefix}_cabinet", "Cabinet"),
        Object(f"{prefix}_drawer_1", "Drawer"),
        Object(f"{prefix}_drawer_2", "Drawer"),
    ]
    relations = [
        Relation(f"{prefix}_cabinet", f"{prefix}_drawer_1"),
        Relation(f"{prefix}_cabinet", f"{prefix}_drawer_2"),
    ]
    return objects, relations


# %% counting what each graph holds


def test_a_relation_is_counted_by_the_classes_at_its_ends():
    """
    Said in classes, two graphs are comparable without pairing up their cabinets.
    """
    predicted, predicted_relations = cabinet_with_two_drawers("run")
    modelled, modelled_relations = cabinet_with_two_drawers("apartment")

    comparison = StructuralComparison.between(
        predicted, predicted_relations, modelled, modelled_relations
    )

    assert comparison.relations == [
        CountedPattern(
            pattern=RelationPattern(
                whole_class="Cabinet",
                field_name="drawers",
                part_class="Drawer",
                relation="part",
            ),
            counted=Tally(modelled=2, predicted=2),
        )
    ]


def test_objects_of_each_class_are_counted_in_both_graphs():
    """
    The class counts are the other half of what a graph is made of.
    """
    predicted, predicted_relations = cabinet_with_two_drawers("run")
    modelled, modelled_relations = cabinet_with_two_drawers("apartment")
    predicted = predicted + [Object("run_drawer_3", "Drawer")]

    comparison = StructuralComparison.between(
        predicted, predicted_relations, modelled, modelled_relations
    )

    assert comparison.classes["Drawer"] == Tally(modelled=2, predicted=3)
    assert comparison.classes["Cabinet"] == Tally(modelled=1, predicted=1)


def test_an_object_neither_graph_named_is_counted_rather_than_dropped():
    """
    A body nobody classified is a finding about the run, not something to leave out.
    """
    comparison = StructuralComparison.between([Object("run_body_1")], [], [], [])

    assert comparison.classes[UNCLASSIFIED] == Tally(modelled=0, predicted=1)


def test_a_relation_whose_field_was_never_recorded_is_kept_apart_from_named_ones():
    """
    Counting it as one that named a field would credit the run with agreeing where it
    never said anything.
    """
    comparison = StructuralComparison.between(
        [Object("run_cabinet", "Cabinet"), Object("run_drawer", "Drawer")],
        [Relation("run_cabinet", "run_drawer", field_name="")],
        [Object("apartment_cabinet", "Cabinet"), Object("apartment_drawer", "Drawer")],
        [Relation("apartment_cabinet", "apartment_drawer", field_name="drawers")],
    )

    assert [row.pattern.field_name for row in comparison.relations] == [
        UNNAMED_FIELD,
        "drawers",
    ]
    assert all(row.counted.agreeing == 0 for row in comparison.relations)


# %% what the counts come to


def test_a_graph_is_credited_only_for_as_many_as_the_other_holds():
    """
    Building nine of something the modelled world holds seven of is seven agreed and two
    over, not nine agreed.
    """
    counted = Tally(modelled=7, predicted=9)

    assert counted.agreeing == 7
    assert counted.recall == 1.0
    assert counted.precision == 7 / 9


def test_nothing_expected_and_nothing_built_is_not_a_division_by_zero():
    """
    Agreeing about nothing is vacuous rather than wrong, and must not raise.
    """
    counted = Tally(modelled=0, predicted=0)

    assert counted.agreeing == 0
    assert counted.precision == 1.0
    assert counted.recall == 1.0


def test_the_patterns_only_one_graph_holds_are_reported_apart():
    """
    A structure one graph builds and the other never does is where a disagreement shows
    most plainly, so it is worth asking for on its own.
    """
    comparison = StructuralComparison.between(
        [Object("run_door", "Door"), Object("run_handle", "Handle")],
        [Relation("run_door", "run_handle", field_name="handle")],
        [Object("apartment_cabinet", "Cabinet"), Object("apartment_drawer", "Drawer")],
        [Relation("apartment_cabinet", "apartment_drawer", field_name="drawers")],
    )

    only_one = comparison.relations_only_one_graph_holds()

    assert {str(row.pattern) for row in only_one} == {
        "Door --handle--> Handle",
        "Cabinet --drawers--> Drawer",
    }


def test_the_totals_say_how_far_the_two_graphs_agree_overall():
    """
    The headline number, which the per-pattern rows explain.
    """
    predicted, predicted_relations = cabinet_with_two_drawers("run")
    modelled, modelled_relations = cabinet_with_two_drawers("apartment")

    comparison = StructuralComparison.between(
        predicted,
        predicted_relations + [Relation("run_cabinet", "run_drawer_1")],
        modelled,
        modelled_relations,
    )

    assert comparison.relation_totals == Tally(modelled=2, predicted=3)
    assert comparison.class_totals == Tally(modelled=3, predicted=3)


def test_a_comparison_reads_back_as_what_was_written():
    """
    It is written beside the run it was made from and read again to build a report.
    """
    predicted, predicted_relations = cabinet_with_two_drawers("run")
    modelled, modelled_relations = cabinet_with_two_drawers("apartment")

    comparison = StructuralComparison.between(
        predicted, predicted_relations, modelled, modelled_relations
    )

    assert StructuralComparison.from_json(comparison.to_json()) == comparison
