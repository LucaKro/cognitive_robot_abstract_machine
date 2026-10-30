"""
How much structure each graph has, per class, without pairing objects up.

Two graphs can use the same classes and the same kinds of relation and still be built
differently: one sink against three, or every drawer held by a cabinet against half of
them held by nothing. Both are visible by counting per class, which needs no
correspondence between the graphs and so can be reported for a scene that has no
alignment.
"""

from __future__ import annotations

from experiments.warsaw.evaluation.composition import CompositionComparison

from .test_structure import Object, Relation

# %% a modelled world and a run of it


MODELLED_OBJECTS = [
    Object("cabinet", "Cabinet"),
    Object("drawer_a", "Drawer"),
    Object("drawer_b", "Drawer"),
    Object("sink", "Sink"),
]
MODELLED_RELATIONS = [
    Relation("cabinet", "drawer_a", field_name="drawers"),
    Relation("cabinet", "drawer_b", field_name="drawers"),
]


def comparison(objects, relations) -> CompositionComparison:
    """
    Compare a run against that world.
    """
    return CompositionComparison.between(
        predicted_objects=objects,
        predicted_relations=relations,
        modelled_objects=MODELLED_OBJECTS,
        modelled_relations=MODELLED_RELATIONS,
    )


# %% how many objects of a class each graph has


def test_a_class_split_into_more_objects_than_modelled_is_visible_as_that():
    """
    Three sinks where the modelled world has one is a segmentation failure that every
    identity-free count of relations would miss.
    """
    judged = comparison(
        [Object(f"sink_{one}", "Sink") for one in range(3)],
        [],
    )

    sink = judged.by_class["Sink"]
    assert (sink.modelled_objects, sink.predicted_objects) == (1, 3)
    assert sink.granularity == 3.0


def test_a_class_neither_graph_divides_differently_has_a_granularity_of_one():
    """
    One for one is the value a reader compares the rest against.
    """
    judged = comparison([Object("sink_1", "Sink")], [])

    assert judged.by_class["Sink"].granularity == 1.0


def test_a_class_the_run_never_found_is_a_granularity_of_zero():
    """
    Finding none of what was modelled is a number a reader wants, not a gap.
    """
    judged = comparison([], [])

    assert judged.by_class["Sink"].predicted_objects == 0
    assert judged.by_class["Sink"].granularity == 0.0


def test_a_class_the_modelled_world_lacks_has_no_granularity_at_all():
    """
    There is nothing to divide, so there is no ratio; any number here would invite a
    comparison against a modelled count of zero.
    """
    judged = comparison([Object("mug_1", "Mug")], [])

    assert judged.by_class["Mug"].granularity is None


# %% how many of them are held by something


def test_the_share_of_a_class_that_is_held_by_something_is_counted_per_graph():
    """
    Whether a drawer ends up in a cabinet is the question the evaluation exists for, and
    the share of drawers that ended up in anything can be read without knowing which
    cabinet each belongs in.
    """
    judged = comparison(
        [
            Object("cabinet_1", "Cabinet"),
            Object("drawer_1", "Drawer"),
            Object("drawer_2", "Drawer"),
        ],
        [Relation("cabinet_1", "drawer_1", field_name="drawers")],
    )

    drawer = judged.by_class["Drawer"]
    assert (drawer.modelled_held, drawer.modelled_objects) == (2, 2)
    assert (drawer.predicted_held, drawer.predicted_objects) == (1, 2)
    assert drawer.modelled_share_held == 1.0
    assert drawer.predicted_share_held == 0.5


def test_a_class_with_no_objects_at_all_has_no_share_held():
    """
    A share of nothing is not zero, and reporting it as zero would read as a class whose
    objects were all left unheld.
    """
    judged = comparison([], [])

    assert judged.by_class["Cabinet"].predicted_share_held is None


# %% reading it back


def test_a_comparison_reads_back_as_what_was_written():
    """
    The numbers are written beside the run and read by whatever reports on them.
    """
    judged = comparison([Object("sink_1", "Sink")], [])

    assert CompositionComparison.from_json(judged.to_json()) == judged


def test_a_class_only_one_graph_has_is_still_reported():
    """
    A class the run invented, or one it never found, is exactly what a reader is looking
    for; dropping either would leave the table agreeing by omission.
    """
    judged = comparison([Object("mug_1", "Mug")], [])

    assert judged.by_class["Mug"].modelled_objects == 0
    assert judged.by_class["Sink"].predicted_objects == 0
