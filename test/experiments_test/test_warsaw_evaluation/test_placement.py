"""
Bringing both worlds into one frame so their objects can be paired by where they are.

Class and size barely separate one drawer of a run of drawers from the next, so where
each sits is what decides which is which. That evidence only exists once the two frames
are related, and it has to survive the reconstruction not being metric.
"""

from __future__ import annotations

import numpy as np
import trimesh

from experiments.warsaw.evaluation.ground_truth import GroundTruthGraph, GroundTruthNode
from experiments.warsaw.evaluation.placement import (
    PlacedObject,
    placed_modelled_objects,
)

# %% a modelled world to place


def modelled_node(name: str, bounds: list[list[float]] | None) -> GroundTruthNode:
    """
    One object of a modelled world, spanning what the export recorded.
    """
    return GroundTruthNode(
        name=name,
        source_id=name,
        parent=None,
        semantic_classes=["Drawer"],
        faces=12,
        world_transform=np.eye(4).tolist(),
        bounds=bounds,
    )


# %% where an object is


def test_an_object_sits_in_the_middle_of_what_it_spans():
    """
    The middle of the box rather than of the surface: a scan covers the front of a
    cabinet far more densely than its sides, so a centroid would follow the sampling.
    """
    lopsided = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    lopsided.apply_translation((5.0, 0.0, 0.5))

    placed = PlacedObject.of("drawer_1", ["Drawer"], lopsided)

    assert placed.centre == [5.0, 0.0, 0.5]


def test_an_object_with_no_geometry_is_placed_nowhere():
    """
    A body the split left empty has no position, and inventing one would pair it with
    whatever happens to sit at the origin.
    """
    placed = PlacedObject.of("drawer_1", ["Drawer"], None)

    assert placed.centre is None
    assert placed.size is None


def test_placing_an_object_measures_it_where_it_now_stands():
    """
    The transform that brings a run into the modelled frame carries the scale, so an
    object placed through it is in metres however the scan happened to be scaled.
    """
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    box.apply_transform(np.diag([0.5, 0.5, 0.5, 1.0]))

    placed = PlacedObject.of("drawer_1", ["Drawer"], box)

    assert placed.size.extents == [1.0, 1.0, 1.0]


# %% a modelled world


def test_modelled_objects_are_placed_where_the_export_recorded_them():
    """
    The modelled world is already in the frame both are compared in, so it moves for
    nothing.
    """
    graph = GroundTruthGraph(
        scene="apartment",
        frame="apartment/root",
        geometry_source="visual",
        nodes=[modelled_node("apartment/drawer_a", [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])],
    )

    [placed] = placed_modelled_objects(graph)

    assert placed.centre == [0.5, 1.0, 1.5]


def test_a_modelled_object_that_spans_nothing_is_placed_nowhere():
    """
    A world's root carries no geometry, and it is not a thing a run can be paired with.
    """
    graph = GroundTruthGraph(
        scene="apartment",
        frame="apartment/root",
        geometry_source="visual",
        nodes=[modelled_node("apartment/root", None)],
    )

    [placed] = placed_modelled_objects(graph)

    assert placed.centre is None


def test_an_object_carries_what_it_spans_and_not_only_where_its_middle_is():
    """
    A scan sees the front of a cabinet while the modelled world is a solid box, so how
    near the two are has to be measured against the box rather than between the middles.
    """
    box = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    box.apply_translation((5.0, 0.0, 0.5))

    placed = PlacedObject.of("cabinet_1", ["Cabinet"], box)

    assert placed.bounds == [[4.0, -0.5, 0.0], [6.0, 0.5, 1.0]]
