"""
Putting the reconstruction and the modelled world in one scene.

Seeing whether an object is where the model says it is means looking at both at once, in
one frame, with every body still telling you which world it came from and what it is.
"""

from __future__ import annotations

import numpy as np
import trimesh

from experiments.warsaw.evaluation.overlay import (
    MODELLED,
    RECONSTRUCTION,
    overlaid,
)

# %% two small scenes


def scene_of(name: str, at: float) -> trimesh.Scene:
    """
    A scene holding one box whose node is named and whose geometry is not.

    This is the shape a GLB comes back in: the name worth reading is on the node, while
    the geometry under it is called something like ``body_1``. A scene built with the two
    the same would hide the difference.
    """
    scene = trimesh.Scene()
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    scene.add_geometry(
        box,
        node_name=name,
        geom_name="body_1",
        transform=trimesh.transformations.translation_matrix((at, 0, 0)),
    )
    return scene


def halved() -> np.ndarray:
    """
    A transform that halves and shifts, standing for a reconstruction that is not
    metric.
    """
    matrix = np.eye(4)
    matrix[:3, :3] *= 0.5
    matrix[0, 3] = 10.0
    return matrix


# %% what comes out


def test_both_worlds_arrive_and_say_which_they_are():
    """
    A body clicked in the overlay is worth nothing if it does not say whether it is what
    was scanned or what was modelled.
    """
    together = overlaid(scene_of("drawer_1", 0.0), scene_of("cabinet", 0.0), np.eye(4))

    names = set(together.graph.nodes)
    assert f"{RECONSTRUCTION}/drawer_1" in names
    assert f"{MODELLED}/cabinet" in names


def test_the_reconstruction_is_moved_into_the_modelled_frame():
    """
    The whole point: the scan is not metric and sits in its own frame, so it is the one
    that moves.
    """
    together = overlaid(scene_of("drawer_1", 2.0), scene_of("cabinet", 0.0), halved())

    moved = together.geometry[f"{RECONSTRUCTION}/drawer_1"]
    assert np.isclose(moved.bounding_box.extents[0], 0.5)
    assert np.isclose(moved.bounding_box.centroid[0], 11.0)


def test_the_modelled_world_is_left_where_it_is():
    """
    It is the frame everything else is being brought into, so moving it would move the
    answer.
    """
    together = overlaid(scene_of("drawer_1", 2.0), scene_of("cabinet", 3.0), halved())

    kept = together.geometry[f"{MODELLED}/cabinet"]
    assert np.isclose(kept.bounding_box.extents[0], 1.0)
    assert np.isclose(kept.bounding_box.centroid[0], 3.0)


def test_the_two_worlds_are_told_apart_by_more_than_their_names():
    """
    Overlaid geometry is read by eye before it is read by name, so the modelled world is
    given one flat colour against the run's per-class colouring.
    """
    together = overlaid(scene_of("drawer_1", 0.0), scene_of("cabinet", 0.0), np.eye(4))

    painted = together.geometry[f"{MODELLED}/cabinet"].visual.main_color
    assert tuple(painted) != tuple(
        together.geometry[f"{RECONSTRUCTION}/drawer_1"].visual.main_color
    )
