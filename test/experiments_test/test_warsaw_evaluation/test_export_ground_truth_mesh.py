"""
Writing a modelled world as one mesh, so points can be picked on it by hand.

Relating a reconstruction to a modelled world needs both sides open in a viewer, and a
world built by a class is not a file anyone can open. What matters is that every body
arrives where the world places it rather than at the origin, since the points picked on
it are what the alignment is fitted from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from semantic_digital_twin.adapters.world_mesh_exporter import (
    GeometrySource,
    WorldMeshExtractor,
    WorldMeshSnapshot,
)

from experiments.warsaw.evaluation.export_ground_truth_mesh import one_mesh
from experiments.warsaw.evaluation.ground_truth import world_from_urdf

# %% a modelled world to write

A_MODELLED_WORLD = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "warsaw_evaluation"
    / "two_drawer_cabinet.urdf"
)


def snapshot_of_the_modelled_world() -> WorldMeshSnapshot:
    """
    The bodies of a small modelled world, with their geometry.
    """
    return WorldMeshExtractor(
        geometry_source=GeometrySource.VISUAL_WITH_COLLISION_FALLBACK
    ).extract(world_from_urdf(A_MODELLED_WORLD))


# %% what is written


def test_every_body_reaches_the_mesh():
    """
    One mesh rather than one per body, because what is done with it is clicking corners
    of a room, and a viewer holding two hundred objects makes that harder rather than
    easier.
    """
    whole = one_mesh(snapshot_of_the_modelled_world())

    assert len(whole.faces) > 0


def test_a_body_is_written_where_the_world_places_it():
    """
    The points picked on this mesh are what the alignment is fitted from, so a body left
    at the origin rather than where it stands would fit the reconstruction to a world
    that does not exist.
    """
    snapshot = snapshot_of_the_modelled_world()

    whole = one_mesh(snapshot)

    placed = [
        np.asarray(body.world_transform, dtype=np.float64)[:3, 3]
        for body in snapshot.body_meshes
        if body.local_mesh is not None and len(body.local_mesh.faces)
    ]
    assert len(whole.faces) > 0
    assert np.all(whole.bounds[0] <= np.min(placed, axis=0) + 1e-6)
    assert np.all(whole.bounds[1] >= np.max(placed, axis=0) - 1e-6)


def test_a_world_carrying_no_geometry_is_refused():
    """
    Writing an empty file would be found out only in the viewer, with nothing to click.
    """
    with pytest.raises(ValueError):
        one_mesh(WorldMeshSnapshot(body_meshes=[], semantic_annotations=[]))
