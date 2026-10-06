"""
How far apart a scene's labelled objects are measured to be.

The distance between two objects is the smallest distance between any face centre of one
and any of the other, and every pair a scene reports carries it. It is by far the most
expensive thing a run measures, so it is the thing most likely to be made faster and the
thing that most needs saying what the right answer is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from experiments.warsaw.segment_relations import segment_evidence
from experiments.warsaw.world_loader.loader import WarsawWorldLoader

from .test_warsaw_world_loader import write_scene

# %% a scene of objects standing at known distances


def triangle(corner: np.ndarray) -> np.ndarray:
    """
    :param corner: Where to put the triangle.
    :return: Three vertices of a triangle of its own, so that no two objects of the scene
        share one and nothing is merged on the way in.
    """
    return corner + np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])


@pytest.fixture
def objects_standing_apart(tmp_path) -> Path:
    """
    :return: A directory holding a scene of six objects at unequal spacings, which is
        what makes which of them are nearest a question with one answer.
    """
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [7.0, 7.0, 0.0],
            [0.5, 0.5, 4.0],
        ]
    )
    vertices = np.concatenate([triangle(corner) for corner in corners])
    faces = np.arange(len(vertices)).reshape(-1, 3)
    write_scene(
        tmp_path / "scene.ply",
        vertices,
        faces,
        {"thing": list(range(1, len(corners) + 1))},
    )
    return tmp_path


def smallest_distances(loader: WarsawWorldLoader) -> dict:
    """
    :param loader: The loaded scene.
    :return: Per pair of segment names, the smallest distance between their face
        centres, worked out by comparing every centre with every other.
    """
    centres = loader.scene_mesh.triangles_center
    of_each = {
        str(segment.name): centres[segment.face_indices]
        for segment in loader.label_segments
    }
    return {
        (one, other): float(cdist(of_each[one], of_each[other]).min())
        for one in of_each
        for other in of_each
        if one < other
    }


# %% what every reported pair says


def test_every_pair_carries_the_smallest_distance_between_the_two(
    objects_standing_apart,
):
    """
    Whatever prunes the search, a pair it does report has to carry the distance that
    comparing every face centre with every other one would have found.
    """
    loader = WarsawWorldLoader(input_directory=objects_standing_apart)
    by_hand = smallest_distances(loader)

    measured = segment_evidence(loader, nearest=3)

    assert measured.pairs
    for pair in measured.pairs:
        assert pair.distance == pytest.approx(by_hand[(pair.one, pair.other)])


def test_the_nearest_objects_are_the_ones_reported_as_nearest(objects_standing_apart):
    """
    The search stops early on purpose, so what it must not do is stop before the objects
    that really are nearest have been found.
    """
    loader = WarsawWorldLoader(input_directory=objects_standing_apart)
    by_hand = smallest_distances(loader)
    wanted = 2

    measured = segment_evidence(loader, nearest=wanted)

    for segment in loader.label_segments:
        name = str(segment.name)
        others = sorted(
            (distance, pair) for pair, distance in by_hand.items() if name in pair
        )[:wanted]
        nearest = {one for _, pair in others for one in pair if one != name}
        ranked = {
            pair.other if pair.one == name else pair.one
            for pair in measured.pairs_of(name)
            if (pair.rank_from_one if pair.one == name else pair.rank_from_other)
            is not None
        }
        assert nearest <= ranked
