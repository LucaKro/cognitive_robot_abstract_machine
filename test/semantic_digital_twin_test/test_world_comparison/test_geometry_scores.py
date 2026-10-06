from copy import deepcopy

import pytest

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_comparison.geometry_scores import (
    GeometryScore,
    GeometryScorer,
    SurfaceNearReconstruction,
    WholeSurface,
)
from semantic_digital_twin.world_comparison.matching import BodyMatcher
from semantic_digital_twin.world_comparison.surface_samples import SurfaceSampler
from .worlds import box, square_face, world_of

# %% fixtures

CUBE_SIDE = 0.3
"""
The edge length of the ground truth cube, in metres.
"""


@pytest.fixture
def matcher() -> BodyMatcher:
    return BodyMatcher(
        distance_tolerance=0.05,
        minimum_overlap=0.5,
        minimum_partial_overlap=0.2,
        sampler=SurfaceSampler(spacing=0.005, seed=0),
    )


@pytest.fixture
def scorer() -> GeometryScorer:
    return GeometryScorer(distance_tolerance=0.02, observed_region=WholeSurface())


@pytest.fixture
def identity() -> HomogeneousTransformationMatrix:
    return HomogeneousTransformationMatrix()


@pytest.fixture
def cube_world():
    return world_of(box("cube", CUBE_SIDE, 0.0))


def score_of_face(
    cube_world, matcher, scorer, identity, offset: float
) -> GeometryScore:
    """
    :return: The score of a scanned front face of the cube, standing the given distance
        in front of the cube's real front face.
    """
    scanned_front = world_of(square_face("front", CUBE_SIDE / 2 + offset, CUBE_SIDE))
    correspondence = matcher.match(cube_world, scanned_front, identity)
    [score] = scorer.score(correspondence).scores
    return score


def strip_area(width: float) -> float:
    """
    :return: The area of the four sides of the cube within the given distance of its
        front face.
    """
    return 4 * CUBE_SIDE * width


# %% a perfect reconstruction


def test_perfect_copy_scores_perfectly(cube_world, matcher, scorer, identity):
    correspondence = matcher.match(cube_world, deepcopy(cube_world), identity)
    [score] = scorer.score(correspondence).scores
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f_score == 1.0
    assert score.mean_distance == pytest.approx(0.0, abs=1e-6)
    assert score.ninetieth_percentile_distance == pytest.approx(0.0, abs=1e-6)
    assert score.observed_share == 1.0


# %% an open scan of one face


def test_face_on_the_surface_is_wholly_precise(cube_world, matcher, scorer, identity):
    score = score_of_face(cube_world, matcher, scorer, identity, offset=0.0)
    assert score.precision == 1.0
    assert score.mean_distance == pytest.approx(0.0, abs=1e-6)


def test_recall_over_the_whole_surface_counts_the_unseen_sides(
    cube_world, matcher, scorer, identity
):
    score = score_of_face(cube_world, matcher, scorer, identity, offset=0.0)
    front_area = CUBE_SIDE**2
    covered = front_area + strip_area(scorer.distance_tolerance)
    assert score.recall == pytest.approx(covered / (6 * front_area), abs=0.01)


def test_recall_near_the_reconstruction_counts_only_the_region_around_it(
    cube_world, matcher, identity
):
    observed_distance = 0.05
    scorer = GeometryScorer(
        distance_tolerance=0.02,
        observed_region=SurfaceNearReconstruction(distance=observed_distance),
    )
    score = score_of_face(cube_world, matcher, scorer, identity, offset=0.0)

    front_area = CUBE_SIDE**2
    observed = front_area + strip_area(observed_distance)
    covered = front_area + strip_area(scorer.distance_tolerance)
    assert score.observed_share == pytest.approx(observed / (6 * front_area), abs=0.01)
    assert score.recall == pytest.approx(covered / observed, abs=0.02)


def test_face_off_the_surface_is_measured_at_its_offset(
    cube_world, matcher, scorer, identity
):
    offset = 0.03
    score = score_of_face(cube_world, matcher, scorer, identity, offset=offset)
    assert score.mean_distance == pytest.approx(offset, abs=1e-6)
    assert score.ninetieth_percentile_distance == pytest.approx(offset, abs=1e-6)
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f_score == 0.0


def test_recall_is_undefined_when_nothing_was_observed(cube_world, matcher, identity):
    scorer = GeometryScorer(
        distance_tolerance=0.02,
        observed_region=SurfaceNearReconstruction(distance=0.01),
    )
    score = score_of_face(cube_world, matcher, scorer, identity, offset=0.03)
    assert score.observed_share == 0.0
    assert score.recall is None
    assert score.f_score is None


# %% the whole world


def test_panoptic_quality_combines_recognition_and_geometry(matcher, scorer, identity):
    correspondence = matcher.match(
        world_of(box("a", CUBE_SIDE, 0.0), box("b", CUBE_SIDE, 1.0)),
        world_of(box("a", CUBE_SIDE, 0.0)),
        identity,
    )

    evaluation = scorer.score(correspondence)

    assert evaluation.mean_f_score == 1.0
    assert evaluation.panoptic_quality == pytest.approx(
        correspondence.recognition_quality
    )
