from copy import deepcopy
from dataclasses import replace

import pytest

from semantic_digital_twin.api import BodySpecification, WorldSpecification
from semantic_digital_twin.exceptions import (
    MinimumOverlapOutOfRangeError,
    PartialOverlapOutOfRangeError,
    SampleSpacingNotFinerThanToleranceError,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_comparison.matching import (
    BodyCorrespondence,
    BodyMatcher,
)
from semantic_digital_twin.world_comparison.surface_samples import (
    SurfaceSampler,
    WorldSurfaces,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures


@pytest.fixture
def matcher() -> BodyMatcher:
    return BodyMatcher(
        distance_tolerance=0.02,
        minimum_overlap=0.5,
        minimum_partial_overlap=0.2,
        sampler=SurfaceSampler(spacing=0.01, seed=0),
    )


@pytest.fixture
def box_scale() -> Scale:
    return Scale(0.3, 0.3, 0.3)


@pytest.fixture
def row_of_boxes(box_scale) -> list[BodySpecification]:
    return [
        BodySpecification.box(
            name,
            scale=box_scale,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=x),
        )
        for name, x in [("left", 0.0), ("middle", 1.0), ("right", 2.0)]
    ]


@pytest.fixture
def row_world(row_of_boxes) -> World:
    return WorldSpecification(objects=row_of_boxes).to_domain_object()


@pytest.fixture
def identity() -> HomogeneousTransformationMatrix:
    return HomogeneousTransformationMatrix()


def box(name: str, length: float, centre_x: float) -> BodySpecification:
    """
    A box 0.3 m deep and high, of the given length along x, centred at the given x.
    """
    return BodySpecification.box(
        name,
        scale=Scale(length, 0.3, 0.3),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=centre_x),
    )


def world_of(*bodies: BodySpecification) -> World:
    return WorldSpecification(objects=list(bodies)).to_domain_object()


def names_of(bodies: list[Body]) -> set[str]:
    return {body.name.name for body in bodies}


def matched_name_pairs(correspondence: BodyCorrespondence) -> set[tuple[str, str]]:
    return {
        (match.ground_truth_body.name.name, match.reconstructed_body.name.name)
        for match in correspondence.matches
    }


def each_box_matched_to_itself(
    row_of_boxes: list[BodySpecification],
) -> set[tuple[str, str]]:
    return {(box.name, box.name) for box in row_of_boxes}


# %% identical worlds


def test_world_matches_its_copy_body_for_body(
    row_world, matcher, row_of_boxes, identity
):
    correspondence = matcher.match(row_world, deepcopy(row_world), identity)
    assert matched_name_pairs(correspondence) == each_box_matched_to_itself(
        row_of_boxes
    )
    assert correspondence.unmatched_ground_truth_bodies == []
    assert correspondence.unmatched_reconstructed_bodies == []


def test_kitchen_matches_its_copy_without_leftovers(kitchen_world, identity):
    matcher = BodyMatcher(
        distance_tolerance=0.05,
        minimum_overlap=0.5,
        minimum_partial_overlap=0.2,
        sampler=SurfaceSampler(spacing=0.02, seed=0),
    )
    correspondence = matcher.match(kitchen_world, deepcopy(kitchen_world), identity)
    bodies_with_geometry = [body for body in kitchen_world.bodies if body.visual]
    assert len(correspondence.matches) == len(bodies_with_geometry)
    assert correspondence.unmatched_ground_truth_bodies == []
    assert correspondence.unmatched_reconstructed_bodies == []


def test_bodies_in_contact_are_neither_merged_nor_split(kitchen_world):
    matcher = BodyMatcher(
        distance_tolerance=0.05,
        minimum_overlap=0.5,
        minimum_partial_overlap=0.2,
        sampler=SurfaceSampler(spacing=0.02, seed=0),
    )
    # A few millimetres of misalignment keep the samples of the two worlds from
    # coinciding exactly, as they never do for a real reconstruction.
    slightly_misaligned = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.003, y=0.0015
    )
    correspondence = matcher.match(
        kitchen_world, deepcopy(kitchen_world), slightly_misaligned
    )
    assert correspondence.merged_bodies == []
    assert correspondence.split_bodies == []


# %% differences between the worlds


def test_body_missing_from_reconstruction_is_unmatched(row_world, matcher, identity):
    reconstructed_world = deepcopy(row_world)
    missing = reconstructed_world.get_body_by_name("middle")
    with reconstructed_world.modify_world():
        reconstructed_world.remove_connection(missing.parent_connection)
        reconstructed_world.remove_kinematic_structure_entity(missing)

    correspondence = matcher.match(row_world, reconstructed_world, identity)

    assert names_of(correspondence.unmatched_ground_truth_bodies) == {"middle"}
    assert correspondence.unmatched_reconstructed_bodies == []


def test_alignment_brings_shifted_reconstruction_onto_ground_truth(
    row_world, matcher, row_of_boxes
):
    shift = HomogeneousTransformationMatrix.from_xyz_rpy(x=5.0)
    shifted_world = WorldSpecification(
        objects=[
            replace(box, parent_T_self=shift @ box.parent_T_self)
            for box in row_of_boxes
        ]
    ).to_domain_object()

    correspondence = matcher.match(
        row_world, shifted_world, HomogeneousTransformationMatrix.from_xyz_rpy(x=-5.0)
    )

    assert matched_name_pairs(correspondence) == each_box_matched_to_itself(
        row_of_boxes
    )


def test_body_origin_plays_no_part(
    row_world, matcher, row_of_boxes, box_scale, identity
):
    body_T_box = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.1, y=0.2, z=-0.05)
    box_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.1, y=-0.2, z=0.05)
    moved_origins_world = WorldSpecification(
        objects=[
            BodySpecification.box(
                box.name,
                scale=box_scale,
                origin=body_T_box,
                parent_T_self=box.parent_T_self @ box_T_body,
            )
            for box in row_of_boxes
        ]
    ).to_domain_object()

    correspondence = matcher.match(row_world, moved_origins_world, identity)

    assert matched_name_pairs(correspondence) == each_box_matched_to_itself(
        row_of_boxes
    )


def test_body_moved_beyond_tolerance_is_unmatched_on_both_sides(
    row_world, matcher, row_of_boxes, identity
):
    left, middle, right = row_of_boxes
    raised_middle = replace(
        middle,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, z=0.5),
    )
    moved_world = WorldSpecification(
        objects=[left, raised_middle, right]
    ).to_domain_object()

    correspondence = matcher.match(row_world, moved_world, identity)

    assert names_of(correspondence.unmatched_ground_truth_bodies) == {"middle"}
    assert names_of(correspondence.unmatched_reconstructed_bodies) == {"middle"}


# %% splits


@pytest.fixture
def split_correspondence(matcher, identity) -> BodyCorrespondence:
    return matcher.match(
        world_of(box("whole", 0.6, 0.0)),
        world_of(box("left_half", 0.3, -0.15), box("right_half", 0.3, 0.15)),
        identity,
    )


def test_split_body_matches_one_piece_and_leaves_the_other(split_correspondence):
    assert len(split_correspondence.matches) == 1
    assert len(split_correspondence.unmatched_reconstructed_bodies) == 1
    assert split_correspondence.unmatched_ground_truth_bodies == []


def test_split_body_is_reported_with_its_pieces(split_correspondence):
    [split] = split_correspondence.split_bodies
    assert split.ground_truth_body.name.name == "whole"
    assert names_of([piece.reconstructed_body for piece in split.pieces]) == {
        "left_half",
        "right_half",
    }
    assert split_correspondence.merged_bodies == []


# %% merges


def test_body_merged_from_a_large_and_a_small_part_matches_the_large_one(
    matcher, identity
):
    correspondence = matcher.match(
        world_of(box("carcass", 0.9, -0.15), box("front", 0.3, 0.45)),
        world_of(box("merged", 1.2, 0.0)),
        identity,
    )

    assert matched_name_pairs(correspondence) == {("carcass", "merged")}
    assert names_of(correspondence.unmatched_ground_truth_bodies) == {"front"}
    [merge] = correspondence.merged_bodies
    assert merge.reconstructed_body.name.name == "merged"
    assert [part.ground_truth_body.name.name for part in merge.parts] == [
        "carcass",
        "front",
    ]


def test_body_merged_from_three_equal_parts_is_reported_although_unmatched(
    matcher, identity
):
    correspondence = matcher.match(
        world_of(box("a", 0.3, -0.3), box("b", 0.3, 0.0), box("c", 0.3, 0.3)),
        world_of(box("merged", 0.9, 0.0)),
        identity,
    )

    assert correspondence.matches == []
    [merge] = correspondence.merged_bodies
    assert names_of([part.ground_truth_body for part in merge.parts]) == {
        "a",
        "b",
        "c",
    }


def test_merged_parts_divide_the_surface_between_them(matcher, identity):
    correspondence = matcher.match(
        world_of(box("a", 0.3, -0.15), box("b", 0.3, 0.15)),
        world_of(box("merged", 0.6, 0.0)),
        identity,
    )

    [merge] = correspondence.merged_bodies
    assert sum(part.overlap for part in merge.parts) <= 1.0


# %% minimum overlap


@pytest.fixture
def partly_overlapping_worlds() -> tuple[World, World]:
    return world_of(box("box", 0.3, 0.0)), world_of(box("box", 0.3, 0.1))


def overlap_of_the_two_boxes(
    ground_truth_world: World, reconstructed_world: World, matcher: BodyMatcher
) -> float:
    ground_truth_body = ground_truth_world.get_body_by_name("box")
    reconstructed_body = reconstructed_world.get_body_by_name("box")
    ground_truth_surfaces = WorldSurfaces(
        [
            matcher.sampler.sample(
                ground_truth_body,
                ground_truth_world.compute_forward_kinematics(
                    ground_truth_world.root, ground_truth_body
                ),
            )
        ]
    )
    reconstructed_surface = matcher.sampler.sample(
        reconstructed_body,
        reconstructed_world.compute_forward_kinematics(
            reconstructed_world.root, reconstructed_body
        ),
    )
    [overlap] = ground_truth_surfaces.nearest_body_shares(
        reconstructed_surface, matcher.distance_tolerance
    )
    return float(overlap)


@pytest.mark.parametrize("margin, expected_matches", [(-0.01, 1), (0.01, 0)])
def test_pair_counts_only_from_the_minimum_overlap_on(
    partly_overlapping_worlds, matcher, margin, expected_matches, identity
):
    ground_truth_world, reconstructed_world = partly_overlapping_worlds
    overlap = overlap_of_the_two_boxes(ground_truth_world, reconstructed_world, matcher)
    strict_matcher = replace(matcher, minimum_overlap=overlap + margin)

    correspondence = strict_matcher.match(
        ground_truth_world, reconstructed_world, identity
    )

    assert len(correspondence.matches) == expected_matches


# %% parameters


def test_sample_spacing_must_be_finer_than_the_tolerance(matcher):
    with pytest.raises(SampleSpacingNotFinerThanToleranceError):
        replace(matcher, sampler=SurfaceSampler(spacing=0.02, seed=0))


@pytest.mark.parametrize("minimum_overlap", [0.0, 1.5])
def test_minimum_overlap_must_be_a_fraction(matcher, minimum_overlap):
    with pytest.raises(MinimumOverlapOutOfRangeError):
        replace(matcher, minimum_overlap=minimum_overlap)


@pytest.mark.parametrize("minimum_partial_overlap", [0.0, 0.6])
def test_partial_overlap_must_lie_between_zero_and_the_minimum_overlap(
    matcher, minimum_partial_overlap
):
    with pytest.raises(PartialOverlapOutOfRangeError):
        replace(matcher, minimum_partial_overlap=minimum_partial_overlap)
