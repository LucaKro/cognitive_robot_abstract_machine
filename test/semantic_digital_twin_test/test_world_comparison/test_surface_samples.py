import math

import numpy as np
import pytest

from semantic_digital_twin.api import BodySpecification, WorldSpecification
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_comparison.surface_samples import (
    SurfaceSampler,
    SurfaceSamples,
    WorldSurfaces,
)
from semantic_digital_twin.world_description.geometry import Scale

# %% fixtures


@pytest.fixture
def sampler() -> SurfaceSampler:
    return SurfaceSampler(spacing=0.01, seed=0)


@pytest.fixture
def two_distant_boxes() -> World:
    return WorldSpecification(
        objects=[
            BodySpecification.box("near", scale=Scale(0.3, 0.3, 0.3)),
            BodySpecification.box(
                "far",
                scale=Scale(0.3, 0.3, 0.3),
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=5.0),
            ),
        ]
    ).to_domain_object()


def surface_of(world: World, body_name: str, sampler: SurfaceSampler) -> SurfaceSamples:
    body = world.get_body_by_name(body_name)
    return sampler.sample(body, world.compute_forward_kinematics(world.root, body))


# %% sampling


def test_sample_count_follows_surface_area(two_distant_boxes, sampler):
    body = two_distant_boxes.get_body_by_name("near")
    surface = surface_of(two_distant_boxes, "near", sampler)
    expected = math.ceil(body.visual.combined_mesh.area / sampler.spacing**2)
    assert len(surface.ground_truth_root_P_samples) == expected


def test_samples_lie_on_the_surface_in_the_root_frame(two_distant_boxes, sampler):
    body = two_distant_boxes.get_body_by_name("far")
    root_T_body = two_distant_boxes.compute_forward_kinematics(
        two_distant_boxes.root, body
    )
    surface = sampler.sample(body, root_T_body)
    mesh = body.visual.combined_mesh.copy()
    mesh.apply_transform(root_T_body.to_np())
    _, distances, _ = mesh.nearest.on_surface(surface.ground_truth_root_P_samples)
    assert np.max(distances) == pytest.approx(0.0, abs=1e-9)


def test_same_seed_gives_same_samples(two_distant_boxes, sampler):
    first = surface_of(two_distant_boxes, "near", sampler)
    second = surface_of(two_distant_boxes, "near", sampler)
    assert np.array_equal(
        first.ground_truth_root_P_samples, second.ground_truth_root_P_samples
    )


# %% nearest bodies


@pytest.fixture
def two_touching_boxes() -> World:
    return WorldSpecification(
        objects=[
            BodySpecification.box(
                name,
                scale=Scale(0.3, 0.3, 0.3),
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=x),
            )
            for name, x in [("left", -0.15), ("right", 0.15)]
        ]
    ).to_domain_object()


def world_surfaces_of(world: World, sampler: SurfaceSampler) -> WorldSurfaces:
    return WorldSurfaces(
        [
            surface_of(world, body.name.name, sampler)
            for body in world.bodies
            if body.visual
        ]
    )


def test_surface_lies_wholly_on_its_own_body(two_distant_boxes, sampler):
    world_surfaces = world_surfaces_of(two_distant_boxes, sampler)
    near = surface_of(two_distant_boxes, "near", sampler)
    assert world_surfaces.nearest_body_shares(near, distance=0.02).tolist() == [
        1.0,
        0.0,
    ]


def test_surface_far_from_every_body_lies_on_none(two_distant_boxes, sampler):
    world_surfaces = WorldSurfaces([surface_of(two_distant_boxes, "near", sampler)])
    far = surface_of(two_distant_boxes, "far", sampler)
    assert world_surfaces.nearest_body_shares(far, distance=0.02).tolist() == [0.0]


def test_shares_divide_a_surface_spanning_touching_bodies(two_touching_boxes, sampler):
    spanning_world = WorldSpecification(
        objects=[BodySpecification.box("spanning", scale=Scale(0.6, 0.3, 0.3))]
    ).to_domain_object()
    spanning = surface_of(spanning_world, "spanning", sampler)

    shares = world_surfaces_of(two_touching_boxes, sampler).nearest_body_shares(
        spanning, distance=0.02
    )

    assert shares.sum() == pytest.approx(1.0, abs=0.01)
    assert shares == pytest.approx([0.5, 0.5], abs=0.05)
