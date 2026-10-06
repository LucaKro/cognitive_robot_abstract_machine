"""
Points spread evenly over the surface of a body, for measuring how much of one surface
lies on the bodies of another world.

A mesh's vertices are no such points: how densely a surface is triangulated depends on
how the mesh was made, not on how much surface there is. Samples drawn in proportion to
area make every point stand for the same patch of surface, so a share of samples is a
share of area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
import open3d
import trimesh
from scipy.spatial import cKDTree
from trimesh.sample import sample_surface
from trimesh.transformations import transform_points

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

# %% sampling


@dataclass
class SurfaceSampler:
    """
    Draws reproducible samples from the visual surface of a body, spread evenly by area.
    """

    spacing: float
    """
    The distance between neighbouring samples on average, in metres.
    """

    seed: int
    """
    The seed of the random draw, so the same body always yields the same samples.
    """

    def sample(
        self, body: Body, ground_truth_root_T_body: HomogeneousTransformationMatrix
    ) -> SurfaceSamples:
        """
        Sample the visual surface of a body and place the samples in the root frame of
        the ground truth world.

        :param body: The body whose visual surface is sampled.
        :param ground_truth_root_T_body: The pose of the body in the root frame of the
            ground truth world.
        :return: The samples, one per square of :attr:`spacing` side length.
        """
        mesh = body.visual.combined_mesh
        sample_count = max(1, math.ceil(mesh.area / self.spacing**2))
        body_P_samples, _ = sample_surface(mesh, sample_count, seed=self.seed)
        return SurfaceSamples(
            body=body,
            ground_truth_root_T_body=ground_truth_root_T_body,
            ground_truth_root_P_samples=transform_points(
                body_P_samples, ground_truth_root_T_body.to_np()
            ),
        )


# %% samples


@dataclass
class SurfaceSamples:
    """
    Samples of a body's visual surface, in the root frame of the ground truth world.
    """

    body: Body
    """
    The body whose surface was sampled.
    """

    ground_truth_root_T_body: HomogeneousTransformationMatrix
    """
    The pose of the body in the root frame of the ground truth world.
    """

    ground_truth_root_P_samples: npt.NDArray[np.float64]
    """
    The sampled points in the root frame of the ground truth world, one per row.
    """

    @cached_property
    def distance_to_surface(self) -> DistanceToSurface:
        """
        :return: The distances to the body's visual surface, placed in the root frame of
            the ground truth world.
        """
        surface = self.body.visual.combined_mesh.copy()
        surface.apply_transform(self.ground_truth_root_T_body.to_np())
        return DistanceToSurface(surface)

    def distances_to(self, other_surface: SurfaceSamples) -> npt.NDArray[np.float64]:
        """
        :param other_surface: The surface the distances are measured to.
        :return: For each of these samples, its distance to the nearest point of the
            other body's surface, in metres.
        """
        return other_surface.distance_to_surface.distances_of(
            self.ground_truth_root_P_samples
        )


# %% exact distances


@dataclass
class DistanceToSurface:
    """
    Measures how far points lie from a triangle surface, to the nearest point on any of
    its triangles.

    The distances are computed in single precision around the centre of the surface, so
    they are exact to about a micrometre for surfaces the size of furniture.
    """

    surface: trimesh.Trimesh
    """
    The surface the distances are measured to.
    """

    @cached_property
    def _centre(self) -> npt.NDArray[np.float64]:
        """
        :return: The centre of the surface's bounding box, which the computation is done
            relative to, so single precision is spent on the surface's own extent.
        """
        return self.surface.bounds.mean(axis=0)

    @cached_property
    def _scene(self) -> open3d.t.geometry.RaycastingScene:
        """
        :return: An Open3D scene holding the surface, relative to :attr:`_centre`.
        """
        scene = open3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            open3d.core.Tensor(
                (self.surface.vertices - self._centre).astype(np.float32)
            ),
            open3d.core.Tensor(self.surface.faces.astype(np.uint32)),
        )
        return scene

    def distances_of(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        :param points: The points to measure, one per row, in the frame of the surface.
        :return: For each point, its distance to the nearest point of the surface, in
            metres.
        """
        relative_points = open3d.core.Tensor((points - self._centre).astype(np.float32))
        return self._scene.compute_distance(relative_points).numpy().astype(np.float64)


# %% the surfaces of a world


@dataclass
class WorldSurfaces:
    """
    The sampled surfaces of the bodies of one world, for finding which body a point lies
    on.
    """

    surfaces: list[SurfaceSamples]
    """
    The sampled surface of each body that has one.
    """

    @cached_property
    def _surface_by_body(self) -> dict[int, SurfaceSamples]:
        """
        :return: The sampled surfaces, keyed by the identity of their body.
        """
        return {id(surface.body): surface for surface in self.surfaces}

    def surface_of(self, body: Body) -> SurfaceSamples:
        """
        :return: The sampled surface of the body.
        """
        return self._surface_by_body[id(body)]

    @cached_property
    def search_tree(self) -> cKDTree:
        """
        :return: A k-d tree over the samples of every surface, for finding the sample
            nearest to a point.
        """
        return cKDTree(
            np.vstack(
                [surface.ground_truth_root_P_samples for surface in self.surfaces]
            )
        )

    @cached_property
    def surface_indices(self) -> npt.NDArray[np.int_]:
        """
        :return: For each sample in :attr:`search_tree`, the index of the surface in
            :attr:`surfaces` it was drawn from.
        """
        return np.concatenate(
            [
                np.full(len(surface.ground_truth_root_P_samples), index)
                for index, surface in enumerate(self.surfaces)
            ]
        )

    def nearest_body_shares(
        self, other_surface: SurfaceSamples, distance: float
    ) -> npt.NDArray[np.float64]:
        """
        Divide another surface among the bodies of this world, giving each of its
        samples to the body with the nearest sample.

        Each sample goes to one body only, so a surface that lies on two touching bodies
        is split between them, and the shares never add up to more than one.

        :param other_surface: The surface to divide.
        :param distance: How close the nearest sample has to be for a sample of the
            other surface to lie on a body at all, in metres.
        :return: For each surface in :attr:`surfaces`, the share of the other surface's
            samples that lie on it.
        """
        if not self.surfaces:
            return np.zeros(0)
        distances, nearest = self.search_tree.query(
            other_surface.ground_truth_root_P_samples, distance_upper_bound=distance
        )
        lying_on_a_body = np.isfinite(distances)
        counts = np.bincount(
            self.surface_indices[nearest[lying_on_a_body]],
            minlength=len(self.surfaces),
        )
        return counts / len(distances)
