"""
Deciding which body of a reconstructed world stands for which body of the ground truth
world, from their geometry alone.

Every score that compares the two worlds body by body rests on this correspondence. It
is worked out from where the surfaces lie, never from what the bodies are called or
annotated as, so a later score of those labels is not graded on pairs chosen because
their labels agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from semantic_digital_twin.exceptions import (
    MinimumOverlapOutOfRangeError,
    PartialOverlapOutOfRangeError,
    SampleSpacingNotFinerThanToleranceError,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_comparison.surface_samples import (
    SurfaceSampler,
    WorldSurfaces,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% correspondence


@dataclass
class BodyOverlap:
    """
    How much of a reconstructed body's surface lies on a ground truth body.
    """

    ground_truth_body: Body
    """
    The body of the ground truth world.
    """

    reconstructed_body: Body
    """
    The body of the reconstructed world.
    """

    overlap: float
    """
    The share of the reconstructed body's surface that lies on the ground truth body.
    """


@dataclass
class MergedBody:
    """
    A reconstructed body that partly stands for several ground truth bodies, such as a
    drawer front scanned as one piece with its cabinet.
    """

    reconstructed_body: Body
    """
    The reconstructed body that several ground truth bodies were merged into.
    """

    parts: list[BodyOverlap]
    """
    The ground truth bodies it partly lies on, the one with the most overlap first.
    """


@dataclass
class SplitBody:
    """
    A ground truth body that several reconstructed bodies partly stand for, such as a
    table top scanned as two pieces.
    """

    ground_truth_body: Body
    """
    The ground truth body that was split.
    """

    pieces: list[BodyOverlap]
    """
    The reconstructed bodies that partly lie on it, the one with the most overlap first.
    """


@dataclass
class BodyCorrespondence:
    """
    Which bodies of a reconstructed world stand for which bodies of the ground truth
    world, which have no counterpart, and which were merged or split.
    """

    overlap_table: OverlapTable
    """
    How much of each reconstructed body lies on each ground truth body, from which the
    correspondence was decided.
    """

    matches: list[BodyOverlap] = field(default_factory=list)
    """
    The pairs of bodies that stand for each other, each body in at most one pair.
    """

    unmatched_ground_truth_bodies: list[Body] = field(default_factory=list)
    """
    The bodies of the ground truth world that no reconstructed body stands for.
    """

    unmatched_reconstructed_bodies: list[Body] = field(default_factory=list)
    """
    The reconstructed bodies that stand for no ground truth body.
    """

    merged_bodies: list[MergedBody] = field(default_factory=list)
    """
    The reconstructed bodies that partly stand for several ground truth bodies.

    All but one of those ground truth bodies are unmatched, so this tells a body the
    reconstruction merged into another apart from one it missed.
    """

    split_bodies: list[SplitBody] = field(default_factory=list)
    """
    The ground truth bodies that several reconstructed bodies partly stand for.

    All but one of those reconstructed bodies are unmatched, so this tells a piece of a
    split body apart from a body that stands for nothing.
    """

    @property
    def recognition_quality(self) -> float | None:
        """
        :return: How well the bodies were found, from 0 to 1: the matches divided by the
            matches plus half of the unmatched bodies of either world, as in panoptic
            quality. ``None`` when neither world has a body.
        """
        errors = len(self.unmatched_ground_truth_bodies) + len(
            self.unmatched_reconstructed_bodies
        )
        if not self.matches and not errors:
            return None
        return len(self.matches) / (len(self.matches) + errors / 2)


# %% overlap table


@dataclass
class OverlapTable:
    """
    How much of each reconstructed body's surface lies on each ground truth body.
    """

    ground_truth_surfaces: WorldSurfaces
    """
    The sampled surfaces of the ground truth world.
    """

    reconstructed_surfaces: WorldSurfaces
    """
    The sampled surfaces of the reconstructed world, in the ground truth root frame.
    """

    distance_tolerance: float
    """
    How close a point of a reconstructed surface has to be to a ground truth surface to
    lie on it, in metres.
    """

    @cached_property
    def overlap(self) -> npt.NDArray[np.float64]:
        """
        :return: For each reconstructed body (row) and ground truth body (column), the
            share of the reconstructed body's surface that lies on the ground truth body.
        """
        return np.array(
            [
                self.ground_truth_surfaces.nearest_body_shares(
                    reconstructed_surface, self.distance_tolerance
                )
                for reconstructed_surface in self.reconstructed_surfaces.surfaces
            ]
        ).reshape(
            len(self.reconstructed_surfaces.surfaces),
            len(self.ground_truth_surfaces.surfaces),
        )

    def body_overlap(
        self, reconstructed_index: int, ground_truth_index: int
    ) -> BodyOverlap:
        """
        :return: The overlap of one cell of the table, with the bodies it belongs to.
        """
        return BodyOverlap(
            ground_truth_body=self.ground_truth_surfaces.surfaces[
                ground_truth_index
            ].body,
            reconstructed_body=self.reconstructed_surfaces.surfaces[
                reconstructed_index
            ].body,
            overlap=float(self.overlap[reconstructed_index, ground_truth_index]),
        )

    def overlaps_of_reconstructed_body(
        self, reconstructed_index: int, minimum_overlap: float
    ) -> list[BodyOverlap]:
        """
        :return: The ground truth bodies the reconstructed body lies on by at least the
            minimum overlap, the one with the most overlap first.
        """
        return self._largest_first(
            [
                self.body_overlap(reconstructed_index, ground_truth_index)
                for ground_truth_index in np.flatnonzero(
                    self.overlap[reconstructed_index] >= minimum_overlap
                )
            ]
        )

    def overlaps_on_ground_truth_body(
        self, ground_truth_index: int, minimum_overlap: float
    ) -> list[BodyOverlap]:
        """
        :return: The reconstructed bodies that lie on the ground truth body by at least
            the minimum overlap, the one with the most overlap first.
        """
        return self._largest_first(
            [
                self.body_overlap(reconstructed_index, ground_truth_index)
                for reconstructed_index in np.flatnonzero(
                    self.overlap[:, ground_truth_index] >= minimum_overlap
                )
            ]
        )

    @staticmethod
    def _largest_first(overlaps: list[BodyOverlap]) -> list[BodyOverlap]:
        return sorted(overlaps, key=lambda body_overlap: -body_overlap.overlap)


# %% matching


@dataclass
class BodyMatcher:
    """
    Pairs the bodies of a reconstructed world one to one with those of the ground truth
    world, by how much of each reconstructed body's surface lies on a ground truth body,
    and reports the bodies the reconstruction merged or split.

    Each sample of a reconstructed surface lies on the ground truth body with the
    nearest sample, if that is within :attr:`distance_tolerance`. A reconstructed
    surface is so divided among the ground truth bodies, and a small body mounted on a
    larger one, such as a knob on a panel, lies on itself rather than on both.

    Only bodies with a visual surface take part; bodies without one, such as a world's
    root, are frames rather than physical objects. The overlap is measured from the
    reconstructed body to the ground truth body, so ground truth surface the
    reconstruction never saw, such as the back of a cabinet, does not count against a
    pair.
    """

    distance_tolerance: float
    """
    How close a point of a reconstructed surface has to be to a ground truth surface to
    lie on it, in metres.
    """

    minimum_overlap: float
    """
    The share of a reconstructed body's surface that has to lie on a ground truth body
    for the two to count as a pair.
    """

    minimum_partial_overlap: float
    """
    The share of a reconstructed body's surface that has to lie on a ground truth body
    for it to count as partly standing for that body, when telling merged and split
    bodies.
    """

    sampler: SurfaceSampler
    """
    Draws the surface samples the overlap is measured on.
    """

    def __post_init__(self):
        if self.sampler.spacing >= self.distance_tolerance:
            raise SampleSpacingNotFinerThanToleranceError(
                spacing=self.sampler.spacing,
                distance_tolerance=self.distance_tolerance,
            )
        if not 0.0 < self.minimum_overlap <= 1.0:
            raise MinimumOverlapOutOfRangeError(minimum_overlap=self.minimum_overlap)
        if not 0.0 < self.minimum_partial_overlap <= self.minimum_overlap:
            raise PartialOverlapOutOfRangeError(
                minimum_partial_overlap=self.minimum_partial_overlap,
                minimum_overlap=self.minimum_overlap,
            )

    def match(
        self,
        ground_truth_world: World,
        reconstructed_world: World,
        ground_truth_root_T_reconstructed_root: HomogeneousTransformationMatrix,
    ) -> BodyCorrespondence:
        """
        Pair the bodies of the reconstructed world with those of the ground truth world.

        :param ground_truth_world: The world taken as correct.
        :param reconstructed_world: The world measured against it.
        :param ground_truth_root_T_reconstructed_root: The pose of the reconstructed
            world's root in the ground truth world's root frame, which brings both
            worlds into one frame.
        :return: The pairs, with the most overlap in total, whose overlap reaches
            :attr:`minimum_overlap`, the bodies left without a counterpart, and the
            bodies that were merged or split.
        """
        table = OverlapTable(
            ground_truth_surfaces=self._surfaces_of(ground_truth_world),
            reconstructed_surfaces=self._surfaces_of(
                reconstructed_world,
                ground_truth_root_T_reconstructed_root.copy_with_new_reference_frames(
                    new_reference_frame=ground_truth_world.root,
                    new_child_frame=reconstructed_world.root,
                ),
            ),
            distance_tolerance=self.distance_tolerance,
        )
        matches = self._matches_in(table)
        matched_ground_truth_bodies = {id(match.ground_truth_body) for match in matches}
        matched_reconstructed_bodies = {
            id(match.reconstructed_body) for match in matches
        }
        return BodyCorrespondence(
            overlap_table=table,
            matches=matches,
            unmatched_ground_truth_bodies=[
                surface.body
                for surface in table.ground_truth_surfaces.surfaces
                if id(surface.body) not in matched_ground_truth_bodies
            ],
            unmatched_reconstructed_bodies=[
                surface.body
                for surface in table.reconstructed_surfaces.surfaces
                if id(surface.body) not in matched_reconstructed_bodies
            ],
            merged_bodies=self._merged_bodies_in(table),
            split_bodies=self._split_bodies_in(table),
        )

    def _matches_in(self, table: OverlapTable) -> list[BodyOverlap]:
        """
        :return: The one-to-one pairs with the most overlap in total, among those whose
            overlap reaches :attr:`minimum_overlap`.
        """
        cost = np.where(table.overlap >= self.minimum_overlap, 1.0 - table.overlap, 1.0)
        reconstructed_indices, ground_truth_indices = linear_sum_assignment(cost)
        return [
            table.body_overlap(reconstructed_index, ground_truth_index)
            for reconstructed_index, ground_truth_index in zip(
                reconstructed_indices, ground_truth_indices
            )
            if table.overlap[reconstructed_index, ground_truth_index]
            >= self.minimum_overlap
        ]

    def _merged_bodies_in(self, table: OverlapTable) -> list[MergedBody]:
        """
        :return: The reconstructed bodies that lie on several ground truth bodies by at
            least :attr:`minimum_partial_overlap`.
        """
        return [
            MergedBody(reconstructed_body=surface.body, parts=parts)
            for reconstructed_index, surface in enumerate(
                table.reconstructed_surfaces.surfaces
            )
            if len(
                parts := table.overlaps_of_reconstructed_body(
                    reconstructed_index, self.minimum_partial_overlap
                )
            )
            > 1
        ]

    def _split_bodies_in(self, table: OverlapTable) -> list[SplitBody]:
        """
        :return: The ground truth bodies that several reconstructed bodies lie on by at
            least :attr:`minimum_partial_overlap`.
        """
        return [
            SplitBody(ground_truth_body=surface.body, pieces=pieces)
            for ground_truth_index, surface in enumerate(
                table.ground_truth_surfaces.surfaces
            )
            if len(
                pieces := table.overlaps_on_ground_truth_body(
                    ground_truth_index, self.minimum_partial_overlap
                )
            )
            > 1
        ]

    def _surfaces_of(
        self,
        world: World,
        ground_truth_root_T_world_root: HomogeneousTransformationMatrix | None = None,
    ) -> WorldSurfaces:
        """
        :param world: The world whose bodies are sampled.
        :param ground_truth_root_T_world_root: The pose of the world's root in the
            ground truth world's root frame. ``None`` means the world is the ground
            truth world itself.
        :return: The surface samples of every body of the world that has a visual
            surface.
        """
        if ground_truth_root_T_world_root is None:
            ground_truth_root_T_world_root = HomogeneousTransformationMatrix(
                reference_frame=world.root, child_frame=world.root
            )
        return WorldSurfaces(
            [
                self.sampler.sample(
                    body,
                    ground_truth_root_T_world_root
                    @ world.compute_forward_kinematics(world.root, body),
                )
                for body in world.bodies
                if body.visual
            ]
        )
