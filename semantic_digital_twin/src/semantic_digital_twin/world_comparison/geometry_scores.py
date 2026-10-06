"""
How well the surface of each matched reconstructed body fits its ground truth body.

The scores follow the benchmarks of multi-view stereo reconstruction: precision is the
share of the reconstructed surface within a distance tolerance of the ground truth
surface, recall the share of the ground truth surface within that tolerance of the
reconstruction, and the F-score combines both. Recall is taken only over the part of the
ground truth surface the reconstruction could have seen, which an
:class:`ObservedRegion` decides; a scan never sees the back of a cabinet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from semantic_digital_twin.world_comparison.matching import (
    BodyCorrespondence,
    BodyOverlap,
)
from semantic_digital_twin.world_comparison.surface_samples import (
    SurfaceSamples,
    WorldSurfaces,
)

# %% which ground truth surface counts


class ObservedRegion(ABC):
    """
    Decides which part of a ground truth surface the reconstruction could have seen, so
    that recall counts only that part.
    """

    @abstractmethod
    def observed(
        self,
        ground_truth_surface: SurfaceSamples,
        reconstructed_surfaces: WorldSurfaces,
    ) -> npt.NDArray[np.bool_]:
        """
        :param ground_truth_surface: The ground truth surface to decide for.
        :param reconstructed_surfaces: Every surface of the reconstructed world.
        :return: For each sample of the ground truth surface, whether it counts as
            observed.
        """


@dataclass
class WholeSurface(ObservedRegion):
    """
    Counts the whole ground truth surface as observed, for a reconstruction that could
    have seen every side of every body.
    """

    def observed(
        self,
        ground_truth_surface: SurfaceSamples,
        reconstructed_surfaces: WorldSurfaces,
    ) -> npt.NDArray[np.bool_]:
        return np.ones(
            len(ground_truth_surface.ground_truth_root_P_samples), dtype=bool
        )


@dataclass
class SurfaceNearReconstruction(ObservedRegion):
    """
    Counts as observed the ground truth surface that lies within a distance of any
    reconstructed surface, for a scan whose camera poses are unknown.

    .. warning:: The region is read off the reconstruction itself, so recall measures how
        closely the reconstruction follows the ground truth where it has surface, not how
        much of what a camera saw it captured. With a :attr:`distance` at or below the
        distance tolerance of the scores, recall is close to 1 by construction.
    """

    distance: float
    """
    How close a ground truth sample has to be to any reconstructed sample to count as
    observed, in metres.
    """

    def observed(
        self,
        ground_truth_surface: SurfaceSamples,
        reconstructed_surfaces: WorldSurfaces,
    ) -> npt.NDArray[np.bool_]:
        if not reconstructed_surfaces.surfaces:
            return np.zeros(
                len(ground_truth_surface.ground_truth_root_P_samples), dtype=bool
            )
        distances, _ = reconstructed_surfaces.search_tree.query(
            ground_truth_surface.ground_truth_root_P_samples,
            distance_upper_bound=self.distance,
        )
        return np.isfinite(distances)


# %% scores


@dataclass
class GeometryScore:
    """
    How well the surface of one reconstructed body fits the ground truth body it was
    matched to.
    """

    match: BodyOverlap
    """
    The pair of bodies the score is for.
    """

    precision: float
    """
    The share of the reconstructed surface within the distance tolerance of the ground
    truth surface.
    """

    recall: float | None
    """
    The share of the observed ground truth surface within the distance tolerance of the
    reconstructed surface; ``None`` when none of it was observed.
    """

    mean_distance: float
    """
    The mean distance from the reconstructed surface to the ground truth surface, in
    metres.
    """

    ninetieth_percentile_distance: float
    """
    The distance from the reconstructed surface to the ground truth surface that 90 % of
    the reconstructed surface lies within, in metres.
    """

    observed_share: float
    """
    The share of the ground truth surface that counted as observed, which recall is
    taken over.
    """

    @property
    def f_score(self) -> float | None:
        """
        :return: The harmonic mean of :attr:`precision` and :attr:`recall`, 0 when both
            are 0, and ``None`` when recall is undefined.
        """
        if self.recall is None:
            return None
        if self.precision + self.recall == 0.0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


@dataclass
class GeometryEvaluation:
    """
    The geometry scores of every matched pair of a correspondence, and what they add up
    to for the whole world.
    """

    correspondence: BodyCorrespondence
    """
    The correspondence whose matched pairs were scored.
    """

    scores: list[GeometryScore] = field(default_factory=list)
    """
    One score per matched pair.
    """

    @property
    def mean_f_score(self) -> float | None:
        """
        :return: The mean F-score of the pairs that have one, or ``None`` when none has.
        """
        f_scores = [score.f_score for score in self.scores if score.f_score is not None]
        if not f_scores:
            return None
        return float(np.mean(f_scores))

    @property
    def panoptic_quality(self) -> float | None:
        """
        :return: The recognition quality of the correspondence times the mean F-score of
            its pairs, as panoptic quality combines how well bodies were found with how
            well the found ones fit. ``None`` when either is undefined.
        """
        recognition_quality = self.correspondence.recognition_quality
        mean_f_score = self.mean_f_score
        if recognition_quality is None or mean_f_score is None:
            return None
        return recognition_quality * mean_f_score


# %% scoring


@dataclass
class GeometryScorer:
    """
    Scores how well the surface of each matched reconstructed body fits its ground truth
    body, with exact distances to the surfaces.
    """

    distance_tolerance: float
    """
    How close a point of one surface has to be to the other to count as fitting, in
    metres.
    """

    observed_region: ObservedRegion
    """
    Decides which part of each ground truth surface recall is taken over.
    """

    def score(self, correspondence: BodyCorrespondence) -> GeometryEvaluation:
        """
        :param correspondence: The correspondence whose matched pairs are scored.
        :return: The score of every matched pair.
        """
        table = correspondence.overlap_table
        return GeometryEvaluation(
            correspondence=correspondence,
            scores=[
                self._score_of(
                    match,
                    table.ground_truth_surfaces.surface_of(match.ground_truth_body),
                    table.reconstructed_surfaces.surface_of(match.reconstructed_body),
                    table.reconstructed_surfaces,
                )
                for match in correspondence.matches
            ],
        )

    def _score_of(
        self,
        match: BodyOverlap,
        ground_truth_surface: SurfaceSamples,
        reconstructed_surface: SurfaceSamples,
        reconstructed_surfaces: WorldSurfaces,
    ) -> GeometryScore:
        """
        :return: The score of one matched pair.
        """
        reconstructed_distances = reconstructed_surface.distances_to(
            ground_truth_surface
        )
        observed = self.observed_region.observed(
            ground_truth_surface, reconstructed_surfaces
        )
        recall = None
        if observed.any():
            observed_distances = ground_truth_surface.distances_to(
                reconstructed_surface
            )[observed]
            recall = float(np.mean(observed_distances <= self.distance_tolerance))
        return GeometryScore(
            match=match,
            precision=float(
                np.mean(reconstructed_distances <= self.distance_tolerance)
            ),
            recall=recall,
            mean_distance=float(np.mean(reconstructed_distances)),
            ninetieth_percentile_distance=float(
                np.percentile(reconstructed_distances, 90)
            ),
            observed_share=float(np.mean(observed)),
        )
