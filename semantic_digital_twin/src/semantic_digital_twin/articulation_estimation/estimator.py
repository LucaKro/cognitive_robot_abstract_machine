"""
Selection of the kind of motion that best explains observed poses.
"""

from __future__ import annotations

# %% Imports
import inspect
from dataclasses import dataclass

from krrood.utils import recursive_subclasses
from semantic_digital_twin.articulation_estimation.fits import ArticulationFit
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)


# %% Model selection
@dataclass(kw_only=True)
class ArticulationEstimator:
    """
    Fits every concrete :class:`ArticulationFit` to observed poses and selects the one
    with the lowest :attr:`ArticulationFit.bayesian_information_criterion`, so that a
    simpler kind of motion wins unless a richer one explains the poses clearly better.
    """

    noise: ObservationNoise
    """
    The noise the observations carry.
    """

    def most_likely_fit(self, *, observations: PoseObservations) -> ArticulationFit:
        """
        :param observations: The observed poses of a child relative to its parent.
        :return: The fit of the kind of motion that best explains them.
        """
        fits = [
            fit_type.fit(observations=observations, noise=self.noise)
            for fit_type in recursive_subclasses(ArticulationFit)
            if not inspect.isabstract(fit_type)
        ]
        return min(fits, key=lambda fit: fit.bayesian_information_criterion)
