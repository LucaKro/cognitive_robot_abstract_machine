"""
Errors in estimating articulations from observed poses.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.articulation_estimation.motion_kinds import MotionKind


# %% Observations
@dataclass(kw_only=True)
class TooFewObservationsError(UsageError):
    """
    Raised when fewer poses are observed than it takes to see how a child moves.
    """

    observation_count: int
    """
    The number of poses observed.
    """

    required_observation_count: int
    """
    The fewest poses an estimate needs.
    """

    def error_message(self) -> str:
        return (
            f"{self.observation_count} pose(s) were observed, but estimating how a "
            f"child moves needs at least {self.required_observation_count}."
        )

    def suggest_correction(self) -> str:
        return f"observe the child in at least {self.required_observation_count} poses."


@dataclass(kw_only=True)
class NotHomogeneousTransformationError(UsageError):
    """
    Raised when an observed pose is not a homogeneous transformation matrix: its last
    row is not [0, 0, 0, 1].
    """

    observation_indices: list[int]
    """
    The indices of the observed poses whose last row is not [0, 0, 0, 1].
    """

    def error_message(self) -> str:
        return (
            f"The observed pose(s) at {self.observation_indices} have a last row other "
            f"than [0, 0, 0, 1], so they are not homogeneous transformation matrices."
        )

    def suggest_correction(self) -> str:
        return "pass each pose as a homogeneous transformation matrix."


# %% Kinds of motion
@dataclass(kw_only=True)
class KindNotFittedError(UsageError):
    """
    Raised when the fit of a kind of motion is asked for that was not fitted, because a
    classifier found it too improbable.
    """

    kind: MotionKind
    """
    The kind whose fit was asked for.
    """

    fitted_kinds: list[MotionKind]
    """
    The kinds that were fitted.
    """

    def error_message(self) -> str:
        return (
            f"The {self.kind} kind was not fitted; only {self.fitted_kinds} were, "
            f"because the classifier found the others too improbable."
        )

    def suggest_correction(self) -> str:
        return (
            "ask only for a fitted kind, or fit again with a lower minimum_probability."
        )
