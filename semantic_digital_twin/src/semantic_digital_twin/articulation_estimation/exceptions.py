"""
Errors in estimating articulations from observed poses.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from semantic_digital_twin.exceptions import UsageError


# %% Observations
@dataclass(kw_only=True)
class TooFewObservationsError(UsageError):
    """
    Raised when fewer poses are observed than it takes to tell a moving child from a
    rigidly attached one.
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
