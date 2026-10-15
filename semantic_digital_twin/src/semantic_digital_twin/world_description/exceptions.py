from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from semantic_digital_twin.exceptions import UsageError

# %% inertia tensors


@dataclass
class NegativeInertiaDiagonalError(UsageError):
    """
    Raised when an inertia tensor has a negative moment about one of its axes.
    """

    data: NDArray
    """
    The tensor's matrix.
    """

    def error_message(self) -> str:
        return (
            "The moments on an inertia tensor's diagonal must be non-negative, got "
            f"{np.diag(self.data).tolist()}."
        )

    def suggest_correction(self) -> str:
        return "Check the inertia the body's description declares."


@dataclass
class AsymmetricInertiaTensorError(UsageError):
    """
    Raised when an inertia tensor is not symmetric.
    """

    data: NDArray
    """
    The tensor's matrix.
    """

    def error_message(self) -> str:
        return f"An inertia tensor must be symmetric, got {self.data.tolist()}."

    def suggest_correction(self) -> str:
        return "Give each product of inertia once, so it fills both mirrored entries."
