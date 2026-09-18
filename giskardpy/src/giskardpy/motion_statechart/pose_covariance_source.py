from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from semantic_digital_twin.spatial_types import PoseCovariance


@dataclass
class PoseCovarianceSource(ABC):
    """
    Something that reports how uncertain the pose it most recently observed was.

    Implemented by whatever receives poses from outside the process, so that a node
    reading the uncertainty does not depend on where the poses come from.
    """

    @property
    @abstractmethod
    def pose_covariance(self) -> PoseCovariance | None:
        """
        :return: The covariance of the most recently observed pose, or ``None`` while
            nothing has been observed yet.
        """
