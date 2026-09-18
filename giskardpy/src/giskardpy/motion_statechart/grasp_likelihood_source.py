from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from krrood.symbolic_math.symbolic_math import FloatVariable


@dataclass
class GraspLikelihoodSource(ABC):
    """
    Something that reports how strongly a body is currently held by a gripper.

    Implemented by whatever measures that, so a node filtering the measurement does not
    depend on how it was taken.
    """

    @property
    @abstractmethod
    def likelihood(self) -> FloatVariable:
        """
        :return: The variable the share of rays that hit the body is published to.
        """

    @property
    @abstractmethod
    def sample_size(self) -> int:
        """
        :return: How many rays that share is reported out of, which is what says how far
            it can be trusted.
        """
