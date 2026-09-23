from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from random_events.variable import Variable
from typing_extensions import Dict, Generic, Sequence, Tuple, TypeVar

PredictionT = TypeVar("PredictionT")
"""
How a kind of belief is told what happens to its variables over one control cycle.
"""

EvidenceT = TypeVar("EvidenceT")
"""
What a kind of belief is updated with.
"""


class Statistic(StrEnum):
    """
    A number that summarizes what a belief holds about one variable.
    """

    MEAN = "mean"
    """
    The expected value.
    """

    VARIANCE = "variance"
    """
    How far the value is expected to scatter around the mean.
    """

    PROBABILITY = "probability"
    """
    The probability that a binary state holds.
    """


@dataclass(frozen=True)
class VariableStatistic:
    """
    One statistic of one variable of a belief.
    """

    variable: Variable
    """
    The variable summarized.
    """

    statistic: Statistic
    """
    The number that summarizes it.
    """


@dataclass
class Belief(Generic[PredictionT, EvidenceT], ABC):
    """
    What is believed about some variables of the world, carried across control cycles
    and refined by evidence.
    """

    @property
    @abstractmethod
    def variables(self) -> Tuple[Variable, ...]:
        """
        :return: The variables this belief is about.
        """

    @abstractmethod
    def predict(self, prediction: PredictionT):
        """
        Advance the belief by one control cycle, without evidence.

        :param prediction: What happens to the variables over the cycle.
        """

    @abstractmethod
    def update(self, evidence: Sequence[EvidenceT]):
        """
        Refine the belief by evidence gathered in one control cycle.

        :param evidence: The evidence, which leaves the belief unchanged if empty.
        """

    @abstractmethod
    def statistics(self) -> Dict[VariableStatistic, float]:
        """
        :return: The numbers that summarize this belief, each by the variable and
            statistic it summarizes.
        """
