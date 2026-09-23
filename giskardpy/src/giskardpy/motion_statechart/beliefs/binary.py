from __future__ import annotations

from dataclasses import dataclass

from random_events.set import Set
from random_events.variable import Symbolic
from typing_extensions import Dict, Self, Sequence, Tuple

from giskardpy.motion_statechart.beliefs.belief import (
    Belief,
    Statistic,
    VariableStatistic,
)
from giskardpy.motion_statechart.exceptions import (
    ImpossibleEvidenceError,
    NegativeLikelihoodError,
    ProbabilityOutOfRangeError,
)

# %% what a binary belief is told


def _require_probability(probability: float):
    """
    :param probability: A number meant as a probability.
    :raises ProbabilityOutOfRangeError: If it lies outside of the unit interval.
    """
    if not 0 <= probability <= 1:
        raise ProbabilityOutOfRangeError(probability=probability)


@dataclass
class BinaryTransition:
    """
    How a binary state changes over one control cycle.
    """

    persists: float
    """
    The probability that the state still holds if it held before the cycle.
    """

    arises: float
    """
    The probability that the state holds if it did not hold before the cycle.
    """

    def __post_init__(self):
        """
        :raises ProbabilityOutOfRangeError: If either probability lies outside of the
            unit interval.
        """
        _require_probability(self.persists)
        _require_probability(self.arises)


@dataclass
class BinaryEvidence:
    """
    An observation about a binary state, stated as how likely it is either way.
    """

    likelihood_if_holds: float
    """
    How likely the observation is if the state holds.
    """

    likelihood_if_not: float
    """
    How likely the observation is if the state does not hold.
    """

    def __post_init__(self):
        """
        :raises NegativeLikelihoodError: If either likelihood is negative.
        """
        for likelihood in (self.likelihood_if_holds, self.likelihood_if_not):
            if likelihood < 0:
                raise NegativeLikelihoodError(likelihood=likelihood)


# %% the belief


@dataclass
class BinaryBelief(Belief[BinaryTransition, BinaryEvidence]):
    """
    A belief about whether a binary state of the world holds, filtered by a discrete
    Bayes filter.
    """

    variable: Symbolic
    """
    The state, as a variable whose two values are whether it holds.
    """

    probability: float
    """
    The probability that the state holds.
    """

    def __post_init__(self):
        """
        :raises ProbabilityOutOfRangeError: If the probability lies outside of the unit
            interval.
        """
        _require_probability(self.probability)

    @classmethod
    def about(cls, name: str, probability: float) -> Self:
        """
        :param name: The name of the state.
        :param probability: The probability that the state holds.
        :return: A belief about a new variable for the state.
        """
        return cls(
            variable=Symbolic(name, domain=Set.from_iterable((False, True))),
            probability=probability,
        )

    @property
    def variables(self) -> Tuple[Symbolic, ...]:
        return (self.variable,)

    def predict(self, prediction: BinaryTransition):
        self.probability = (
            self.probability * prediction.persists
            + (1 - self.probability) * prediction.arises
        )

    def update(self, evidence: Sequence[BinaryEvidence]):
        """
        :raises ImpossibleEvidenceError: If the evidence is impossible under this
            belief, in which case the belief is left unchanged.
        """
        probability = self.probability
        for observation in evidence:
            supporting = probability * observation.likelihood_if_holds
            opposing = (1 - probability) * observation.likelihood_if_not
            if supporting + opposing == 0:
                raise ImpossibleEvidenceError(belief=self)
            probability = supporting / (supporting + opposing)
        self.probability = probability

    def statistics(self) -> Dict[VariableStatistic, float]:
        return {
            VariableStatistic(self.variable, Statistic.PROBABILITY): self.probability
        }
