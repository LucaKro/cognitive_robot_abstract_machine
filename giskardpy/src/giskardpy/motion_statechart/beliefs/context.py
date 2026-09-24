from __future__ import annotations

from dataclasses import dataclass, field

from probabilistic_model.probabilistic_model import ProbabilisticModel
from random_events.variable import Variable
from typing_extensions import Dict, Self

from giskardpy.motion_statechart.context import (
    ContextExtension,
    MotionStatechartContext,
)
from giskardpy.motion_statechart.exceptions import (
    DuplicateBeliefError,
    VariableWithoutBeliefError,
)


@dataclass
class BeliefContext(ContextExtension):
    """
    What a motion statechart believes about the world: one distribution per group of
    variables it estimates, each reachable by the variables it is over.
    """

    distributions: Dict[Variable, ProbabilisticModel] = field(default_factory=dict)
    """
    The distribution over each variable.
    """

    @classmethod
    def from_context(cls, context: MotionStatechartContext) -> Self:
        """
        :param context: The context of a motion statechart.
        :return: The beliefs of the statechart, added to its context first if it held
            none yet.
        """
        if cls not in context.extensions:
            context.add_extension(cls())
        return context.require_extension(cls)

    def add(self, distribution: ProbabilisticModel):
        """
        :param distribution: A distribution over variables nothing is believed about yet.
        :raises DuplicateBeliefError: If something is already believed about one of its
            variables, in which case none of them is added.
        """
        for variable in distribution.variables:
            if variable in self.distributions:
                raise DuplicateBeliefError(variable=variable)
        self._assign(distribution)

    def replace(self, distribution: ProbabilisticModel):
        """
        :param distribution: What is now believed about variables something was already
            believed about.
        :raises VariableWithoutBeliefError: If nothing is believed about one of its
            variables yet, in which case none of them is replaced.
        """
        for variable in distribution.variables:
            if variable not in self.distributions:
                raise VariableWithoutBeliefError(variable=variable)
        self._assign(distribution)

    def distribution_of(self, variable: Variable) -> ProbabilisticModel:
        """
        :param variable: A variable something is believed about.
        :return: The distribution over it.
        :raises VariableWithoutBeliefError: If nothing is believed about it.
        """
        if variable not in self.distributions:
            raise VariableWithoutBeliefError(variable=variable)
        return self.distributions[variable]

    def _assign(self, distribution: ProbabilisticModel):
        """
        Lead each variable of the distribution to it.
        """
        for variable in distribution.variables:
            self.distributions[variable] = distribution
