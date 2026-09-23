from __future__ import annotations

from dataclasses import dataclass, field

from random_events.variable import Variable
from typing_extensions import Dict, Self

from giskardpy.motion_statechart.beliefs.belief import Belief
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
    The beliefs a motion statechart holds about the world, each reachable by the
    variables it is about.
    """

    beliefs: Dict[Variable, Belief] = field(default_factory=dict)
    """
    The belief about each variable.
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

    def add(self, belief: Belief):
        """
        :param belief: A belief about variables nothing is believed about yet.
        :raises DuplicateBeliefError: If something is already believed about one of
            its variables, in which case none of them is added.
        """
        for variable in belief.variables:
            if variable in self.beliefs:
                raise DuplicateBeliefError(variable=variable)
        for variable in belief.variables:
            self.beliefs[variable] = belief

    def belief_of(self, variable: Variable) -> Belief:
        """
        :param variable: A variable something is believed about.
        :return: The belief about it.
        :raises VariableWithoutBeliefError: If nothing is believed about it.
        """
        if variable not in self.beliefs:
            raise VariableWithoutBeliefError(variable=variable)
        return self.beliefs[variable]
