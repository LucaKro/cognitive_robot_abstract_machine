from __future__ import annotations

from dataclasses import dataclass, field

from random_events.variable import Continuous
from typing_extensions import Dict

from typing_extensions import Self

from giskardpy.motion_statechart.beliefs.gaussian import GaussianBelief
from giskardpy.motion_statechart.context import (
    ContextExtension,
    MotionStatechartContext,
)
from giskardpy.motion_statechart.exceptions import (
    DuplicateBeliefError,
    UnknownBeliefError,
)


@dataclass
class BeliefContext(ContextExtension):
    """
    The beliefs a motion statechart carries from one control cycle to the next.

    Add it to a
    :class:`~giskardpy.motion_statechart.context.MotionStatechartContext` with
    ``add_extension``, and read it back inside a node with ``require_extension``.
    """

    beliefs: Dict[Continuous, GaussianBelief] = field(default_factory=dict)
    """
    The belief about each quantity, with a belief about several quantities listed under
    every one of them.
    """

    @classmethod
    def of(cls, context: MotionStatechartContext) -> Self:
        """
        :param context: The statechart context whose beliefs to read.
        :return: The beliefs it carries, which it starts carrying now if it had none.
        """
        beliefs = context.extensions.get(cls)
        if beliefs is None:
            beliefs = cls()
            context.add_extension(beliefs)
        return beliefs

    def add(self, belief: GaussianBelief) -> None:
        """
        Make `belief` the belief about every quantity it is about.

        :param belief: The belief to carry from now on.
        :raises DuplicateBeliefError: If one of those quantities already has a belief,
            in which case nothing is added at all.
        """
        for variable in belief.quantities:
            if variable in self.beliefs:
                raise DuplicateBeliefError(variable=variable)
        for variable in belief.quantities:
            self.beliefs[variable] = belief

    def require(self, variable: Continuous) -> GaussianBelief:
        """
        :param variable: The quantity to read a belief about.
        :return: The belief about it.
        :raises UnknownBeliefError: If nothing estimates that quantity.
        """
        belief = self.beliefs.get(variable)
        if belief is None:
            raise UnknownBeliefError(variable=variable)
        return belief
