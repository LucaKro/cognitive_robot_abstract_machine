from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import Union, FrozenSet, List

import numpy as np
from krrood.symbolic_math.symbolic_math import Scalar

goal_parameter = Union[str, float, bool, dict, list, IntEnum, None]


# %% life cycle states


class LifeCycleValues(IntEnum):
    """
    Where a node is in its own execution, see
    :class:`~giskardpy.motion_statechart.motion_statechart.MotionStatechart`.
    """

    NOT_STARTED = 0
    """
    The node has not run yet.

    Its observation is forced back to unknown every tick.
    """

    RUNNING = 1
    """
    The node is active: its constraints carry weight, its observation expression is
    evaluated and
    :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_tick` is
    called.
    """

    PAUSED = 2
    """
    The node was running and its pause condition became true.

    Its constraints are deactivated and its observation is frozen at the last value.
    """

    SUCCEEDED = 3
    """
    The node ended because its own success condition became true.
    """

    FAILED = 4
    """
    The node ended because its own failure condition became true.
    """

    INTERRUPTED = 5
    """
    The node was cut off by an ancestor ending, so it neither succeeded nor failed on
    its own terms.
    """

    @classmethod
    def terminal_states(cls) -> FrozenSet[LifeCycleValues]:
        """
        :return: The states a node can only leave by being reset.
        """
        return frozenset({cls.SUCCEEDED, cls.FAILED, cls.INTERRUPTED})

    @property
    def is_terminal(self) -> bool:
        """
        :return: Whether a node in this state has ended.
        """
        return self in self.terminal_states()


class FloatEnum(float, Enum):
    """
    Enum where members are also (and must be) floats.
    """


class ObservationStateValues(FloatEnum):
    """
    The trinary truth values used throughout the motion statechart.
    """

    FALSE = float(Scalar.const_false())
    UNKNOWN = float(Scalar.const_trinary_unknown())
    TRUE = float(Scalar.const_true())


# %% life cycle predicates


@dataclass(frozen=True)
class LifeCyclePredicateDefinition:
    """
    The truth table of a test on a node's life cycle state.
    """

    true_states: FrozenSet[LifeCycleValues]
    """
    The states in which the predicate is true.
    """

    unknown_states: FrozenSet[LifeCycleValues] = frozenset()
    """
    The states in which the predicate has no answer yet.

    Every state that is neither here nor in :attr:`true_states` makes the predicate
    false.
    """

    def truth_value(self, life_cycle_value: LifeCycleValues) -> ObservationStateValues:
        """
        :param life_cycle_value: The state to evaluate the predicate in.
        :return: The trinary value the predicate takes in that state.
        """
        if life_cycle_value in self.true_states:
            return ObservationStateValues.TRUE
        if life_cycle_value in self.unknown_states:
            return ObservationStateValues.UNKNOWN
        return ObservationStateValues.FALSE

    def lookup_table(self) -> np.ndarray:
        """
        :return: The truth value per life cycle state, indexed by that state's value.
        """
        return np.array(
            [float(self.truth_value(state)) for state in sorted(LifeCycleValues)],
            dtype=np.float64,
        )


class LifeCyclePredicate(Enum):
    """
    A test on a node's life cycle state that may be used in transition conditions.

    Verdict predicates are trinary: they stay unknown until the node terminates, because
    *how* a node ended has no answer before it ends. Phase predicates are binary, because
    *where* a node is right now always has one.
    """

    IS_NOT_STARTED = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.NOT_STARTED})
    )
    IS_RUNNING = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.RUNNING})
    )
    IS_PAUSED = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.PAUSED})
    )
    IS_TERMINATED = LifeCyclePredicateDefinition(
        true_states=LifeCycleValues.terminal_states()
    )
    IS_SUCCEEDED = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.SUCCEEDED}),
        unknown_states=frozenset(LifeCycleValues) - LifeCycleValues.terminal_states(),
    )
    IS_FAILED = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.FAILED}),
        unknown_states=frozenset(LifeCycleValues) - LifeCycleValues.terminal_states(),
    )
    IS_INTERRUPTED = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.INTERRUPTED}),
        unknown_states=frozenset(LifeCycleValues) - LifeCycleValues.terminal_states(),
    )

    @property
    def attribute_name(self) -> str:
        """
        :return: The name this predicate is reached under on a node, also used to render
            it inside a condition.
        """
        return self.name.lower()

    @classmethod
    def from_attribute_name(cls, attribute_name: str) -> LifeCyclePredicate:
        """
        :param attribute_name: The name produced by :attr:`attribute_name`.
        :return: The matching predicate.
        """
        return cls[attribute_name.upper()]


# %% weights and transitions


class DefaultWeights(FloatEnum):
    WEIGHT_MAXIMUM = 10000.0
    WEIGHT_ABOVE_COLLISION_AVOIDANCE = 2500.0
    WEIGHT_COLLISION_AVOIDANCE = 50.0
    WEIGHT_BELOW_COLLISION_AVOIDANCE = 1.0
    WEIGHT_MINIMUM = 0.0


class TransitionKind(Enum):
    START = 1
    """
    Transitions nodes from NOT_STARTED to RUNNING.
    """

    PAUSE = 2
    """
    Transitions nodes from RUNNING to PAUSED if True, or back if False.
    """

    SUCCESS = 3
    """
    Transitions nodes from RUNNING or PAUSED to SUCCEEDED, and their descendants to
    INTERRUPTED.
    """

    RESET = 4
    """
    Transitions nodes from any state to NOT_STARTED.
    """

    FAILURE = 5
    """
    Transitions nodes from RUNNING or PAUSED to FAILED, and their descendants to
    INTERRUPTED.
    """

    @classmethod
    def verdict_kinds(cls) -> List[TransitionKind]:
        """
        :return: The transitions that end a node, in the order they take priority.
        """
        return [cls.FAILURE, cls.SUCCESS]
