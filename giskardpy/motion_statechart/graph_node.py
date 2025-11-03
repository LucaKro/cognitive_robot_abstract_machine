from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from enum import Enum

from random_events.utils import SubclassJSONSerializer
from typing_extensions import (
    Dict,
    Any,
    Self,
    Optional,
    TYPE_CHECKING,
    List,
    TypeVar,
    Union,
)

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.utils.utils import string_shortener
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import (
        MotionStatechart,
        ObservationState,
    )


class TransitionKind(Enum):
    START = 1
    PAUSE = 2
    END = 3
    RESET = 4


@dataclass
class StateTransitionCondition:
    """Encapsulates both string and expression representations of a condition."""

    motion_statechart: MotionStatechart
    kind: TransitionKind
    expression: cas.Expression = cas.TrinaryUnknown
    _parents: List[MotionStatechartNode] = field(default_factory=list, init=False)
    _child: MotionStatechartNode = field(default=None, init=False)

    @classmethod
    def create_true(
        cls, motion_statechart: MotionStatechart, kind: TransitionKind
    ) -> Self:
        return cls(
            expression=cas.TrinaryTrue, motion_statechart=motion_statechart, kind=kind
        )

    @classmethod
    def create_false(
        cls, motion_statechart: MotionStatechart, kind: TransitionKind
    ) -> Self:
        return cls(
            expression=cas.TrinaryFalse, motion_statechart=motion_statechart, kind=kind
        )

    @classmethod
    def create_unknown(
        cls, motion_statechart: MotionStatechart, kind: TransitionKind
    ) -> Self:
        return cls(
            expression=cas.TrinaryUnknown,
            motion_statechart=motion_statechart,
            kind=kind,
        )

    def update_expression(
        self, new_expression: cas.Expression, child: MotionStatechartNode
    ) -> None:
        self.expression = new_expression
        self._parents = [
            x.motion_statechart_node
            for x in new_expression.free_symbols()
            if isinstance(x, ObservationSymbol)
        ]
        self._child = child
        self.apply_to_motion_state_chart()

    def apply_to_motion_state_chart(self):
        self.motion_statechart.add_transition(self)

    def __str__(self):
        """
        Takes a logical expression, replaces the state symbols with monitor names and formats it nicely.
        """
        free_symbols = self.expression.free_symbols()
        if not free_symbols:
            return str(cas.is_true_symbol(self.expression))
        return cas.trinary_logic_to_str(self.expression)

    def __repr__(self):
        return str(self)


@dataclass(repr=False, eq=False)
class ObservationSymbol(cas.Symbol):
    name: PrefixedName = field(kw_only=True)
    motion_statechart_node: MotionStatechartNode

    def resolve(self) -> float:
        return self.motion_statechart_node.observation_state


@dataclass(repr=False, eq=False)
class LifeCycleSymbol(cas.Symbol):
    name: PrefixedName = field(kw_only=True)
    motion_statechart_node: MotionStatechartNode

    def resolve(self) -> LifeCycleValues:
        return self.motion_statechart_node.life_cycle_state


@dataclass(repr=False, eq=False)
class MotionStatechartNode(SubclassJSONSerializer):
    name: PrefixedName = field(kw_only=True)

    motion_statechart: MotionStatechart = field(kw_only=True)
    """
    Back reference to the motion statechart that owns this node.
    """
    index: Optional[int] = field(default=None, init=False)
    """
    The index of the entity in `_world.kinematic_structure`.
    """

    parent_node: MotionStatechartNode = field(default=None, init=False)

    life_cycle_symbol: LifeCycleSymbol = field(init=False)
    observation_symbol: ObservationSymbol = field(init=False)
    observation_expression: cas.Expression = field(
        default_factory=lambda: cas.TrinaryUnknown, init=False
    )

    _plot: bool = field(default=True, kw_only=True)
    _plot_style: str = field(kw_only=True)
    _plot_shape: str = field(kw_only=True)
    _plot_extra_boarder_styles: List[str] = field(default_factory=list, kw_only=True)

    _start_condition: StateTransitionCondition = field(init=False)
    _pause_condition: StateTransitionCondition = field(init=False)
    _end_condition: StateTransitionCondition = field(init=False)
    _reset_condition: StateTransitionCondition = field(init=False)

    def __post_init__(self):
        self.observation_symbol = ObservationSymbol(
            motion_statechart_node=self,
            name=PrefixedName("observation", str(self.name)),
        )
        self.life_cycle_symbol = LifeCycleSymbol(
            motion_statechart_node=self,
            name=PrefixedName("life_cycle", str(self.name)),
        )
        self._start_condition = StateTransitionCondition.create_true(
            motion_statechart=self.motion_statechart, kind=TransitionKind.START
        )
        self._pause_condition = StateTransitionCondition.create_false(
            motion_statechart=self.motion_statechart, kind=TransitionKind.PAUSE
        )
        self._end_condition = StateTransitionCondition.create_false(
            motion_statechart=self.motion_statechart, kind=TransitionKind.END
        )
        self._reset_condition = StateTransitionCondition.create_false(
            motion_statechart=self.motion_statechart, kind=TransitionKind.RESET
        )
        self.motion_statechart.add_node(self)

    @property
    def world(self) -> World:
        return self.motion_statechart.world

    def __hash__(self):
        return hash(self.name)

    @property
    def life_cycle_state(self) -> LifeCycleValues:
        return LifeCycleValues(self.motion_statechart.life_cycle_state[self])

    @property
    def observation_state(self) -> float:
        return self.motion_statechart.observation_state[self]

    @property
    def start_condition(self) -> cas.Expression:
        return self._start_condition.expression

    @start_condition.setter
    def start_condition(self, expression: cas.Expression) -> None:
        self._start_condition.update_expression(expression, self)

    @property
    def pause_condition(self) -> cas.Expression:
        return self._pause_condition.expression

    @pause_condition.setter
    def pause_condition(self, expression: cas.Expression) -> None:
        self._pause_condition.update_expression(expression, self)

    @property
    def end_condition(self) -> cas.Expression:
        return self._end_condition.expression

    @end_condition.setter
    def end_condition(self, expression: cas.Expression) -> None:
        self._end_condition.update_expression(expression, self)

    @property
    def reset_condition(self) -> cas.Expression:
        return self._reset_condition.expression

    @reset_condition.setter
    def reset_condition(self, expression: cas.Expression) -> None:
        self._reset_condition.update_expression(expression, self)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "name": self.name, "_plot": self._plot}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(name=data["name"], prefix=data["prefix"])

    def formatted_name(self, quoted: bool = False) -> str:
        formatted_name = string_shortener(
            original_str=str(self.name), max_lines=4, max_line_length=25
        )
        result = (
            f"{formatted_name}\n"
            f"----start_condition----\n"
            f"{str(self._start_condition)}\n"
            f"----pause_condition----\n"
            f"{str(self._pause_condition)}\n"
            f"----end_condition----\n"
            f"{str(self._end_condition)}\n"
            f"----reset_condition----\n"
            f"{str(self._end_condition)}"
        )
        if quoted:
            return '"' + result + '"'
        return result

    def update_expression_on_starting(
        self, expression: cas.GenericSymbolicType
    ) -> cas.GenericSymbolicType:
        if len(expression.free_symbols()) == 0:
            return expression
        return god_map.motion_statechart_manager.register_expression_updater(
            expression, self
        )


GenericMotionStatechartNode = TypeVar(
    "GenericMotionStatechartNode", bound=MotionStatechartNode
)


@dataclass(eq=False, repr=False)
class Monitor(MotionStatechartNode):
    _plot_style: str = field(default="filled, rounded", kw_only=True)
    _plot_shape: str = field(default="rectangle", kw_only=True)


@dataclass(eq=False, repr=False)
class Goal(MotionStatechartNode):
    nodes: List[MotionStatechartNode] = field(default_factory=list)
    _plot_style: str = field(default="filled", kw_only=True)
    _plot_shape: str = field(default="none", kw_only=True)

    def add_node(self, node: MotionStatechartNode) -> None:
        self.nodes.append(node)
        node.parent_node = self

    def arrange_in_sequence(self, nodes: List[MotionStatechartNode]) -> None:
        first_node = nodes[0]
        first_node.end_condition = first_node
        for node in nodes[1:]:
            node.start_condition = first_node
            node.end_condition = node
            first_node = node

    def apply_goal_conditions_to_children(self):
        for node in self.nodes:
            self.apply_start_condition_to_node(node)
            self.apply_pause_condition_to_node(node)
            self.apply_end_condition_to_node(node)
            self.apply_reset_condition_to_node(node)
            if isinstance(node, Goal):
                node.apply_goal_conditions_to_children()

    def apply_start_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_trinary_true_symbol(node.start_condition):
            node.start_condition = self.start_condition

    def apply_pause_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_trinary_false_symbol(node.pause_condition):
            node.pause_condition = self.pause_condition
        elif not cas.is_trinary_false_symbol(node.pause_condition):
            node.pause_condition = cas.trinary_logic_or(
                node.pause_condition, node.pause_condition
            )

    def apply_end_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_trinary_false_symbol(node.end_condition):
            node.end_condition = self.end_condition
        elif not cas.is_trinary_false_symbol(self.end_condition):
            node.end_condition = cas.trinary_logic_or(
                node.end_condition, self.end_condition
            )

    def apply_reset_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_trinary_false_symbol(node.reset_condition):
            node.reset_condition = self.reset_condition
        elif not cas.is_trinary_false_symbol(node.pause_condition):
            node.reset_condition = cas.trinary_logic_or(
                node.reset_condition, node.reset_condition
            )


@dataclass(eq=False, repr=False)
class PayloadMonitor(Monitor, ABC):
    """
    A monitor which uses regular python code to compute an observation state.

    Inherit from this class to implement a monitor that cannot be computed with a casadi expression.
    Implement the _compute_observation method to compute and return the observation state.
    .. warning:: Don't touch anything else in this class.
    """

    observation_expression: cas.Expression = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.observation_expression = cas.Expression(self.observation_symbol)

    def compute_observation(
        self,
    ) -> Union[
        ObservationState.TrinaryFalse,
        ObservationState.TrinaryTrue,
        ObservationState.TrinaryUnknown,
    ]:
        """
        Computes and returns the current observation state by calling _compute_observation.
        Don't override this function.

        :return: The computed observation state from the ObservationState enumeration.
        :rtype: Union[ObservationState.TrinaryFalse, ObservationState.TrinaryTrue,
                ObservationState.TrinaryUnknown]
        """
        return self._compute_observation()

    @abstractmethod
    def _compute_observation(
        self,
    ) -> Union[
        ObservationState.TrinaryFalse,
        ObservationState.TrinaryTrue,
        ObservationState.TrinaryUnknown,
    ]:
        """
        Override this function to compute the observation state, if it can't be done with a casadi expression.
        .. warning:: This method must return essentially instantly and not block the main thread.
        :return:
        """


@dataclass(eq=False, repr=False)
class ThreadPayloadMonitor(PayloadMonitor, ABC):
    """
    Payload monitor that evaluates _compute_observation in a background thread.

    - compute_observation triggers an async evaluation and immediately returns.
    - Until the first successful completion, returns TrinaryUnknown.
    - Afterwards, returns the last successfully computed value.
    """

    # Internal threading primitives
    _request_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _thread: threading.Thread = field(init=False, repr=False)

    # Cache of last successful result from _compute_observation
    _has_result: bool = field(default=False, init=False, repr=False)
    _last_result: float = field(
        default=float(cas.TrinaryUnknown.to_np()), init=False, repr=False
    )

    def __post_init__(self):
        super().__post_init__()
        # Start a daemon worker thread that computes observations when requested
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.__class__.__name__}-worker",
            daemon=True,
        )
        self._thread.start()

    def compute_observation(
        self,
    ) -> float:
        # Signal the worker to compute a fresh value if it is not already signaled.
        self._request_event.set()
        # Return the last known result (initialized to Unknown until first success)
        return self._last_result

    def _worker_loop(self):
        while not self._stop_event.is_set():
            # Wait until a request is made (wake periodically to check for stop)
            triggered = self._request_event.wait(timeout=0.1)
            if not triggered:
                continue
            # Clear early to allow new requests while we compute
            self._request_event.clear()
            try:
                result = self._compute_observation()
                # Accept only valid trinary values (floats expected)
                self._last_result = result
                self._has_result = True
            except Exception:
                # On failure, keep previous result and mark as having no new value
                pass


@dataclass(eq=False, repr=False)
class EndMotion(MotionStatechartNode):
    observation_expression: cas.Expression = field(
        default_factory=lambda: cas.TrinaryTrue, init=False
    )
    _plot_boarder_styles: List[str] = field(
        default_factory=lambda: ["rounded"], kw_only=True
    )
    _plot_style: str = field(default="filled, rounded", kw_only=True)
    _plot_shape: str = field(default="rectangle", kw_only=True)


@dataclass(eq=False, repr=False)
class CancelMotion(MotionStatechartNode):
    exception: Exception = field(kw_only=True)
    observation_expression: cas.Expression = field(
        default_factory=lambda: cas.TrinaryTrue, init=False
    )
    _plot_extra_boarder_styles: List[str] = field(
        default_factory=lambda: ["dashed, rounded"], kw_only=True
    )
    _plot_style: str = field(default="filled, rounded", kw_only=True)
    _plot_shape: str = field(default="rectangle", kw_only=True)
