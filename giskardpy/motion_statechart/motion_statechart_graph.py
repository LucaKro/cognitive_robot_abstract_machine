from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rustworkx as rx
from typing_extensions import List, MutableMapping, ClassVar, Self, Type

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    StateTransitionCondition,
    EndMotion,
    CancelMotion,
    GenericMotionStatechartNode,
)


@dataclass
class State(MutableMapping[MotionStatechartNode, float]):
    motion_statechart_graph: MotionStatechartGraph
    default_value: float
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def grow(self) -> None:
        self.data = np.append(self.data, self.default_value)

    def life_cycle_symbols(self) -> List[cas.Symbol]:
        return [node.life_cycle_symbol for node in self.motion_statechart_graph.nodes]

    def observation_symbols(self) -> List[MotionStatechartNode]:
        return self.motion_statechart_graph.nodes

    def __getitem__(self, node: MotionStatechartNode) -> float:
        return float(self.data[node.index])

    def __setitem__(self, node: MotionStatechartNode, value: float) -> None:
        self.data[node.index] = value

    def __delitem__(self, node: MotionStatechartNode) -> None:
        self.data = np.delete(self.data, node.index, axis=1)

    def __iter__(self) -> iter:
        return iter(self.data)

    def __len__(self) -> int:
        return self.data.shape[0]

    def keys(self) -> List[MotionStatechartNode]:
        return self.motion_statechart_graph.nodes

    def items(self) -> List[tuple[MotionStatechartNode, float]]:
        return [(node, self[node]) for node in self.motion_statechart_graph.nodes]

    def values(self) -> List[float]:
        return [self[node] for node in self.keys()]

    def __contains__(self, node: MotionStatechartNode) -> bool:
        return node in self.motion_statechart_graph.nodes

    def __deepcopy__(self, memo) -> Self:
        """
        Create a deep copy of the WorldState.
        """
        return State(
            motion_statechart_graph=self.motion_statechart_graph,
            default_value=self.default_value,
            data=self.data.copy(),
        )


@dataclass
class LifeCycleState(State):
    NOT_STARTED: ClassVar[float] = 0.0
    RUNNING: ClassVar[float] = 1.0
    PAUSED: ClassVar[float] = 2.0
    DONE: ClassVar[float] = 3.0

    default_value: float = NOT_STARTED
    _compiled_updater: cas.CompiledFunction = field(init=False)

    def compile(self):
        state_updater = []
        for node in self.motion_statechart_graph.nodes:
            state_symbol = node.life_cycle_symbol

            not_started_transitions = cas.if_else(
                condition=cas.is_trinary_true(node.start_condition),
                if_result=cas.Expression(LifeCycleState.RUNNING),
                else_result=cas.Expression(LifeCycleState.NOT_STARTED),
            )
            running_transitions = cas.if_cases(
                cases=[
                    (
                        cas.is_trinary_true(node.reset_condition),
                        cas.Expression(LifeCycleState.NOT_STARTED),
                    ),
                    (
                        cas.is_trinary_true(node.end_condition),
                        cas.Expression(LifeCycleState.DONE),
                    ),
                    (
                        cas.is_trinary_true(node.pause_condition),
                        cas.Expression(LifeCycleState.PAUSED),
                    ),
                ],
                else_result=cas.Expression(LifeCycleState.RUNNING),
            )
            pause_transitions = cas.if_cases(
                cases=[
                    (
                        cas.is_trinary_true(node.reset_condition),
                        cas.Expression(LifeCycleState.NOT_STARTED),
                    ),
                    (
                        cas.is_trinary_true(node.end_condition),
                        cas.Expression(LifeCycleState.DONE),
                    ),
                    (
                        cas.is_trinary_false(node.pause_condition),
                        cas.Expression(LifeCycleState.RUNNING),
                    ),
                ],
                else_result=cas.Expression(LifeCycleState.PAUSED),
            )
            ended_transitions = cas.if_else(
                condition=cas.is_trinary_true(node.reset_condition),
                if_result=cas.Expression(LifeCycleState.NOT_STARTED),
                else_result=cas.Expression(LifeCycleState.DONE),
            )

            state_machine = cas.if_eq_cases(
                a=state_symbol,
                b_result_cases=[
                    (LifeCycleState.NOT_STARTED, not_started_transitions),
                    (LifeCycleState.RUNNING, running_transitions),
                    (LifeCycleState.PAUSED, pause_transitions),
                    (LifeCycleState.DONE, ended_transitions),
                ],
                else_result=cas.Expression(state_symbol),
            )
            state_updater.append(state_machine)
        state_updater = cas.Expression(state_updater)
        self._compiled_updater = state_updater.compile(
            parameters=[self.observation_symbols(), self.life_cycle_symbols()],
            sparse=False,
        )

    def update_state(self, observation_state: np.ndarray):
        self.data = self._compiled_updater(observation_state, self.data)


@dataclass
class ObservationState(State):
    TrinaryFalse: ClassVar[float] = float(cas.TrinaryFalse.to_np())
    TrinaryUnknown: ClassVar[float] = float(cas.TrinaryUnknown.to_np())
    TrinaryTrue: ClassVar[float] = float(cas.TrinaryTrue.to_np())

    default_value: float = float(cas.TrinaryUnknown.to_np())

    _compiled_updater: cas.CompiledFunction = field(init=False)

    def compile(self):
        observation_state_updater = []
        for node in self.motion_statechart_graph.nodes:
            state_f = cas.if_eq_cases(
                a=node.life_cycle_symbol,
                b_result_cases=[
                    (int(LifeCycleState.RUNNING), node.observation_expression),
                    (
                        int(LifeCycleState.NOT_STARTED),
                        cas.TrinaryUnknown,
                    ),
                ],
                else_result=cas.Expression(node),
            )
            observation_state_updater.append(state_f)
        self._compiled_updater = cas.Expression(observation_state_updater).compile(
            parameters=[self.observation_symbols(), self.life_cycle_symbols()],
            sparse=False,
        )

    def update_state(self, life_cycle_state: np.ndarray):
        self.data = self._compiled_updater(self.data, life_cycle_state)


@dataclass
class MotionStatechartGraph:
    rx_graph: rx.PyDiGraph[MotionStatechartNode] = field(
        default_factory=lambda: rx.PyDAG(multigraph=True), init=False, repr=False
    )
    observation_state: ObservationState = field(init=False)
    life_cycle_state: LifeCycleState = field(init=False)
    """
    1. evaluate observation state
        input: anything
        output: observation state
    2. evaluate life cycle state
        input: observation state, life cycle state
        output: life cycle state
    """

    def __post_init__(self):
        self.life_cycle_state = LifeCycleState(self)
        self.observation_state = ObservationState(self)

    @property
    def nodes(self) -> List[MotionStatechartNode]:
        return list(self.rx_graph.nodes())

    @property
    def edges(self) -> List[StateTransitionCondition]:
        return self.rx_graph.edges()

    def add_node(self, node: MotionStatechartNode):
        node._motion_statechart = self
        node.index = self.rx_graph.add_node(node)
        self.life_cycle_state.grow()
        self.observation_state.grow()

    def add_transition(
        self,
        condition: StateTransitionCondition,
    ):
        for parent in condition._parents:
            self.rx_graph.add_edge(condition._child.index, parent.index, condition)

    def remove_transition(self, condition: StateTransitionCondition):
        to_delete = [
            e
            for e in self.rx_graph.edges()
            if self.rx_graph.get_edge_data_by_index(e) is condition
        ]

        for e in to_delete:
            self.rx_graph.remove_edge_from_index(e)

    def compile(self):
        self.observation_state.compile()
        self.life_cycle_state.compile()

    def update_observation_state(self):
        self.observation_state.update_state(self.life_cycle_state.data)

    def update_life_cycle_state(self):
        self.life_cycle_state.update_state(self.observation_state.data)

    def tick(self):
        self.update_observation_state()
        self.update_life_cycle_state()
        self.raise_if_cancel_motion()

    def get_nodes_by_type(
        self, node_type: Type[GenericMotionStatechartNode]
    ) -> List[GenericMotionStatechartNode]:
        return [node for node in self.nodes if isinstance(node, node_type)]

    def is_end_motion(self) -> bool:
        return any(
            self.observation_state[node] == ObservationState.TrinaryTrue
            for node in self.get_nodes_by_type(EndMotion)
        )

    def raise_if_cancel_motion(self):
        for node in self.get_nodes_by_type(CancelMotion):
            if self.observation_state[node] == ObservationState.TrinaryTrue:
                raise node.exception
