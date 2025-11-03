from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rustworkx as rx
from typing_extensions import List, MutableMapping, ClassVar, Self, Type

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    StateTransitionCondition,
    Goal,
    EndMotion,
    CancelMotion,
    GenericMotionStatechartNode,
    ObservationSymbol,
    LifeCycleSymbol,
)
from giskardpy.motion_statechart.plotters.graphviz import MotionStatechartGraphviz
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World


@dataclass(repr=False)
class State(MutableMapping[MotionStatechartNode, float]):
    motion_statechart: MotionStatechart
    default_value: float
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def grow(self) -> None:
        self.data = np.append(self.data, self.default_value)

    def life_cycle_symbols(self) -> List[LifeCycleSymbol]:
        return [node.life_cycle_symbol for node in self.motion_statechart.nodes]

    def observation_symbols(self) -> List[ObservationSymbol]:
        return [node.observation_symbol for node in self.motion_statechart.nodes]

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
        return self.motion_statechart.nodes

    def items(self) -> List[tuple[MotionStatechartNode, float]]:
        return [(node, self[node]) for node in self.motion_statechart.nodes]

    def values(self) -> List[float]:
        return [self[node] for node in self.keys()]

    def __contains__(self, node: MotionStatechartNode) -> bool:
        return node in self.motion_statechart.nodes

    def __deepcopy__(self, memo) -> Self:
        """
        Create a deep copy of the WorldState.
        """
        return State(
            motion_statechart=self.motion_statechart,
            default_value=self.default_value,
            data=self.data.copy(),
        )

    def __str__(self) -> str:
        return str({str(symbol.name): value for symbol, value in self.items()})

    def __repr__(self) -> str:
        return str(self)


@dataclass(repr=False)
class LifeCycleState(State):

    default_value: float = LifeCycleValues.NOT_STARTED
    _compiled_updater: cas.CompiledFunction = field(init=False)

    def compile(self):
        state_updater = []
        for node in self.motion_statechart.nodes:
            state_symbol = node.life_cycle_symbol

            not_started_transitions = cas.if_else(
                condition=cas.is_trinary_true(node.start_condition),
                if_result=cas.Expression(LifeCycleValues.RUNNING),
                else_result=cas.Expression(LifeCycleValues.NOT_STARTED),
            )
            running_transitions = cas.if_cases(
                cases=[
                    (
                        cas.is_trinary_true(node.reset_condition),
                        cas.Expression(LifeCycleValues.NOT_STARTED),
                    ),
                    (
                        cas.is_trinary_true(node.end_condition),
                        cas.Expression(LifeCycleValues.DONE),
                    ),
                    (
                        cas.is_trinary_true(node.pause_condition),
                        cas.Expression(LifeCycleValues.PAUSED),
                    ),
                ],
                else_result=cas.Expression(LifeCycleValues.RUNNING),
            )
            pause_transitions = cas.if_cases(
                cases=[
                    (
                        cas.is_trinary_true(node.reset_condition),
                        cas.Expression(LifeCycleValues.NOT_STARTED),
                    ),
                    (
                        cas.is_trinary_true(node.end_condition),
                        cas.Expression(LifeCycleValues.DONE),
                    ),
                    (
                        cas.is_trinary_false(node.pause_condition),
                        cas.Expression(LifeCycleValues.RUNNING),
                    ),
                ],
                else_result=cas.Expression(LifeCycleValues.PAUSED),
            )
            ended_transitions = cas.if_else(
                condition=cas.is_trinary_true(node.reset_condition),
                if_result=cas.Expression(LifeCycleValues.NOT_STARTED),
                else_result=cas.Expression(LifeCycleValues.DONE),
            )

            state_machine = cas.if_eq_cases(
                a=state_symbol,
                b_result_cases=[
                    (LifeCycleValues.NOT_STARTED, not_started_transitions),
                    (LifeCycleValues.RUNNING, running_transitions),
                    (LifeCycleValues.PAUSED, pause_transitions),
                    (LifeCycleValues.DONE, ended_transitions),
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

    def __str__(self) -> str:
        return str(
            {
                str(symbol.name): LifeCycleValues(value).name
                for symbol, value in self.items()
            }
        )


@dataclass(repr=False)
class ObservationState(State):
    TrinaryFalse: ClassVar[float] = float(cas.TrinaryFalse.to_np())
    TrinaryUnknown: ClassVar[float] = float(cas.TrinaryUnknown.to_np())
    TrinaryTrue: ClassVar[float] = float(cas.TrinaryTrue.to_np())

    default_value: float = float(cas.TrinaryUnknown.to_np())

    _compiled_updater: cas.CompiledFunction = field(init=False)

    def compile(self):
        observation_state_updater = []
        for node in self.motion_statechart.nodes:
            state_f = cas.if_eq_cases(
                a=node.life_cycle_symbol,
                b_result_cases=[
                    (
                        int(LifeCycleValues.RUNNING),
                        node.create_observation_expression(),
                    ),
                    (
                        int(LifeCycleValues.NOT_STARTED),
                        cas.TrinaryUnknown,
                    ),
                ],
                else_result=cas.Expression(node.observation_symbol),
            )
            observation_state_updater.append(state_f)
        self._compiled_updater = cas.Expression(observation_state_updater).compile(
            parameters=[
                self.observation_symbols(),
                self.life_cycle_symbols(),
                self.motion_statechart.world.get_world_state_symbols(),
            ],
            sparse=False,
        )

    def update_state(self, life_cycle_state: np.ndarray, world_state: np.ndarray):
        self.data = self._compiled_updater(self.data, life_cycle_state, world_state)


@dataclass
class MotionStatechart:
    """
    Represents a motion statechart.
    A motion statechart is a directed graph of nodes and edges.
    Nodes have two states: observation state and life cycle state.
    Life cycle states indicate the current state in the life cycle of the node:
        - NOT_STARTED: the node has not started yet.
        - RUNNING: the node is running.
        - PAUSED: the node is paused.
        - DONE: the node has ended.
    Out of these 4 states, nodes are only "active" if they are in the RUNNING state.
    Observation states indicate the current observation of the node:
        - TrinaryFalse: the thing the node is observing is not True.
        - TrinaryUnknown: the node has not yet made an observation or it cannot determine its truth value yet.
        - TrinaryTrue: the thing the node is observing is True.
    Nodes are connected with edges, or transitions.
    There are 4 types of transitions:
        - start condition: If True, the node transitions from NOT_STARTED to RUNNING.
        - pause condition: If True, the node transitions from RUNNING to PAUSED.
                           If False, the node transitions from PAUSED to RUNNING.
        - end condition: If True, the node transitions from RUNNING or PAUSED to DONE.
        - reset condition: If True, the node transitions from any state to NOT_STARTED.
    If multiple conditions are met, the following order is used:
        1. reset condition
        2. end condition
        3. pause condition
        4. start condition
    How to use this class:
        1. initialized with a world
        2. add nodes.
        3. set the transition conditions of nodes
        4. compile the motion statechart.
        5. call tick() to update the observation state and life cycle state.
            tick() will raise an exception if the cancel motion condition is met.
        6. call is_end_motion() to check if the motion is done.
    """

    world: World
    """
    Reference to the world, where the motion statechart is defined.
    Symbols to the degree of freedom of the world can be used by nodes.
    """

    rx_graph: rx.PyDiGraph[MotionStatechartNode] = field(
        default_factory=lambda: rx.PyDAG(multigraph=True), init=False, repr=False
    )
    """
    The underlying graph of the motion statechart.
    """

    observation_state: ObservationState = field(init=False)
    """
    Combined representation of the observation state of the motion statechart, to enable an efficient tick().
    """

    life_cycle_state: LifeCycleState = field(init=False)
    """
    Combined representation of the life cycle state of the motion statechart, to enable an efficient tick().
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
        if self.get_node_by_name(node.name):
            raise ValueError(f"Node {node.name} already exists.")
        node._motion_statechart = self
        node.index = self.rx_graph.add_node(node)
        self.life_cycle_state.grow()
        self.observation_state.grow()

    def get_node_by_name(self, name: PrefixedName) -> List[MotionStatechartNode]:
        return [node for node in self.nodes if node.name == name]

    def add_transition(
        self,
        condition: StateTransitionCondition,
    ):
        for parent in condition._parents:
            self.rx_graph.add_edge(condition._child.index, parent.index, condition)

    def compile(self):
        for goal in self.get_nodes_by_type(Goal):
            goal.apply_goal_conditions_to_children()
        for node in self.nodes:
            node.build_common()
        self.observation_state.compile()
        self.life_cycle_state.compile()

    def _update_observation_state(self):
        self.observation_state.update_state(
            self.life_cycle_state.data, self.world.state.data
        )
        for node in self.nodes:
            if self.life_cycle_state[node] == LifeCycleValues.RUNNING:
                observation_overwrite = node.on_running()
                if observation_overwrite is not None:
                    self.observation_state[node] = observation_overwrite

    def _update_life_cycle_state(self):
        self.life_cycle_state.update_state(self.observation_state.data)

    def tick(self):
        self._update_observation_state()
        self._update_life_cycle_state()
        self._raise_if_cancel_motion()

    def get_nodes_by_type(
        self, node_type: Type[GenericMotionStatechartNode]
    ) -> List[GenericMotionStatechartNode]:
        return [node for node in self.nodes if isinstance(node, node_type)]

    def is_end_motion(self) -> bool:
        return any(
            self.observation_state[node] == ObservationState.TrinaryTrue
            for node in self.get_nodes_by_type(EndMotion)
        )

    def _raise_if_cancel_motion(self):
        for node in self.get_nodes_by_type(CancelMotion):
            if self.observation_state[node] == ObservationState.TrinaryTrue:
                raise node.exception

    def draw(self):
        graph = MotionStatechartGraphviz(self).to_dot_graph()
        file_name = "muh.pdf"
        # create_path(file_name)
        graph.write_pdf(file_name)
        print(f"Saved task graph at {file_name}.")
