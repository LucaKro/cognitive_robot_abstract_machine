from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    EndMotion,
)
from giskardpy.motion_statechart.motion_statechart_graph import MotionStatechartGraph
import semantic_digital_twin.spatial_types.spatial_types as cas
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


def test_motion_statechart():
    msg = MotionStatechartGraph()

    node1 = MotionStatechartNode(name=PrefixedName("muh"), motion_statechart=msg)
    node1.observation_expression = cas.TrinaryTrue
    node2 = MotionStatechartNode(name=PrefixedName("muh2"), motion_statechart=msg)
    node2.observation_expression = cas.TrinaryTrue
    node3 = MotionStatechartNode(name=PrefixedName("muh3"), motion_statechart=msg)
    node3.observation_expression = cas.TrinaryTrue
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)

    node1.start_condition = cas.trinary_logic_or(node3, node2)
    end.start_condition = node1
    assert len(msg.nodes) == 4
    assert len(msg.edges) == 3

    msg.compile()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node3] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node3] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert msg.is_end_motion()


def test_duplicate_name(): ...
