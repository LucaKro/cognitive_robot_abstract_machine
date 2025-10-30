import pytest

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.monitors.monitors import TrueMonitor
from giskardpy.motion_statechart.monitors.payload_monitors import Print
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body


def test_motion_statechart():
    msg = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = TrueMonitor(name=PrefixedName("muh2"), motion_statechart=msg)
    node3 = TrueMonitor(name=PrefixedName("muh3"), motion_statechart=msg)
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


def test_duplicate_name():
    msg = MotionStatechart(World())

    with pytest.raises(ValueError):
        cas.Symbol(name=PrefixedName("muh"))
        MotionStatechartNode(name=PrefixedName("muh"), motion_statechart=msg)
        MotionStatechartNode(name=PrefixedName("muh"), motion_statechart=msg)


def test_print():
    msg = MotionStatechart(World())
    print_node1 = Print(name=PrefixedName("cow"), message="muh", motion_statechart=msg)
    print_node2 = Print(name=PrefixedName("cow2"), message="muh", motion_statechart=msg)

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node1.start_condition = print_node1
    print_node2.start_condition = node1
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    end.start_condition = print_node2
    assert len(msg.nodes) == 4
    assert len(msg.edges) == 3

    msg.compile()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert msg.is_end_motion()


def test_cancel_motion():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    cancel = CancelMotion(
        name=PrefixedName("done"), motion_statechart=msg, exception=Exception("test")
    )
    cancel.start_condition = node1

    msg.compile()
    msg.tick()  # first tick, cancel motion node1 turns true
    msg.tick()  # second tick, cancel goes into running
    with pytest.raises(Exception):
        msg.tick()  # third tick, cancel goes true and triggers


def test_joint_goal():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        tip = Body(name=PrefixedName("tip"))
        ul = DerivativeMap()
        ul.velocity = 1
        ll = DerivativeMap()
        ll.velocity = -1
        dof = DegreeOfFreedom(
            name=PrefixedName("dof"), lower_limits=ll, upper_limits=ul
        )
        world.add_degree_of_freedom(dof)
        root_C_tip = RevoluteConnection(
            parent=root, child=tip, axis=cas.Vector3.Z(), dof_name=dof.name
        )
        world.add_connection(root_C_tip)

    msg = MotionStatechart(world)

    task1 = JointPositionList(
        name=PrefixedName("task1"), goal_state={root_C_tip: 1}, motion_statechart=msg
    )
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    end.start_condition = task1

    msg.compile()
    msg.tick()
    assert msg.observation_state[task1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[task1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED

    root_C_tip.position = 1

    msg.tick()
    assert msg.observation_state[task1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[task1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    msg.tick()
    assert msg.observation_state[task1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue
    assert msg.life_cycle_state[task1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING


def test_reset():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = TrueMonitor(name=PrefixedName("muh2"), motion_statechart=msg)
    node3 = TrueMonitor(name=PrefixedName("muh3"), motion_statechart=msg)
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    node1.reset_condition = node2
    node2.start_condition = node1
    node3.start_condition = node2
    node2.end_condition = node2
    end.start_condition = cas.trinary_logic_and(node1, node2, node3)

    msg.compile()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.DONE
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.DONE
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.DONE
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.DONE
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert msg.is_end_motion()


@pytest.mark.skip(reason="not yet implemented")
def test_assign_node_to_observation_state():
    assert False


@pytest.mark.skip(reason="not yet implemented")
def test_nested_goals():
    assert False


def test_goal():
    msg = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)

    goal = Goal(name=PrefixedName("goal"), motion_statechart=msg)
    sub_node1 = TrueMonitor(name=PrefixedName("sub muh1"), motion_statechart=msg)
    sub_node2 = TrueMonitor(name=PrefixedName("sub muh2"), motion_statechart=msg)
    goal.add_node(sub_node1)
    goal.add_node(sub_node2)
    sub_node1.end_condition = sub_node1
    sub_node2.start_condition = sub_node1
    goal.observation_expression = sub_node2
    goal.start_condition = node1

    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    end.start_condition = goal

    msg.compile()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[goal] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[sub_node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[sub_node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[goal] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[goal] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[sub_node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[sub_node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[goal] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[goal] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[sub_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[sub_node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[goal] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[goal] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[sub_node1] == msg.life_cycle_state.DONE
    assert msg.life_cycle_state[sub_node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[goal] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[goal] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[sub_node1] == msg.life_cycle_state.DONE
    assert msg.life_cycle_state[sub_node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[goal] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[goal] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[sub_node1] == msg.life_cycle_state.DONE
    assert msg.life_cycle_state[sub_node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[goal] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[goal] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[sub_node1] == msg.life_cycle_state.DONE
    assert msg.life_cycle_state[sub_node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[goal] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert msg.is_end_motion()
