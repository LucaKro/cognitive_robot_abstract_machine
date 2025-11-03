import pytest
import time
from dataclasses import dataclass

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.graph_node import Goal
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.graph_node import ThreadPayloadMonitor
from giskardpy.motion_statechart.monitors.monitors import TrueMonitor
from giskardpy.motion_statechart.monitors.payload_monitors import Print
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    ObservationState,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body


def test_condition_to_str():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = TrueMonitor(name=PrefixedName("muh2"), motion_statechart=msg)
    node3 = TrueMonitor(name=PrefixedName("muh3"), motion_statechart=msg)
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)

    end.start_condition = cas.trinary_logic_and(
        node1.observation_symbol,
        cas.trinary_logic_or(
            node2.observation_symbol, cas.trinary_logic_not(node3.observation_symbol)
        ),
    )
    a = str(end._start_condition)
    assert a == '("muh/observation" and ("muh2/observation" or not "muh3/observation"))'


def test_motion_statechart_to_dot():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = TrueMonitor(name=PrefixedName("muh2"), motion_statechart=msg)
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    node1.end_condition = node2.observation_symbol
    end.start_condition = cas.trinary_logic_and(
        node1.observation_symbol, node2.observation_symbol
    )
    msg.draw()


@pytest.mark.skip(reason="not implemented yet")
def test_self_start_condition():
    pass


def test_motion_statechart():
    msg = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = TrueMonitor(name=PrefixedName("muh2"), motion_statechart=msg)
    node3 = TrueMonitor(name=PrefixedName("muh3"), motion_statechart=msg)
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)

    node1.start_condition = cas.trinary_logic_or(
        node3.observation_symbol, node2.observation_symbol
    )
    end.start_condition = node1.observation_symbol
    assert len(msg.nodes) == 4
    assert len(msg.edges) == 3

    msg.compile()
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert node2.observation_state == msg.observation_state.TrinaryUnknown
    assert node3.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node3.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert node2.observation_state == msg.observation_state.TrinaryUnknown
    assert node3.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert node2.observation_state == msg.observation_state.TrinaryTrue
    assert node3.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert node2.observation_state == msg.observation_state.TrinaryTrue
    assert node3.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msg.is_end_motion()
    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert node2.observation_state == msg.observation_state.TrinaryTrue
    assert node3.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryTrue
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msg.is_end_motion()


def test_duplicate_name():
    msg = MotionStatechart(World())

    with pytest.raises(ValueError):
        cas.Symbol(name=PrefixedName("muh"))
        TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
        TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)


def test_print():
    msg = MotionStatechart(World())
    print_node1 = Print(name=PrefixedName("cow"), message="muh", motion_statechart=msg)
    print_node2 = Print(name=PrefixedName("cow2"), message="muh", motion_statechart=msg)

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node1.start_condition = print_node1.observation_symbol
    print_node2.start_condition = node1.observation_symbol
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    end.start_condition = print_node2.observation_symbol
    assert len(msg.nodes) == 4
    assert len(msg.edges) == 3

    msg.compile()
    assert print_node1.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert print_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert print_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert print_node1.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert print_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert print_node1.observation_state == msg.observation_state.TrinaryTrue
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert print_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert print_node1.observation_state == msg.observation_state.TrinaryTrue
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert print_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert print_node1.observation_state == msg.observation_state.TrinaryTrue
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert print_node2.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msg.is_end_motion()

    msg.tick()
    assert print_node1.observation_state == msg.observation_state.TrinaryTrue
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert print_node2.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryTrue

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msg.is_end_motion()


def test_cancel_motion():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    cancel = CancelMotion(
        name=PrefixedName("done"), motion_statechart=msg, exception=Exception("test")
    )
    cancel.start_condition = node1.observation_symbol

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
    end.start_condition = task1.observation_symbol

    msg.compile()
    msg.tick()
    assert task1.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert task1.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    root_C_tip.position = 1

    msg.tick()
    assert task1.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert task1.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    msg.tick()
    assert task1.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryTrue
    assert task1.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING


def test_reset():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = TrueMonitor(name=PrefixedName("muh2"), motion_statechart=msg)
    node3 = TrueMonitor(name=PrefixedName("muh3"), motion_statechart=msg)
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    node1.reset_condition = node2.observation_symbol
    node2.start_condition = node1.observation_symbol
    node3.start_condition = node2.observation_symbol
    node2.end_condition = node2.observation_symbol
    end.start_condition = cas.trinary_logic_and(
        node1.observation_symbol, node2.observation_symbol, node3.observation_symbol
    )

    msg.compile()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert node2.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert node2.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert node2.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert node2.observation_state == msg.observation_state.TrinaryTrue
    assert node3.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert node2.observation_state == msg.observation_state.TrinaryTrue
    assert node3.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert node2.observation_state == msg.observation_state.TrinaryTrue
    assert node3.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryTrue
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msg.is_end_motion()


def test_nested_goals():
    msg = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("start"), motion_statechart=msg)

    # inner goal with two sub-nodes
    inner = Goal(name=PrefixedName("inner"), motion_statechart=msg)
    sub_node1 = TrueMonitor(name=PrefixedName("inner sub 1"), motion_statechart=msg)
    sub_node2 = TrueMonitor(name=PrefixedName("inner sub 2"), motion_statechart=msg)
    inner.add_node(sub_node1)
    inner.add_node(sub_node2)
    sub_node1.end_condition = sub_node1.observation_symbol
    sub_node2.start_condition = sub_node1.observation_symbol
    inner.create_observation_expression = lambda: sub_node2.observation_symbol

    # outer goal that contains the inner goal as a node
    outer = Goal(name=PrefixedName("outer"), motion_statechart=msg)
    outer.add_node(inner)
    outer.create_observation_expression = lambda: inner.observation_symbol
    outer.start_condition = node1.observation_symbol

    end = EndMotion(name=PrefixedName("done nested"), motion_statechart=msg)
    end.start_condition = outer.observation_symbol

    # compile and check initial states
    msg.compile()
    msg.draw()
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert sub_node1.observation_state == msg.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert inner.observation_state == msg.observation_state.TrinaryUnknown
    assert outer.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert inner.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 1: start trigger begins running
    msg.tick()
    msg.draw()
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert sub_node1.observation_state == msg.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert inner.observation_state == msg.observation_state.TrinaryUnknown
    assert outer.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert inner.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 2: start trigger turns true; inner sub_node1 already resolves to True and sub_node2 starts
    msg.tick()
    msg.draw()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msg.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert inner.observation_state == msg.observation_state.TrinaryUnknown
    assert outer.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 3: inner sub_node2 turns true (inner goal still evaluating)
    msg.tick()
    msg.draw()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert inner.observation_state == msg.observation_state.TrinaryUnknown
    assert outer.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 4: inner sub_node2 turns true
    msg.tick()
    msg.draw()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msg.observation_state.TrinaryTrue
    assert inner.observation_state == msg.observation_state.TrinaryUnknown
    assert outer.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 5: inner goal becomes true, outer still running, end starts running
    msg.tick()
    msg.draw()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msg.observation_state.TrinaryTrue
    assert inner.observation_state == msg.observation_state.TrinaryTrue
    assert outer.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 6: outer goal becomes true; end still running
    msg.tick()
    msg.draw()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msg.observation_state.TrinaryTrue
    assert inner.observation_state == msg.observation_state.TrinaryTrue
    assert outer.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msg.is_end_motion()

    # tick 7: end motion becomes true
    msg.tick()
    msg.draw()
    assert end.observation_state == msg.observation_state.TrinaryTrue
    assert msg.is_end_motion()


@dataclass(eq=False, repr=False)
class _TestThreadMonitor(ThreadPayloadMonitor):
    delay: float = 0.05
    return_value: float = ObservationState.TrinaryTrue

    def _compute_observation(self):
        time.sleep(self.delay)
        return self.return_value


def test_thread_payload_monitor_non_blocking_and_caching():
    msg = MotionStatechart(World())
    mon = _TestThreadMonitor(
        name=PrefixedName("thread_mon"),
        motion_statechart=msg,
        delay=0.05,
        return_value=ObservationState.TrinaryTrue,
    )
    # First call should be non-blocking and return Unknown until worker completes at least once
    start = time.perf_counter()
    val0 = mon.compute_observation()
    elapsed = time.perf_counter() - start
    assert elapsed < mon.delay / 4.0
    assert val0 == ObservationState.TrinaryUnknown
    # Wait for worker to finish and cache
    time.sleep(mon.delay * 2)
    val1 = mon.compute_observation()
    assert val1 == ObservationState.TrinaryTrue


def test_thread_payload_monitor_integration():
    msg = MotionStatechart(World())
    mon = _TestThreadMonitor(
        name=PrefixedName("thread_mon2"),
        motion_statechart=msg,
        delay=0.03,
        return_value=ObservationState.TrinaryTrue,
    )
    end = EndMotion(name=PrefixedName("done thread"), motion_statechart=msg)
    end.start_condition = mon.observation_symbol

    msg.compile()

    # tick 1: monitor not started yet becomes RUNNING; end not started
    msg.tick()
    assert mon.observation_state == ObservationState.TrinaryUnknown
    assert mon.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    # tick 2: compute_observation is triggered asynchronously; still Unknown immediately
    msg.tick()
    assert mon.observation_state == ObservationState.TrinaryUnknown
    assert mon.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    # allow background to finish and propagate on next tick
    time.sleep(mon.delay * 2)
    msg.tick()
    assert mon.observation_state == ObservationState.TrinaryTrue
    assert end.life_cycle_state == LifeCycleValues.RUNNING

    # next tick the EndMotion should turn true
    msg.tick()
    assert end.observation_state == ObservationState.TrinaryTrue


def test_goal():
    msg = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)

    goal = Goal(name=PrefixedName("goal"), motion_statechart=msg)
    sub_node1 = TrueMonitor(name=PrefixedName("sub muh1"), motion_statechart=msg)
    sub_node2 = TrueMonitor(name=PrefixedName("sub muh2"), motion_statechart=msg)
    goal.add_node(sub_node1)
    goal.add_node(sub_node2)
    sub_node1.end_condition = sub_node1.observation_symbol
    sub_node2.start_condition = sub_node1.observation_symbol
    goal.create_observation_expression = lambda: sub_node2.observation_symbol
    goal.start_condition = node1.observation_symbol

    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    end.start_condition = goal.observation_symbol

    msg.compile()
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert sub_node1.observation_state == msg.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert goal.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert goal.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryUnknown
    assert sub_node1.observation_state == msg.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert goal.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert goal.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msg.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert goal.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msg.observation_state.TrinaryUnknown
    assert goal.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msg.observation_state.TrinaryTrue
    assert goal.observation_state == msg.observation_state.TrinaryUnknown
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msg.observation_state.TrinaryTrue
    assert goal.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msg.is_end_motion()

    msg.tick()
    assert node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msg.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msg.observation_state.TrinaryTrue
    assert goal.observation_state == msg.observation_state.TrinaryTrue
    assert end.observation_state == msg.observation_state.TrinaryTrue
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msg.is_end_motion()
