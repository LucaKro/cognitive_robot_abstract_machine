"""
Tests that a motion statechart life cycle state maps onto the task status a plan
reports.
"""

import pytest
from giskardpy.motion_statechart.data_types import LifeCycleValues

from coraplex.datastructures.enums import TaskStatus


@pytest.mark.parametrize(
    "life_cycle_state, status",
    [
        (LifeCycleValues.NOT_STARTED, TaskStatus.CREATED),
        (LifeCycleValues.RUNNING, TaskStatus.RUNNING),
        (LifeCycleValues.PAUSED, TaskStatus.PAUSE),
        (LifeCycleValues.SUCCEEDED, TaskStatus.SUCCEEDED),
        (LifeCycleValues.FAILED, TaskStatus.FAILED),
        (LifeCycleValues.INTERRUPTED, TaskStatus.INTERRUPTED),
    ],
)
def test_every_life_cycle_state_maps_to_a_status(life_cycle_state, status):
    assert TaskStatus.from_life_cycle_state(life_cycle_state) is status


def test_the_mapping_is_total_and_injective():
    """
    The two enums are meant to stay in step, so every state has its own status.
    """
    statuses = {TaskStatus.from_life_cycle_state(state) for state in LifeCycleValues}
    assert len(statuses) == len(LifeCycleValues)
