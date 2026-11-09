"""
Tests for ordering the steps of GPSR tasks into room visits.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from challenges.GPSR.dataclasses.task_definitions import Command
from challenges.GPSR.plan_transformation.transformation_by_rooms.exceptions import (
    MissingTasksOrRooms,
    NoRoomVisitOrderFound,
)
from challenges.GPSR.plan_transformation.transformation_by_rooms.mock_gpsr_tasks import (
    MOCK_GPSR_PLAN,
)
from challenges.GPSR.plan_transformation.transformation_by_rooms.transformation import (
    RoomVisitPlanner,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Kitchen,
    Room,
)

# %% room visits of the mock plan


@pytest.fixture
def room_visits() -> list[tuple[type[Room], list[tuple[Command, int]]]]:
    """
    :return: The room visits the mock plan is grouped into.
    """
    planner = RoomVisitPlanner()
    return planner.transform_plan_by_rooms(MOCK_GPSR_PLAN)


def test_steps_of_each_task_are_performed_once_and_in_order(
    room_visits: list[tuple[type[Room], list[tuple[Command, int]]]],
):
    for task in MOCK_GPSR_PLAN:
        performed_step_numbers = [
            step_number
            for _, steps in room_visits
            for performed_task, step_number in steps
            if performed_task is task
        ]
        assert performed_step_numbers == list(range(1, len(task.rooms) + 1))


def test_every_step_is_performed_in_an_allowed_room(
    room_visits: list[tuple[type[Room], list[tuple[Command, int]]]],
):
    for room, steps in room_visits:
        for task, step_number in steps:
            assert room in task.rooms[step_number - 1]


def test_consecutive_visits_go_to_different_rooms(
    room_visits: list[tuple[type[Room], list[tuple[Command, int]]]],
):
    visited_rooms = [room for room, _ in room_visits]
    assert all(room != next_room for room, next_room in pairwise(visited_rooms))


def test_steps_sharing_a_visit_are_ordered_by_task_and_then_by_step(
    room_visits: list[tuple[type[Room], list[tuple[Command, int]]]],
):
    for _, steps in room_visits:
        positions = [
            (MOCK_GPSR_PLAN.index(task), step_number) for task, step_number in steps
        ]
        assert positions == sorted(positions)


def test_mock_plan_needs_the_fewest_room_visits(
    room_visits: list[tuple[type[Room], list[tuple[Command, int]]]],
):
    # The remote leaves the living room before the milk can arrive there from the
    # kitchen, so the living room is visited twice, the kitchen and bedroom once each.
    assert len(room_visits) == 4


# %% special cases


def test_tasks_in_a_single_room_need_a_single_visit():
    two_step_task = Command(
        instruction="two steps in the kitchen", rooms=[{Kitchen}, {Kitchen}], actions=[[], []]
    )
    one_step_task = Command(
        instruction="one step in the kitchen", rooms=[{Kitchen}], actions=[[]]
    )
    planner = RoomVisitPlanner()
    room_visits = planner.transform_plan_by_rooms([two_step_task, one_step_task])

    assert room_visits == [
        (Kitchen, [(two_step_task, 1), (two_step_task, 2), (one_step_task, 1)])
    ]


# %% missing tasks or rooms


def test_no_tasks_cannot_be_transformed():
    planner = RoomVisitPlanner()
    with pytest.raises(MissingTasksOrRooms):
        planner.transform_plan_by_rooms([])


def test_tasks_without_allowed_rooms_cannot_be_transformed():
    task = Command(instruction="go nowhere", rooms=[set()], actions=[[]])
    planner = RoomVisitPlanner()

    with pytest.raises(MissingTasksOrRooms):
        planner.transform_plan_by_rooms([task])


# %% unsolvable tasks


def test_step_without_an_allowed_room_next_to_others_cannot_be_transformed():
    task = Command(
        instruction="cook, then go nowhere", rooms=[{Kitchen}, set()], actions=[[], []]
    )
    planner = RoomVisitPlanner()

    with pytest.raises(NoRoomVisitOrderFound):
        planner.transform_plan_by_rooms([task])


def test_unsolvable_tasks_report_the_search_time_limit_of_the_planner():
    task = Command(
        instruction="cook, then go nowhere", rooms=[{Kitchen}, set()], actions=[[], []]
    )
    planner = RoomVisitPlanner(search_time_limit=5.0)

    with pytest.raises(NoRoomVisitOrderFound) as error:
        planner.transform_plan_by_rooms([task])

    assert error.value.search_time_limit == planner.search_time_limit
