"""
Tests for ordering the steps of GPSR tasks into room visits.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from coraplex.exceptions import MissingTasksOrRooms, NoRoomVisitOrderFound
from coraplex.robot_plans.task_definitions import Command, Step
from coraplex.robot_plans.transformation_by_room import RoomVisit, RoomVisitPlanner
from semantic_digital_twin.semantic_annotations.semantic_annotations import Kitchen
from ...dataset.mock_gpsr_tasks import MOCK_GPSR_PLAN


# %% room visits of the mock plan


@pytest.fixture
def room_visits() -> list[RoomVisit]:
    """
    :return: The room visits the mock plan is grouped into.
    """
    planner = RoomVisitPlanner()
    return planner.transform_plan_by_rooms(MOCK_GPSR_PLAN)


def test_steps_of_each_task_are_performed_once_and_in_order(
    room_visits: list[RoomVisit],
):
    for command in MOCK_GPSR_PLAN:
        performed_steps = [
            step
            for room_visit in room_visits
            for step in room_visit.steps
            if step in command.steps
        ]
        assert performed_steps == command.steps


def test_every_step_is_performed_in_an_allowed_room(room_visits: list[RoomVisit]):
    for room_visit in room_visits:
        for step in room_visit.steps:
            assert room_visit.room in step.rooms


def test_consecutive_visits_go_to_different_rooms(room_visits: list[RoomVisit]):
    visited_rooms = [room_visit.room for room_visit in room_visits]
    assert all(room != next_room for room, next_room in pairwise(visited_rooms))


def test_steps_sharing_a_visit_are_ordered_by_task_and_then_by_step(
    room_visits: list[RoomVisit],
):
    arrival_order = [step for command in MOCK_GPSR_PLAN for step in command.steps]
    for room_visit in room_visits:
        positions = [arrival_order.index(step) for step in room_visit.steps]
        assert positions == sorted(positions)


def test_mock_plan_needs_the_fewest_room_visits(room_visits: list[RoomVisit]):
    # The remote leaves the living room before the milk can arrive there from the
    # kitchen, so the living room is visited twice, the kitchen and bedroom once each.
    assert len(room_visits) == 4


def test_actions_of_a_visit_are_those_of_its_steps_in_order(
    room_visits: list[RoomVisit],
):
    for room_visit in room_visits:
        assert room_visit.actions == [
            action for step in room_visit.steps for action in step.actions
        ]


# %% special cases


def test_tasks_in_a_single_room_need_a_single_visit():
    first_step = Step(rooms={Kitchen}, actions=[])
    second_step = Step(rooms={Kitchen}, actions=[])
    only_step = Step(rooms={Kitchen}, actions=[])
    two_step_task = Command(
        instruction="two steps in the kitchen", steps=[first_step, second_step]
    )
    one_step_task = Command(instruction="one step in the kitchen", steps=[only_step])
    planner = RoomVisitPlanner()

    room_visits = planner.transform_plan_by_rooms([two_step_task, one_step_task])

    assert room_visits == [
        RoomVisit(room=Kitchen, steps=[first_step, second_step, only_step])
    ]


# %% missing tasks or rooms


def test_no_tasks_cannot_be_transformed():
    planner = RoomVisitPlanner()
    with pytest.raises(MissingTasksOrRooms):
        planner.transform_plan_by_rooms([])


def test_tasks_without_allowed_rooms_cannot_be_transformed():
    task = Command(instruction="go nowhere", steps=[Step(rooms=set(), actions=[])])
    planner = RoomVisitPlanner()

    with pytest.raises(MissingTasksOrRooms):
        planner.transform_plan_by_rooms([task])


# %% unsolvable tasks


def cook_then_go_nowhere() -> Command:
    """
    :return: A task whose second step has no room it may be performed in.
    """
    return Command(
        instruction="cook, then go nowhere",
        steps=[Step(rooms={Kitchen}, actions=[]), Step(rooms=set(), actions=[])],
    )


def test_step_without_an_allowed_room_next_to_others_cannot_be_transformed():
    planner = RoomVisitPlanner()

    with pytest.raises(NoRoomVisitOrderFound):
        planner.transform_plan_by_rooms([cook_then_go_nowhere()])


def test_unsolvable_tasks_report_the_search_time_limit_of_the_planner():
    planner = RoomVisitPlanner(search_time_limit=5.0)

    with pytest.raises(NoRoomVisitOrderFound) as error:
        planner.transform_plan_by_rooms([cook_then_go_nowhere()])

    assert error.value.search_time_limit == planner.search_time_limit
