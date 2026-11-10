"""
A mock GPSR plan: independently issued commands, most of them fetching something from
one room and delivering it to another, listed in arrival order rather than grouped by
room.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from coraplex.robot_plans.transformation_by_room import RoomVisitPlanner
from coraplex.robot_plans.task_definitions import Command

from coraplex.robot_plans.actions.base import SequentialAction
from cramph.node import StatechartNode
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bathroom,
    Bedroom,
    Kitchen,
    LivingRoom,
)

# %% stand-in actions


@dataclass(eq=False)
class MockAction(SequentialAction):
    """
    A named stand-in for a real :class:`SequentialAction`, such as
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`, so a mock plan
    reads as what each step does without a live world and robot to build the real action
    against.
    """

    description: str
    """
    What this action does, e.g. ``"pick up the milk"``.
    """

    def create_steps(self) -> list[StatechartNode]:
        raise NotImplementedError(
            "MockAction stands in for a real action; it is never run."
        )

    def __repr__(self) -> str:
        return self.description


# %% the commands: one set of allowed rooms and one list of actions per step

lower_torso = Command(
    instruction="lower the torso and park the arms",
    rooms=[{Kitchen, LivingRoom, Bedroom, Bathroom}],
    actions=[[MockAction("move torso low"), MockAction("park arms")]],
)

bring_milk = Command(
    instruction="bring the milk to the living room",
    rooms=[{Kitchen}, {LivingRoom}],
    actions=[
        [MockAction("move torso high"), MockAction("pick up the milk")],
        [MockAction("place the milk")],
    ],
)

bring_cup = Command(
    instruction="bring the cup to the bedroom",
    rooms=[{LivingRoom}, {Bedroom}],
    actions=[
        [MockAction("park arms"), MockAction("pick up the cup")],
        [MockAction("place the cup")],
    ],
)

bring_remote = Command(
    instruction="bring the remote to the kitchen",
    rooms=[{LivingRoom}, {Kitchen}],
    actions=[
        [MockAction("pick up the remote")],
        [MockAction("place the remote")],
    ],
)

# %% the plan: arrival order, not room order

MOCK_GPSR_PLAN: list[Command] = [lower_torso, bring_milk, bring_cup, bring_remote]
"""
The commands queued as they arrived, so performing them in that order visits a room,
leaves it, and visits it again.
"""


# %% printing the room visits


def main() -> list[list[MockAction]]:
    """
    Collects the room visits :data:`MOCK_GPSR_PLAN` is grouped into, first as task steps
    and then as their actions and the actions in execution order.
    """
    logger = logging.getLogger(__name__)
    planner = RoomVisitPlanner()
    room_visits = planner.transform_plan_by_rooms(MOCK_GPSR_PLAN)
    optimal_plan = []
    logger.info("Steps per room visit:")
    for room, steps in room_visits:
        logger.info(f"  In room {room.__name__}:")
        for task, step_number in steps:
            logger.info(
                f"    part {step_number} of {len(task.rooms)} of task '{task.instruction}'"
            )

    logger.info("Actions per room visit, grouped by task:")
    for room, steps in room_visits:
        logger.info(f"  In room {room.__name__}:")
        for task, step_number in steps:
            logger.info(f"    {task.instruction}: {task.actions[step_number - 1]}")
            optimal_plan.append(task.actions[step_number - 1])
    return optimal_plan


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    main()
