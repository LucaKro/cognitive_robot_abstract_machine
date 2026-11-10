"""
A mock GPSR plan: independently issued commands, most of them fetching something from
one room and delivering it to another, listed in arrival order rather than grouped by
room.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from coraplex.robot_plans.transformation_by_room import RoomVisitPlanner
from coraplex.robot_plans.task_definitions import Command, Step

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


# %% the commands: one step per room they are performed in

lower_torso = Command(
    instruction="lower the torso and park the arms",
    steps=[
        Step(
            rooms={Kitchen, LivingRoom, Bedroom, Bathroom},
            actions=[MockAction("move torso low"), MockAction("park arms")],
        )
    ],
)

bring_milk = Command(
    instruction="bring the milk to the living room",
    steps=[
        Step(
            rooms={Kitchen},
            actions=[MockAction("move torso high"), MockAction("pick up the milk")],
        ),
        Step(rooms={LivingRoom}, actions=[MockAction("place the milk")]),
    ],
)

bring_cup = Command(
    instruction="bring the cup to the bedroom",
    steps=[
        Step(
            rooms={LivingRoom},
            actions=[MockAction("park arms"), MockAction("pick up the cup")],
        ),
        Step(rooms={Bedroom}, actions=[MockAction("place the cup")]),
    ],
)

bring_remote = Command(
    instruction="bring the remote to the kitchen",
    steps=[
        Step(rooms={LivingRoom}, actions=[MockAction("pick up the remote")]),
        Step(rooms={Kitchen}, actions=[MockAction("place the remote")]),
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
    command_of_step = {
        step: command for command in MOCK_GPSR_PLAN for step in command.steps
    }
    optimal_plan = []
    logger.info("Steps per room visit:")
    for room_visit in room_visits:
        logger.info(f"  In room {room_visit.room.__name__}:")
        for step in room_visit.steps:
            task = command_of_step[step]
            logger.info(
                f"    part {task.steps.index(step) + 1} of {len(task.steps)} of task "
                f"'{task.instruction}'"
            )

    logger.info("Actions per room visit, grouped by task:")
    for room_visit in room_visits:
        logger.info(f"  In room {room_visit.room.__name__}:")
        for step in room_visit.steps:
            logger.info(f"    {command_of_step[step].instruction}: {step.actions}")
            optimal_plan.append(step.actions)
    return optimal_plan


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    main()
