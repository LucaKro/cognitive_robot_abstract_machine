"""
A mock GPSR plan: independently issued commands, most of them fetching something from
one room and delivering it to another, listed in arrival order rather than grouped by
room.

Run it from this folder::

    python mock_gpsr_tasks.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from task_definitions import GPSRTask, transform_plan

from coraplex.robot_plans.actions.base import SequentialAction

# %% stand-in actions


@dataclass(eq=False)
class MockAction(SequentialAction):
    """
    A named stand-in for a real :class:`SequentialAction`, such as
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`, so a mock plan
    reads as what each step does without a live world and robot to build the real
    action against.
    """

    description: str
    """
    What this action does, e.g. ``"pick up the milk"``.
    """

    def create_steps(self) -> list:
        raise NotImplementedError("MockAction stands in for a real action; it is never run.")

    def __repr__(self) -> str:
        return self.description


# %% the commands: one set of allowed rooms and one list of actions per step

lower_torso = GPSRTask(
    name="lower the torso and park the arms",
    rooms=[{"kitchen", "living_room", "bed_room", "office"}],
    actions=[[MockAction("move torso low"), MockAction("park arms")]],
)

bring_milk = GPSRTask(
    name="bring the milk to the living room",
    rooms=[{"kitchen"}, {"living_room"}],
    actions=[
        [MockAction("move torso high"), MockAction("pick up the milk")],
        [MockAction("place the milk")],
    ],
)

bring_cup = GPSRTask(
    name="bring the cup to the bedroom",
    rooms=[{"living_room"}, {"bed_room"}],
    actions=[
        [MockAction("park arms"), MockAction("pick up the cup")],
        [MockAction("place the cup")],
    ],
)

bring_remote = GPSRTask(
    name="bring the remote to the kitchen",
    rooms=[{"living_room"}, {"kitchen"}],
    actions=[
        [MockAction("pick up the remote")],
        [MockAction("place the remote")],
    ],
)

# %% the plan: arrival order, not room order

MOCK_GPSR_PLAN: list[GPSRTask] = [lower_torso, bring_milk, bring_cup, bring_remote]
"""
The commands queued as they arrived, so performing them in that order visits a room,
leaves it, and visits it again. :func:`~task_definitions.transform_plan` groups them
into fewer room visits.
"""


def main() -> None:
    start = time.time()
    room_visits = transform_plan(MOCK_GPSR_PLAN)
    optimal_plan = []
    print("Steps per room visit:")
    for room, steps in room_visits:
        print(f"  In room {room}:")
        for task, step_number in steps:
            print(f"    part {step_number} of {len(task.rooms)} of task '{task.name}'")

    print("\nActions per room visit, grouped by task:")
    for room, steps in room_visits:
        print(f"  In room {room}:")
        for task, step_number in steps:
            print(f"    {task.name}: {task.actions[step_number - 1]}")
            optimal_plan.append(task.actions[step_number - 1])

    print(time.time()-start)
    print(optimal_plan)



if __name__ == "__main__":
    main()