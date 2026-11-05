from __future__ import annotations

from dataclasses import dataclass

from transformation import transform_by_rooms
from coraplex.robot_plans.actions.base import Action


@dataclass(eq=False)
class GPSRTask:
    """
    One GPSR command, such as "bring the milk to the living room", split into the
    actions it performs room by room.
    """

    name: str
    """
    What this task does, e.g. ``"bring the milk to the living room"``.
    """

    rooms: list[set[str]]
    """
    The rooms the task is performed in, one entry per step and in the order of the
    steps; each entry holds the rooms that step may be performed in.
    """

    actions: list[list[Action]]
    """
    The actions of each step, in execution order: ``actions[i]`` is performed in one
    of the rooms in ``rooms[i]``.
    """


def transform_plan(
    tasks: list[GPSRTask],
) -> list[tuple[str, list[tuple[GPSRTask, int]]]]:
    """
    :param tasks: The tasks in the order they arrived.
    :return: One entry per room visit, in the order the rooms are visited: the room
        and the steps performed during that visit, in execution order, each as its
        task and its step number counted from 1.
    """
    all_rooms = sorted({room for task in tasks for step in task.rooms for room in step})
    route, plan = transform_by_rooms([task.rooms for task in tasks], all_rooms)
    return [
        (
            room,
            [
                (tasks[task_number - 1], step_number)
                for task_number, step_number in steps
            ],
        )
        for room, steps in plan
    ]
