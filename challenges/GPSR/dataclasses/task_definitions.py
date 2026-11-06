from __future__ import annotations

from dataclasses import dataclass

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


