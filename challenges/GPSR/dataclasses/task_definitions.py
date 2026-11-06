from __future__ import annotations

from dataclasses import dataclass

from coraplex.robot_plans.actions.base import Action
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room

# %% tasks


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

    rooms: list[set[type[Room]]]
    """
    The kinds of room the task is performed in, one entry per step and in the order of
    the steps; each entry holds the kinds of room that step may be performed in.
    """

    actions: list[list[Action]]
    """
    The actions of each step, in execution order: ``actions[i]`` is performed in one
    of the rooms in ``rooms[i]``.
    """


