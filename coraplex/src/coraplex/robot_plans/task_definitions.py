"""
Commands given to the robot in the GPSR challenge, split into the actions they perform
room by room.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.robot_plans.actions.base import Action
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room

# %% tasks


@dataclass(eq=False)
class Step:
    """
    One part of a command that is performed within a single room, such as picking up the
    milk in the kitchen.
    """

    rooms: set[type[Room]]
    """
    The kinds of room this step may be performed in.
    """

    actions: list[Action]
    """
    The actions of this step, in execution order.
    """


@dataclass(eq=False)
class Command:
    """
    One command, such as "bring the milk to the living room", split into the steps it
    performs room by room.
    """

    instruction: str
    """
    What the goal of this task is, e.g. ``"bring the milk to the living room"``.
    """

    steps: list[Step]
    """
    The steps of this command, in the order they are performed.
    """
