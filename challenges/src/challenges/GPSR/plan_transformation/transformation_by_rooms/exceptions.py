"""
Exceptions raised when the steps of Command are ordered into room visits.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.semantic_annotations import Room

# %% transformation by rooms


@dataclass
class MissingTasksOrRooms(DataclassException):
    """
    Raised when the steps of tasks are to be ordered into room visits, but there are no
    tasks or no rooms to perform their steps in.
    """

    task_list: list[list[set[type[Room]]]]
    """
    The tasks that were to be ordered, each as one set of allowed rooms per step.
    """

    room_list: list[type[Room]]
    """
    The rooms the steps were allowed to be performed in.
    """

    def error_message(self) -> str:
        if len(self.task_list) == 0:
            return "No tasks were given, so there is nothing to plan room visits for."
        return (
            "No rooms were given, so none of the steps of the "
            f"{len(self.task_list)} tasks can be performed anywhere."
        )

    def suggest_correction(self) -> str:
        if len(self.task_list) == 0:
            return "Pass at least one task."
        return "Give every step of every task at least one room it may be performed in."


@dataclass
class NoRoomVisitOrderFound(DataclassException):
    """
    Raised when the solver finds no order of room visits that performs every step of the
    given tasks.
    """

    task_list: list[list[set[type[Room]]]]
    """
    The tasks that were to be ordered, each as one set of allowed rooms per step.
    """

    search_time_limit: float
    """
    How long the solver was allowed to search, in seconds.
    """

    def error_message(self) -> str:
        return (
            f"No order of room visits performs every step of the {len(self.task_list)} "
            f"tasks within {self.search_time_limit} seconds."
        )

    def suggest_correction(self) -> str:
        return "Give every step at least one allowed room, or raise the time limit."
