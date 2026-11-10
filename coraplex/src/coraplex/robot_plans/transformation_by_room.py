"""
Orders the steps of GPSR tasks into room visits, so the robot changes rooms as rarely as
possible while the steps of each task keep their order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

from ortools.sat.python import cp_model

from coraplex.exceptions import MissingTasksOrRooms, NoRoomVisitOrderFound
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.task_definitions import Command, Step
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bedroom,
    Kitchen,
    LivingRoom,
    Room,
)

# %% room visits


@dataclass
class RoomVisit:
    """
    One stay of the robot in a room, during which it performs one or more steps.
    """

    room: type[Room]
    """
    The kind of room the robot stays in.
    """

    steps: list[Step]
    """
    The steps performed during this stay, ordered by the task they belong to and then by
    their order within that task.
    """

    @property
    def actions(self) -> list[Action]:
        """
        :return: The actions of :attr:`steps`, in execution order.
        """
        return [action for step in self.steps for action in step.actions]


# %% room visit model


@dataclass
class RoomVisitModel:
    """
    The constraint model whose solutions assign every step of :attr:`commands` to a
    visit of one of :attr:`rooms`, with as few visits as possible.

    There is one potential visit per step, since that many always suffice; visits that
    are not needed trail at the end and go to no room.
    """

    commands: list[Command]
    """
    The tasks whose steps are assigned, in the order they arrived.
    """

    rooms: list[type[Room]]
    """
    Every room a step may be performed in.
    """

    model: cp_model.CpModel = field(default_factory=cp_model.CpModel, init=False)
    """
    The constraints and the objective.
    """

    visit_goes_to: dict[tuple[int, type[Room]], cp_model.IntVar] = field(
        default_factory=dict, init=False
    )
    """
    Whether a visit, by its index, goes to a room.
    """

    step_performed_during: dict[tuple[Step, int], cp_model.IntVar] = field(
        default_factory=dict, init=False
    )
    """
    Whether a step is performed during a visit, by the visit's index.
    """

    def __post_init__(self) -> None:
        self.visit_goes_to = {
            (visit, room): self.model.NewBoolVar(
                f"visit_{visit}_goes_to_{room.__name__}"
            )
            for visit in self.visits
            for room in self.rooms
        }
        self.step_performed_during = {
            (step, visit): self.model.NewBoolVar(f"step_{index}_during_visit_{visit}")
            for index, step in enumerate(self.steps)
            for visit in self.visits
        }
        self.add_visit_constraints()
        self.add_step_constraints()
        self.add_task_order_constraints()
        self.model.Minimize(sum(self.visit_goes_to.values()))

    @property
    def steps(self) -> list[Step]:
        """
        :return: The steps of every command, by command and then in step order.
        """
        return [step for command in self.commands for step in command.steps]

    @property
    def visits(self) -> range:
        """
        :return: The indices of the potential visits.
        """
        return range(len(self.steps))

    def rooms_entered(self, visit: int) -> cp_model.LinearExpr:
        """
        :param visit: The index of a visit.
        :return: How many rooms the visit goes to, which is 1 for a used visit and 0
            otherwise.
        """
        return sum(self.visit_goes_to[visit, room] for room in self.rooms)

    def visit_index_of(self, step: Step) -> cp_model.LinearExpr:
        """
        :param step: One of :attr:`steps`.
        :return: The index of the visit the step is performed during.
        """
        return sum(
            visit * self.step_performed_during[step, visit] for visit in self.visits
        )

    def add_visit_constraints(self) -> None:
        """
        Every visit goes to at most one room, the used visits come first, and two
        consecutive visits go to different rooms, since staying in a room is one visit.
        """
        for visit in self.visits:
            self.model.AddAtMostOne(
                self.visit_goes_to[visit, room] for room in self.rooms
            )
        for visit, next_visit in pairwise(self.visits):
            self.model.Add(self.rooms_entered(visit) >= self.rooms_entered(next_visit))
            for room in self.rooms:
                self.model.Add(
                    self.visit_goes_to[visit, room]
                    + self.visit_goes_to[next_visit, room]
                    <= 1
                )

    def add_step_constraints(self) -> None:
        """
        Every step is performed during exactly one visit, and only during a visit to one
        of the rooms it may be performed in.
        """
        for step in self.steps:
            self.model.AddExactlyOne(
                self.step_performed_during[step, visit] for visit in self.visits
            )
            for visit in self.visits:
                self.model.Add(
                    self.step_performed_during[step, visit]
                    <= sum(self.visit_goes_to[visit, room] for room in step.rooms)
                )

    def add_task_order_constraints(self) -> None:
        """
        The steps of every task keep their order; consecutive steps may share a visit.
        """
        for command in self.commands:
            for step, next_step in pairwise(command.steps):
                self.model.Add(
                    self.visit_index_of(step) <= self.visit_index_of(next_step)
                )

    def solve(self, search_time_limit: float) -> list[RoomVisit]:
        """
        :param search_time_limit: How long the solver may search, in seconds.
        :return: The used visits in the order the rooms are visited.
        :raises NoRoomVisitOrderFound: If the solver finds no assignment within
            ``search_time_limit``.
        """
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = search_time_limit
        if solver.Solve(self.model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise NoRoomVisitOrderFound(self.commands, search_time_limit)
        return [
            RoomVisit(
                room=room,
                steps=[
                    step
                    for step in self.steps
                    if solver.Value(self.step_performed_during[step, visit])
                ],
            )
            for visit in self.visits
            for room in self.rooms
            if solver.Value(self.visit_goes_to[visit, room])
        ]


# %% planning by rooms


@dataclass
class RoomVisitPlanner:
    """
    Orders the steps of tasks into as few room visits as possible, while the steps of
    each task keep their order.
    """

    search_time_limit: float = 30.0
    """
    How long the solver may search for an order of room visits, in seconds.
    """

    def transform_plan_by_rooms(self, commands: list[Command]) -> list[RoomVisit]:
        """
        Groups the steps of the given tasks into as few room visits as possible, while
        the steps of each task keep their order.

        :param commands: The tasks in the order they arrived.
        :return: The room visits, in the order the rooms are visited.
        :raises MissingTasksOrRooms: If there are no tasks, or no step of any task has
            an allowed room.
        :raises NoRoomVisitOrderFound: If the solver finds no order of room visits that
            performs every step within :attr:`search_time_limit`.
        """
        rooms = sorted(
            {
                room
                for command in commands
                for step in command.steps
                for room in step.rooms
            },
            key=lambda room: room.__name__,
        )
        if len(rooms) == 0:
            raise MissingTasksOrRooms(commands, rooms)
        return RoomVisitModel(commands, rooms).solve(self.search_time_limit)


# %% example


def main() -> None:
    """
    Prints the room visits three example tasks in three rooms are grouped into.
    """
    commands = [
        Command(
            instruction="first the kitchen, then the living room",
            steps=[Step({Kitchen}, []), Step({LivingRoom}, [])],
        ),
        Command(
            instruction="any room",
            steps=[Step({Kitchen, LivingRoom, Bedroom}, [])],
        ),
        Command(
            instruction="first the bedroom, then the kitchen",
            steps=[Step({Bedroom}, []), Step({Kitchen}, [])],
        ),
    ]
    room_visits = RoomVisitPlanner().transform_plan_by_rooms(commands)
    print(
        [room_visit.room.__name__ for room_visit in room_visits],
        "-> room changes:",
        len(room_visits) - 1,
    )
    for room_visit in room_visits:
        print(f"In room {room_visit.room.__name__}:")
        for step in room_visit.steps:
            command = next(command for command in commands if step in command.steps)
            print(
                f"    step {command.steps.index(step) + 1} of "
                f"{len(command.steps)} of '{command.instruction}'"
            )


if __name__ == "__main__":
    main()
