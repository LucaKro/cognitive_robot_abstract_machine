"""
Orders the steps of GPSR tasks into room visits, so the robot changes rooms as rarely
as possible while the steps of each task keep their order.
"""

from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model
from challenges.GPSR.dataclasses.task_definitions import Command
from challenges.GPSR.plan_transformation.transformation_by_rooms.exceptions import (
    MissingTasksOrRooms,
    NoRoomVisitOrderFound,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bedroom,
    Kitchen,
    LivingRoom,
    Room,
)


# %% room visit model


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

    def _transform_by_rooms(
        self, task_list: list[list[set[type[Room]]]], room_list: list[type[Room]]
    ) -> tuple[list[type[Room]], list[tuple[type[Room], list[tuple[int, int]]]]]:
        """
        Orders the steps of the given tasks into as few room visits as possible, while the
        steps of each task keep their order.

        :param task_list: The tasks in the order they arrived, each as one set of allowed
            rooms per step, in step order.
        :param room_list: Every room a step may be performed in.
        :return: The visited rooms in visiting order, and one entry per room visit: the room
            and the steps performed during that visit, each as its task number and step
            number counted from 1, ordered by task and then by step.
        :raises MissingTasksOrRooms: If there are no tasks or no rooms.
        :raises NoRoomVisitOrderFound: If the solver finds no order of room visits that
            performs every step within
            :attr:`search_time_limit`.
        """
        if len(task_list) == 0 or len(room_list) == 0:
            raise MissingTasksOrRooms(task_list, room_list)

        # One visit per step is always enough, so the number of steps bounds the visits.
        maximum_visit_count = sum(len(element) for element in task_list)
        model = cp_model.CpModel()
        # visit_goes_to[visit, room] == 1: specific visit goes to specific room
        visit_goes_to = {
            (visit, room): model.NewBoolVar(f"visit_goes_to_{visit}_{room.__name__}")
            for visit in range(maximum_visit_count)
            for room in room_list
        }

        for visit in range(maximum_visit_count):
            model.AddAtMostOne(visit_goes_to[visit, room] for room in room_list)
            if visit + 1 < maximum_visit_count:
                # The used visits come first; unused visits only trail at the end.
                model.Add(
                    sum(visit_goes_to[visit, room] for room in room_list)
                    >= sum(visit_goes_to[visit + 1, room] for room in room_list)
                )
                # Staying in a room is one visit, so the next visit goes to another room.
                for room in room_list:
                    model.Add(
                        visit_goes_to[visit, room] + visit_goes_to[visit + 1, room] <= 1
                    )

        # pos[task, step]: the index of the visit step of task is performed in
        pos = {}
        # step_performed_during[task, step, visit] == 1: step of task is performed during visit
        step_performed_during = {}
        for task, stages in enumerate(task_list):
            for step, allowed in enumerate(stages):
                for visit in range(maximum_visit_count):
                    step_performed_during[task, step, visit] = model.NewBoolVar(
                        f"step_performed_during_{task}_{step}_{visit}"
                    )
                model.AddExactlyOne(
                    step_performed_during[task, step, visit]
                    for visit in range(maximum_visit_count)
                )
                # A step is only performed during a visit to one of its allowed rooms.
                for visit in range(maximum_visit_count):
                    model.Add(
                        step_performed_during[task, step, visit]
                        <= sum(visit_goes_to[visit, room] for room in allowed)
                    )
                pos[task, step] = sum(
                    visit * step_performed_during[task, step, visit]
                    for visit in range(maximum_visit_count)
                )
                # The steps of a task keep their order; consecutive steps may share a visit.
                if step > 0:
                    model.Add(pos[task, step - 1] <= pos[task, step])

        # The fewest visits are the fewest room changes.
        model.Minimize(sum(visit_goes_to.values()))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.search_time_limit
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise NoRoomVisitOrderFound(task_list, self.search_time_limit)

        route = [
            room
            for visit in range(maximum_visit_count)
            for room in room_list
            if solver.Value(visit_goes_to[visit, room])
        ]

        # one entry per room visit: (room, [(task number, step number), ...]), both
        # counted from 1; sorted so that two steps of one task in the same visit keep
        # their order
        plan = [
            (
                room,
                sorted(
                    (task + 1, step + 1)
                    for task, stages in enumerate(task_list)
                    for step in range(len(stages))
                    if solver.Value(step_performed_during[task, step, visit])
                ),
            )
            for task, room in enumerate(route)
        ]

        return route, plan

    # %% planning by rooms

    def transform_plan_by_rooms(
        self,
        tasks: list[Command],
    ) -> list[tuple[type[Room], list[tuple[Command, int]]]]:
        """
        Groups the steps of the given tasks into as few room visits as possible, while the
        steps of each task keep their order.

        :param tasks: The tasks in the order they arrived.
        :return: One entry per room visit, in the order the rooms are visited: the room
            and the steps performed during that visit, in execution order, each as its
            task and its step number counted from 1.
        :raises MissingTasksOrRooms: If there are no tasks, or no step of any task has an
            allowed room.
        :raises NoRoomVisitOrderFound: If the solver finds no order of room visits that
            performs every step within
            :attr:`search_time_limit`.
        """
        all_rooms = sorted(
            {room for task in tasks for step in task.rooms for room in step},
            key=lambda room: room.__name__,
        )
        route, plan = self._transform_by_rooms(
            [task.rooms for task in tasks], all_rooms
        )
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


# %% example


def main() -> None:
    """
    Prints the room visits three example tasks in three rooms are grouped into.
    """
    planner_main = RoomVisitPlanner()
    # The rooms and tasks scheduled.
    rooms = [Kitchen, LivingRoom, Bedroom]
    # A task is a list of steps, a step is the set of rooms it may be performed in.
    tasks = [
        [{Kitchen}, {LivingRoom}],  # first Kitchen, then LivingRoom
        [{Kitchen, LivingRoom, Bedroom}],  # any room
        [{Bedroom}, {Kitchen}],  # first Bedroom, then Kitchen
    ]

    route, plan = planner_main._transform_by_rooms(tasks, rooms)
    print([room.__name__ for room in route], "-> room changes:", len(route) - 1)

    for visit, (room, steps) in enumerate(plan):
        visit_of_room = route[: visit + 1].count(room)
        print(f"In room {room.__name__} (visit {visit_of_room} of {room.__name__}):")
        for task_number, step_number in steps:
            print(
                f"    task {task_number}, step {step_number} of "
                f"{len(tasks[task_number - 1])}"
            )


if __name__ == "__main__":
    main()
