# pip install ortools
from ortools.sat.python import cp_model
from challenges.GPSR.dataclasses.task_definitions import GPSRTask

rooms = ["A", "B", "C"]
# task = list of stages, stage = set of allowed rooms
tasks = [
    [{"A"}, {"B"}],        # first A, then B
    [{"A", "B", "C"}],     # any room
    [{"C"}, {"A"}],        # first C, then A
]

def _transform_by_rooms(task_list: list[list[set[str]]], room_list: list[str]):
    T = sum(len(t) for t in task_list)          # upper bound on visits
    m = cp_model.CpModel()
    x = {(t, r): m.NewBoolVar(f"x_{t}_{r}") for t in range(T) for r in room_list}

    for t in range(T):
        m.AddAtMostOne(x[t, r] for r in room_list)
        if t + 1 < T:
            # used slots form a prefix
            m.Add(sum(x[t, r] for r in room_list) >= sum(x[t + 1, r] for r in room_list))
            # staying in the same room = same visit, so forbid repeats
            for r in room_list:
                m.Add(x[t, r] + x[t + 1, r] <= 1)

    pos = {}
    # y[k, s, t] == 1: stage s of task k is executed during visit t
    y = {}
    for k, stages in enumerate(task_list):
        for s, allowed in enumerate(stages):
            for t in range(T):
                y[k, s, t] = m.NewBoolVar(f"y_{k}_{s}_{t}")
            m.AddExactlyOne(y[k, s, t] for t in range(T))
            for t in range(T):
                m.Add(y[k, s, t] <= sum(x[t, r] for r in allowed))
            pos[k, s] = sum(t * y[k, s, t] for t in range(T))
            if s > 0:
                m.Add(pos[k, s - 1] <= pos[k, s])

    m.Minimize(sum(x.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(m)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        route = [r for t in range(T) for r in room_list if solver.Value(x[t, r])]

        # one entry per room visit: (room, [(task number, step number), ...]), both
        # counted from 1; sorted so that two steps of one task in the same visit keep
        # their order
        plan = [
            (
                room,
                sorted(
                    (k + 1, s + 1)
                    for k, stages in enumerate(task_list)
                    for s in range(len(stages))
                    if solver.Value(y[k, s, t])
                ),
            )
            for t, room in enumerate(route)
        ]


        return route, plan

def transform_plan_by_rooms(
    tasks: list[GPSRTask],
) -> list[tuple[str, list[tuple[GPSRTask, int]]]]:
    """
    :param tasks: The tasks in the order they arrived.
    :return: One entry per room visit, in the order the rooms are visited: the room
        and the steps performed during that visit, in execution order, each as its
        task and its step number counted from 1.
    """
    all_rooms = sorted({room for task in tasks for step in task.rooms for room in step})
    route, plan = _transform_by_rooms([task.rooms for task in tasks], all_rooms)
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

# test function
def main():
    route, plan = _transform_by_rooms(tasks, rooms)
    print(route, "-> room changes:", len(route) - 1)

    for t, (room, steps) in enumerate(plan):
        visit_of_room = route[: t + 1].count(room)
        print(f"In room {room} (visit {visit_of_room} of {room}):")
        for task_number, step_number in steps:
            print(
                f"    task {task_number}, step {step_number} of "
                f"{len(tasks[task_number - 1])}"
            )

if __name__ == "__main__":
    main()