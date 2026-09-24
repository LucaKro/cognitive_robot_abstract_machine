"""
Tests for when an episode's disturbances strike and are released.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from experiments.articulated_manipulation.disturbance_protocol import (
    AfterSimulatedTime,
    CabinetMoved,
    OnceOpenedBy,
    PartPushed,
)
from experiments.articulated_manipulation.episode import DisturbanceSchedule
from semantic_digital_twin.spatial_types.spatial_types import Pose2D


@dataclass
class RecordingPhysics:
    """
    Stands in for the ground truth of a running episode, at a state a test sets, and
    records what the disturbances do to it.
    """

    simulated_time: timedelta = timedelta()
    """
    How much simulated time has passed.
    """

    opened_fraction: float = 0.0
    """
    The share of its travel the part is open.
    """

    moves: list[Pose2D] = field(default_factory=list)
    """
    Every displacement the cabinet was moved by.
    """

    pushes: list[float] = field(default_factory=list)
    """
    Every force the part was pushed with.
    """

    def move_cabinet(self, displacement: Pose2D) -> None:
        self.moves.append(displacement)

    def push_part(self, force: float) -> None:
        self.pushes.append(force)


def pushed_shut_once_half_open() -> PartPushed:
    """
    :return: The part pushed shut for a second once it is half open.
    """
    return PartPushed(
        trigger=OnceOpenedBy(0.5), duration=timedelta(seconds=1), force=-10.0
    )


def test_a_disturbance_waits_until_it_is_due():
    physics = RecordingPhysics(opened_fraction=0.4)
    schedule = DisturbanceSchedule(pending=[pushed_shut_once_half_open()])

    schedule.advance(physics)

    assert physics.pushes == []


def test_a_disturbance_strikes_once_it_is_due():
    push = pushed_shut_once_half_open()
    physics = RecordingPhysics(opened_fraction=0.5)
    schedule = DisturbanceSchedule(pending=[push])

    schedule.advance(physics)

    assert physics.pushes == [push.force]


def test_a_disturbance_lasts_its_duration():
    push = pushed_shut_once_half_open()
    physics = RecordingPhysics(opened_fraction=0.5)
    schedule = DisturbanceSchedule(pending=[push])
    schedule.advance(physics)

    physics.simulated_time = push.duration / 2
    schedule.advance(physics)

    assert physics.pushes == [push.force]


def test_a_disturbance_is_released_after_its_duration():
    push = pushed_shut_once_half_open()
    physics = RecordingPhysics(opened_fraction=0.5)
    schedule = DisturbanceSchedule(pending=[push])
    schedule.advance(physics)

    physics.simulated_time = push.duration
    schedule.advance(physics)

    assert physics.pushes == [push.force, 0.0]


def test_a_disturbance_strikes_only_once():
    push = pushed_shut_once_half_open()
    physics = RecordingPhysics(opened_fraction=0.5)
    schedule = DisturbanceSchedule(pending=[push])
    schedule.advance(physics)
    physics.simulated_time = push.duration
    schedule.advance(physics)

    physics.simulated_time = 3 * push.duration
    schedule.advance(physics)

    assert physics.pushes == [push.force, 0.0]


def test_the_cabinet_is_moved_once_its_time_has_come():
    moved = CabinetMoved(
        trigger=AfterSimulatedTime(timedelta(seconds=2)),
        displacement=Pose2D(x=0.05),
    )
    physics = RecordingPhysics(simulated_time=timedelta(seconds=1))
    schedule = DisturbanceSchedule(pending=[moved])
    schedule.advance(physics)
    assert physics.moves == []

    physics.simulated_time = moved.trigger.elapsed
    schedule.advance(physics)

    assert physics.moves == [moved.displacement]
