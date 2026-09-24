"""
Tests for the benchmark's conditions and their seeded episodes.
"""

from __future__ import annotations

import math

import numpy
import pytest

from experiments.articulated_manipulation.cabinet_scene import (
    ArticulatedPart,
    CabinetSceneSpecification,
)
from experiments.articulated_manipulation.disturbance_protocol import (
    ArmStartPose,
    CabinetMoved,
    ConditionKind,
    DisturbanceProtocol,
    OnceOpenedBy,
    PartPushed,
    PriorError,
    PriorErrorLevel,
)


@pytest.fixture
def protocol() -> DisturbanceProtocol:
    return DisturbanceProtocol(episodes_per_condition=3)


def front_distance(
    first: CabinetSceneSpecification, second: CabinetSceneSpecification
) -> float:
    """
    :return: How far apart the centres of the two cabinets' fronts lie on the table.
    """
    return math.hypot(
        float(first.table_T_cabinet_front.x) - float(second.table_T_cabinet_front.x),
        float(first.table_T_cabinet_front.y) - float(second.table_T_cabinet_front.y),
    )


# %% prior error


@pytest.mark.parametrize("level", list(PriorErrorLevel), ids=lambda level: level.name)
def test_the_believed_front_lies_the_levels_distance_from_the_true_one(protocol, level):
    magnitude = protocol.prior_error_magnitudes[level]
    truth = CabinetSceneSpecification(articulated_part=ArticulatedPart.DRAWER)

    error = PriorError.sample(
        location=magnitude.location,
        joint_axis=0.0,
        random=numpy.random.default_rng(0),
    )

    assert front_distance(error.believed(truth), truth) == pytest.approx(
        magnitude.location
    )


@pytest.mark.parametrize("level", list(PriorErrorLevel), ids=lambda level: level.name)
def test_the_believed_joint_axis_is_turned_by_the_levels_angle(protocol, level):
    magnitude = protocol.prior_error_magnitudes[level]
    truth = CabinetSceneSpecification(articulated_part=ArticulatedPart.DRAWER)

    error = PriorError.sample(
        location=0.0,
        joint_axis=magnitude.joint_axis,
        random=numpy.random.default_rng(0),
    )

    believed = error.believed(truth)
    assert abs(
        believed.mechanism_axis_deviation - truth.mechanism_axis_deviation
    ) == pytest.approx(magnitude.joint_axis)


def test_a_location_error_keeps_the_cabinets_yaw(protocol):
    magnitude = protocol.prior_error_magnitudes[PriorErrorLevel.HIGH]
    truth = CabinetSceneSpecification(articulated_part=ArticulatedPart.DRAWER)

    error = PriorError.sample(
        location=magnitude.location,
        joint_axis=magnitude.joint_axis,
        random=numpy.random.default_rng(0),
    )

    believed_yaw = float(error.believed(truth).table_T_cabinet_front.yaw)
    assert believed_yaw == pytest.approx(float(truth.table_T_cabinet_front.yaw))


# %% the conditions


def test_every_condition_kind_is_run(protocol):
    kinds = {condition.kind for condition in protocol.conditions()}

    assert kinds == set(ConditionKind)


@pytest.mark.parametrize(
    "kind, level_of",
    [
        (
            ConditionKind.LOCATION_PRIOR_ERROR,
            lambda condition: condition.location_error,
        ),
        (ConditionKind.JOINT_PRIOR_ERROR, lambda condition: condition.joint_error),
    ],
    ids=lambda value: value.name if isinstance(value, ConditionKind) else "",
)
def test_each_prior_error_runs_every_wrong_level(protocol, kind, level_of):
    levels = [
        level_of(condition)
        for condition in protocol.conditions()
        if condition.kind is kind
    ]

    assert levels == [
        PriorErrorLevel.LOW,
        PriorErrorLevel.MEDIUM,
        PriorErrorLevel.HIGH,
    ]


def test_the_cabinet_is_moved_after_the_protocols_time(protocol):
    [condition] = [
        condition
        for condition in protocol.conditions()
        if condition.kind is ConditionKind.CABINET_MOVED
    ]

    [disturbance] = condition.disturbances
    assert isinstance(disturbance, CabinetMoved)
    assert disturbance.trigger.elapsed == protocol.cabinet_moved_after
    assert disturbance.displacement is protocol.cabinet_displacement


@pytest.mark.parametrize(
    "kind, opened_by",
    [
        (
            ConditionKind.PULLED_FROM_THE_HAND,
            lambda protocol: protocol.pulled_once_opened_by,
        ),
        (ConditionKind.CLOSED_AGAIN, lambda protocol: protocol.closed_once_opened_by),
    ],
    ids=lambda value: value.name if isinstance(value, ConditionKind) else "",
)
def test_the_part_is_pushed_shut_once_opened_far_enough(protocol, kind, opened_by):
    [condition] = [
        condition for condition in protocol.conditions() if condition.kind is kind
    ]

    [disturbance] = condition.disturbances
    assert isinstance(disturbance, PartPushed)
    assert disturbance.trigger == OnceOpenedBy(opened_by(protocol))
    assert disturbance.force == protocol.pull_force
    assert disturbance.duration == protocol.pull_duration


# %% the episodes


def test_each_condition_runs_its_number_of_episodes(protocol):
    setups = protocol.episode_setups(seed=0)

    assert len(setups) == len(protocol.conditions()) * protocol.episodes_per_condition


def test_the_same_seed_draws_the_same_episodes(protocol):
    first = protocol.episode_setups(seed=7)
    second = protocol.episode_setups(seed=7)

    assert [setup.seed for setup in first] == [setup.seed for setup in second]
    assert [
        front_distance(one.believed_specification, other.believed_specification)
        for one, other in zip(first, second)
    ] == [0.0] * len(first)


def test_the_episodes_of_a_condition_differ(protocol):
    setups = [
        setup
        for setup in protocol.episode_setups(seed=0)
        if setup.condition.location_error is PriorErrorLevel.HIGH
    ]

    first, second, *_ = setups
    assert (
        front_distance(first.believed_specification, second.believed_specification)
        > 0.0
    )


def test_an_undisturbed_episode_believes_the_truth(protocol):
    setup = next(
        setup
        for setup in protocol.episode_setups(seed=0)
        if setup.condition.kind is ConditionKind.UNDISTURBED
    )

    believed = setup.believed_specification
    assert front_distance(believed, setup.true_specification) == 0.0
    assert (
        believed.mechanism_axis_deviation
        == setup.true_specification.mechanism_axis_deviation
    )


def test_the_yaw_sweep_turns_the_true_cabinet(protocol):
    yaws = [
        float(setup.true_specification.table_T_cabinet_front.yaw)
        for setup in protocol.episode_setups(seed=0)
        if setup.condition.kind is ConditionKind.CABINET_YAW
    ]

    assert yaws == pytest.approx(
        [
            yaw
            for yaw in protocol.cabinet_yaws
            for _ in range(protocol.episodes_per_condition)
        ]
    )


def test_episodes_open_the_protocols_part():
    protocol = DisturbanceProtocol(articulated_part=ArticulatedPart.DOOR)

    parts = {
        setup.true_specification.articulated_part
        for setup in protocol.episode_setups(seed=0)
    }

    assert parts == {ArticulatedPart.DOOR}


# %% the arm's start pose


def test_the_arm_starts_within_its_deviation():
    start = ArmStartPose(deviation=0.1, seed=3)

    offsets = start.offsets(joints=[object()] * 6)

    assert len(offsets) == 6
    assert max(abs(offset) for offset in offsets) <= start.deviation
    assert max(abs(offset) for offset in offsets) > 0.0


def test_an_arm_without_deviation_starts_parked():
    offsets = ArmStartPose().offsets(joints=[object()] * 6)

    assert offsets == [0.0] * 6
