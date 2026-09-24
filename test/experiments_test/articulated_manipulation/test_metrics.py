"""
Tests for what the benchmark reports about a task.
"""

from __future__ import annotations

import statistics
from datetime import timedelta

import pytest

from experiments.articulated_manipulation.disturbance_protocol import (
    DisturbanceProtocol,
    EpisodeSetup,
)
from experiments.articulated_manipulation.metrics import (
    ConditionReport,
    EpisodeOutcome,
    EpisodeVerdict,
    ProtocolReport,
)

REQUIRED_OPENED_FRACTION = 0.8
"""
The share of its travel the part has to be open in these tests.
"""


def outcome(
    setup: EpisodeSetup,
    verdict: EpisodeVerdict,
    completion_time: timedelta | None = None,
    cycle_durations: list[float] | None = None,
) -> EpisodeOutcome:
    """
    :return: An outcome of the given setup that ended with the given verdict.
    """
    return EpisodeOutcome(
        setup=setup,
        verdict=verdict,
        completion_time=completion_time,
        opened_fraction=0.0,
        control_cycle_durations=cycle_durations or [0.01],
    )


@pytest.fixture
def setups() -> list[EpisodeSetup]:
    return DisturbanceProtocol(episodes_per_condition=4).episode_setups(seed=0)


# %% the verdict


def test_a_task_that_never_reports_done_is_unfinished():
    verdict = EpisodeVerdict.judge(
        reported_done=False,
        opened_fraction=1.0,
        required_opened_fraction=REQUIRED_OPENED_FRACTION,
    )

    assert verdict is EpisodeVerdict.UNFINISHED


def test_reporting_done_with_the_part_open_is_a_success():
    verdict = EpisodeVerdict.judge(
        reported_done=True,
        opened_fraction=REQUIRED_OPENED_FRACTION,
        required_opened_fraction=REQUIRED_OPENED_FRACTION,
    )

    assert verdict is EpisodeVerdict.SUCCESS


def test_reporting_done_with_the_part_not_open_enough_is_a_false_success():
    verdict = EpisodeVerdict.judge(
        reported_done=True,
        opened_fraction=REQUIRED_OPENED_FRACTION - 0.01,
        required_opened_fraction=REQUIRED_OPENED_FRACTION,
    )

    assert verdict is EpisodeVerdict.FALSE_SUCCESS


# %% one condition


def test_the_rates_count_the_share_of_each_verdict(setups):
    first, second, third, fourth = setups[:4]
    report = ConditionReport(
        condition=first.condition,
        outcomes=[
            outcome(first, EpisodeVerdict.SUCCESS, timedelta(seconds=1)),
            outcome(second, EpisodeVerdict.FALSE_SUCCESS, timedelta(seconds=2)),
            outcome(third, EpisodeVerdict.FALSE_SUCCESS, timedelta(seconds=2)),
            outcome(fourth, EpisodeVerdict.UNFINISHED),
        ],
    )

    assert report.success_rate == 1 / 4
    assert report.false_success_rate == 2 / 4


def test_the_completion_time_averages_the_successes_only(setups):
    first, second, third, _ = setups[:4]
    report = ConditionReport(
        condition=first.condition,
        outcomes=[
            outcome(first, EpisodeVerdict.SUCCESS, timedelta(seconds=1)),
            outcome(second, EpisodeVerdict.SUCCESS, timedelta(seconds=3)),
            outcome(third, EpisodeVerdict.FALSE_SUCCESS, timedelta(seconds=10)),
        ],
    )

    assert report.mean_completion_time == timedelta(seconds=2)


def test_without_a_success_there_is_no_completion_time(setups):
    first = setups[0]
    report = ConditionReport(
        condition=first.condition,
        outcomes=[outcome(first, EpisodeVerdict.UNFINISHED)],
    )

    assert report.mean_completion_time is None


def test_the_cycle_time_averages_every_cycle_of_every_episode(setups):
    first, second = setups[:2]
    first_cycles = [0.01, 0.02]
    second_cycles = [0.03]
    report = ConditionReport(
        condition=first.condition,
        outcomes=[
            outcome(first, EpisodeVerdict.SUCCESS, timedelta(seconds=1), first_cycles),
            outcome(second, EpisodeVerdict.UNFINISHED, cycle_durations=second_cycles),
        ],
    )

    assert report.mean_control_cycle_duration == pytest.approx(
        statistics.fmean(first_cycles + second_cycles)
    )


# %% the whole protocol


def test_outcomes_are_grouped_by_their_condition(setups):
    outcomes = [outcome(setup, EpisodeVerdict.UNFINISHED) for setup in setups]

    report = ProtocolReport.from_outcomes(outcomes, recovery_transitions_authored=0)

    assert [
        condition_report.condition.kind for condition_report in report.condition_reports
    ] == [condition.kind for condition in DisturbanceProtocol().conditions()]
    assert all(
        len(condition_report.outcomes) == 4
        for condition_report in report.condition_reports
    )
    assert all(
        episode.setup.condition is condition_report.condition
        for condition_report in report.condition_reports
        for episode in condition_report.outcomes
    )


def test_the_report_carries_the_recovery_transitions_authored(setups):
    authored = 3

    report = ProtocolReport.from_outcomes(
        [outcome(setups[0], EpisodeVerdict.SUCCESS, timedelta(seconds=1))],
        recovery_transitions_authored=authored,
    )

    assert report.recovery_transitions_authored == authored
