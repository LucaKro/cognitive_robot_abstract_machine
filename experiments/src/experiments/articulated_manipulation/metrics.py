"""
What the drawer-opening benchmark reports: whether each episode succeeded, falsely
claimed success or never finished, and per condition the success and false-success
rates, the time to completion and the control-cycle time.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto

from experiments.articulated_manipulation.disturbance_protocol import (
    Condition,
    EpisodeSetup,
)

# %% one episode


class EpisodeVerdict(Enum):
    """
    How an episode ended, judged against the physics.
    """

    SUCCESS = auto()
    """
    The task reported that it is done, and the part is open.
    """

    FALSE_SUCCESS = auto()
    """
    The task reported that it is done, but the part is not open.
    """

    UNFINISHED = auto()
    """
    The task did not report that it is done in time.
    """

    @classmethod
    def judge(
        cls,
        reported_done: bool,
        opened_fraction: float,
        required_opened_fraction: float,
    ) -> EpisodeVerdict:
        """
        :param reported_done: Whether the task reported that it is done.
        :param opened_fraction: The share of its travel the part is open in the physics.
        :param required_opened_fraction: The share of its travel the part has to be open
            to count as open.
        :return: The verdict on the episode.
        """
        if not reported_done:
            return cls.UNFINISHED
        if opened_fraction < required_opened_fraction:
            return cls.FALSE_SUCCESS
        return cls.SUCCESS


@dataclass
class EpisodeOutcome:
    """
    How one episode ended.
    """

    setup: EpisodeSetup
    """
    The episode's setup.
    """

    verdict: EpisodeVerdict
    """
    How the episode ended.
    """

    completion_time: timedelta | None
    """
    How much simulated time passed until the task reported that it is done, or None if
    it never did.
    """

    opened_fraction: float
    """
    The share of its travel the part was open in the physics when the episode ended.
    """

    control_cycle_durations: list[float]
    """
    How long each of the controller's cycles took, in seconds.
    """


# %% one condition


@dataclass
class ConditionReport:
    """
    How a task fared in every episode of one condition.
    """

    condition: Condition
    """
    The condition.
    """

    outcomes: list[EpisodeOutcome]
    """
    How each of its episodes ended.
    """

    @property
    def success_rate(self) -> float:
        """
        The share of episodes that succeeded.
        """
        return self._share_of(EpisodeVerdict.SUCCESS)

    @property
    def false_success_rate(self) -> float:
        """
        The share of episodes in which the task reported success while the part was not
        open.
        """
        return self._share_of(EpisodeVerdict.FALSE_SUCCESS)

    @property
    def mean_completion_time(self) -> timedelta | None:
        """
        The mean simulated time the successful episodes took, or None if none succeeded.
        """
        times = [
            outcome.completion_time
            for outcome in self.outcomes
            if outcome.verdict is EpisodeVerdict.SUCCESS
        ]
        if not times:
            return None
        return sum(times, timedelta()) / len(times)

    @property
    def mean_control_cycle_duration(self) -> float:
        """
        The mean time one control cycle took over every episode, in seconds.
        """
        return statistics.fmean(
            duration
            for outcome in self.outcomes
            for duration in outcome.control_cycle_durations
        )

    def _share_of(self, verdict: EpisodeVerdict) -> float:
        """
        :param verdict: The verdict to count.
        :return: The share of episodes that ended with it.
        """
        return sum(outcome.verdict is verdict for outcome in self.outcomes) / len(
            self.outcomes
        )


# %% the whole protocol


@dataclass
class ProtocolReport:
    """
    How a task fared under every condition of the benchmark.
    """

    recovery_transitions_authored: int
    """
    How many transitions the task's author wrote to recover from a specific disturbance;
    the benchmark's target is none.
    """

    condition_reports: list[ConditionReport]
    """
    How the task fared under each condition.
    """

    @classmethod
    def from_outcomes(
        cls, outcomes: list[EpisodeOutcome], recovery_transitions_authored: int
    ) -> ProtocolReport:
        """
        :param outcomes: How every episode ended.
        :param recovery_transitions_authored: How many recovery transitions the task's
            author wrote.
        :return: The outcomes grouped by the condition they ran under, in the order the
            conditions first appear.
        """
        conditions: list[Condition] = []
        for outcome in outcomes:
            if not any(outcome.setup.condition is known for known in conditions):
                conditions.append(outcome.setup.condition)
        return cls(
            recovery_transitions_authored=recovery_transitions_authored,
            condition_reports=[
                ConditionReport(
                    condition=condition,
                    outcomes=[
                        outcome
                        for outcome in outcomes
                        if outcome.setup.condition is condition
                    ],
                )
                for condition in conditions
            ],
        )
