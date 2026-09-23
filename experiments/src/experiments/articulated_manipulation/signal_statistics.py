"""
How often a signal arrives and how much it varies while the robot is at rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from probabilistic_model.distributions.distributions import (
    ContinuousDistribution,
    DiracDeltaDistribution,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from random_events.variable import Continuous
from typing_extensions import List, Optional

from krrood.exceptions import DataclassException

# %% exceptions


@dataclass
class TooFewSamplesError(DataclassException):
    """
    Raised when a recording has too few samples to measure how often they arrive.
    """

    sample_count: int
    """
    How many samples the recording has.
    """

    def error_message(self) -> str:
        return (
            "A recording needs an interval between two samples to be timed, but has "
            f"{self.sample_count} sample(s)."
        )

    def suggest_correction(self) -> str:
        return "Record for longer, or check that the signal is being published."


# %% statistics


class TimingVariable(StrEnum):
    """
    The variables a recording's timing is described over.
    """

    INTERVAL = "interval"
    """
    The time between two consecutive samples, in seconds.
    """


@dataclass
class ChannelStatistics:
    """
    How one channel of a signal varies over a recording.
    """

    distribution: ContinuousDistribution
    """
    The channel's values: a Gaussian whose location is the channel's bias at rest and
    whose scale is its noise, or a Dirac delta if the channel never changed.
    """

    smallest_step: Optional[float]
    """
    The smallest difference between two distinct values the channel took, which shows
    its quantisation; ``None`` if it only ever took one value.
    """


@dataclass
class SignalStatistics:
    """
    How often a signal arrived and how each of its channels varied over a recording.
    """

    sample_count: int
    """
    How many samples were recorded.
    """

    interval: ContinuousDistribution
    """
    The time between consecutive samples: a Gaussian whose scale is the timing
    jitter, or a Dirac delta if every interval was the same.
    """

    longest_interval: float
    """
    The longest time between two consecutive samples, in seconds, which shows dropouts.
    """

    channels: List[ChannelStatistics]
    """
    The statistics of each channel, in the order of the signal's channels.
    """

    @property
    def rate(self) -> float:
        """
        Samples per second, on average.
        """
        return 1.0 / self.interval.location


# %% recordings


@dataclass
class SignalRecording:
    """
    The samples of one signal, in the order they were taken.
    """

    stamps: npt.NDArray[np.float64]
    """
    When each sample was taken, in seconds.
    """

    values: npt.NDArray[np.float64]
    """
    The value of every channel at every sample, one row per sample.
    """

    channel_names: List[str]
    """
    The name of each channel, in the order of the columns of :attr:`values`.
    """

    def statistics(self) -> SignalStatistics:
        """
        :return: How often the samples arrived and how each channel varied.
        :raises TooFewSamplesError: If the recording has too few samples to time them.
        """
        if not self.is_timed:
            raise TooFewSamplesError(sample_count=self.stamps.size)
        intervals = self.intervals
        return SignalStatistics(
            sample_count=self.stamps.size,
            interval=self.distribution_of(TimingVariable.INTERVAL, intervals),
            longest_interval=float(np.max(intervals)),
            channels=[
                ChannelStatistics(
                    distribution=self.distribution_of(name, channel),
                    smallest_step=self.smallest_step_of(channel),
                )
                for name, channel in zip(self.channel_names, self.values.T)
            ],
        )

    @property
    def intervals(self) -> npt.NDArray[np.float64]:
        """
        The time between each two consecutive samples, in seconds.
        """
        return np.diff(self.stamps)

    @property
    def is_timed(self) -> bool:
        """
        Whether the recording has an interval between two samples to time.
        """
        return self.intervals.size > 0

    @staticmethod
    def distribution_of(
        name: str, values: npt.NDArray[np.float64]
    ) -> ContinuousDistribution:
        """
        :param name: The name of the variable the values are samples of.
        :param values: The samples.
        :return: The Gaussian with the samples' mean and standard deviation, or a Dirac
            delta at their value if they never vary.
        """
        variable = Continuous(name)
        mean, standard_deviation = float(np.mean(values)), float(np.std(values))
        if standard_deviation == 0.0:
            return DiracDeltaDistribution(variable=variable, location=mean)
        return GaussianDistribution(
            variable=variable, location=mean, scale=standard_deviation
        )

    @staticmethod
    def smallest_step_of(values: npt.NDArray[np.float64]) -> Optional[float]:
        """
        :param values: The samples of one channel.
        :return: The smallest difference between two distinct values, or ``None`` if
            the values never differ.
        """
        steps = np.diff(np.unique(values))
        if steps.size == 0:
            return None
        return float(np.min(steps))
