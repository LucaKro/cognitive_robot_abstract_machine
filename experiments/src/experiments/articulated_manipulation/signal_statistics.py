"""
How often a signal arrives and how much it varies while the robot is at rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import List, Optional

from krrood.exceptions import DataclassException

MINIMUM_SAMPLE_COUNT = 2
"""
The fewest samples a recording needs, since its timing is measured between samples.
"""

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
            f"A recording needs at least {MINIMUM_SAMPLE_COUNT} samples, "
            f"but has {self.sample_count}."
        )

    def suggest_correction(self) -> str:
        return "Record for longer, or check that the signal is being published."


# %% statistics


@dataclass
class ChannelStatistics:
    """
    How one channel of a signal varies over a recording.
    """

    mean: float
    """
    The channel's average value; at rest, its bias.
    """

    standard_deviation: float
    """
    The channel's spread around its mean; at rest, its noise.
    """

    smallest_step: Optional[float]
    """
    The smallest difference between two distinct values the channel took, which shows
    its quantisation; ``None`` if it only ever took one value.
    """

    @classmethod
    def from_values(cls, values: np.ndarray) -> ChannelStatistics:
        """
        :param values: The channel's values, one per sample.
        :return: The statistics of those values.
        """
        distinct_values = np.unique(values)
        smallest_step = (
            float(np.min(np.diff(distinct_values)))
            if distinct_values.size >= MINIMUM_SAMPLE_COUNT
            else None
        )
        return cls(
            mean=float(np.mean(values)),
            standard_deviation=float(np.std(values)),
            smallest_step=smallest_step,
        )


@dataclass
class SignalStatistics:
    """
    How often a signal arrived and how each of its channels varied over a recording.
    """

    sample_count: int
    """
    How many samples were recorded.
    """

    rate: float
    """
    Samples per second, over the whole recording.
    """

    interval_standard_deviation: float
    """
    The spread of the time between consecutive samples, in seconds.
    """

    longest_interval: float
    """
    The longest time between two consecutive samples, in seconds, which shows dropouts.
    """

    channels: List[ChannelStatistics]
    """
    The statistics of each channel, in the order of the signal's channels.
    """


# %% recordings


@dataclass
class SignalRecording:
    """
    The samples of one signal, in the order they were taken.
    """

    stamps: np.ndarray
    """
    When each sample was taken, in seconds.
    """

    values: np.ndarray
    """
    The value of every channel at every sample, one row per sample.
    """

    def statistics(self) -> SignalStatistics:
        """
        :return: How often the samples arrived and how each channel varied.
        :raises TooFewSamplesError: If the recording has too few samples to time them.
        """
        if self.stamps.size < MINIMUM_SAMPLE_COUNT:
            raise TooFewSamplesError(sample_count=self.stamps.size)
        intervals = np.diff(self.stamps)
        return SignalStatistics(
            sample_count=self.stamps.size,
            rate=float(intervals.size / (self.stamps[-1] - self.stamps[0])),
            interval_standard_deviation=float(np.std(intervals)),
            longest_interval=float(np.max(intervals)),
            channels=[
                ChannelStatistics.from_values(channel) for channel in self.values.T
            ],
        )
