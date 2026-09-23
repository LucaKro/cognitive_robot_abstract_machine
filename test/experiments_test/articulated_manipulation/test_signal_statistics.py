"""
Tests for the statistics of a signal recorded while the robot is at rest.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.distributions.gaussian import GaussianDistribution

from experiments.articulated_manipulation.signal_statistics import (
    SignalRecording,
    TooFewSamplesError,
)

# %% helpers


def recording_of_one_channel(
    stamps: npt.NDArray[np.float64], values: npt.NDArray[np.float64] | None = None
) -> SignalRecording:
    """
    A recording of a single channel named ``value``, zero unless values are given.
    """
    channel_values = np.zeros(stamps.size) if values is None else values
    return SignalRecording(
        stamps=stamps,
        values=channel_values[:, np.newaxis],
        channel_names=["value"],
    )


# %% timing


def test_rate_is_the_frequency_the_samples_were_taken_at():
    frequency = 500.0
    recording = recording_of_one_channel(np.arange(1_000) / frequency)

    assert recording.statistics().rate == pytest.approx(frequency)


def test_interval_is_a_gaussian_over_the_time_between_samples():
    short_interval, long_interval = 0.001, 0.003
    intervals = np.tile([short_interval, long_interval], 50)
    recording = recording_of_one_channel(np.concatenate([[0.0], np.cumsum(intervals)]))

    interval = recording.statistics().interval

    assert isinstance(interval, GaussianDistribution)
    assert interval.location == pytest.approx((short_interval + long_interval) / 2)
    assert interval.scale == pytest.approx((long_interval - short_interval) / 2)


def test_longest_interval_finds_a_dropout():
    regular_interval, dropout = 0.002, 0.05
    intervals = np.full(100, regular_interval)
    intervals[40] = dropout
    recording = recording_of_one_channel(np.concatenate([[0.0], np.cumsum(intervals)]))

    assert recording.statistics().longest_interval == pytest.approx(dropout)


def test_sample_count_counts_every_sample():
    stamps = np.arange(37) * 0.01
    recording = recording_of_one_channel(stamps)

    assert recording.statistics().sample_count == stamps.size


# %% channels


def test_each_channel_is_a_gaussian_over_its_own_values():
    low, high = 1.0, 3.0
    first_channel = np.tile([low, high], 50)
    second_channel = np.tile([-1.0, 1.0], 50)
    recording = SignalRecording(
        stamps=np.arange(first_channel.size) * 0.01,
        values=np.column_stack([first_channel, second_channel]),
        channel_names=["first", "second"],
    )

    first, second = recording.statistics().channels

    assert first.distribution.variable.name == recording.channel_names[0]
    assert first.distribution.location == pytest.approx((low + high) / 2)
    assert first.distribution.scale == pytest.approx((high - low) / 2)
    assert second.distribution.variable.name == recording.channel_names[1]
    assert second.distribution.location == pytest.approx(0.0)


def test_constant_channel_is_a_dirac_delta_without_an_observable_step():
    level = 2.0
    stamps = np.arange(10) * 0.01
    recording = recording_of_one_channel(stamps, np.full(stamps.size, level))

    (channel,) = recording.statistics().channels

    assert isinstance(channel.distribution, DiracDeltaDistribution)
    assert channel.distribution.location == level
    assert channel.smallest_step is None


def test_smallest_step_is_the_quantisation_of_a_register_value():
    quantisation = 0.25
    register_counts = np.array([4, 5, 7, 5, 4, 6, 4])
    recording = recording_of_one_channel(
        np.arange(register_counts.size) * 0.01, register_counts * quantisation
    )

    (channel,) = recording.statistics().channels

    assert channel.smallest_step == pytest.approx(quantisation)


# %% invalid recordings


def test_a_single_sample_has_no_statistics():
    recording = recording_of_one_channel(np.zeros(1))

    with pytest.raises(TooFewSamplesError):
        recording.statistics()
