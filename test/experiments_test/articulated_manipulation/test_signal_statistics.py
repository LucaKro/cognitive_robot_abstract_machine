"""
Tests for the statistics of a signal recorded while the robot is at rest.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.articulated_manipulation.signal_statistics import (
    MINIMUM_SAMPLE_COUNT,
    SignalRecording,
    TooFewSamplesError,
)

# %% timing


def test_rate_is_the_frequency_the_samples_were_taken_at():
    frequency = 500.0
    stamps = np.arange(1_000) / frequency
    recording = SignalRecording(stamps=stamps, values=np.zeros((stamps.size, 1)))

    assert recording.statistics().rate == pytest.approx(frequency)


def test_interval_spread_measures_uneven_spacing():
    short_interval, long_interval = 0.001, 0.003
    intervals = np.tile([short_interval, long_interval], 50)
    stamps = np.concatenate([[0.0], np.cumsum(intervals)])
    recording = SignalRecording(stamps=stamps, values=np.zeros((stamps.size, 1)))

    assert recording.statistics().interval_standard_deviation == pytest.approx(
        (long_interval - short_interval) / 2
    )


def test_longest_interval_finds_a_dropout():
    regular_interval, dropout = 0.002, 0.05
    intervals = np.full(100, regular_interval)
    intervals[40] = dropout
    stamps = np.concatenate([[0.0], np.cumsum(intervals)])
    recording = SignalRecording(stamps=stamps, values=np.zeros((stamps.size, 1)))

    assert recording.statistics().longest_interval == pytest.approx(dropout)


def test_sample_count_counts_every_sample():
    stamps = np.arange(37) * 0.01
    recording = SignalRecording(stamps=stamps, values=np.zeros((stamps.size, 1)))

    assert recording.statistics().sample_count == stamps.size


# %% channels


def test_each_channel_reports_its_own_mean_and_spread():
    low, high = 1.0, 3.0
    first_channel = np.tile([low, high], 50)
    second_channel = np.full(first_channel.size, -2.0)
    stamps = np.arange(first_channel.size) * 0.01
    recording = SignalRecording(
        stamps=stamps, values=np.column_stack([first_channel, second_channel])
    )

    first, second = recording.statistics().channels

    assert first.mean == pytest.approx((low + high) / 2)
    assert first.standard_deviation == pytest.approx((high - low) / 2)
    assert second.mean == pytest.approx(-2.0)
    assert second.standard_deviation == pytest.approx(0.0)


def test_smallest_step_is_the_quantisation_of_a_register_value():
    quantisation = 0.25
    register_counts = np.array([4, 5, 7, 5, 4, 6, 4])
    stamps = np.arange(register_counts.size) * 0.01
    recording = SignalRecording(
        stamps=stamps, values=(register_counts * quantisation)[:, np.newaxis]
    )

    (channel,) = recording.statistics().channels

    assert channel.smallest_step == pytest.approx(quantisation)


def test_constant_channel_has_no_observable_step():
    stamps = np.arange(10) * 0.01
    recording = SignalRecording(stamps=stamps, values=np.full((stamps.size, 1), 2.0))

    (channel,) = recording.statistics().channels

    assert channel.smallest_step is None


# %% invalid recordings


def test_too_few_samples_have_no_statistics():
    stamps = np.arange(MINIMUM_SAMPLE_COUNT - 1) * 0.01
    recording = SignalRecording(stamps=stamps, values=np.zeros((stamps.size, 1)))

    with pytest.raises(TooFewSamplesError):
        recording.statistics()
