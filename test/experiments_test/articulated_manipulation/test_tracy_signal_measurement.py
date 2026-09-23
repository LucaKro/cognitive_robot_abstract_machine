"""
Tests for recording Tracy's signals and reporting which of them arrive, how often and
how noisily.
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState

from krrood.adapters.json_serializer import from_json, to_json

from experiments.articulated_manipulation.signal_statistics import (
    MINIMUM_SAMPLE_COUNT,
    SignalRecording,
)
from experiments.articulated_manipulation.tracy_signal_measurement import (
    AtRestMeasurement,
    SignalMeasurement,
    SignalRecorder,
    TracySignalReport,
)
from experiments.articulated_manipulation.tracy_signals import (
    FingerPosition,
    TracySide,
    TracySignalInventory,
    WristWrench,
)

# %% helpers


def wrench_at(stamp_nanoseconds: int, force_x: float) -> WrenchStamped:
    """
    A wrench taken at the given time, pushing along x.
    """
    message = WrenchStamped()
    message.header.stamp.sec = stamp_nanoseconds // 1_000_000_000
    message.header.stamp.nanosec = stamp_nanoseconds % 1_000_000_000
    message.wrench.force.x = force_x
    return message


def recording_of(sample_count: int) -> SignalRecording:
    """
    A recording of the left wrist wrench with the given number of samples, taken every
    two milliseconds.
    """
    signal = WristWrench(side=TracySide.LEFT)
    recorder = SignalRecorder(signal=signal)
    for index in range(sample_count):
        recorder.receive(wrench_at(index * 2_000_000, float(index)))
    return recorder.recording()


# %% recording


def test_recorder_keeps_each_sample_with_its_stamp():
    signal = WristWrench(side=TracySide.LEFT)
    recorder = SignalRecorder(signal=signal)
    messages = [wrench_at(2_000_000, 1.0), wrench_at(4_000_000, 2.0)]

    for message in messages:
        recorder.receive(message)
    recording = recorder.recording()

    np.testing.assert_array_equal(
        recording.stamps, [signal.stamp(message) for message in messages]
    )
    np.testing.assert_array_equal(
        recording.values, [signal.read(message) for message in messages]
    )


def test_recorder_skips_messages_that_do_not_carry_the_signal():
    recorder = SignalRecorder(signal=FingerPosition(side=TracySide.LEFT))
    message = JointState()
    message.name = ["unrelated_joint"]
    message.position = [0.1]

    recorder.receive(message)

    assert recorder.recording().stamps.size == 0
    assert recorder.recording().values.shape == (0, 1)


# %% measurements


def test_signal_that_barely_arrived_is_not_exposed():
    recording = recording_of(MINIMUM_SAMPLE_COUNT - 1)

    measurement = SignalMeasurement.from_recording(
        signal=WristWrench(side=TracySide.LEFT), recording=recording
    )

    assert not measurement.is_exposed
    assert measurement.statistics is None


def test_signal_that_arrived_carries_its_statistics():
    recording = recording_of(10)

    measurement = SignalMeasurement.from_recording(
        signal=WristWrench(side=TracySide.LEFT), recording=recording
    )

    assert measurement.is_exposed
    assert measurement.statistics == recording.statistics()


def test_report_survives_a_round_trip_through_json():
    report = TracySignalReport(
        duration=1.0,
        measurements=[
            SignalMeasurement.from_recording(
                signal=WristWrench(side=TracySide.LEFT), recording=recording_of(10)
            ),
            SignalMeasurement.from_recording(
                signal=WristWrench(side=TracySide.RIGHT), recording=recording_of(0)
            ),
        ],
    )

    assert from_json(to_json(report)) == report


# %% measuring on a live ROS graph


def test_measurement_reports_published_signals_as_exposed_and_silent_ones_not(
    rclpy_node,
):
    published = WristWrench(side=TracySide.LEFT)
    silent = WristWrench(side=TracySide.RIGHT)
    publisher = rclpy_node.create_publisher(WrenchStamped, published.topic, 10)

    def publish_wrench() -> None:
        message = WrenchStamped()
        message.header.stamp = rclpy_node.get_clock().now().to_msg()
        publisher.publish(message)

    timer = rclpy_node.create_timer(0.01, publish_wrench)
    report = AtRestMeasurement(
        node=rclpy_node,
        inventory=TracySignalInventory(signals=[published, silent]),
        duration=1.0,
    ).run()
    rclpy_node.destroy_timer(timer)
    rclpy_node.destroy_publisher(publisher)

    exposure = [
        (measurement.signal, measurement.is_exposed)
        for measurement in report.measurements
    ]
    assert exposure == [(published, True), (silent, False)]
