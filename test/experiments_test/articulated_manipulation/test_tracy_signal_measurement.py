"""
Tests for recording Tracy's signals and reporting which of them arrive, how often and
how noisily.

Skipped where Tracy's description is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState

from krrood.adapters.json_serializer import from_json, to_json

from experiments.articulated_manipulation.signal_statistics import SignalRecording
from experiments.articulated_manipulation.tracy_signal_measurement import (
    AtRestMeasurement,
    SignalMeasurement,
    SignalRecorder,
    TracySignalReport,
)
from experiments.articulated_manipulation.tracy_signals import (
    FingerPosition,
    TracySignalInventory,
    WristWrench,
)
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.utils import tracy_installed

pytestmark = pytest.mark.skipif(
    not tracy_installed(), reason="iai_tracy_description is not installed"
)

# %% helpers


@pytest.fixture
def tracy(tracy_world) -> Tracy:
    return tracy_world.get_semantic_annotations_by_type(Tracy)[0]


def wrench_at(stamp_nanoseconds: int, force_x: float) -> WrenchStamped:
    """
    A wrench taken at the given time, pushing along x.
    """
    message = WrenchStamped()
    message.header.stamp.sec = stamp_nanoseconds // 1_000_000_000
    message.header.stamp.nanosec = stamp_nanoseconds % 1_000_000_000
    message.wrench.force.x = force_x
    return message


def wrench_recorder(signal: WristWrench, sample_count: int) -> SignalRecorder:
    """
    A recorder of the given wrench signal that received the given number of samples,
    taken every two milliseconds.
    """
    recorder = SignalRecorder(signal=signal)
    for index in range(sample_count):
        recorder.receive(wrench_at(index * 2_000_000, float(index)))
    return recorder


# %% recording


def test_recorder_keeps_each_sample_with_its_stamp(tracy):
    signal = WristWrench(sensor=tracy.left_arm.sensors[0])
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
    assert recording.channel_names == [channel.name for channel in signal.channels]


def test_recorder_skips_messages_that_do_not_carry_the_signal(tracy):
    recorder = SignalRecorder(
        signal=FingerPosition(gripper=tracy.left_arm.end_effector)
    )
    message = JointState()
    message.name = ["unrelated_joint"]
    message.position = [0.1]

    recorder.receive(message)

    assert recorder.recording().stamps.size == 0
    assert recorder.recording().values.shape == (0, 1)


# %% measurements


def test_signal_that_barely_arrived_is_not_exposed(tracy):
    recorder = wrench_recorder(
        WristWrench(sensor=tracy.left_arm.sensors[0]),
        SignalRecording.minimum_sample_count - 1,
    )

    measurement = SignalMeasurement.from_recorder(recorder)

    assert not measurement.is_exposed
    assert measurement.statistics is None


def test_signal_that_arrived_carries_its_statistics(tracy):
    signal = WristWrench(sensor=tracy.left_arm.sensors[0])
    recorder = wrench_recorder(signal, 10)

    measurement = SignalMeasurement.from_recorder(recorder)

    assert measurement.is_exposed
    assert measurement.statistics == recorder.recording().statistics()
    assert measurement.topic == signal.topic
    assert measurement.channels == signal.channels


def test_report_survives_a_round_trip_through_json(tracy):
    report = TracySignalReport(
        duration=1.0,
        measurements=[
            SignalMeasurement.from_recorder(
                wrench_recorder(WristWrench(sensor=tracy.left_arm.sensors[0]), 10)
            ),
            SignalMeasurement.from_recorder(
                wrench_recorder(WristWrench(sensor=tracy.right_arm.sensors[0]), 0)
            ),
        ],
    )

    assert from_json(to_json(report)) == report


# %% measuring on a live ROS graph


def test_measurement_reports_published_signals_as_exposed_and_silent_ones_not(
    rclpy_node, tracy
):
    published = WristWrench(sensor=tracy.left_arm.sensors[0])
    silent = WristWrench(sensor=tracy.right_arm.sensors[0])
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
        (measurement.topic, measurement.is_exposed)
        for measurement in report.measurements
    ]
    assert exposure == [(published.topic, True), (silent.topic, False)]
