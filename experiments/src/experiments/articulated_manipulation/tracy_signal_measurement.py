"""
Measure which of Tracy's signals its drivers actually publish, how often they arrive and
how noisy they are, and write the result as a JSON report.

Run it on the machine that runs Tracy's drivers while the arms stand still and the
grippers are open and hold nothing, so that each channel's spread is its noise::

    python -m experiments.articulated_manipulation.tracy_signal_measurement \\
        --duration 60 --output tracy_signals.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from typing_extensions import Any, List, Optional

from krrood.adapters.json_serializer import to_json

from experiments.articulated_manipulation.signal_statistics import (
    MINIMUM_SAMPLE_COUNT,
    SignalRecording,
    SignalStatistics,
)
from experiments.articulated_manipulation.tracy_signals import (
    TracySignal,
    TracySignalInventory,
)

RECEIVE_QUEUE_DEPTH = 1_000
"""
How many messages a subscription holds before dropping the oldest: two seconds of the
fastest signal, so that a late callback does not drop samples and make the driver look
slower than it is.
"""

NODE_NAME = "tracy_signal_measurement"
"""
The name of the node that records the signals.
"""

JSON_INDENTATION = 2
"""
Spaces per nesting level in the written report, so it can be read and diffed.
"""

# %% recording


@dataclass
class SignalRecorder:
    """
    Collects the samples of one signal from the messages it is published in.
    """

    signal: TracySignal
    """
    The signal being recorded.
    """

    stamps: List[float] = field(default_factory=list, init=False)
    """
    When each recorded sample was taken, in seconds.
    """

    values: List[np.ndarray] = field(default_factory=list, init=False)
    """
    The channels' values of each recorded sample.
    """

    def receive(self, message: Any) -> None:
        """
        Record the sample the message carries, unless it does not carry the signal.

        :param message: A message published on the signal's topic.
        """
        if not self.signal.provides(message):
            return
        self.stamps.append(self.signal.stamp(message))
        self.values.append(self.signal.read(message))

    def recording(self) -> SignalRecording:
        """
        :return: Every sample recorded so far.
        """
        return SignalRecording(
            stamps=np.array(self.stamps),
            values=np.array(self.values).reshape(
                len(self.stamps), len(self.signal.channels)
            ),
        )


# %% report


@dataclass
class SignalMeasurement:
    """
    Whether one signal arrived during a measurement, and if so, how it behaved.
    """

    signal: TracySignal
    """
    The signal measured.
    """

    statistics: Optional[SignalStatistics]
    """
    How often the signal arrived and how its channels varied; ``None`` if too few of
    its samples arrived to tell.
    """

    @classmethod
    def from_recording(
        cls, signal: TracySignal, recording: SignalRecording
    ) -> SignalMeasurement:
        """
        :param signal: The signal measured.
        :param recording: Every sample of it that arrived.
        :return: The measurement of the signal.
        """
        if recording.stamps.size < MINIMUM_SAMPLE_COUNT:
            return cls(signal=signal, statistics=None)
        return cls(signal=signal, statistics=recording.statistics())

    @property
    def is_exposed(self) -> bool:
        """
        Whether the drivers publish the signal.
        """
        return self.statistics is not None


@dataclass
class TracySignalReport:
    """
    The measurement of every signal of Tracy over one recording period.
    """

    duration: float
    """
    How long the signals were recorded, in seconds.
    """

    measurements: List[SignalMeasurement]
    """
    The measurement of each signal, in the order of the inventory.
    """


# %% measuring


@dataclass
class AtRestMeasurement:
    """
    Records every signal of an inventory for a while and reports what arrived.

    The node has to be spun by an executor for the whole measurement.
    """

    node: Node
    """
    The node that subscribes to the signals' topics.
    """

    inventory: TracySignalInventory
    """
    The signals to measure.
    """

    duration: float
    """
    How long to record, in seconds.
    """

    def run(self) -> TracySignalReport:
        """
        :return: The measurement of every signal, recorded for :attr:`duration`.
        """
        recorders = [SignalRecorder(signal=signal) for signal in self.inventory.signals]
        subscriptions = [
            self.node.create_subscription(
                recorder.signal.message_type(),
                recorder.signal.topic,
                recorder.receive,
                RECEIVE_QUEUE_DEPTH,
            )
            for recorder in recorders
        ]
        time.sleep(self.duration)
        for subscription in subscriptions:
            self.node.destroy_subscription(subscription)
        return TracySignalReport(
            duration=self.duration,
            measurements=[
                SignalMeasurement.from_recording(
                    signal=recorder.signal, recording=recorder.recording()
                )
                for recorder in recorders
            ],
        )


# %% command line


class CommandLineOption(StrEnum):
    """
    The options the measurement is run with.
    """

    DURATION = "--duration"
    """
    How long to record, in seconds.
    """

    OUTPUT = "--output"
    """
    Where to write the JSON report.
    """


@dataclass
class MeasurementSettings:
    """
    What the measurement is run with, as given on the command line.
    """

    duration: float
    """
    How long to record, in seconds.
    """

    output: Path
    """
    Where to write the JSON report.
    """

    @classmethod
    def from_command_line(cls) -> MeasurementSettings:
        """
        :return: The settings given on the command line.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(CommandLineOption.DURATION, type=float, required=True)
        parser.add_argument(CommandLineOption.OUTPUT, type=Path, required=True)
        arguments = parser.parse_args()
        return cls(duration=arguments.duration, output=arguments.output)


def main() -> None:
    """
    Measure every signal of Tracy and write the report.
    """
    settings = MeasurementSettings.from_command_line()
    rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    report = AtRestMeasurement(
        node=node,
        inventory=TracySignalInventory.of_tracy(),
        duration=settings.duration,
    ).run()
    settings.output.write_text(json.dumps(to_json(report), indent=JSON_INDENTATION))
    executor.shutdown()
    spinner.join()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
