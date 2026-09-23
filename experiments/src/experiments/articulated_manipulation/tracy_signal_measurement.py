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
import numpy.typing as npt
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from typing_extensions import ClassVar, Generic, List, Optional, TypeVar

from krrood.adapters.json_serializer import to_json

from experiments.articulated_manipulation.signal_statistics import (
    SignalRecording,
    SignalStatistics,
)
from experiments.articulated_manipulation.tracy_signals import (
    SignalChannel,
    TracySignal,
    TracySignalInventory,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.tracy import Tracy

MessageType = TypeVar("MessageType")

# %% recording


@dataclass
class SignalRecorder(Generic[MessageType]):
    """
    Collects the samples of one signal from the messages it is published in.
    """

    signal: TracySignal[MessageType]
    """
    The signal being recorded.
    """

    stamps: List[float] = field(default_factory=list, init=False)
    """
    When each recorded sample was taken, in seconds.
    """

    values: List[npt.NDArray[np.float64]] = field(default_factory=list, init=False)
    """
    The channels' values of each recorded sample.
    """

    def receive(self, message: MessageType) -> None:
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
        channels = self.signal.channels
        return SignalRecording(
            stamps=np.array(self.stamps),
            values=np.array(self.values).reshape(len(self.stamps), len(channels)),
            channel_names=[channel.name for channel in channels],
        )


# %% report


@dataclass
class SignalMeasurement:
    """
    Whether one signal arrived during a measurement, and if so, how it behaved.
    """

    topic: str
    """
    The topic the signal was expected on.
    """

    channels: List[SignalChannel]
    """
    The values the signal carries.
    """

    statistics: Optional[SignalStatistics]
    """
    How often the signal arrived and how its channels varied; ``None`` if too few of
    its samples arrived to tell.
    """

    @classmethod
    def from_recorder(cls, recorder: SignalRecorder) -> SignalMeasurement:
        """
        :param recorder: The recorder of the signal, once the measurement is over.
        :return: The measurement of the signal.
        """
        recording = recorder.recording()
        arrived = recording.stamps.size >= SignalRecording.minimum_sample_count
        return cls(
            topic=recorder.signal.topic,
            channels=recorder.signal.channels,
            statistics=recording.statistics() if arrived else None,
        )

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

    json_indentation: ClassVar[int] = 2
    """
    Spaces per nesting level in the written report, so it can be read and diffed.
    """

    def write(self, path: Path) -> None:
        """
        :param path: Where to write the report as JSON.
        """
        path.write_text(json.dumps(to_json(self), indent=self.json_indentation))


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

    receive_queue_depth: ClassVar[int] = 1_000
    """
    How many messages a subscription holds before dropping the oldest: two seconds of
    the fastest signal, so that a late callback does not drop samples and make the
    driver look slower than it is.
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
                self.receive_queue_depth,
            )
            for recorder in recorders
        ]
        time.sleep(self.duration)
        for subscription in subscriptions:
            self.node.destroy_subscription(subscription)
        return TracySignalReport(
            duration=self.duration,
            measurements=[
                SignalMeasurement.from_recorder(recorder) for recorder in recorders
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

    node_name: ClassVar[str] = "tracy_signal_measurement"
    """
    The name of the node that records the signals.
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
    world = URDFParser.from_file(file_path=Tracy.get_ros_file_path()).parse()
    Tracy.from_world(world)
    tracy = world.get_semantic_annotations_by_type(Tracy)[0]
    rclpy.init()
    node = rclpy.create_node(settings.node_name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    report = AtRestMeasurement(
        node=node,
        inventory=TracySignalInventory.of_tracy(tracy),
        duration=settings.duration,
    ).run()
    report.write(settings.output)
    executor.shutdown()
    spinner.join()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
