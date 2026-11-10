"""
The ``nlp-listen`` command, which publishes heard speech on ROS topics. This module
needs a sourced ROS installation.
"""

from __future__ import annotations

import argparse
import signal
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from types import FrameType

import rclpy
from krrood.ormatic.utils import classproperty
from rclpy.context import Context
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import String

from cramspeech.microphone import InputDevice, Microphone
from cramspeech.speech_recognizer import SpeechRecognizer
from cramspeech.transcription import (
    WhisperComputeType,
    WhisperLanguage,
    WhisperModelSize,
    WhisperTranscriber,
)


# %% Topics
class SpeechTopic(StrEnum):
    """
    The ROS topics heard sentences are published on.
    """

    HEARD_TEXT = "whisper_out"
    """
    The text of each utterance, for anyone who acts on what was said.
    """

    TEXT_TO_SAY = "/tts_text"
    """
    Sentences the robot's text-to-speech node says out loud.
    """


# %% Publishing
@dataclass
class TranscriptPublisher:
    """
    Publishes the text of every utterance the microphone hears on
    :attr:`SpeechTopic.HEARD_TEXT`, and also on :attr:`SpeechTopic.TEXT_TO_SAY` when
    echoing.
    """

    node: Node
    """
    The ROS node the publishers belong to.
    """

    recognizer: SpeechRecognizer
    """
    Hears the utterances and transcribes them.
    """

    message_queue_depth: int = 10
    """
    Number of messages each publisher keeps for subscribers that fall behind.
    """

    echo_heard_speech: bool = False
    """
    Whether the robot's text-to-speech node also receives what was heard, and so says
    it back.
    """

    publishers: dict[SpeechTopic, Publisher] = field(init=False)
    """
    One publisher per topic the heard text goes to.
    """

    def __post_init__(self) -> None:
        topics = [SpeechTopic.HEARD_TEXT]
        if self.echo_heard_speech:
            topics.append(SpeechTopic.TEXT_TO_SAY)
        self.publishers = {
            topic: self.node.create_publisher(String, topic, self.message_queue_depth)
            for topic in topics
        }

    def publish_heard_speech(self) -> None:
        """
        Listen from now on and publish the text of every utterance in which words were
        heard.

        :raises MicrophoneStopped: When the microphone stops delivering audio.
        """
        logger = self.node.get_logger()
        logger.info("Listening.")
        for transcript in self.recognizer.hear_utterances():
            if not transcript.text:
                continue
            logger.info(f"Heard: {transcript.text}")
            for publisher in self.publishers.values():
                publisher.publish(String(data=transcript.text))


# %% Command line
class CommandLineOption(StrEnum):
    """
    The options of ``nlp-listen``.
    """

    DEVICE = "--device"
    """
    Sets :attr:`ListenCommandLine.device`.
    """

    MODEL_SIZE = "--model-size"
    """
    Sets :attr:`ListenCommandLine.model_size`.
    """

    COMPUTE_TYPE = "--compute-type"
    """
    Sets :attr:`ListenCommandLine.compute_type`.
    """

    LIST_DEVICES = "--list-devices"
    """
    Prints the available input devices instead of listening.
    """

    LANGUAGE = "--language"
    """
    Sets :attr:`ListenCommandLine.language`.
    """

    ECHO = "--echo"
    """
    Sets :attr:`ListenCommandLine.echo_heard_speech`.
    """


@dataclass
class ListenCommandLine:
    """
    The settings of a speech listener started from the command line.
    """

    device: int | str | None = None
    """
    Index or name of the input device as ``python -m sounddevice`` lists it, ``None``
    for the system's default input.
    """

    model_size: WhisperModelSize = WhisperModelSize.SMALL
    """
    Which Whisper model transcribes the speech.
    """

    compute_type: WhisperComputeType = WhisperComputeType.INT8
    """
    Number format the Whisper model runs in.
    """

    language: WhisperLanguage = WhisperLanguage.ENGLISH
    """
    The language the Whisper model expects the speech in.
    """

    node_name: str = "nlp_listener"
    """
    Name of the ROS node that publishes the heard text.
    """

    echo_heard_speech: bool = False
    """
    Whether the robot's text-to-speech node also receives what was heard, and so says
    it back.
    """

    @classproperty
    def signal_exit_status_base(cls) -> int:
        """
        :return: The number shells add to a signal's number to report a process that
            the signal ended.
        """
        return 128

    @classmethod
    def run(cls) -> None:
        """
        Read the settings from the command line and publish what is heard until the
        microphone stops or the process is stopped by Ctrl+C or SIGTERM.

        Stopping releases the microphone and shuts down ROS.
        """
        settings = cls.from_arguments(sys.argv[1:])
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            signal.signal(stop_signal, cls.stop)
        with Context() as context:
            settings.publisher(context).publish_heard_speech()

    @classmethod
    def from_arguments(cls, arguments: list[str]) -> ListenCommandLine:
        """
        :param arguments: The command line after the program name.
        :return: The settings the options set, the defaults for the others.
        """
        defaults = cls()
        parser = argparse.ArgumentParser(
            description=f"Listen to a microphone and publish every heard sentence on "
            f"{SpeechTopic.HEARD_TEXT}."
        )
        parser.add_argument(
            CommandLineOption.DEVICE,
            default=defaults.device,
            help=f"index, or any unique part of the name, of the input device "
            f"{CommandLineOption.LIST_DEVICES} shows (default: the system's input)",
        )
        parser.add_argument(
            CommandLineOption.LIST_DEVICES,
            action="store_true",
            help="print the input devices, the default one starred, and exit",
        )
        parser.add_argument(
            CommandLineOption.MODEL_SIZE,
            type=WhisperModelSize,
            choices=list(WhisperModelSize),
            default=defaults.model_size,
            help=f"Whisper model (default: {defaults.model_size})",
        )
        parser.add_argument(
            CommandLineOption.COMPUTE_TYPE,
            type=WhisperComputeType,
            choices=list(WhisperComputeType),
            default=defaults.compute_type,
            help=f"number format of the model (default: {defaults.compute_type})",
        )
        parser.add_argument(
            CommandLineOption.LANGUAGE,
            type=WhisperLanguage,
            choices=list(WhisperLanguage),
            default=defaults.language,
            help=f"language of the speech (default: {defaults.language})",
        )
        parser.add_argument(
            CommandLineOption.ECHO,
            action="store_true",
            default=defaults.echo_heard_speech,
            help=f"also publish every heard sentence on {SpeechTopic.TEXT_TO_SAY}, so "
            f"the robot says it back",
        )
        options = parser.parse_args(arguments)
        if options.list_devices:
            print("\n".join(str(device) for device in InputDevice.list_available()))
            parser.exit()
        device = options.device
        return cls(
            device=int(device) if device is not None and device.isdigit() else device,
            model_size=options.model_size,
            compute_type=options.compute_type,
            language=options.language,
            echo_heard_speech=options.echo,
        )

    @classmethod
    def stop(cls, signal_number: int, frame: FrameType | None) -> None:
        """
        Exit with the status shells give a process the signal ended, unwinding the
        listener so it releases the microphone.

        :param signal_number: The signal that stops the listener.
        :param frame: The frame the signal interrupted.
        """
        sys.exit(cls.signal_exit_status_base + signal_number)

    def publisher(self, context: Context) -> TranscriptPublisher:
        """
        :param context: The initialized ROS context the node joins.
        :return: A publisher with these settings on a new ROS node, not listening yet.
        """
        return TranscriptPublisher(
            node=rclpy.create_node(self.node_name, context=context),
            recognizer=SpeechRecognizer(
                microphone=Microphone(device=self.device),
                transcriber=WhisperTranscriber(
                    model_size=self.model_size,
                    language=self.language,
                    compute_type=self.compute_type,
                ),
            ),
            echo_heard_speech=self.echo_heard_speech,
        )
