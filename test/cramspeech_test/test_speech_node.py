"""
What reaches the ROS topics, and how the options of ``nlp-listen`` become its settings.
"""

from __future__ import annotations

import signal
import time

import pytest
from cramspeech.exceptions import MicrophoneStopped
from cramspeech.microphone import InputDevice, Recording
from cramspeech.speech_node import (
    CommandLineOption,
    ListenCommandLine,
    SpeechTopic,
    TranscriptPublisher,
)
from cramspeech.speech_recognizer import SpeechDetector, SpeechRecognizer
from cramspeech.transcription import (
    WhisperComputeType,
    WhisperLanguage,
    WhisperModelSize,
    WhisperTranscriber,
)
from rclpy.node import Node
from std_msgs.msg import String
from typing_extensions import Callable

from .dataset.non_speech import SoundSequence
from .dataset.stand_ins import (
    AudioDevicesOfADesktop,
    ModelReturningSegments,
    ReplayingInputDevice,
)


# %% Fixtures
@pytest.fixture
def two_utterances(silence_then_speech_then_pause: SoundSequence) -> Recording:
    """
    Two tones that stand for speech, each followed by a pause.
    """
    _, speech, pause = silence_then_speech_then_pause.sounds
    return SoundSequence(
        sounds=[*silence_then_speech_then_pause.sounds, speech, pause]
    ).recording(sample_rate=SpeechDetector.sample_rate)


@pytest.fixture
def received_messages(rclpy_node: Node) -> dict[SpeechTopic, list[str]]:
    """
    The text of every message published on each :class:`SpeechTopic` during the test, in
    arrival order.
    """
    received = {topic: [] for topic in SpeechTopic}
    for topic in SpeechTopic:
        rclpy_node.create_subscription(
            String,
            topic,
            lambda message, topic=topic: received[topic].append(message.data),
            10,
        )
    return received


def publisher_hearing(
    node: Node, recognizer: SpeechRecognizer, echo_heard_speech: bool = False
) -> TranscriptPublisher:
    """
    :param node: The ROS node the transcript publisher belongs to.
    :param recognizer: Hears the utterances for the transcript publisher.
    :param echo_heard_speech: Whether the transcript publisher also sends what it hears
        to the text-to-speech node.
    :return: The transcript publisher, once each of its topics has a subscriber.
    """
    publisher = TranscriptPublisher(
        node=node, recognizer=recognizer, echo_heard_speech=echo_heard_speech
    )
    wait_until(
        lambda: all(
            topic_publisher.get_subscription_count() > 0
            for topic_publisher in publisher.publishers.values()
        )
    )
    return publisher


def wait_until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    """
    Wait for messages to be delivered by the node's executor thread.

    :param condition: Holds once the expected messages arrived.
    :param timeout: Seconds after which the test goes on regardless.
    """
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)


# %% Publishing
def test_transcripts_go_only_to_the_heard_text_topic_by_default(
    rclpy_node: Node,
    recognizer_hearing_every_sound: SpeechRecognizer,
    model_hearing_a_request: ModelReturningSegments,
    two_utterances: Recording,
    received_messages: dict[SpeechTopic, list[str]],
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    Without echoing, the robot's text-to-speech node is not told what was heard.
    """
    publisher = publisher_hearing(rclpy_node, recognizer_hearing_every_sound)
    replay_through_the_microphone(two_utterances)

    with pytest.raises(MicrophoneStopped):
        publisher.publish_heard_speech()
    expected = {
        SpeechTopic.HEARD_TEXT: 2 * model_hearing_a_request.texts,
        SpeechTopic.TEXT_TO_SAY: [],
    }
    wait_until(lambda: received_messages == expected)

    assert received_messages == expected


def test_every_transcript_is_published_on_both_topics_when_echoing(
    rclpy_node: Node,
    recognizer_hearing_every_sound: SpeechRecognizer,
    model_hearing_a_request: ModelReturningSegments,
    two_utterances: Recording,
    received_messages: dict[SpeechTopic, list[str]],
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    Each utterance reaches every topic as its text, in spoken order.
    """
    publisher = publisher_hearing(
        rclpy_node, recognizer_hearing_every_sound, echo_heard_speech=True
    )
    replay_through_the_microphone(two_utterances)

    with pytest.raises(MicrophoneStopped):
        publisher.publish_heard_speech()
    expected = {topic: 2 * model_hearing_a_request.texts for topic in SpeechTopic}
    wait_until(lambda: received_messages == expected)

    assert received_messages == expected


def test_empty_transcripts_are_not_published(
    rclpy_node: Node,
    recognizer_hearing_every_sound: SpeechRecognizer,
    two_utterances: Recording,
    received_messages: dict[SpeechTopic, list[str]],
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    An utterance Whisper finds no words in, such as a cough, is not sent anywhere.
    """
    recognizer_hearing_every_sound.transcriber.model = ModelReturningSegments(texts=[])
    publisher = publisher_hearing(
        rclpy_node, recognizer_hearing_every_sound, echo_heard_speech=True
    )
    replay_through_the_microphone(two_utterances)

    with pytest.raises(MicrophoneStopped):
        publisher.publish_heard_speech()
    wait_until(lambda: any(received_messages.values()), timeout=0.5)

    assert received_messages == {topic: [] for topic in SpeechTopic}


# %% Command line
def test_command_line_without_options_keeps_the_defaults() -> None:
    """
    An option that is not given leaves its setting at the default of
    :class:`ListenCommandLine`.
    """
    assert ListenCommandLine.from_arguments([]) == ListenCommandLine()


def test_command_line_does_not_echo_by_default() -> None:
    """
    Started without the option, the robot does not say back what it heard.
    """
    assert ListenCommandLine.from_arguments([]).echo_heard_speech is False


def test_command_line_defaults_match_the_transcriber() -> None:
    """
    Started without options, the listener transcribes as a default
    :class:`WhisperTranscriber` does.
    """
    settings = ListenCommandLine.from_arguments([])
    transcriber = WhisperTranscriber()

    assert settings.model_size is transcriber.model_size
    assert settings.compute_type is transcriber.compute_type
    assert settings.language is transcriber.language


def test_command_line_reads_a_device_index_as_a_number() -> None:
    """
    Sounddevice takes a number as a device index and text as part of a device name.
    """
    settings = ListenCommandLine.from_arguments([CommandLineOption.DEVICE, "6"])

    assert settings.device == 6


def test_command_line_keeps_a_device_name_as_text() -> None:
    """
    A device given by name reaches sounddevice unchanged.
    """
    settings = ListenCommandLine.from_arguments(
        [CommandLineOption.DEVICE, "HyperX Cloud"]
    )

    assert settings.device == "HyperX Cloud"


def test_command_line_reads_the_whisper_settings() -> None:
    """
    The model size, the number format and the language arrive as plain text and are read
    as their enum members.
    """
    settings = ListenCommandLine.from_arguments(
        [
            CommandLineOption.MODEL_SIZE.value,
            WhisperModelSize.BASE.value,
            CommandLineOption.COMPUTE_TYPE.value,
            WhisperComputeType.FLOAT32.value,
            CommandLineOption.LANGUAGE.value,
            WhisperLanguage.GERMAN.value,
        ]
    )

    assert settings.model_size is WhisperModelSize.BASE
    assert settings.compute_type is WhisperComputeType.FLOAT32
    assert settings.language is WhisperLanguage.GERMAN


def test_command_line_reads_the_echo_option() -> None:
    """
    With the option, the robot says back what it heard.
    """
    settings = ListenCommandLine.from_arguments([CommandLineOption.ECHO.value])

    assert settings.echo_heard_speech is True


def test_command_line_lists_the_input_devices_and_exits(
    audio_devices_of_a_desktop: AudioDevicesOfADesktop,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Listing the devices prints one line per input device and starts no listener.
    """
    with pytest.raises(SystemExit) as stopped:
        ListenCommandLine.from_arguments([CommandLineOption.LIST_DEVICES.value])

    printed_lines = capsys.readouterr().out.splitlines()
    assert stopped.value.code == 0
    assert printed_lines == [str(device) for device in InputDevice.list_available()]


def test_publisher_hears_with_the_command_line_settings(rclpy_node: Node) -> None:
    """
    The node, the microphone, the transcriber and the echoing follow the settings.
    """
    settings = ListenCommandLine(
        device=6,
        model_size=WhisperModelSize.BASE,
        compute_type=WhisperComputeType.FLOAT32,
        language=WhisperLanguage.GERMAN,
        echo_heard_speech=True,
    )

    publisher = settings.publisher(rclpy_node.context)

    transcriber = publisher.recognizer.transcriber
    assert publisher.node.get_name() == settings.node_name
    assert publisher.echo_heard_speech is settings.echo_heard_speech
    assert publisher.recognizer.microphone.device == settings.device
    assert transcriber.model_size is settings.model_size
    assert transcriber.compute_type is settings.compute_type
    assert transcriber.language is settings.language
    publisher.node.destroy_node()


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_stopping_exits_with_the_status_of_the_signal(
    stop_signal: signal.Signals,
) -> None:
    """
    A listener stopped by Ctrl+C or by its launcher exits with the status shells give a
    process ended by that signal.
    """
    with pytest.raises(SystemExit) as stopped:
        ListenCommandLine.stop(stop_signal, None)

    assert stopped.value.code == ListenCommandLine.signal_exit_status_base + stop_signal
