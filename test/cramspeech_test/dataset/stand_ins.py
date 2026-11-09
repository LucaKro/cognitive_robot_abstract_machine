"""
Stand-ins for the microphone driver, the Whisper model and the Silero model, so tests
run without audio hardware or a model download.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

import numpy as np
import sounddevice
from cramspeech.microphone import Recording
from cramspeech.speech_recognizer import SpeechDetector
from faster_whisper.transcribe import Segment
from typing_extensions import Any, Callable, Iterator, Self


# %% Microphone driver
@dataclass
class SilentInputDevice:
    """
    Takes the place of :func:`sounddevice.rec` and :func:`sounddevice.wait`: every
    recording is silence of the requested shape, and every request is kept for
    inspection.
    """

    requests: list[dict[str, Any]] = field(default_factory=list)
    """
    The keyword arguments of each :meth:`rec` call, in call order.
    """

    def rec(self, **arguments: Any) -> np.ndarray:
        """
        :return: Zeros with one row per requested frame and one column per channel.
        """
        self.requests.append(arguments)
        return np.zeros(
            (arguments["frames"], arguments["channels"]), dtype=arguments["dtype"]
        )

    def wait(self) -> None:
        """
        Returns at once, since :meth:`rec` finishes its recording immediately.
        """


@dataclass
class ReplayingInputDevice:
    """
    Takes the place of :class:`sounddevice.InputStream`: as soon as the stream opens,
    plays a whole recording into its callback, as if the caller had been busy all along,
    then reports the stream finished.
    """

    recording: Recording
    """
    The audio the stream delivers, from its first sample on.
    """

    requests: list[dict[str, Any]] = field(default_factory=list)
    """
    The keyword arguments of each opened stream, its callbacks left out, in opening
    order.
    """

    is_open: bool = False
    """
    Whether a stream is currently open.
    """

    callback: (
        Callable[[np.ndarray, int, Any, sounddevice.CallbackFlags], None] | None
    ) = None
    """
    The function the latest stream hands each block of samples to.
    """

    finished_callback: Callable[[], None] | None = None
    """
    The function the latest stream calls once it delivers no more audio.
    """

    drops_audio: bool = False
    """
    Whether every block reports that audio before it was lost, as when the device's
    buffer overflowed.
    """

    def input_stream(
        self,
        callback: Callable[[np.ndarray, int, Any, sounddevice.CallbackFlags], None],
        finished_callback: Callable[[], None],
        **arguments: Any,
    ) -> ReplayingInputDevice:
        """
        :return: This device as the stream, keeping its callbacks and the keyword
            arguments it was opened with.
        """
        self.requests.append(arguments)
        self.callback = callback
        self.finished_callback = finished_callback
        return self

    def __enter__(self) -> Self:
        """
        Opens the stream and delivers the recording in blocks of the requested size,
        silence filling up the last one.

        Like PortAudio, every block reuses one buffer.
        """
        self.is_open = True
        block_length = self.requests[-1]["blocksize"]
        buffer = np.zeros((block_length, 1), dtype=np.float32)
        status = sounddevice.CallbackFlags()
        status.input_overflow = self.drops_audio
        waveform = self.recording.waveform
        for start in range(0, len(waveform), block_length):
            block = waveform[start : start + block_length]
            buffer[:, 0] = np.pad(block, (0, block_length - len(block)))
            self.callback(buffer, block_length, None, status)
        self.finished_callback()
        return self

    def __exit__(self, *exception_information: object) -> None:
        """
        Closes the stream, also when the caller raised.
        """
        self.is_open = False


@dataclass
class AudioDevicesOfADesktop:
    """
    Takes the place of :func:`sounddevice.query_devices`: a desktop's devices, of which
    some only play, in the shape sounddevice lists them.
    """

    devices: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"index": 0, "name": "HDMI output", "max_input_channels": 0},
            {"index": 1, "name": "USB headset", "max_input_channels": 1},
            {"index": 2, "name": "Built-in microphone", "max_input_channels": 2},
        ]
    )
    """
    Every device with its index, name and number of channels it records.
    """

    default_input_index: int = 2
    """
    The index of the system's default input.
    """

    def query_devices(self) -> list[dict[str, Any]]:
        """
        :return: :attr:`devices`.
        """
        return self.devices


@dataclass
class DefaultDevicesOfADesktop:
    """
    Takes the place of :data:`sounddevice.default`, holding only the default devices.
    """

    device: tuple[int, int]
    """
    The indices of the default input and output device.
    """


# %% Whisper model
@dataclass
class ModelReturningSegments:
    """
    Takes the place of :class:`faster_whisper.WhisperModel`: hears the same texts in
    every recording and keeps the arguments of each transcription.
    """

    texts: list[str]
    """
    The text of each segment, in spoken order.
    """

    calls: list[tuple[np.ndarray, dict[str, Any]]] = field(default_factory=list)
    """
    The audio and keyword arguments of each :meth:`transcribe` call, in call order.
    """

    def transcribe(
        self, audio: np.ndarray, **settings: Any
    ) -> tuple[Iterator[Segment], None]:
        """
        :return: One :class:`~faster_whisper.transcribe.Segment` per text, lazily as Whisper
            yields them, and no transcription info.
        """
        self.calls.append((audio, settings))
        segments = (
            Segment(
                id=index,
                seek=0,
                start=0.0,
                end=0.0,
                text=text,
                tokens=[],
                avg_logprob=0.0,
                compression_ratio=0.0,
                no_speech_prob=0.0,
                words=None,
                temperature=None,
            )
            for index, text in enumerate(self.texts)
        )
        return segments, None


# %% Silero model
@dataclass
class ModelHearingSpeechInEverySound:
    """
    Takes the place of :class:`faster_whisper.vad.SileroVADModel`: every frame that
    holds any sound scores as certain speech, and every silent frame as certainly none.
    """

    scored_sample_counts: list[int] = field(default_factory=list)
    """
    The number of samples of each scored audio, in call order.
    """

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        :param audio: Whole frames of :attr:`SpeechDetector.frame_length` samples.
        :return: One score per frame, 1 for sound and 0 for silence.
        """
        self.scored_sample_counts.append(len(audio))
        frames = audio.reshape(-1, SpeechDetector.frame_length)
        return np.any(frames != 0, axis=1).astype(np.float32)
