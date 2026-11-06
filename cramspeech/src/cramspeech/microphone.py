"""
Audio input: finding the input devices, and recording or streaming mono audio from one
of them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from queue import Queue

import numpy as np
import sounddevice
from typing_extensions import Any, Iterator

from cramspeech.exceptions import MicrophoneStopped


# %% Recording
@dataclass
class Recording:
    """
    Mono audio as a sequence of samples taken at a fixed rate.
    """

    waveform: np.ndarray
    """
    One float32 sample in [-1, 1] per frame, in recording order.
    """

    sample_rate: int
    """
    Samples per second, in hertz.
    """


# %% Input devices
@dataclass
class InputDevice:
    """
    An audio device that can record, as sounddevice lists it.
    """

    index: int
    """
    Position of the device in sounddevice's list, which can change when devices are
    plugged in or out.
    """

    name: str
    """
    The device's name.

    Any part of it that no other device's name contains selects it.
    """

    is_default: bool
    """
    Whether the device is the system's default input.
    """

    @classmethod
    def list_available(cls) -> list[InputDevice]:
        """
        :return: Every device with at least one input channel, in sounddevice's order.
        """
        default_index = sounddevice.default.device[0]
        return [
            cls(
                index=device["index"],
                name=device["name"],
                is_default=device["index"] == default_index,
            )
            for device in sounddevice.query_devices()
            if device["max_input_channels"] > 0
        ]

    def __str__(self) -> str:
        """
        :return: The index and the name, after a star for the default input.
        """
        marker = "*" if self.is_default else " "
        return f"{marker} {self.index:>3}  {self.name}"


# %% Microphone
@dataclass
class Microphone:
    """
    An audio input device that records mono audio on request.
    """

    device: int | str | None = None
    """
    Index or name of the input device as ``python -m sounddevice`` lists it.

    ``None`` selects the system's default input.
    """

    def record(self, duration: float, sample_rate: int) -> Recording:
        """
        Record from the device, returning once the recording is complete.

        :param duration: Length of the recording, in seconds.
        :param sample_rate: Samples per second to record, in hertz.
        :return: The recorded audio.
        """
        frames = sounddevice.rec(
            frames=round(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            device=self.device,
        )
        sounddevice.wait()
        return Recording(waveform=frames[:, 0], sample_rate=sample_rate)

    def stream(self, sample_rate: int, block_duration: float) -> Iterator[np.ndarray]:
        """
        Listen to the device until the caller stops iterating, which releases it.

        Audio that arrives while the caller is busy is kept until it is read. A block
        after audio the device lost is logged as a warning.

        :param sample_rate: Samples per second to record, in hertz.
        :param block_duration: Length of each block, in seconds.
        :return: Consecutive blocks of mono float32 samples, in recording order.
        :raises MicrophoneStopped: When the device stops delivering audio.
        """
        block_length = round(block_duration * sample_rate)
        arrived_blocks: Queue[np.ndarray | None] = Queue()

        def keep_block(
            samples: np.ndarray,
            frame_count: int,
            time: Any,
            status: sounddevice.CallbackFlags,
        ) -> None:
            """
            Queue a copy of the block that just arrived, warning first when audio was
            lost before it.
            """
            if status.input_overflow:
                logging.getLogger(__name__).warning(
                    f"The input device {self.device!r} lost audio before this block."
                )
            arrived_blocks.put(samples[:, 0].copy())

        with sounddevice.InputStream(
            samplerate=sample_rate,
            blocksize=block_length,
            channels=1,
            dtype=np.float32,
            device=self.device,
            callback=keep_block,
            finished_callback=lambda: arrived_blocks.put(None),
        ):
            while (block := arrived_blocks.get()) is not None:
                yield block
        raise MicrophoneStopped(device=self.device)
