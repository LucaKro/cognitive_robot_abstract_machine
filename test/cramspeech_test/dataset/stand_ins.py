"""
Stand-ins for the microphone driver and the Whisper model, so tests run without audio
hardware or a model download.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

import numpy as np
from faster_whisper.transcribe import Segment
from typing_extensions import Any, Iterator


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
