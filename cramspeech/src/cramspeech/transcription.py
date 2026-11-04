from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

from faster_whisper import WhisperModel
from krrood.ormatic.utils import classproperty

from cramspeech.microphone import Recording


# %% Transcript
@dataclass
class Transcript:
    """
    What a :class:`Transcriber` understood in a recording.
    """

    text: str
    """
    The recognized words, empty when nothing was said.
    """


# %% Transcriber
class Transcriber(ABC):
    """
    A speech recognition model that turns a recording into text.
    """

    @classproperty
    @abstractmethod
    def sample_rate(cls) -> int:
        """
        :return: Samples per second, in hertz, that the model expects its recordings at.
        """

    @abstractmethod
    def transcribe(self, recording: Recording) -> Transcript:
        """
        :param recording: Speech recorded at :attr:`sample_rate`.
        :return: The text spoken in the recording.
        """


# %% Whisper
class WhisperModelSize(StrEnum):
    """
    Whisper models as faster-whisper names them. Larger models are more accurate and slower.
    """

    TINY = "tiny"
    """
    Smallest multilingual model.
    """

    BASE = "base"
    """
    Second smallest multilingual model.
    """

    SMALL = "small"
    """
    Mid-sized multilingual model.
    """


@dataclass
class WhisperTranscriber(Transcriber):
    """
    Transcribes speech with faster-whisper.
    """

    model_size: WhisperModelSize = WhisperModelSize.SMALL
    """
    Which Whisper model to run.
    """

    language: str = "en"
    """
    The spoken language.
    """

    beam_size: int = 8
    """
    Number of candidate transcriptions kept during decoding.
    """

    @cached_property
    def model(self) -> WhisperModel:
        """
        The loaded Whisper model. faster-whisper downloads it from Hugging Face on first use.
        """
        return WhisperModel(self.model_size)

    @classproperty
    def sample_rate(cls) -> int:
        return 16000

    def transcribe(self, recording: Recording) -> Transcript:
        segments, _ = self.model.transcribe(
            recording.waveform, language=self.language, beam_size=self.beam_size
        )
        return Transcript(text="".join(segment.text for segment in segments))
