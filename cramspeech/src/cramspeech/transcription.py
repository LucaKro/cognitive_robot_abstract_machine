"""
Turning recorded speech into text with faster-whisper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

from faster_whisper import WhisperModel
from faster_whisper.feature_extractor import FeatureExtractor
from krrood.ormatic.utils import classproperty

from cramspeech.exceptions import SampleRateMismatch
from cramspeech.microphone import Recording


# %% Transcript
@dataclass
class Transcript:
    """
    What a :class:`WhisperTranscriber` understood in a recording.
    """

    text: str
    """
    The recognized words, empty when nothing was said.
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


class WhisperLanguage(StrEnum):
    """
    The languages a Whisper model can be told to expect.
    """

    ENGLISH = "en"
    """
    English speech.
    """

    GERMAN = "de"
    """
    German speech.
    """


class WhisperComputeType(StrEnum):
    """
    Number formats faster-whisper can run a Whisper model in. Smaller formats run faster
    and use less memory, at a small cost in accuracy.
    """

    INT8 = "int8"
    """
    8-bit integer weights, the fastest format on a CPU.
    """

    FLOAT16 = "float16"
    """
    16-bit floating point, the usual format on a GPU.
    """

    FLOAT32 = "float32"
    """
    32-bit floating point, the most accurate and slowest format.
    """


@dataclass
class WhisperTranscriber:
    """
    Transcribes speech with faster-whisper.
    """

    model_size: WhisperModelSize = WhisperModelSize.SMALL
    """
    Which Whisper model to run.
    """

    language: WhisperLanguage = WhisperLanguage.ENGLISH
    """
    The spoken language.
    """

    beam_size: int = 8
    """
    Number of candidate transcriptions kept during decoding.
    """

    compute_type: WhisperComputeType = WhisperComputeType.INT8
    """
    Number format the model runs in.
    """

    @cached_property
    def model(self) -> WhisperModel:
        """
        The loaded Whisper model. faster-whisper downloads it from Hugging Face on first use.
        """
        return WhisperModel(self.model_size, compute_type=self.compute_type)

    @classproperty
    def sample_rate(cls) -> int:
        """
        :return: Samples per second, in hertz, that Whisper expects its recordings at.
        """
        return FeatureExtractor().sampling_rate

    def transcribe(self, recording: Recording) -> Transcript:
        """
        :param recording: Speech recorded at :attr:`sample_rate`.
        :return: The text spoken in the recording, empty when nobody spoke.
        :raises SampleRateMismatch: When the recording is not at :attr:`sample_rate`.
        """
        if recording.sample_rate != self.sample_rate:
            raise SampleRateMismatch(
                recording_sample_rate=recording.sample_rate,
                model_sample_rate=self.sample_rate,
            )
        segments, _ = self.model.transcribe(
            recording.waveform,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
        )
        return Transcript(text=" ".join(segment.text.strip() for segment in segments))
