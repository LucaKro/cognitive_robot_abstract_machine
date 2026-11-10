"""
Turning the captured audio into text, detect where somebody spoke and paused, and
transcribing each utterance.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from functools import cached_property
from math import ceil

import numpy as np
from faster_whisper.vad import SileroVADModel, get_vad_model
from krrood.ormatic.utils import classproperty
from typing_extensions import Iterator

from cramspeech.exceptions import NothingHeard, SampleRateMismatch
from cramspeech.microphone import Microphone, Recording
from cramspeech.transcription import Transcript, WhisperTranscriber


# %% Speech detection
@dataclass
class UtteranceSearch:
    """
    What a :class:`SpeechDetector` found when it searched audio for an utterance.
    """

    utterance: Recording | None
    """
    The first finished utterance with its lead-in, ``None`` while nobody spoke or the
    speaker is still talking.
    """

    holds_speech: bool
    """
    Whether the searched audio holds any speech.
    """

    first_sample_to_keep: int
    """
    Index of the first sample of the searched audio that later searches still need.
    """


@dataclass
class SpeechDetector:
    """
    Finds where somebody spoke and then paused, using the Silero voice activity model
    that ships with faster-whisper.
    """

    speech_threshold: float = 0.5
    """
    Score from which the model counts a frame as speech, between 0 and 1.
    """

    pause_duration: float = 1.0
    """
    Seconds without speech after which an utterance is finished.
    """

    kept_before_speech: float = 0.4
    """
    Seconds of audio kept before the first speech frame, so the first word is not cut
    off.
    """

    maximum_duration: float = 30.0
    """
    Longest utterance, in seconds.

    Speech that goes on longer is cut off there.
    """

    @cached_property
    def model(self) -> SileroVADModel:
        """
        The Silero model that scores how likely each frame holds speech.
        """
        return get_vad_model()

    @classproperty
    def sample_rate(cls) -> int:
        """
        :return: Samples per second, in hertz, that Silero expects its recordings at.
        """
        return 16000

    @classproperty
    def frame_length(cls) -> int:
        """
        :return: Number of samples the model scores at once.
        """
        return 512

    def search_for_utterance(self, recording: Recording) -> UtteranceSearch:
        """
        :param recording: Audio at :attr:`sample_rate`, oldest sample first.
        :return: The first utterance that ended in a pause of :attr:`pause_duration` or
            reached :attr:`maximum_duration`, with :attr:`kept_before_speech` of audio
            before it, and the part of the audio later searches still need.
        :raises SampleRateMismatch: When the recording is not at :attr:`sample_rate`.
        """
        speech_frames = self.speech_frames_in(recording)
        frame_count = len(recording.waveform) // self.frame_length
        lead_in = round(self.kept_before_speech * self.sample_rate)
        if speech_frames.size == 0:
            frames_before_lead_in = (len(recording.waveform) - lead_in) // (
                self.frame_length
            )
            return UtteranceSearch(
                utterance=None,
                holds_speech=False,
                first_sample_to_keep=max(0, frames_before_lead_in * self.frame_length),
            )

        start = max(0, int(speech_frames[0]) * self.frame_length - lead_in)
        maximum_end = start + round(self.maximum_duration * self.sample_rate)
        pause_start = self.start_of_first_pause(speech_frames, frame_count)
        if pause_start is None and len(recording.waveform) < maximum_end:
            return UtteranceSearch(
                utterance=None, holds_speech=True, first_sample_to_keep=0
            )
        end = maximum_end if pause_start is None else min(pause_start, maximum_end)
        utterance = Recording(
            waveform=recording.waveform[start:end], sample_rate=recording.sample_rate
        )
        return UtteranceSearch(
            utterance=utterance, holds_speech=True, first_sample_to_keep=end
        )

    def speech_frames_in(self, recording: Recording) -> np.ndarray:
        """
        :param recording: Audio at :attr:`sample_rate`, oldest sample first.
        :return: Indices of the whole frames that hold speech, in ascending order.
        :raises SampleRateMismatch: When the recording is not at :attr:`sample_rate`.
        """
        if recording.sample_rate != self.sample_rate:
            raise SampleRateMismatch(
                recording_sample_rate=recording.sample_rate,
                model_sample_rate=self.sample_rate,
            )
        frame_count = len(recording.waveform) // self.frame_length
        if frame_count == 0:
            return np.zeros(0, dtype=np.int64)
        scores = self.model(recording.waveform[: frame_count * self.frame_length])
        return np.flatnonzero(scores >= self.speech_threshold)

    def start_of_first_pause(
        self, speech_frames: np.ndarray, frame_count: int
    ) -> int | None:
        """
        :param speech_frames: Indices of the frames that hold speech, in ascending order.
        :param frame_count: Number of frames scored.
        :return: The sample at which the first silence of at least
            :attr:`pause_duration` begins, ``None`` if the speaker has not paused that
            long yet.
        """
        pause_frame_count = ceil(
            self.pause_duration * self.sample_rate / self.frame_length
        )
        frames_to_next_speech = np.diff(speech_frames, append=frame_count)
        paused_after = np.flatnonzero(frames_to_next_speech > pause_frame_count)
        if paused_after.size == 0:
            return None
        return (int(speech_frames[paused_after[0]]) + 1) * self.frame_length


# %% Speech recognizer
@dataclass
class SpeechRecognizer:
    """
    Hears speech from a microphone and transcribes it, either for a fixed time or one
    utterance at a time.
    """

    microphone: Microphone
    """
    The device the speech is recorded from.
    """

    transcriber: WhisperTranscriber
    """
    The speech-to-text model that turns the recording into text.
    """

    detector: SpeechDetector = field(default_factory=SpeechDetector)
    """
    Finds where an utterance in the streamed audio begins and ends.
    """

    block_duration: float = 0.25
    """
    Seconds of audio streamed from the microphone between two searches for a finished
    utterance.
    """

    def hear(self, duration: float) -> Transcript:
        """
        Record for a fixed time, starting now, and return what was said.

        :param duration: How long to record, in seconds.
        :return: The text spoken during the recording.
        """
        recording = self.microphone.record(
            duration=duration, sample_rate=self.transcriber.sample_rate
        )
        return self.transcriber.transcribe(recording)

    def hear_next_utterance(self, timeout: float) -> Transcript:
        """
        Listen until somebody has spoken and paused, and return what was said.

        :param timeout: Seconds to wait for speech to start. Speech that has started is
            heard to its end.
        :return: The text of the first utterance.
        :raises NothingHeard: When no speech starts within *timeout*.
        :raises MicrophoneStopped: When the microphone stops delivering audio.
        """
        timeout_block_count = ceil(timeout / self.block_duration)
        with closing(self._listen()) as searches:
            for block_count, search in enumerate(searches, start=1):
                if search.utterance is not None:
                    return self.transcriber.transcribe(search.utterance)
                if block_count >= timeout_block_count and not search.holds_speech:
                    raise NothingHeard(timeout=timeout)

    def hear_utterances(self) -> Iterator[Transcript]:
        """
        Listen until the caller stops iterating, which releases the microphone.

        :return: The text of every utterance, in spoken order, each as soon as its pause
            has been heard.
        :raises MicrophoneStopped: When the microphone stops delivering audio.
        """
        with closing(self._listen()) as searches:
            for search in searches:
                if search.utterance is not None:
                    yield self.transcriber.transcribe(search.utterance)

    def _listen(self) -> Iterator[UtteranceSearch]:
        """
        Listen to the microphone until the caller stops iterating, which releases it.

        :return: After each block of :attr:`block_duration`, the search of the audio
            heard so far, without the utterances already found.
        :raises MicrophoneStopped: When the microphone stops delivering audio.
        """
        sample_rate = self.transcriber.sample_rate
        heard = Recording(
            waveform=np.zeros(0, dtype=np.float32), sample_rate=sample_rate
        )
        with closing(
            self.microphone.stream(
                sample_rate=sample_rate, block_duration=self.block_duration
            )
        ) as blocks:
            for block in blocks:
                heard.waveform = np.concatenate((heard.waveform, block))
                search = self.detector.search_for_utterance(heard)
                heard.waveform = heard.waveform[search.first_sample_to_keep :]
                yield search
