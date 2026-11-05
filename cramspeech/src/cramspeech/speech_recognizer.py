from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import ceil

from faster_whisper.vad import SileroVADModel, get_vad_model
from krrood.ormatic.utils import classproperty

from cramspeech.exceptions import SampleRateMismatch
import numpy as np
from cramspeech.microphone import Microphone, Recording
from cramspeech.transcription import Transcript, WhisperTranscriber


# %% Speech recognizer
@dataclass
class SpeechRecognizer:
    """
    Records from a microphone on request and transcribes the recording.
    """

    microphone: Microphone
    """
    The device the speech is recorded from.
    """

    transcriber: WhisperTranscriber
    """
    The ASR-model that turns the recording into text.
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


# %% Speech detection
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

    def utterance_in(self, recording: Recording) -> Recording | None:
        """
        :param recording: Audio at :attr:`sample_rate`, oldest sample first.
        :return: The first utterance that ended in a pause of :attr:`pause_duration` or
            reached :attr:`maximum_duration`, with :attr:`kept_before_speech` of audio
            before it. ``None`` while nobody spoke or the speaker is still talking.
        """
        if recording.sample_rate != self.sample_rate:
            raise SampleRateMismatch(
                recording_sample_rate=recording.sample_rate,
                model_sample_rate=self.sample_rate,
            )
        frame_count = len(recording.waveform) // self.frame_length
        if frame_count == 0:
            return None
        scores = self.model(recording.waveform[: frame_count * self.frame_length])
        speech_frames = np.flatnonzero(scores >= self.speech_threshold)
        if speech_frames.size == 0:
            return None

        lead_in = round(self.kept_before_speech * self.sample_rate)
        start = max(0, int(speech_frames[0]) * self.frame_length - lead_in)
        maximum_end = start + round(self.maximum_duration * self.sample_rate)
        pause_start = self.start_of_first_pause(speech_frames, frame_count)
        if pause_start is None and len(recording.waveform) < maximum_end:
            return None
        end = maximum_end if pause_start is None else min(pause_start, maximum_end)
        return Recording(
            waveform=recording.waveform[start:end], sample_rate=recording.sample_rate
        )

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
