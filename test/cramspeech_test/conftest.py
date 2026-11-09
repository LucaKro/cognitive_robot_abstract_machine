"""
Fixtures that replace the microphone driver and the models, and the recordings tests
play through them.
"""

from __future__ import annotations

import pytest
import sounddevice
from cramspeech.microphone import Microphone, Recording
from cramspeech.speech_recognizer import SpeechDetector, SpeechRecognizer
from cramspeech.transcription import WhisperTranscriber
from typing_extensions import Callable

from .dataset.non_speech import NonSpeechRecording, NonSpeechSound, SoundSequence
from .dataset.stand_ins import (
    AudioDevicesOfADesktop,
    DefaultDevicesOfADesktop,
    ModelHearingSpeechInEverySound,
    ModelReturningSegments,
    ReplayingInputDevice,
    SilentInputDevice,
)


# %% Microphone driver
@pytest.fixture
def silent_input_device(monkeypatch: pytest.MonkeyPatch) -> SilentInputDevice:
    """
    Replaces the recording calls of :mod:`sounddevice` with silence for the duration of
    a test.
    """
    device = SilentInputDevice()
    monkeypatch.setattr(sounddevice, "rec", device.rec)
    monkeypatch.setattr(sounddevice, "wait", device.wait)
    return device


@pytest.fixture
def replay_through_the_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Recording], ReplayingInputDevice]:
    """
    Makes every :class:`sounddevice.InputStream` opened during a test play the given
    recording.
    """

    def replay(recording: Recording) -> ReplayingInputDevice:
        """
        :param recording: The audio the microphone delivers from now on.
        :return: The device that plays it, for inspecting its streams.
        """
        device = ReplayingInputDevice(recording=recording)
        monkeypatch.setattr(sounddevice, "InputStream", device.input_stream)
        return device

    return replay


@pytest.fixture
def audio_devices_of_a_desktop(
    monkeypatch: pytest.MonkeyPatch,
) -> AudioDevicesOfADesktop:
    """
    Makes :mod:`sounddevice` report a desktop's audio devices for the duration of a
    test.
    """
    devices = AudioDevicesOfADesktop()
    monkeypatch.setattr(sounddevice, "query_devices", devices.query_devices)
    monkeypatch.setattr(
        sounddevice,
        "default",
        DefaultDevicesOfADesktop(device=(devices.default_input_index, 0)),
    )
    return devices


# %% Recordings
@pytest.fixture
def one_second_of_silence() -> Recording:
    """
    One second of silence at the sample rate Whisper takes.
    """
    return NonSpeechRecording(sound=NonSpeechSound.SILENCE, duration=1.0).recording(
        sample_rate=WhisperTranscriber.sample_rate
    )


@pytest.fixture(params=list(NonSpeechSound))
def non_speech_recording(request: pytest.FixtureRequest) -> Recording:
    """
    Each :class:`NonSpeechSound` in turn, at the sample rate Whisper takes.
    """
    return NonSpeechRecording(sound=request.param).recording(
        sample_rate=WhisperTranscriber.sample_rate
    )


@pytest.fixture
def silence_then_speech_then_pause(frame_duration: float) -> SoundSequence:
    """
    Silence, a tone that stands for speech, then silence longer than a pause.
    """
    silence = NonSpeechRecording(
        sound=NonSpeechSound.SILENCE, duration=40 * frame_duration
    )
    speech = NonSpeechRecording(sound=NonSpeechSound.TONE, duration=30 * frame_duration)
    return SoundSequence(sounds=[silence, speech, silence])


# %% Whisper model
@pytest.fixture
def model_hearing_a_request() -> ModelReturningSegments:
    """
    A Whisper stand-in that hears one spoken request in every recording.
    """
    return ModelReturningSegments(texts=["Bring me a cup."])


@pytest.fixture
def transcriber_hearing() -> Callable[[ModelReturningSegments], WhisperTranscriber]:
    """
    Builds transcribers with default settings that hear what a Whisper stand-in hears.
    """

    def hearing(model: ModelReturningSegments) -> WhisperTranscriber:
        """
        :param model: Takes the place of the Whisper model.
        :return: A transcriber that runs *model*.
        """
        transcriber = WhisperTranscriber()
        transcriber.model = model
        return transcriber

    return hearing


# %% Speech detection
@pytest.fixture
def frame_duration() -> float:
    """
    Seconds of audio in one frame the speech detector scores.
    """
    return SpeechDetector.frame_length / SpeechDetector.sample_rate


@pytest.fixture
def detector_hearing_every_sound() -> SpeechDetector:
    """
    A speech detector that takes every sound for speech and silence for a pause.
    """
    detector = SpeechDetector()
    detector.model = ModelHearingSpeechInEverySound()
    return detector


# %% Speech recognizer
@pytest.fixture
def recognizer_hearing_every_sound(
    detector_hearing_every_sound: SpeechDetector,
    model_hearing_a_request: ModelReturningSegments,
    transcriber_hearing: Callable[[ModelReturningSegments], WhisperTranscriber],
) -> SpeechRecognizer:
    """
    A recognizer on the default microphone whose detector takes every sound for speech
    and whose transcriber hears :func:`model_hearing_a_request`'s text.
    """
    return SpeechRecognizer(
        microphone=Microphone(),
        transcriber=transcriber_hearing(model_hearing_a_request),
        detector=detector_hearing_every_sound,
    )
