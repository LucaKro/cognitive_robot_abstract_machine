"""
Hearing for a fixed time, until the next utterance and continuously, with the microphone
replaying a recording.
"""

from __future__ import annotations

from itertools import islice

import numpy as np
import pytest
from cramspeech.exceptions import NothingHeard
from cramspeech.microphone import Microphone, Recording
from cramspeech.speech_recognizer import SpeechDetector, SpeechRecognizer
from cramspeech.transcription import Transcript, WhisperTranscriber
from typing_extensions import Callable

from .dataset.non_speech import NonSpeechRecording, NonSpeechSound, SoundSequence
from .dataset.stand_ins import (
    ModelHearingSpeechInEverySound,
    ModelReturningSegments,
    ReplayingInputDevice,
    SilentInputDevice,
)


# %% Hearing for a fixed time
def test_hear_returns_the_transcribed_text(
    silent_input_device: SilentInputDevice,
    model_hearing_a_request: ModelReturningSegments,
    transcriber_hearing: Callable[[ModelReturningSegments], WhisperTranscriber],
) -> None:
    """
    One call records from the microphone and hands the recording to the transcriber.
    """
    recognizer = SpeechRecognizer(
        microphone=Microphone(),
        transcriber=transcriber_hearing(model_hearing_a_request),
    )

    assert recognizer.hear(duration=1.0) == Transcript(
        text=model_hearing_a_request.texts[0]
    )


# %% Hearing the next utterance
def test_hear_next_utterance_returns_the_transcribed_speech(
    recognizer_hearing_every_sound: SpeechRecognizer,
    model_hearing_a_request: ModelReturningSegments,
    silence_then_speech_then_pause: SoundSequence,
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    The utterance the detector finds in the streamed audio is what gets transcribed.
    """
    sample_rate = SpeechDetector.sample_rate
    recording = silence_then_speech_then_pause.recording(sample_rate=sample_rate)
    replay_through_the_microphone(recording)

    transcript = recognizer_hearing_every_sound.hear_next_utterance(timeout=5.0)

    [(transcribed_audio, _)] = model_hearing_a_request.calls
    expected_utterance = recognizer_hearing_every_sound.detector.search_for_utterance(
        recording
    ).utterance
    assert transcript == Transcript(text=model_hearing_a_request.texts[0])
    assert np.array_equal(transcribed_audio, expected_utterance.waveform)


def test_hear_next_utterance_stops_listening_once_it_heard_an_utterance(
    recognizer_hearing_every_sound: SpeechRecognizer,
    silence_then_speech_then_pause: SoundSequence,
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    The microphone stream is closed when the utterance is returned.
    """
    device = replay_through_the_microphone(
        silence_then_speech_then_pause.recording(sample_rate=SpeechDetector.sample_rate)
    )

    recognizer_hearing_every_sound.hear_next_utterance(timeout=5.0)

    assert len(device.requests) == 1
    assert not device.is_open


def test_hear_next_utterance_raises_when_nobody_speaks(
    recognizer_hearing_every_sound: SpeechRecognizer,
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    Silence until the timeout means nothing was heard, and the stream is closed.
    """
    device = replay_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.SILENCE).recording(
            sample_rate=SpeechDetector.sample_rate
        )
    )

    with pytest.raises(NothingHeard):
        recognizer_hearing_every_sound.hear_next_utterance(timeout=1.0)
    assert not device.is_open


def test_hear_next_utterance_listens_past_the_timeout_once_speech_started(
    recognizer_hearing_every_sound: SpeechRecognizer,
    model_hearing_a_request: ModelReturningSegments,
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    The timeout bounds when the speech starts, not when it ends.
    """
    replay_through_the_microphone(
        SoundSequence(
            sounds=[
                NonSpeechRecording(sound=NonSpeechSound.SILENCE, duration=0.25),
                NonSpeechRecording(sound=NonSpeechSound.TONE, duration=2.0),
                NonSpeechRecording(sound=NonSpeechSound.SILENCE, duration=1.5),
            ]
        ).recording(sample_rate=SpeechDetector.sample_rate)
    )

    transcript = recognizer_hearing_every_sound.hear_next_utterance(timeout=0.5)

    assert transcript == Transcript(text=model_hearing_a_request.texts[0])


# %% Hearing every utterance
def test_hear_utterances_yields_each_utterance_in_order(
    recognizer_hearing_every_sound: SpeechRecognizer,
    model_hearing_a_request: ModelReturningSegments,
    silence_then_speech_then_pause: SoundSequence,
    frame_duration: float,
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    Every utterance is transcribed on its own, as soon as its pause has been heard.
    """
    sample_rate = SpeechDetector.sample_rate
    *_, pause = silence_then_speech_then_pause.sounds
    shorter_speech = NonSpeechRecording(
        sound=NonSpeechSound.TONE, duration=20 * frame_duration
    )
    sequence = SoundSequence(
        sounds=[*silence_then_speech_then_pause.sounds, shorter_speech, pause]
    )
    recording = sequence.recording(sample_rate=sample_rate)
    replay_through_the_microphone(recording)

    transcripts = list(islice(recognizer_hearing_every_sound.hear_utterances(), 2))

    lead_in = round(
        recognizer_hearing_every_sound.detector.kept_before_speech * sample_rate
    )
    expected_utterances = [
        recording.waveform[
            sequence.start_of(speech, sample_rate=sample_rate) - lead_in : (
                sequence.start_of(speech + 1, sample_rate=sample_rate)
            )
        ]
        for speech in (1, 3)
    ]
    transcribed_audio = [audio for audio, _ in model_hearing_a_request.calls]
    assert transcripts == 2 * [Transcript(text=model_hearing_a_request.texts[0])]
    assert all(map(np.array_equal, transcribed_audio, expected_utterances))
    assert len(transcribed_audio) == len(expected_utterances)


def test_hear_utterances_releases_the_microphone_when_the_caller_stops(
    recognizer_hearing_every_sound: SpeechRecognizer,
    silence_then_speech_then_pause: SoundSequence,
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    Listening goes on between utterances and ends when the caller stops iterating.
    """
    device = replay_through_the_microphone(
        silence_then_speech_then_pause.recording(sample_rate=SpeechDetector.sample_rate)
    )
    utterances = recognizer_hearing_every_sound.hear_utterances()
    next(utterances)

    utterances.close()

    assert not device.is_open


def test_listening_through_silence_keeps_only_the_lead_in(
    recognizer_hearing_every_sound: SpeechRecognizer,
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    However long nobody speaks, each search scores no more than the lead-in and the
    newest block.
    """
    sample_rate = SpeechDetector.sample_rate
    replay_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.SILENCE, duration=10.0).recording(
            sample_rate=sample_rate
        )
    )
    detector = recognizer_hearing_every_sound.detector
    model: ModelHearingSpeechInEverySound = detector.model

    with pytest.raises(NothingHeard):
        recognizer_hearing_every_sound.hear_next_utterance(timeout=9.0)

    lead_in = round(detector.kept_before_speech * sample_rate)
    block_length = round(recognizer_hearing_every_sound.block_duration * sample_rate)
    assert max(model.scored_sample_counts) < lead_in + SpeechDetector.frame_length + (
        block_length
    )
