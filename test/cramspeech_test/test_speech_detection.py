"""
Finding finished utterances in audio, and which audio a search keeps for the next one.
"""

from __future__ import annotations

import numpy as np
import pytest
from cramspeech.exceptions import SampleRateMismatch
from cramspeech.microphone import Recording
from cramspeech.speech_recognizer import SpeechDetector

from .dataset.non_speech import NonSpeechRecording, NonSpeechSound, SoundSequence


# %% No utterance
def test_silence_holds_no_utterance(
    detector_hearing_every_sound: SpeechDetector, frame_duration: float
) -> None:
    """
    Nothing is found while nobody speaks.
    """
    silence = NonSpeechRecording(
        sound=NonSpeechSound.SILENCE, duration=100 * frame_duration
    )

    utterance = detector_hearing_every_sound.search_for_utterance(
        silence.recording(sample_rate=SpeechDetector.sample_rate)
    ).utterance

    assert utterance is None


def test_speech_without_a_pause_is_not_finished(
    detector_hearing_every_sound: SpeechDetector, frame_duration: float
) -> None:
    """
    Speech followed by a silence shorter than the pause duration may still go on.
    """
    sequence = SoundSequence(
        sounds=[
            NonSpeechRecording(
                sound=NonSpeechSound.SILENCE, duration=40 * frame_duration
            ),
            NonSpeechRecording(sound=NonSpeechSound.TONE, duration=30 * frame_duration),
            NonSpeechRecording(
                sound=NonSpeechSound.SILENCE, duration=20 * frame_duration
            ),
        ]
    )

    utterance = detector_hearing_every_sound.search_for_utterance(
        sequence.recording(sample_rate=SpeechDetector.sample_rate)
    ).utterance

    assert utterance is None


# %% Finished utterance
def test_speech_followed_by_a_pause_is_an_utterance(
    detector_hearing_every_sound: SpeechDetector,
    silence_then_speech_then_pause: SoundSequence,
) -> None:
    """
    The utterance runs from the lead-in before the speech to the end of the speech.
    """
    sample_rate = SpeechDetector.sample_rate
    sequence = silence_then_speech_then_pause
    recording = sequence.recording(sample_rate=sample_rate)

    utterance = detector_hearing_every_sound.search_for_utterance(recording).utterance

    lead_in = round(detector_hearing_every_sound.kept_before_speech * sample_rate)
    start = sequence.start_of(1, sample_rate=sample_rate) - lead_in
    end = sequence.start_of(2, sample_rate=sample_rate)
    assert np.array_equal(utterance.waveform, recording.waveform[start:end])
    assert utterance.sample_rate == sample_rate


def test_utterance_ends_at_the_first_pause(
    detector_hearing_every_sound: SpeechDetector,
    silence_then_speech_then_pause: SoundSequence,
) -> None:
    """
    Speech after the first long enough pause belongs to the next utterance.
    """
    sample_rate = SpeechDetector.sample_rate
    _, speech, pause = silence_then_speech_then_pause.sounds
    sequence = SoundSequence(
        sounds=[*silence_then_speech_then_pause.sounds, speech, pause]
    )
    recording = sequence.recording(sample_rate=sample_rate)

    utterance = detector_hearing_every_sound.search_for_utterance(recording).utterance

    assert len(utterance.waveform) == sequence.start_of(2, sample_rate=sample_rate) - (
        sequence.start_of(1, sample_rate=sample_rate)
        - round(detector_hearing_every_sound.kept_before_speech * sample_rate)
    )


def test_lead_in_stops_at_the_start_of_the_recording(
    detector_hearing_every_sound: SpeechDetector, frame_duration: float
) -> None:
    """
    Speech that starts sooner than the lead-in is kept from the first sample on.
    """
    sample_rate = SpeechDetector.sample_rate
    sequence = SoundSequence(
        sounds=[
            NonSpeechRecording(
                sound=NonSpeechSound.SILENCE, duration=2 * frame_duration
            ),
            NonSpeechRecording(sound=NonSpeechSound.TONE, duration=30 * frame_duration),
            NonSpeechRecording(
                sound=NonSpeechSound.SILENCE, duration=35 * frame_duration
            ),
        ]
    )
    recording = sequence.recording(sample_rate=sample_rate)

    utterance = detector_hearing_every_sound.search_for_utterance(recording).utterance

    end = sequence.start_of(2, sample_rate=sample_rate)
    assert np.array_equal(utterance.waveform, recording.waveform[:end])


def test_speech_reaching_the_maximum_duration_is_an_utterance(
    detector_hearing_every_sound: SpeechDetector, frame_duration: float
) -> None:
    """
    Speech that has not paused yet is cut off once the utterance reaches the maximum
    duration.
    """
    sample_rate = SpeechDetector.sample_rate
    detector_hearing_every_sound.maximum_duration = 50 * frame_duration
    sequence = SoundSequence(
        sounds=[
            NonSpeechRecording(
                sound=NonSpeechSound.SILENCE, duration=40 * frame_duration
            ),
            NonSpeechRecording(
                sound=NonSpeechSound.TONE, duration=100 * frame_duration
            ),
        ]
    )
    recording = sequence.recording(sample_rate=sample_rate)

    utterance = detector_hearing_every_sound.search_for_utterance(recording).utterance

    lead_in = round(detector_hearing_every_sound.kept_before_speech * sample_rate)
    start = sequence.start_of(1, sample_rate=sample_rate) - lead_in
    end = start + round(detector_hearing_every_sound.maximum_duration * sample_rate)
    assert np.array_equal(utterance.waveform, recording.waveform[start:end])


def test_speech_longer_than_the_maximum_duration_is_cut_off_before_its_pause(
    detector_hearing_every_sound: SpeechDetector, frame_duration: float
) -> None:
    """
    An utterance never runs past the maximum duration, even when its pause is already
    recorded.
    """
    sample_rate = SpeechDetector.sample_rate
    detector_hearing_every_sound.maximum_duration = 50 * frame_duration
    sequence = SoundSequence(
        sounds=[
            NonSpeechRecording(
                sound=NonSpeechSound.SILENCE, duration=40 * frame_duration
            ),
            NonSpeechRecording(
                sound=NonSpeechSound.TONE, duration=100 * frame_duration
            ),
            NonSpeechRecording(
                sound=NonSpeechSound.SILENCE, duration=35 * frame_duration
            ),
        ]
    )
    recording = sequence.recording(sample_rate=sample_rate)

    utterance = detector_hearing_every_sound.search_for_utterance(recording).utterance

    assert len(utterance.waveform) == round(
        detector_hearing_every_sound.maximum_duration * sample_rate
    )


# %% Audio kept for later searches
def test_silence_keeps_only_the_lead_in(
    detector_hearing_every_sound: SpeechDetector, frame_duration: float
) -> None:
    """
    While nobody speaks, later searches need only enough audio for the next lead-in, cut
    at a frame boundary so the frames stay where they were.
    """
    recording = NonSpeechRecording(
        sound=NonSpeechSound.SILENCE, duration=100 * frame_duration
    ).recording(sample_rate=SpeechDetector.sample_rate)

    search = detector_hearing_every_sound.search_for_utterance(recording)

    lead_in = round(
        detector_hearing_every_sound.kept_before_speech * SpeechDetector.sample_rate
    )
    kept = len(recording.waveform) - search.first_sample_to_keep
    assert lead_in <= kept < lead_in + SpeechDetector.frame_length
    assert search.first_sample_to_keep % SpeechDetector.frame_length == 0
    assert not search.holds_speech


def test_unfinished_speech_is_kept_whole(
    detector_hearing_every_sound: SpeechDetector,
    silence_then_speech_then_pause: SoundSequence,
) -> None:
    """
    Speech that may still go on is kept with everything before it.
    """
    sequence = SoundSequence(sounds=silence_then_speech_then_pause.sounds[:2])

    search = detector_hearing_every_sound.search_for_utterance(
        sequence.recording(sample_rate=SpeechDetector.sample_rate)
    )

    assert search.first_sample_to_keep == 0
    assert search.holds_speech


def test_audio_after_an_utterance_is_kept(
    detector_hearing_every_sound: SpeechDetector,
    silence_then_speech_then_pause: SoundSequence,
) -> None:
    """
    Later searches start where the utterance ended, at the beginning of its pause.
    """
    sample_rate = SpeechDetector.sample_rate

    search = detector_hearing_every_sound.search_for_utterance(
        silence_then_speech_then_pause.recording(sample_rate=sample_rate)
    )

    assert search.first_sample_to_keep == silence_then_speech_then_pause.start_of(
        2, sample_rate=sample_rate
    )


# %% Sample rate
def test_search_for_utterance_rejects_a_recording_at_another_sample_rate(
    detector_hearing_every_sound: SpeechDetector,
) -> None:
    """
    Silero scores frames correctly only at its own sample rate.
    """
    recording = NonSpeechRecording(sound=NonSpeechSound.SILENCE).recording(
        sample_rate=SpeechDetector.sample_rate // 2
    )

    with pytest.raises(SampleRateMismatch):
        detector_hearing_every_sound.search_for_utterance(recording)


# %% Real Silero model
def test_silero_finds_no_speech_without_speech(
    non_speech_recording: Recording,
) -> None:
    """
    The bundled Silero model hears no utterance in silence, noise or a tone.
    """
    assert SpeechDetector().search_for_utterance(non_speech_recording).utterance is None


def test_silero_finds_no_speech_in_less_than_a_frame(frame_duration: float) -> None:
    """
    A recording too short to score holds no utterance yet.
    """
    recording = NonSpeechRecording(
        sound=NonSpeechSound.TONE, duration=frame_duration / 2
    ).recording(sample_rate=SpeechDetector.sample_rate)

    assert SpeechDetector().search_for_utterance(recording).utterance is None
