"""
What the transcriber passes to and takes from the Whisper model, and what the real model
hears in audio without speech.
"""

from __future__ import annotations

import pytest
from cramspeech.exceptions import SampleRateMismatch
from cramspeech.microphone import Recording
from cramspeech.transcription import (
    Transcript,
    WhisperLanguage,
    WhisperModelSize,
    WhisperTranscriber,
)
from typing_extensions import Callable

from ..pytest_environment import runs_in_continuous_integration
from .dataset.non_speech import NonSpeechRecording, NonSpeechSound
from .dataset.stand_ins import ModelReturningSegments


# %% Settings
def test_transcriber_expects_english_by_default() -> None:
    """
    Speech is taken for English unless the transcriber is told otherwise.
    """
    assert WhisperTranscriber().language is WhisperLanguage.ENGLISH


def test_transcribe_passes_the_settings_to_the_model(
    one_second_of_silence: Recording,
) -> None:
    """
    The language and beam size of the transcriber reach the model with the recorded
    audio.
    """
    model = ModelReturningSegments(texts=[])
    transcriber = WhisperTranscriber(language=WhisperLanguage.GERMAN, beam_size=2)
    transcriber.model = model

    transcriber.transcribe(one_second_of_silence)

    [(audio, settings)] = model.calls
    assert audio is one_second_of_silence.waveform
    assert settings == {
        "language": transcriber.language,
        "beam_size": transcriber.beam_size,
        "vad_filter": True,
    }


# %% Segments
def test_transcribe_joins_the_segments(
    one_second_of_silence: Recording,
    transcriber_hearing: Callable[[ModelReturningSegments], WhisperTranscriber],
) -> None:
    """
    Whisper's segments start with a space.

    The transcript joins them with single spaces.
    """
    transcriber = transcriber_hearing(
        ModelReturningSegments(texts=[" Bring me", " a cup."])
    )

    assert transcriber.transcribe(one_second_of_silence) == Transcript(
        text="Bring me a cup."
    )


# %% Sample rate
def test_transcribe_rejects_a_recording_at_another_sample_rate(
    transcriber_hearing: Callable[[ModelReturningSegments], WhisperTranscriber],
) -> None:
    """
    Whisper misreads audio at any rate but its own, so such a recording is refused.
    """
    transcriber = transcriber_hearing(ModelReturningSegments(texts=[]))
    recording = NonSpeechRecording(sound=NonSpeechSound.SILENCE).recording(
        sample_rate=transcriber.sample_rate // 2
    )

    with pytest.raises(SampleRateMismatch):
        transcriber.transcribe(recording)


# %% No speech
@pytest.mark.skipif(
    runs_in_continuous_integration(),
    reason="Downloads the Whisper model from Hugging Face.",
)
def test_whisper_hears_nothing_without_speech(non_speech_recording: Recording) -> None:
    """
    Whisper on its own makes up words from silence or a beep, such as "You".

    The transcriber returns an empty transcript instead.
    """
    transcriber = WhisperTranscriber(model_size=WhisperModelSize.BASE)

    assert transcriber.transcribe(non_speech_recording) == Transcript(text="")


# %% Compute type
@pytest.mark.skipif(
    runs_in_continuous_integration(),
    reason="Downloads the Whisper model from Hugging Face.",
)
def test_whisper_runs_in_the_chosen_compute_type() -> None:
    """
    The model is loaded in the transcriber's number format.

    The hardware may keep the activations in a wider float, so ctranslate2 reports, for
    example, ``int8_float32`` on a CPU.
    """
    transcriber = WhisperTranscriber(model_size=WhisperModelSize.TINY)

    assert transcriber.model.model.compute_type.startswith(transcriber.compute_type)
