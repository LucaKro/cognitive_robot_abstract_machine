"""
Tests for hearing speech: recording from the microphone, then transcribing it.
"""

from __future__ import annotations

# %% Imports
from cramspeech.microphone import Microphone
from cramspeech.speech_recognizer import SpeechRecognizer
from cramspeech.transcription import Transcript, WhisperTranscriber

from .dataset.stand_ins import ModelReturningSegments, SilentInputDevice


# %% Hearing
def test_hear_returns_the_transcribed_text(
    silent_input_device: SilentInputDevice,
) -> None:
    """
    One call records from the microphone and hands the recording to the transcriber.
    """
    model = ModelReturningSegments(texts=["Bring me a cup."])
    transcriber = WhisperTranscriber()
    transcriber.model = model
    recognizer = SpeechRecognizer(microphone=Microphone(), transcriber=transcriber)

    assert recognizer.hear(duration=1.0) == Transcript(text=model.texts[0])
