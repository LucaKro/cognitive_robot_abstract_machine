from __future__ import annotations

from dataclasses import dataclass

from cramspeech.microphone import Microphone
from cramspeech.transcription import Transcriber, Transcript


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

    transcriber: Transcriber
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
