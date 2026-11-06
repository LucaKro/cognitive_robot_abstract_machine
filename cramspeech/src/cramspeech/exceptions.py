"""
Errors of listening and transcribing: audio at the wrong sample rate, nobody speaking in
time, and a microphone that stopped delivering audio.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


# %% Transcription
@dataclass
class SampleRateMismatch(DataclassException):
    """
    Raised when a recording reaches a model at a sample rate the model does not take.
    """

    recording_sample_rate: int
    """
    Samples per second of the recording, in hertz.
    """

    model_sample_rate: int
    """
    Samples per second the model takes, in hertz.
    """

    def error_message(self) -> str:
        return (
            f"The recording has {self.recording_sample_rate} Hz, but the model takes "
            f"{self.model_sample_rate} Hz."
        )

    def suggest_correction(self) -> str:
        return f"Record at {self.model_sample_rate} Hz, the sample_rate of the model's class."


# %% Listening
@dataclass
class NothingHeard(DataclassException):
    """
    Raised when nobody starts speaking while the microphone is listening.
    """

    timeout: float
    """
    Seconds of audio listened to without any speech starting.
    """

    def error_message(self) -> str:
        return f"Nobody started speaking within {self.timeout} s."

    def suggest_correction(self) -> str:
        return (
            "Speak while the microphone listens, allow a longer timeout, or check that "
            "the microphone's device is the input being spoken into."
        )


# %% Microphone
@dataclass
class MicrophoneStopped(DataclassException):
    """
    Raised when an input device stops delivering audio while it is being listened to.
    """

    device: int | str | None
    """
    The input device as :attr:`cramspeech.microphone.Microphone.device` names it.
    """

    def error_message(self) -> str:
        return f"The input device {self.device!r} stopped delivering audio."

    def suggest_correction(self) -> str:
        return (
            "Check that the device is still connected and that no other program holds "
            "it, then listen again."
        )
