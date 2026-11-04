from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


# %% Transcription
@dataclass
class SampleRateMismatch(DataclassException):
    """
    Raised when a recording reaches a transcriber at a sample rate its model does not
    take.
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
        return f"Record at {self.model_sample_rate} Hz, the transcriber's sample_rate."
