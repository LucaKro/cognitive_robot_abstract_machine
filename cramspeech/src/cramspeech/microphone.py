from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sounddevice


# %% Recording
@dataclass
class Recording:
    """
    Mono audio as a sequence of samples taken at a fixed rate.
    """

    waveform: np.ndarray
    """
    One float32 sample in [-1, 1] per frame, in recording order.
    """

    sample_rate: int
    """
    Samples per second, in hertz.
    """


# %% Microphone
@dataclass
class Microphone:
    """
    An audio input device that records mono audio on request.
    """

    device: int | str | None = None
    """
    Index or name of the input device as ``python -m sounddevice`` lists it.

    ``None`` selects the system's default input.
    """

    def record(self, duration: float, sample_rate: int) -> Recording:
        """
        Record from the device, returning once the recording is complete.

        :param duration: Length of the recording, in seconds.
        :param sample_rate: Samples per second to record, in hertz.
        :return: The recorded audio.
        """
        frames = sounddevice.rec(
            frames=round(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            device=self.device,
        )
        sounddevice.wait()
        return Recording(waveform=frames[:, 0], sample_rate=sample_rate)
