"""
Generated recordings of silence, white noise and a tone, which tests use in place of
speech.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from cramspeech.microphone import Recording


# %% Sounds
class NonSpeechSound(StrEnum):
    """
    Sounds a microphone picks up when nobody speaks.
    """

    SILENCE = "silence"
    """
    A quiet room.
    """

    WHITE_NOISE = "white noise"
    """
    Steady hiss, such as a fan or the microphone's own noise.
    """

    TONE = "tone"
    """
    A single pure pitch, such as a beep.
    """


# %% Recording
@dataclass(frozen=True)
class NonSpeechRecording:
    """
    A recording of one :class:`NonSpeechSound`, generated with numpy.
    """

    sound: NonSpeechSound
    """
    What the recording contains.
    """

    duration: float = 3.0
    """
    Length of the recording, in seconds.
    """

    noise_amplitude: float = 0.1
    """
    Standard deviation of :attr:`NonSpeechSound.WHITE_NOISE`, in full scale.
    """

    tone_frequency: float = 440.0
    """
    Pitch of :attr:`NonSpeechSound.TONE`, in hertz (concert pitch A).
    """

    tone_amplitude: float = 0.3
    """
    Peak of :attr:`NonSpeechSound.TONE`, in full scale.
    """

    seed: int = 0
    """
    Seed of the noise generator, so every run hears the same noise.
    """

    def recording(self, sample_rate: int) -> Recording:
        """
        :param sample_rate: Samples per second, in hertz.
        :return: :attr:`duration` seconds of :attr:`sound`.
        """
        sample_count = round(self.duration * sample_rate)
        match self.sound:
            case NonSpeechSound.SILENCE:
                waveform = np.zeros(sample_count)
            case NonSpeechSound.WHITE_NOISE:
                generator = np.random.default_rng(self.seed)
                waveform = self.noise_amplitude * generator.standard_normal(
                    sample_count
                )
            case NonSpeechSound.TONE:
                times = np.arange(sample_count) / sample_rate
                waveform = self.tone_amplitude * np.sin(
                    2 * np.pi * self.tone_frequency * times
                )
        return Recording(waveform=waveform.astype(np.float32), sample_rate=sample_rate)


# %% Sequence
@dataclass
class SoundSequence:
    """
    Sounds recorded one after another.
    """

    sounds: list[NonSpeechRecording]
    """
    The sounds, in recording order.
    """

    def recording(self, sample_rate: int) -> Recording:
        """
        :param sample_rate: Samples per second, in hertz.
        :return: All :attr:`sounds` in one recording.
        """
        waveforms = [sound.recording(sample_rate).waveform for sound in self.sounds]
        return Recording(waveform=np.concatenate(waveforms), sample_rate=sample_rate)

    def start_of(self, position: int, sample_rate: int) -> int:
        """
        :param position: Index into :attr:`sounds`.
        :param sample_rate: Samples per second, in hertz.
        :return: The sample at which the sound at *position* begins.
        """
        earlier_sounds = SoundSequence(sounds=self.sounds[:position])
        return len(earlier_sounds.recording(sample_rate).waveform)
