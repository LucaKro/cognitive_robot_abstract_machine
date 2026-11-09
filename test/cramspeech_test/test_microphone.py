"""
Recording and streaming through a stand-in audio driver, listing the input devices, and,
outside CI, the real microphone.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from cramspeech.exceptions import MicrophoneStopped
from cramspeech.microphone import InputDevice, Microphone, Recording
from cramspeech.transcription import WhisperTranscriber
from typing_extensions import Callable

from ..pytest_environment import runs_in_continuous_integration
from .dataset.non_speech import NonSpeechRecording, NonSpeechSound
from .dataset.stand_ins import (
    AudioDevicesOfADesktop,
    ReplayingInputDevice,
    SilentInputDevice,
)


# %% Recording request
def test_record_requests_the_duration_at_the_sample_rate(
    silent_input_device: SilentInputDevice,
) -> None:
    """
    The driver is asked for one mono float32 channel on the chosen device, with as many
    frames as the duration lasts at the sample rate.
    """
    microphone = Microphone(device="USB microphone")
    sample_rate = WhisperTranscriber.sample_rate

    microphone.record(duration=1.5, sample_rate=sample_rate)

    assert silent_input_device.requests == [
        {
            "frames": 3 * sample_rate // 2,
            "samplerate": sample_rate,
            "channels": 1,
            "dtype": np.float32,
            "device": microphone.device,
        }
    ]


# %% Recorded audio
def test_record_returns_the_frames_as_a_mono_recording(
    silent_input_device: SilentInputDevice,
) -> None:
    """
    The single recorded channel becomes the waveform, one sample per frame, tagged with
    the sample rate it was recorded at.
    """
    sample_rate = WhisperTranscriber.sample_rate

    recording = Microphone().record(duration=1.0, sample_rate=sample_rate)

    [request] = silent_input_device.requests
    assert recording.waveform.shape == (request["frames"],)
    assert recording.waveform.dtype == np.float32
    assert recording.sample_rate == sample_rate


# %% Streaming
def test_stream_opens_a_mono_stream_with_blocks_of_the_duration(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    The driver is asked for one mono float32 channel on the chosen device, delivered in
    blocks as long as the block duration at the sample rate.
    """
    sample_rate = WhisperTranscriber.sample_rate
    device = replay_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.TONE).recording(sample_rate=sample_rate)
    )
    microphone = Microphone(device="USB microphone")

    block = next(microphone.stream(sample_rate=sample_rate, block_duration=0.25))

    assert device.requests == [
        {
            "samplerate": sample_rate,
            "blocksize": sample_rate // 4,
            "channels": 1,
            "dtype": np.float32,
            "device": microphone.device,
        }
    ]
    assert block.shape == (sample_rate // 4,)


def test_stream_keeps_every_block_in_recording_order(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    Blocks that arrive before the caller reads them, as while it transcribes, are kept
    and joined up into the audio the device delivered.
    """
    sample_rate = WhisperTranscriber.sample_rate
    recording = NonSpeechRecording(sound=NonSpeechSound.WHITE_NOISE).recording(
        sample_rate=sample_rate
    )
    replay_through_the_microphone(recording)
    block_duration = 0.25
    blocks = Microphone().stream(sample_rate=sample_rate, block_duration=block_duration)
    block_count = len(recording.waveform) // round(block_duration * sample_rate)

    streamed = np.concatenate([next(blocks) for _ in range(block_count)])

    assert np.array_equal(streamed, recording.waveform)


def test_stream_raises_when_the_device_stops_delivering(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    A stream that ends, such as an unplugged microphone, fails instead of waiting for
    audio forever.
    """
    sample_rate = WhisperTranscriber.sample_rate
    replay_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.TONE, duration=0.5).recording(
            sample_rate=sample_rate
        )
    )
    blocks = Microphone().stream(sample_rate=sample_rate, block_duration=0.25)

    with pytest.raises(MicrophoneStopped):
        list(blocks)


def test_stream_warns_about_every_block_after_lost_audio(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Audio the device dropped is reported, and listening goes on with the next block.
    """
    sample_rate = WhisperTranscriber.sample_rate
    block_duration = 0.25
    recording = NonSpeechRecording(sound=NonSpeechSound.TONE, duration=0.5).recording(
        sample_rate=sample_rate
    )
    device = replay_through_the_microphone(recording)
    device.drops_audio = True
    blocks = Microphone().stream(sample_rate=sample_rate, block_duration=block_duration)
    block_count = len(recording.waveform) // round(block_duration * sample_rate)

    with pytest.raises(MicrophoneStopped):
        list(blocks)
    warnings = [
        record.levelno
        for record in caplog.records
        if record.name == Microphone.__module__
    ]

    assert warnings == block_count * [logging.WARNING]


def test_stream_closes_the_device_when_the_caller_stops(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    The device is released as soon as the caller stops iterating.
    """
    sample_rate = WhisperTranscriber.sample_rate
    device = replay_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.TONE).recording(sample_rate=sample_rate)
    )
    blocks = Microphone().stream(sample_rate=sample_rate, block_duration=0.25)
    next(blocks)

    blocks.close()

    assert not device.is_open


# %% Input devices
def test_only_devices_that_record_are_listed(
    audio_devices_of_a_desktop: AudioDevicesOfADesktop,
) -> None:
    """
    Every device with an input channel is listed with its index and name, and the
    system's default input is marked.
    """
    default_index = audio_devices_of_a_desktop.default_input_index
    expected = [
        InputDevice(
            index=device["index"],
            name=device["name"],
            is_default=device["index"] == default_index,
        )
        for device in audio_devices_of_a_desktop.devices
        if device["max_input_channels"] > 0
    ]

    assert InputDevice.list_available() == expected


def test_input_device_reads_as_its_index_and_name() -> None:
    """
    The line names the index and the name, the two ways to choose the device.
    """
    device = InputDevice(index=6, name="USB headset", is_default=False)

    assert str(device).split() == [str(device.index), *device.name.split()]


def test_default_input_device_is_marked() -> None:
    """
    The system's default input starts with a star, as ``python -m sounddevice`` shows
    it.
    """
    device = InputDevice(index=6, name="USB headset", is_default=True)

    assert str(device).split()[0] == "*"


# %% Real microphone
@pytest.mark.skipif(
    runs_in_continuous_integration(),
    reason="Needs a microphone.",
)
def test_record_from_the_real_microphone() -> None:
    """
    The system's default input records mono float32 samples in full scale.
    """
    sample_rate = WhisperTranscriber.sample_rate

    recording = Microphone().record(duration=0.5, sample_rate=sample_rate)

    assert recording.waveform.shape == (sample_rate // 2,)
    assert recording.waveform.dtype == np.float32
    assert np.abs(recording.waveform).max() <= 1.0
    assert recording.sample_rate == sample_rate


@pytest.mark.skipif(
    runs_in_continuous_integration(),
    reason="Needs a microphone.",
)
def test_stream_from_the_real_microphone() -> None:
    """
    The system's default input streams mono float32 blocks in full scale.
    """
    sample_rate = WhisperTranscriber.sample_rate
    blocks = Microphone().stream(sample_rate=sample_rate, block_duration=0.25)

    first_blocks = [next(blocks), next(blocks)]
    blocks.close()

    for block in first_blocks:
        assert block.shape == (sample_rate // 4,)
        assert block.dtype == np.float32
        assert np.abs(block).max() <= 1.0
