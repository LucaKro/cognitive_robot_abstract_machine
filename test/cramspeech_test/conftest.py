"""
Fixtures for the :mod:`cramspeech` tests.
"""

from __future__ import annotations

# %% Imports
import pytest
import sounddevice

from .dataset.stand_ins import SilentInputDevice


# %% Microphone driver
@pytest.fixture
def silent_input_device(monkeypatch: pytest.MonkeyPatch) -> SilentInputDevice:
    """
    Replaces the recording calls of :mod:`sounddevice` with silence for the duration of
    a test.
    """
    device = SilentInputDevice()
    monkeypatch.setattr(sounddevice, "rec", device.rec)
    monkeypatch.setattr(sounddevice, "wait", device.wait)
    return device
