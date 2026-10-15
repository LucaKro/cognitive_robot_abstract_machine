"""
The whole demonstration: the milk ends up on the kitchen island, carried there by the
finger pads' friction alone.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("robosimstudio")

from experiments.robosimstudio_tiago_demo.demo import RoboSimStudioTiagoDemonstration
from experiments.robosimstudio_tiago_demo.descriptions import EnvironmentVariable

# %% end to end


@pytest.mark.slow
def test_the_milk_ends_up_on_the_kitchen_island(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    The milk is released just above the island and may tip or slide a little before it
    comes to rest.
    """
    # The demonstration extends the package path; restore it for the tests after this.
    monkeypatch.setenv(
        EnvironmentVariable.ROS_PACKAGE_PATH,
        os.environ.get(EnvironmentVariable.ROS_PACKAGE_PATH, ""),
    )
    demonstration = RoboSimStudioTiagoDemonstration(output_directory=tmp_path)

    outcome = demonstration.perform(demonstration.build_simulation())

    assert outcome.distance_to_target <= 0.1
