"""
Timing the phases of loading a scene and measuring how its objects meet.

The point of the tool is to say which phase is worth making faster, so what it must get
right is that every phase is reported and that the totals are the phases' own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from experiments.warsaw.profile_measuring import (
    MeasuringCosts,
    Phase,
    main,
    time_the_phases,
)
from experiments.warsaw.world_loader.loader import WarsawWorldLoader

from .test_warsaw_world_loader import write_scene

# %% a scene small enough to measure in a moment


@pytest.fixture
def small_scene(tmp_path) -> Path:
    """
    :return: A directory holding a scene of two objects, which is enough for every phase
        to have something to do.
    """
    box = trimesh.creation.box(extents=(1, 1, 1))
    faces = box.faces[:4]
    write_scene(
        tmp_path / "scene.ply",
        box.vertices,
        faces,
        {"cabinet": [1, 1, 2, 2]},
    )
    return tmp_path


# %% what a profile reports


def test_every_phase_is_reported_with_what_it_cost(small_scene):
    """
    A phase left out of the report is a phase nobody will think to make faster.
    """
    costs = time_the_phases(small_scene, nearest=1)

    assert [one.phase for one in costs.phases] == [phase.value for phase in Phase]
    assert all(one.seconds >= 0.0 for one in costs.phases)


def test_the_total_is_what_the_phases_add_up_to(small_scene):
    """
    The report is read as shares of a whole, so the whole has to be the parts.
    """
    costs = time_the_phases(small_scene, nearest=1)

    assert costs.total_seconds == pytest.approx(
        sum(one.seconds for one in costs.phases)
    )


def test_the_costliest_phase_comes_first_when_ranked(small_scene):
    """
    What the tool is opened for is the top line.
    """
    costs = time_the_phases(small_scene, nearest=1)

    ranked = costs.ranked()

    assert ranked == sorted(costs.phases, key=lambda one: -one.seconds)


# %% reading one back


def test_a_profile_can_be_written_and_read_again(small_scene, tmp_path):
    """
    Two profiles are compared to say whether a change helped, so one has to outlive the
    process that took it.
    """
    written = tmp_path / "costs.json"

    main(["--scene", str(small_scene), "--nearest", "1", "--output", str(written)])

    read_back = MeasuringCosts.from_json(json.loads(written.read_text()))
    assert read_back.scene == str(small_scene)
    assert read_back.nearest == 1
    assert [one.phase for one in read_back.phases] == [phase.value for phase in Phase]
