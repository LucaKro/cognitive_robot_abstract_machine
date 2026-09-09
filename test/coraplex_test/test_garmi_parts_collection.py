"""
Coverage for the parts the Garmi collection run fetches.

The run exists to show one robot taking differently shaped parts with differently turned
hands, from stations it can actually reach. Both of those are decided by the ``PARTS``
table alone, so they are checked here rather than by driving a simulator.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

DEMO_PATH = (
    Path(__file__).resolve().parents[2]
    / "coraplex"
    / "demos"
    / "coraplex_generated"
    / "garmi_screw_box_shuttle.py"
)
"""
The demonstration under test, which lives outside any importable package.
"""


@pytest.fixture(scope="module")
def demo():
    """
    The demonstration module, loaded from its path.

    It is registered in ``sys.modules`` before execution because it postpones its
    annotations, which dataclass field resolution looks the module up to read.
    """
    spec = importlib.util.spec_from_file_location(DEMO_PATH.stem, DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_part_is_taken_with_its_own_hand_orientation(demo):
    """
    No two parts are grasped the same way: the parts differ in which face a parallel
    gripper can close on, so their grasps have to differ too.
    """
    orientations = [
        (part.approach_direction, part.vertical_alignment, part.rotate_gripper)
        for part in demo.PARTS
    ]

    assert len(set(orientations)) == len(demo.PARTS)


# %% collision avoidance


def test_the_run_avoids_collisions(demo):
    """
    The hall's racks are only kept clear of because every motion of the run carries a
    collision-avoidance goal; without it the arm follows its captured poses through them.
    """
    assert demo.AVOIDS_COLLISIONS
