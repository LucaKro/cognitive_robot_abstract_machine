"""
Coverage for the parts the Garmi collection run fetches.

The run exists to show one robot taking differently shaped parts with differently turned
hands, from stations it can actually reach. Both of those are decided by the ``PARTS``
table alone, so they are checked here rather than by driving a simulator.
"""

import importlib.util
import math
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


def reach(pose, stand) -> float:
    """
    How far the robot's base stands from what it reaches for, in the floor plane.

    :param pose: The pose being reached for.
    :param stand: The base pose it is reached from.
    """
    return math.hypot(float(pose.x) - float(stand.x), float(pose.y) - float(stand.y))


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


def test_every_station_is_within_the_reach_the_run_relies_on(demo):
    """
    Every pick and place is made from a base pose no further off than the one station
    this run already performs, so no leg asks for a longer reach than the arm has shown.
    """
    screw_box = next(
        part for part in demo.PARTS if part.mesh is demo.PartMesh.SCREW_BOX
    )
    proven_reach = reach(screw_box.storage_pose, screw_box.storage_stand)

    for part in demo.PARTS:
        assert reach(part.storage_pose, part.storage_stand) <= proven_reach
        assert reach(part.delivery_pose, part.delivery_stand) <= proven_reach


# %% collision avoidance


def test_the_run_avoids_collisions(demo):
    """
    The hall's racks are only kept clear of because every motion of the run carries a
    collision-avoidance goal; without it the arm follows its captured poses through them.
    """
    assert demo.AVOIDS_COLLISIONS
