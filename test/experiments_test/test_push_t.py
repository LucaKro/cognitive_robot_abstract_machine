"""
Tests for the Push-T scene: a T-shaped block that has to be pushed onto a target pose.

The pushing test opens MuJoCo's viewer and runs at wall-clock speed, so the motion can
be watched while it happens.
"""

import numpy
import pytest

from experiments.push_t.real_time_simulation import (
    RealTimeSimulation,
    SimulationNotStartedError,
)
from experiments.push_t.scene import (
    BLOCK_HEIGHT,
    PlanarPoint,
    PlanarPose,
    PushTScene,
)

# %% the run being exercised

BLOCK_START = PlanarPose(x=0.3, y=0.0)
"""
Where the block lies before it is pushed.
"""

PUSHER_START = PlanarPoint(x=0.46, y=0.0)
"""
Where the pusher waits, just clear of the block's near face.
"""

PUSHER_END = PlanarPoint(x=0.18, y=0.0)
"""
Where the pusher travels to, straight through where the block starts out.
"""

PUSH_DURATION = 3.0
"""
Seconds the pusher takes to travel from its start to its end.
"""

SETTLE_DURATION = 0.5
"""
Seconds the block is left alone afterwards, so it comes to rest.
"""

CONTROL_RATE = 60
"""
How often per second the pusher is given a new set point.
"""

MINIMUM_APPROACH = 0.1
"""
Metres the block has to end up closer to its target for the push to have worked.
"""


def straight_line_position(progress: float) -> PlanarPoint:
    """
    The pusher's set point at some point along its straight run.

    :param progress: How far along the run, from 0 at the start to 1 at the end.
    :return: The point the pusher should hold.
    """
    return PlanarPoint(
        x=PUSHER_START.x + progress * (PUSHER_END.x - PUSHER_START.x),
        y=PUSHER_START.y + progress * (PUSHER_END.y - PUSHER_START.y),
    )


def distance_between(first_pose: numpy.ndarray, second_pose: numpy.ndarray) -> float:
    """
    The distance between the positions two poses describe.

    :param first_pose: A 4x4 homogeneous transformation matrix.
    :param second_pose: A 4x4 homogeneous transformation matrix.
    :return: The distance in metres.
    """
    return float(numpy.linalg.norm(first_pose[:3, 3] - second_pose[:3, 3]))


# %% scene structure


def test_the_target_marker_cannot_be_collided_with():
    """
    The marker only says where the block should end up, so it must not obstruct the
    block on its way there or be knocked aside by it.
    """
    scene = PushTScene.create(block_pose=BLOCK_START, pusher_position=PUSHER_START)

    assert len(scene.target.collision) == 0
    assert len(scene.target.visual) > 0


def test_the_target_marker_has_the_same_shape_as_the_block():
    """
    A marker of a different shape would show a pose the block can never match, so both
    are built from one description.
    """
    scene = PushTScene.create(block_pose=BLOCK_START, pusher_position=PUSHER_START)

    def outlines(shapes):
        return [(shape.scale, shape.origin.to_np().tolist()) for shape in shapes]

    assert outlines(scene.target.visual) == outlines(scene.block.collision)


def test_the_block_starts_where_it_was_placed():
    """
    The block's start pose is given on the plane, so the scene is the one to work out
    the height at which it rests on the ground.
    """
    scene = PushTScene.create(block_pose=BLOCK_START, pusher_position=PUSHER_START)

    block_pose = scene.pose_of(scene.block)
    assert block_pose[0, 3] == pytest.approx(BLOCK_START.x)
    assert block_pose[1, 3] == pytest.approx(BLOCK_START.y)
    assert block_pose[2, 3] == pytest.approx(BLOCK_HEIGHT / 2)


# %% pushing


def test_advancing_a_simulation_that_was_never_started_is_refused(mujoco_scene_file):
    """
    Stepping a simulation whose viewer was never opened and whose clock never started
    would silently run the physics against a reference time of zero, so it is refused
    instead.
    """
    scene = PushTScene.create(block_pose=BLOCK_START, pusher_position=PUSHER_START)
    simulation = RealTimeSimulation(world=scene.world, headless=True)

    with pytest.raises(SimulationNotStartedError):
        simulation.advance(1 / CONTROL_RATE)


def test_the_pusher_pushes_the_block_towards_the_target(mujoco_scene_file):
    """
    Running the pusher straight through where the block lies has to carry the block
    along with it, leaving it flat on the plane and the marker untouched.
    """
    scene = PushTScene.create(block_pose=BLOCK_START, pusher_position=PUSHER_START)
    target_pose = scene.pose_of(scene.target)
    start_distance = distance_between(scene.pose_of(scene.block), target_pose)

    with RealTimeSimulation(world=scene.world) as simulation:
        scene.command_pusher(simulation, PUSHER_START)
        control_steps = round(PUSH_DURATION * CONTROL_RATE)
        for step in range(control_steps):
            scene.command_pusher(
                simulation, straight_line_position((step + 1) / control_steps)
            )
            simulation.advance(1 / CONTROL_RATE)
        simulation.advance(SETTLE_DURATION)

        block_pose = scene.pose_of(scene.block)
        settled_target_pose = scene.pose_of(scene.target)

    assert start_distance - distance_between(block_pose, target_pose) >= (
        MINIMUM_APPROACH
    )
    assert block_pose[2, 3] == pytest.approx(BLOCK_HEIGHT / 2, abs=1e-3)
    numpy.testing.assert_array_equal(settled_target_pose, target_pose)
