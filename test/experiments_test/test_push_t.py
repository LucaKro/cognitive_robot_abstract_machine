"""
Tests for the Push-T scene: a T-shaped block that has to be pushed onto a target pose.

The pushing test opens MuJoCo's viewer and runs at wall-clock speed, so the motion can
be watched while it happens.
"""

import math
import os

import numpy
import pytest

from experiments.push_t.push_contacts import BLOCK_CENTROID, build_push_contacts
from experiments.push_t.real_time_simulation import (
    RealTimeSimulation,
    SimulationNotStartedError,
)
from experiments.push_t.scene import (
    BLOCK_HEIGHT,
    PUSHER_LIFT_HEIGHT,
    PUSHER_RADIUS,
    PUSHER_TRAVEL_HEIGHT,
    PUSHING_HEIGHT,
    PlanarPoint,
    PlanarPose,
    PushTScene,
)
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.pushing import PushSelector, PushToPose
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig

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


def test_a_raised_pusher_travels_over_the_block(mujoco_scene_file):
    """
    Lifting clear of the block is how the pusher gets from one of its faces to another,
    so a raised pusher crossing the block has to leave it where it lies.
    """
    scene = PushTScene.create(block_pose=BLOCK_START, pusher_position=PUSHER_START)
    start_pose = scene.pose_of(scene.block)

    with RealTimeSimulation(world=scene.world, headless=True) as simulation:
        scene.command_pusher(simulation, PUSHER_START, height=PUSHER_LIFT_HEIGHT)
        simulation.advance(SETTLE_DURATION)
        control_steps = round(PUSH_DURATION * CONTROL_RATE)
        for step in range(control_steps):
            scene.command_pusher(
                simulation,
                straight_line_position((step + 1) / control_steps),
                height=PUSHER_LIFT_HEIGHT,
            )
            simulation.advance(1 / CONTROL_RATE)
        simulation.advance(SETTLE_DURATION)

        crossed_pose = scene.pose_of(scene.block)

    numpy.testing.assert_allclose(crossed_pose, start_pose, atol=1e-3)


# %% pushing under a motion statechart

VIEWER_IS_UNAVAILABLE = os.environ.get("CI", "false").lower() == "true"
"""
Whether to run without MuJoCo's viewer, as CI has no display to open one on.
"""

STANDOFF_DISTANCE = 0.05
"""
Metres behind the contact at which a push starts, clear of the block.
"""

MINIMUM_PUSH_DISTANCE = 0.03
"""
The shortest one push may be, in metres.

Below roughly this, friction takes the whole push up and the block does not move at all.
"""

MAXIMUM_PUSH_DISTANCE = 0.08
"""
The furthest past the contact one push may travel, in metres.
"""

ORIENTATION_TOLERANCE = 0.08
"""
Radians of heading error above which turning the block takes priority over moving it.

Kept under the goal's own success threshold, so that every heading the goal stops
correcting is one it would accept, and loose enough that shoving the block gets a turn at
all - turning it always drags it off course, so a tolerance near zero never converges.
"""

CONTROL_FREQUENCY = 50
"""
How often per second the motion statechart is ticked.
"""

RUN_TIME_LIMIT = 150.0
"""
Simulated seconds after which a run that has not converged is given up on.

Generous against the roughly 60 to 100 seconds the two start poses take, since a push is
a coarse way to place something and the count of attempts it takes varies.
"""

ROTATION_HEAVY_START = PlanarPose(x=0.12, y=0.06, yaw=0.9)
"""
A start pose the block mostly has to be turned out of.
"""

TRANSLATION_HEAVY_START = PlanarPose(x=0.25, y=0.12, yaw=0.05)
"""
A start pose the block mostly has to be shoved out of.
"""


def build_push_goal(scene: PushTScene) -> PushToPose:
    """
    Build the goal that pushes a scene's block onto its target marker, with the contacts
    and the centroid the T's own shape gives it.

    :param scene: The scene to act in.
    :return: The goal.
    """
    return PushToPose(
        pushed_body=scene.block,
        target_body=scene.target,
        pusher=scene.pusher,
        selector=PushSelector(
            contacts=build_push_contacts(),
            centroid=BLOCK_CENTROID,
            pusher_radius=PUSHER_RADIUS,
            standoff_distance=STANDOFF_DISTANCE,
            minimum_push_distance=MINIMUM_PUSH_DISTANCE,
            maximum_push_distance=MAXIMUM_PUSH_DISTANCE,
            pushing_height=PUSHING_HEIGHT,
            orientation_tolerance=ORIENTATION_TOLERANCE,
        ),
        travel_height=PUSHER_TRAVEL_HEIGHT,
    )


def planar_error(block_pose: numpy.ndarray, target_pose: numpy.ndarray):
    """
    How far the block still is from its target, on the plane.

    :param block_pose: The block's pose.
    :param target_pose: The pose the block should end up at.
    :return: The distance in metres and the heading error in radians.
    """
    distance = float(numpy.linalg.norm(block_pose[:2, 3] - target_pose[:2, 3]))
    heading = math.remainder(
        math.atan2(target_pose[1, 0], target_pose[0, 0])
        - math.atan2(block_pose[1, 0], block_pose[0, 0]),
        math.tau,
    )
    return distance, heading


@pytest.mark.parametrize("block_start", [ROTATION_HEAVY_START, TRANSLATION_HEAVY_START])
def test_the_statechart_pushes_the_block_onto_the_target(
    block_start, mujoco_scene_file
):
    """
    A statechart given only the block's live pose has to work out where to push it, one
    contact at a time, until it sits on the marker.
    """
    scene = PushTScene.create(block_pose=block_start, pusher_position=PUSHER_START)
    goal = build_push_goal(scene)
    motion_statechart = MotionStatechart()
    motion_statechart.add_nodes([goal, EndMotion.when_true(goal)])

    controller_config = QPControllerConfig(target_frequency=CONTROL_FREQUENCY)
    executor = Executor(
        context=MotionStatechartContext(
            world=scene.world, qp_controller_config=controller_config
        )
    )
    executor.compile(motion_statechart=motion_statechart)
    motion_statechart.draw('/tmp/he_said_tities_hehe.pdf')

    with RealTimeSimulation(
        world=scene.world, headless=VIEWER_IS_UNAVAILABLE
    ) as simulation:
        for _ in range(round(RUN_TIME_LIMIT * CONTROL_FREQUENCY)):
            if motion_statechart.is_end_motion():
                break
            executor.tick()
            simulation.advance(controller_config.control_dt)

        distance, heading = planar_error(
            scene.pose_of(scene.block), scene.pose_of(scene.target)
        )

    assert motion_statechart.is_end_motion()
    assert distance <= goal.position_threshold
    assert abs(heading) <= goal.orientation_threshold
