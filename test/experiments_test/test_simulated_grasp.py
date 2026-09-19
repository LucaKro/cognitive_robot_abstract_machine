"""
Tests for :mod:`experiments.simulated_grasp`.

The geometry of the scene is exercised without a simulator, so the arrangement the grasp
depends on is pinned cheaply. What the grasp itself does is only answerable by physics,
so those tests start MuJoCo and follow the same continuous-integration gating as every
other simulator-backed test in this repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..pytest_environment import runs_in_continuous_integration

from experiments.simulated_grasp.grasp_attempt import (
    CONTROL_FREQUENCY,
    GRIPPED_FINGER_OFFSET,
    GraspOutcome,
    PhysicalGrasp,
)
from experiments.simulated_grasp.tabletop_world import (
    BLOCK_HEIGHT,
    BLOCK_SIDE,
    FINGER_LENGTH,
    GRASP_DEPTH,
    READY_POSTURE,
    TABLE_TOP,
    TabletopWorld,
    WIDEST_FINGER_OFFSET,
)

requires_mujoco = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)


@pytest.fixture
def tabletop() -> TabletopWorld:
    return TabletopWorld.of()


# %% the arrangement the grasp depends on


def test_the_block_starts_resting_on_the_table(tabletop):
    """
    The block is placed on the surface rather than floating above it or sunk into it, so
    the physics does not start by dropping or ejecting it.
    """
    assert tabletop.block_position[2] - BLOCK_HEIGHT / 2 == pytest.approx(TABLE_TOP)


def test_the_commanded_frame_sits_between_the_fingertips(tabletop):
    """
    Commanding that frame onto the block's centre is what puts the fingers around the
    block, so it has to lie along the fingers rather than at the palm.
    """
    fingertip = (
        tabletop.world.compute_forward_kinematics_np(
            tabletop.palm, tabletop.fingers[0].child
        )[2, 3]
        + FINGER_LENGTH / 2
    )

    assert 0.0 < GRASP_DEPTH < fingertip


def test_the_fingers_start_the_same_distance_either_side_of_the_palm(tabletop):
    """
    The gripper closes on whatever the commanded frame is over only while its two
    fingers stay symmetric about that frame.
    """
    offsets = [
        tabletop.world.compute_forward_kinematics_np(tabletop.palm, finger.child)[1, 3]
        for finger in tabletop.fingers
    ]

    assert offsets == pytest.approx([WIDEST_FINGER_OFFSET, -WIDEST_FINGER_OFFSET])


def test_the_open_gripper_clears_the_block_turned_any_way(tabletop):
    """
    The fingers straddle the block whatever way the hand ends up facing, which is what
    lets an approach leave the rotation about the vertical unconstrained.
    """
    finger_thickness = tabletop.fingers[0].child.collision.shapes[0].scale.y
    opening = 2 * WIDEST_FINGER_OFFSET - finger_thickness

    assert opening > BLOCK_SIDE * np.sqrt(2)


def test_a_gripping_finger_is_commanded_past_the_block_surface(tabletop):
    """
    A grip is the servo still pushing against a finger the block has stopped, so the
    face that meets the block has to be commanded inside it - and not so far inside that
    the two fingers would command themselves through each other.
    """
    finger_thickness = tabletop.fingers[0].child.collision.shapes[0].scale.y
    commanded_face = GRIPPED_FINGER_OFFSET - finger_thickness / 2

    assert 0.0 < commanded_face < BLOCK_SIDE / 2


def test_the_arm_starts_with_the_palm_facing_the_table(tabletop):
    """
    The posture every attempt starts from already points the fingers down, to within a
    degree, so reaching the block costs the wrist no half-turn it would have to wind up
    against its limits.
    """
    palm_axis = tabletop.world.compute_forward_kinematics_np(
        tabletop.world.root, tabletop.palm
    )[:3, 2]

    assert palm_axis == pytest.approx([0.0, 0.0, -1.0], abs=np.radians(1.0))


def test_every_joint_carries_a_degree_of_freedom_of_its_own_name(tabletop):
    """
    A joint whose degree of freedom is named differently is read as one that mimics
    another joint of that name, and an actuator is wired to the first joint that carries
    a degree of freedom of the name it drives.

    Both go by the name alone, so a joint that does not carry its own leaves the arm
    silently driven by the wrong servo.
    """
    joints = tabletop.arm + tabletop.fingers
    names = [joint.name.name for joint in joints]

    assert names == [joint.raw_dof.name.name for joint in joints]
    assert len(set(names)) == len(names)


def test_the_ready_posture_states_one_position_per_arm_joint(tabletop):
    """
    The posture is applied by pairing it with the arm's joints, which silently leaves
    the outermost joints wherever they were if it is short.
    """
    assert len(READY_POSTURE) == len(tabletop.arm)


# %% what the physics says the grasp did


@pytest.fixture(scope="module")
def completed_attempt() -> GraspOutcome:
    return PhysicalGrasp().execute()


@requires_mujoco
def test_the_gripper_holds_the_block_while_it_carries_it(completed_attempt):
    """
    Under ideal conditions the grasp is a grasp: the block leaves the table and both
    fingers are on it while it is up there.
    """
    assert completed_attempt.block_was_lifted
    assert completed_attempt.held_by_both_fingers


@requires_mujoco
def test_the_block_ends_up_where_it_was_to_be_put_down(completed_attempt):
    """
    The block is carried across the table and released there, rather than dropped along
    the way.
    """
    assert completed_attempt.reached_its_goals
    assert completed_attempt.placement_error < 10.0
    assert completed_attempt.travelled > 150.0


@requires_mujoco
def test_the_block_stays_put_when_the_gripper_never_closes():
    """
    The same motion moves the block nowhere without the grip, which is what says the
    block was carried by the fingers rather than pushed along by anything else.
    """
    outcome = PhysicalGrasp(closes_the_gripper=False).execute()

    assert not outcome.block_was_lifted
    assert not outcome.held_by_both_fingers
    assert outcome.travelled < 1.0


@requires_mujoco
def test_a_recording_keeps_one_frame_per_control_cycle(tmp_path):
    """
    A recording shows the attempt it was made of, so it carries a frame for every cycle
    the controller ran plus the one taken once the scene has settled.
    """
    import imageio.v2 as imageio

    video_path = tmp_path / "grasp.mp4"

    outcome = PhysicalGrasp(video_path=video_path).execute()

    frames = [frame for frame in imageio.get_reader(video_path)]
    assert len(frames) == outcome.control_cycles + 1
    assert imageio.get_reader(video_path).get_meta_data()["fps"] == CONTROL_FREQUENCY
