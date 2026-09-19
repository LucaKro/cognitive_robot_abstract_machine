"""
Tests for :mod:`experiments.simulated_grasp`.

The arrangement the grasp depends on, and the repairs that make the vendored arm
drivable, are exercised without a simulator. What the grasp itself does is only
answerable by physics, so those tests start MuJoCo and follow the same continuous-
integration gating as every other simulator-backed test in this repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..pytest_environment import runs_in_continuous_integration

from experiments.simulated_grasp.grasp_attempt import (
    CONTROL_FREQUENCY,
    GraspOutcome,
    PhysicalGrasp,
)
from experiments.simulated_grasp.panda_world import (
    ARM_VELOCITY_LIMIT,
    BLOCK_HEIGHT,
    BLOCK_SIDE,
    GRASP_DEPTH,
    FINGER_VELOCITY_LIMIT,
    GRIPPED_FINGER_OFFSET,
    OPEN_FINGER_OFFSET,
    READY_POSTURE,
    TABLE_TOP,
    PandaWorld,
)

requires_mujoco = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)


@pytest.fixture(scope="module")
def panda() -> PandaWorld:
    """
    Reading the arm costs a few seconds of mesh work, and nothing below changes it.
    """
    return PandaWorld.of()


# %% the arrangement the grasp depends on


def test_the_block_starts_resting_on_the_table(panda):
    """
    The block is placed on the surface rather than floating above it or sunk into it, so
    the physics does not start by dropping or ejecting it.
    """
    assert panda.block_position[2] - BLOCK_HEIGHT / 2 == pytest.approx(TABLE_TOP)


def test_the_commanded_frame_sits_between_the_fingertips(panda):
    """
    Commanding that frame over the block is what puts the fingers around it, so it has
    to lie along the fingers rather than at the hand or beyond their tips.
    """
    finger = panda.fingers[0].child
    root = panda.world.compute_forward_kinematics_np(panda.hand, finger)[2, 3]
    tip = root + finger.combined_mesh.bounds[1][2]

    assert root < GRASP_DEPTH < tip


def test_the_fingers_start_the_same_distance_either_side_of_the_palm(panda):
    """
    The gripper closes on whatever the commanded frame is over only while its two
    fingers stay symmetric about that frame.
    """
    offsets = [
        panda.world.compute_forward_kinematics_np(panda.hand, finger.child)[1, 3]
        for finger in panda.fingers
    ]

    assert offsets == pytest.approx([OPEN_FINGER_OFFSET, -OPEN_FINGER_OFFSET])


def test_the_open_gripper_clears_the_block_turned_any_way():
    """
    The fingers straddle the block whatever way the hand ends up facing, which is what
    lets an approach leave the rotation about the vertical unconstrained.
    """
    assert 2 * OPEN_FINGER_OFFSET > BLOCK_SIDE * np.sqrt(2)


def test_a_gripping_finger_is_commanded_inside_the_block():
    """
    A grip is the servo still pushing against a finger the block has stopped, so the
    commanded position lies inside the block - and not so far inside that the two
    fingers would command themselves through each other.
    """
    assert 0.0 < GRIPPED_FINGER_OFFSET < BLOCK_SIDE / 2


def test_the_arm_starts_with_the_hand_facing_the_table(panda):
    """
    The posture every attempt starts from already points the fingers down, to within a
    degree, so reaching the block costs the wrist no half-turn it would have to wind up
    against its limits.
    """
    hand_axis = panda.world.compute_forward_kinematics_np(panda.world.root, panda.hand)[
        :3, 2
    ]

    assert hand_axis == pytest.approx([0.0, 0.0, -1.0], abs=np.radians(1.0))


def test_the_ready_posture_states_one_position_per_arm_joint(panda):
    """
    The posture is applied by pairing it with the arm's joints, which silently leaves
    the outermost joints wherever they were if it is short.
    """
    assert len(READY_POSTURE) == len(panda.arm)


# %% the repairs that make the vendored arm drivable


def test_every_joint_is_driven_by_a_servo_of_its_own(panda):
    """
    The gripper arrives driven by a tendon the parser cannot follow to a joint, and with
    both fingers carrying a degree of freedom named after the first.

    Both a servo's transmission and the coupling that would be restored in its place are
    resolved by that name, so without the repair one finger collects two servos and the
    other none.
    """
    driven = [
        degree_of_freedom.name.name
        for actuator in panda.world.actuators
        for degree_of_freedom in actuator.dofs
    ]
    joints = [connection.name.name for connection in panda.arm + panda.fingers]

    assert sorted(driven) == sorted(joints)


def test_every_joint_states_how_fast_it_may_travel(panda):
    """
    The description carries positions and torques but no velocities, and the controller
    cannot build a task for a joint whose velocity is unbounded.

    The numbers in force are the stated ones, which registering the arm as a robot can
    tighten but not loosen.
    """
    arm_limits = [connection.raw_dof.limits for connection in panda.arm]
    finger_limits = [connection.raw_dof.limits for connection in panda.fingers]

    assert [limit.upper.velocity for limit in arm_limits] == pytest.approx(
        [ARM_VELOCITY_LIMIT] * len(arm_limits)
    )
    assert [limit.upper.velocity for limit in finger_limits] == pytest.approx(
        [FINGER_VELOCITY_LIMIT] * len(finger_limits)
    )


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
def test_only_the_fingers_ever_touch_the_block(completed_attempt):
    """
    The hand's own shell reaches lower than its fingertips, so a grip taken at the
    block's centre presses the palm onto it before the fingers close.
    """
    assert not completed_attempt.hand_touched_the_block


@requires_mujoco
def test_the_block_ends_up_where_it_was_to_be_put_down(completed_attempt):
    """
    The block is carried across the table and released there, rather than dropped along
    the way.
    """
    assert completed_attempt.reached_its_goals
    assert completed_attempt.placement_error < 20.0
    assert completed_attempt.travelled > 200.0


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
