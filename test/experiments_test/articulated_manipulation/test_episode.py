"""
Tests for running a task through an episode of the benchmark: the physics is disturbed
behind the controller's back, and the verdict is judged against the physics.

Skipped where Tracy's description is not installed; the ones that simulate run only in
CI.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import numpy
import pytest

from ...pytest_environment import runs_in_continuous_integration

from experiments.articulated_manipulation.cabinet_physics import CabinetPhysics
from experiments.articulated_manipulation.cabinet_scene import (
    ArticulatedPart,
    CabinetScene,
    CabinetSceneSpecification,
)
from experiments.articulated_manipulation.disturbance_protocol import (
    ArmStartPose,
    ConditionKind,
    DisturbanceProtocol,
    EpisodeSetup,
)
from experiments.articulated_manipulation.episode import Episode, EvaluatedTask
from experiments.articulated_manipulation.metrics import EpisodeVerdict
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types.spatial_types import Pose2D
from semantic_digital_twin.utils import tracy_installed
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

pytestmark = pytest.mark.skipif(
    not tracy_installed(), reason="iai_tracy_description is not installed"
)

mujoco_runs_only_in_continuous_integration = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)

# %% tasks under evaluation


class JointCommandingTask(EvaluatedTask):
    """
    Opens the part by commanding its joint without ever touching it, and reports that it
    is done once its own world says the part is open.
    """

    @property
    def recovery_transitions_authored(self) -> int:
        return 0

    def motion_statechart(self, scene: CabinetScene) -> MotionStatechart:
        open_by_command = JointPositionList(
            name="open by command",
            goal_state=JointState.from_mapping(
                {scene.mechanism: scene.mechanism.dof.limits.upper.position}
            ),
        )
        motion = MotionStatechart()
        motion.add_nodes([open_by_command, EndMotion.when_true(open_by_command)])
        return motion


class NeverFinishingTask(JointCommandingTask):
    """
    Commands the part's joint, but never reports that it is done.
    """

    def motion_statechart(self, scene: CabinetScene) -> MotionStatechart:
        motion = MotionStatechart()
        motion.add_node(
            JointPositionList(
                name="open by command",
                goal_state=JointState.from_mapping(
                    {scene.mechanism: scene.mechanism.dof.limits.upper.position}
                ),
            )
        )
        return motion


# %% fixtures


@pytest.fixture
def protocol() -> DisturbanceProtocol:
    return DisturbanceProtocol(episodes_per_condition=1)


def setup_of(protocol: DisturbanceProtocol, kind: ConditionKind) -> EpisodeSetup:
    """
    :return: The first episode setup of the protocol's condition of the given kind.
    """
    return next(
        setup
        for setup in protocol.episode_setups(seed=0)
        if setup.condition.kind is kind
    )


@pytest.fixture
def specification() -> CabinetSceneSpecification:
    return CabinetSceneSpecification(articulated_part=ArticulatedPart.DRAWER)


# %% the arm's start pose


def test_the_arm_starts_away_from_its_park_pose_by_its_offsets(specification):
    start = ArmStartPose(deviation=0.1, seed=3)
    scene = specification.to_domain_object()
    joints = [
        connection
        for connection in scene.robot.left_arm.connections
        if isinstance(connection, ActiveConnection1DOF)
    ]
    parked = [joint.position for joint in joints]

    start.apply_to(scene.robot.left_arm)

    assert [
        joint.position - park for joint, park in zip(joints, parked)
    ] == pytest.approx(start.offsets(joints))


# %% the ground truth


@mujoco_runs_only_in_continuous_integration
def test_moving_the_cabinet_moves_it_in_the_physics_alone(specification):
    scene = specification.to_domain_object()
    believed_cabinet = scene.cabinet.root.global_transform.to_np()
    displacement = Pose2D(x=0.05, y=-0.03, yaw=0.1)
    simulation = MujocoSim(world=scene.world, headless=True)
    physics = CabinetPhysics(
        simulation=simulation, scene=scene, specification=specification
    )
    simulation.start_stepped_simulation()
    try:
        physics.move_cabinet(displacement)
        simulation.step_simulation(timedelta(seconds=simulation.simulator.step_size))
        simulated_cabinet = simulation.simulator.get_body_position(
            scene.cabinet.root.name.name
        ).result
    finally:
        simulation.stop_simulation()

    moved = dataclasses.replace(
        specification,
        table_T_cabinet_front=Pose2D.from_pose(
            (
                specification.table_T_cabinet_front.to_homogeneous_matrix()
                @ displacement.to_homogeneous_matrix()
            ).to_pose()
        ),
    )
    assert numpy.allclose(
        simulated_cabinet, moved.world_T_cabinet().to_position().to_np()[:3]
    )
    assert numpy.allclose(scene.cabinet.root.global_transform.to_np(), believed_cabinet)


@mujoco_runs_only_in_continuous_integration
def test_pushing_the_part_shut_closes_it_in_the_physics(specification):
    scene = specification.to_domain_object()
    ajar = 0.15
    scene.set_opening(ajar)
    simulation = MujocoSim(world=scene.world, headless=True)
    physics = CabinetPhysics(
        simulation=simulation, scene=scene, specification=specification
    )
    simulation.start_stepped_simulation()
    try:
        physics.push_part(-50.0)
        simulation.step_simulation(timedelta(seconds=1))
        closed_to = physics.opening
    finally:
        simulation.stop_simulation()

    assert closed_to < ajar / 2


# %% episodes


@mujoco_runs_only_in_continuous_integration
def test_opening_the_part_only_by_command_is_a_false_success(protocol):
    """
    The stuck-drawer probe: the task's world opens the drawer, the physics does not.
    """
    episode = Episode(
        setup=setup_of(protocol, ConditionKind.UNDISTURBED),
        task=JointCommandingTask(),
    )

    outcome = episode.run()

    assert outcome.verdict is EpisodeVerdict.FALSE_SUCCESS
    assert outcome.opened_fraction == pytest.approx(0.0, abs=1e-2)
    assert outcome.completion_time is not None
    assert len(outcome.control_cycle_durations) == pytest.approx(
        outcome.completion_time.total_seconds() * episode.control_frequency, abs=1
    )


@mujoco_runs_only_in_continuous_integration
def test_a_task_that_never_reports_done_is_unfinished(protocol):
    episode = Episode(
        setup=setup_of(protocol, ConditionKind.UNDISTURBED),
        task=NeverFinishingTask(),
        timeout=timedelta(seconds=0.5),
    )

    outcome = episode.run()

    assert outcome.verdict is EpisodeVerdict.UNFINISHED
    assert outcome.completion_time is None


@mujoco_runs_only_in_continuous_integration
def test_an_episode_under_a_prior_error_runs_against_the_true_cabinet(protocol):
    """
    The task's world believes the cabinet stands elsewhere; the physics still has the
    true one, whose untouched drawer stays shut.
    """
    episode = Episode(
        setup=setup_of(protocol, ConditionKind.LOCATION_PRIOR_ERROR),
        task=JointCommandingTask(),
    )

    outcome = episode.run()

    assert outcome.verdict is EpisodeVerdict.FALSE_SUCCESS
