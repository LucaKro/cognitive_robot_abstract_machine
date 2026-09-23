"""
Tests for the benchmark scene: the cabinet's moving part is moved by contact alone,
never by giskard commanding its joint.

Skipped where Tracy's description is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ...pytest_environment import runs_in_continuous_integration

from experiments.articulated_manipulation.cabinet_scene import (
    ArticulatedPart,
    CabinetScene,
    CabinetSceneBuilder,
)
from giskardpy.executor import Executor, SteppedSimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.utils import tracy_installed
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

pytestmark = pytest.mark.skipif(
    not tracy_installed(), reason="iai_tracy_description is not installed"
)

mujoco_runs_only_in_continuous_integration = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)

CONTROL_FREQUENCY = 50
"""
The control frequency giskard runs at, in Hz.
"""

# %% fixtures


@pytest.fixture
def builder(request) -> CabinetSceneBuilder:
    return CabinetSceneBuilder(articulated_part=request.param)


@pytest.fixture
def scene(builder) -> CabinetScene:
    return builder.build()


every_part = pytest.mark.parametrize(
    "builder", list(ArticulatedPart), indirect=True, ids=lambda part: part.name
)


def run_live(scene: CabinetScene, motion: MotionStatechart, cycles: int) -> float:
    """
    Run giskard live against the stepped physics for a fixed number of control cycles.

    :param scene: The scene to run in.
    :param motion: What giskard does.
    :param cycles: How many control cycles to run.
    :return: The moving part's opening in the physics afterwards.
    """
    simulation = MujocoSim(world=scene.world, headless=True)
    simulation.start_stepped_simulation()
    try:
        executor = Executor(
            context=MotionStatechartContext(
                world=scene.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=CONTROL_FREQUENCY
                ),
            ),
            pacer=SteppedSimulationPacer(simulation),
        )
        executor.compile(motion_statechart=motion)
        for _ in range(cycles):
            executor.tick()
            executor.pacer.sleep()
        return simulation.simulator.get_joint_value(scene.mechanism.name.name).result
    finally:
        simulation.stop_simulation()


# %% the scene


@every_part
def test_no_controller_commands_the_moving_part(scene):
    assert not scene.mechanism.has_hardware_interface


@every_part
def test_the_handle_moves_with_the_moving_part(scene):
    assert scene.part.handle is scene.handle
    assert (
        scene.handle.root.get_first_parent_connection_of_type(ActiveConnection1DOF)
        is scene.mechanism
    )


# %% moved only through contact


@every_part
@mujoco_runs_only_in_continuous_integration
def test_giskard_commanding_the_moving_parts_joint_does_not_move_it(scene):
    """
    Giskard integrates the joint it commands into the world, but nothing touches the
    part, so the physics leaves it closed.
    """
    fully_open = scene.mechanism.dof.limits.upper.position
    motion = MotionStatechart()
    motion.add_node(
        JointPositionList(
            name="open without touching",
            goal_state=JointState.from_mapping({scene.mechanism: fully_open}),
        )
    )

    simulated = run_live(scene, motion, cycles=100)

    assert simulated == pytest.approx(0.0, abs=1e-3)


@dataclass
class Push:
    """
    How the left hand pushes an ajar part shut: it comes down in front of the part and
    moves straight towards the cabinet.
    """

    opening: float
    """
    How far the part is open before the push.
    """

    sideways_offset: float
    """
    How far to the left of the cabinet's centre line the hand pushes, in metres.
    """

    standoff: float = 0.4
    """
    How far in front of the cabinet's front the hand comes down, in metres.
    """

    height: float = 0.12
    """
    How high above the table top the hand pushes, in metres.
    """

    approach_height: float = 0.42
    """
    How high above the table top the hand travels to the point it comes down at, in
    metres.
    """

    depth: float = 0.13
    """
    How far in front of the cabinet's front the push ends, in metres.
    """


PUSHES = {
    ArticulatedPart.DRAWER: Push(opening=0.15, sideways_offset=0.0),
    ArticulatedPart.DOOR: Push(opening=0.6, sideways_offset=0.05),
}


@every_part
@mujoco_runs_only_in_continuous_integration
def test_the_hand_pushing_the_moving_part_moves_it(builder, scene):
    push = PUSHES[builder.articulated_part]
    scene.set_opening(push.opening)
    table = scene.robot.root
    tool_frame = scene.robot.left_arm.end_effector.tool_frame
    line = builder.sideways_offset + push.sideways_offset
    waypoints = [
        (builder.front_distance - push.standoff, line, push.approach_height),
        (builder.front_distance - push.standoff, line, push.height),
        (builder.front_distance - push.depth, line, push.height),
    ]
    motion = MotionStatechart()
    motion.add_node(
        Sequence(
            [
                CartesianPosition(
                    name=f"waypoint {index}",
                    root_link=scene.world.root,
                    tip_link=tool_frame,
                    goal_point=Point3(x, y, z, reference_frame=table),
                )
                for index, (x, y, z) in enumerate(waypoints)
            ]
        )
    )

    simulated = run_live(scene, motion, cycles=600)

    assert simulated < push.opening / 2
