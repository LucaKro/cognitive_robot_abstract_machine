"""
Tests for the benchmark scene: the cabinet's moving part is moved by contact alone,
never by giskard commanding its joint.

Skipped where Tracy's description is not installed.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy
import pytest

from ...pytest_environment import runs_in_continuous_integration

from experiments.articulated_manipulation.cabinet_scene import (
    ArticulatedPart,
    CabinetScene,
    CabinetSceneSpecification,
)
from giskardpy.executor import Executor, SteppedSimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cabinet
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose2D,
)
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
def specification(request) -> CabinetSceneSpecification:
    return CabinetSceneSpecification(articulated_part=request.param)


@pytest.fixture
def scene(specification) -> CabinetScene:
    return specification.to_domain_object()


every_part = pytest.mark.parametrize(
    "specification", list(ArticulatedPart), indirect=True, ids=lambda part: part.name
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
        drive(simulation, scene, motion, cycles)
        return simulation.simulator.get_joint_value(scene.mechanism.name.name).result
    finally:
        simulation.stop_simulation()


def drive(
    simulation: MujocoSim, scene: CabinetScene, motion: MotionStatechart, cycles: int
) -> None:
    """
    Run giskard in lockstep with a started stepped simulation.

    :param simulation: The running simulation of the scene.
    :param scene: The scene to run in.
    :param motion: What giskard does.
    :param cycles: How many control cycles to run.
    """
    executor = Executor(
        context=MotionStatechartContext(
            world=scene.world,
            qp_controller_config=QPControllerConfig(target_frequency=CONTROL_FREQUENCY),
        ),
        pacer=SteppedSimulationPacer(simulation),
    )
    executor.compile(motion_statechart=motion)
    for _ in range(cycles):
        executor.tick()
        executor.pacer.sleep()


# %% the scene


@every_part
def test_the_world_specification_holds_tracy_and_the_cabinet(specification):
    world_specification = specification.world_specification()

    [robot] = world_specification.robots
    [cabinet] = world_specification.objects
    assert robot.semantic_annotation_type is Tracy
    assert cabinet.semantic_annotation_type is Cabinet


@every_part
def test_the_cabinet_stands_on_tracys_table(specification, scene):
    assert numpy.allclose(
        scene.robot.root.global_transform.to_np(),
        specification.world_T_table.to_np(),
    )
    cabinet_bottom = scene.cabinet.root.global_transform.to_np()[2, 3] - (
        specification.cabinet_scale.z / 2
    )
    assert cabinet_bottom == pytest.approx(specification.world_T_table.to_np()[2, 3])


def test_the_cabinet_front_lies_at_its_pose_on_the_table():
    specification = CabinetSceneSpecification(
        articulated_part=ArticulatedPart.DRAWER,
        table_T_cabinet_front=Pose2D(x=0.8, y=0.2, yaw=math.pi / 6),
    )

    scene = specification.to_domain_object()

    world_T_cabinet_front = (
        scene.cabinet.root.global_transform
        @ HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-specification.cabinet_scale.x / 2,
            z=-specification.cabinet_scale.z / 2,
        )
    )
    assert numpy.allclose(
        world_T_cabinet_front.to_np(),
        (
            specification.world_T_table
            @ specification.table_T_cabinet_front.to_homogeneous_matrix()
        ).to_np(),
    )


@every_part
def test_every_body_has_its_own_name(scene):
    """
    MuJoCo refuses a model in which two bodies share a name.
    """
    names = [body.name.name for body in scene.world.bodies]
    assert len(set(names)) == len(names)


@every_part
def test_no_controller_commands_the_moving_part(scene):
    assert not scene.mechanism.has_hardware_interface


@every_part
def test_the_moving_part_is_the_only_connection_no_controller_drives(scene):
    assert scene.world.uncontrolled_connections == [scene.mechanism]


@every_part
def test_the_handle_moves_with_the_moving_part(scene):
    assert (
        scene.handle.root.get_first_parent_connection_of_type(ActiveConnection1DOF)
        is scene.mechanism
    )


@every_part
def test_the_mechanisms_axis_turns_by_its_deviation(specification, scene):
    deviation = 0.2
    turned = dataclasses.replace(
        specification, mechanism_axis_deviation=deviation
    ).to_domain_object()

    angle = turned.mechanism.axis.angle_between(scene.mechanism.axis)
    assert float(angle.to_np()[0]) == pytest.approx(deviation)


def test_a_turned_drawer_still_slides_level():
    specification = CabinetSceneSpecification(
        articulated_part=ArticulatedPart.DRAWER, mechanism_axis_deviation=0.2
    )

    axis = specification.to_domain_object().mechanism.axis

    assert axis.to_np()[2] == pytest.approx(0.0)


def test_a_turned_doors_hinge_stays_in_the_cabinets_front():
    specification = CabinetSceneSpecification(
        articulated_part=ArticulatedPart.DOOR, mechanism_axis_deviation=0.2
    )

    axis = specification.to_domain_object().mechanism.axis

    assert axis.to_np()[0] == pytest.approx(0.0)


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

    height: float = 0.12
    """
    How high above the cabinet's bottom the hand pushes, in metres.
    """

    approach_height: float = 0.42
    """
    How high above the cabinet's bottom the hand travels to the point it comes down
    at, in metres.
    """

    standoff: float = 0.4
    """
    How far in front of the cabinet's front the hand comes down, in metres.
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
def test_the_hand_pushing_the_moving_part_moves_it(specification, scene):
    push = PUSHES[specification.articulated_part]
    scene.set_opening(push.opening)
    tool_frame = scene.robot.left_arm.end_effector.tool_frame
    front = -specification.cabinet_scale.x / 2
    bottom = -specification.cabinet_scale.z / 2
    waypoints = [
        (front - push.standoff, push.sideways_offset, bottom + push.approach_height),
        (front - push.standoff, push.sideways_offset, bottom + push.height),
        (front - push.depth, push.sideways_offset, bottom + push.height),
    ]
    motion = MotionStatechart()
    motion.add_node(
        Sequence(
            [
                CartesianPosition(
                    name=f"waypoint {index}",
                    root_link=scene.world.root,
                    tip_link=tool_frame,
                    goal_point=Point3(x, y, z, reference_frame=scene.cabinet.root),
                )
                for index, (x, y, z) in enumerate(waypoints)
            ]
        )
    )

    simulated = run_live(scene, motion, cycles=600)

    assert simulated < push.opening / 2


# %% kept from the controller


@every_part
@mujoco_runs_only_in_continuous_integration
def test_the_controllers_world_and_the_physics_diverge_over_the_moving_part(scene):
    """
    Giskard opens the part in its own world while the untouched part stays closed in
    the physics, and the gap is logged.
    """
    fully_open = scene.mechanism.dof.limits.upper.position
    motion = MotionStatechart()
    motion.add_node(
        JointPositionList(
            name="open without touching",
            goal_state=JointState.from_mapping({scene.mechanism: fully_open}),
        )
    )
    simulation = MujocoSim(world=scene.world, headless=True)
    simulation.start_stepped_simulation()
    try:
        drive(simulation, scene, motion, cycles=200)
        simulated = simulation.simulator.get_joint_value(
            scene.mechanism.name.name
        ).result
    finally:
        simulation.stop_simulation()

    believed = scene.world.state[scene.mechanism.raw_dof.id].position
    [divergence] = simulation.synchronizer.divergence_log[-1].divergences
    assert believed == pytest.approx(fully_open, abs=0.01)
    assert simulated == pytest.approx(0.0, abs=1e-3)
    assert divergence.degree_of_freedom is scene.mechanism.raw_dof
    assert divergence.world_position == believed
    assert divergence.physics_position == pytest.approx(simulated)
