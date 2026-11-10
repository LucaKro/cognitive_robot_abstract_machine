"""
Coverage for the scaffolding demonstrations share.

The pieces exercised here are the ones a second demonstration would otherwise re-derive:
which world a run acts on, whether its scene is merged into it, and who owns the ROS
context. None of it needs a controller; a real run is handed the world a controller
would serve.
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from uuid import UUID

import numpy as np
import pytest
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from typing_extensions import Iterator

from coraplex.demonstrations import RobotDemonstration, RobotDemonstrationRosSession
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body
from cramph.threaded_nodes import FunctionCall

from ..plan_running import robot_extensions
from cramph.context import ContextExtension, StatechartContext
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    PhysicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
)
from coraplex.plans.failures import MotionExceededSimulationTimeLimit
from cramph.statechart import Statechart
from ..pytest_environment import runs_in_continuous_integration


class PlanDeliberatelyFailed(Exception):
    """
    Raised by a demonstration whose plan is meant to fail.
    """


@dataclass
class RecordingRobotSpecification(RobotSpecification):
    """
    A robot that records the worlds it is put into instead of being put into them.
    """

    spawned_into: list[World] = field(default_factory=list)
    """
    The worlds the robot was put into, in order.
    """

    def spawn(self, world: World) -> None:
        self.spawned_into.append(world)


@dataclass(kw_only=True)
class RecordingDemonstration(RobotDemonstration):
    """
    A demonstration that records which of its hooks ran and in which environment.
    """

    world: World | None = None
    """
    World a simulated run is handed instead of one it builds itself, and ``None`` to
    build it.
    """

    robot: RecordingRobotSpecification = field(
        default_factory=lambda: RecordingRobotSpecification(
            semantic_annotation_type=MinimalRobot
        )
    )
    """
    The robot a simulated run puts into the scene it builds.
    """

    scene_already_populated: bool = False
    """
    What :meth:`is_scene_populated` reports.
    """

    fail_the_plan: bool = False
    """
    Whether the plan raises :class:`PlanDeliberatelyFailed` instead of recording.
    """

    populated_scene_roots: list[UUID] = field(default_factory=list)
    """
    The ids of the roots of the scenes that were built, in order.
    """

    tear_down_calls: int = 0
    """
    How often the ROS session was released.
    """

    observed_executor: PlanExecutor | None = field(default=None)
    """
    The executor the plan was run with.
    """

    segmenting_events: bool = False
    """
    Whether the events of the run are being segmented right now.
    """

    observed_segmenting_events: bool | None = field(default=None)
    """
    Whether the events of the run were being segmented while the plan ran.
    """

    simulation_running_during_the_plan: bool | None = field(default=None)
    """
    Whether :attr:`simulation` was running while the plan ran.
    """

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification()

    def robot_specification(self) -> RobotSpecification:
        return self.robot

    def build_simulated_world(self) -> World:
        if self.world is not None:
            return self.world
        return super().build_simulated_world()

    def is_scene_populated(self, world: World) -> bool:
        return self.scene_already_populated

    def populate_scene(self, world: World) -> None:
        self.populated_scene_roots.append(world.root.id)

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return robot_extensions(world.get_semantic_annotations_by_type(MinimalRobot)[0])

    def build_statechart(self, context: StatechartContext) -> Statechart:
        statechart = Statechart(context=context)
        statechart.add_node(FunctionCall(function=lambda: self.run_plan_body(context)))
        return statechart

    @contextmanager
    def segment_events(self, world: World) -> Iterator[None]:
        self.segmenting_events = True
        yield
        self.segmenting_events = False

    def run_plan_body(self, context: StatechartContext) -> None:
        """
        Record how the plan is executed, or fail if this demonstration is meant to.

        :param context: The context the plan runs in.
        """
        if self.fail_the_plan:
            raise PlanDeliberatelyFailed()
        self.observed_segmenting_events = self.segmenting_events
        self.simulation_running_during_the_plan = (
            self.simulation is not None and self.simulation.is_running()
        )

    def create_executor(self, world: World) -> PlanExecutor:
        """
        Differs from :meth:`RobotDemonstration.create_executor` in also recording the
        executor in :attr:`observed_executor`.
        """
        self.observed_executor = super().create_executor(world)
        return self.observed_executor

    def tear_down(self) -> None:
        self.tear_down_calls += 1
        super().tear_down()


class HingedSceneBody(StrEnum):
    """
    The bodies of the scene of a :class:`HingedSceneDemonstration`.
    """

    DOOR = "door"
    """
    The door, which turns on a hinge.
    """

    BOX = "box"
    """
    The box, which lies freely in the scene.
    """


@dataclass(kw_only=True)
class HingedSceneDemonstration(RecordingDemonstration):
    """
    A demonstration whose scene holds a door on a hinge, opened by :attr:`door_opening`,
    and a box lying freely next to it.
    """

    door_opening: float = 0.5
    """
    The angle the door is opened to on its hinge, in radians.
    """

    hinge_distance: float = 1.0
    """
    How far the hinge stands from the root of the scene along its x axis, in meters.
    """

    def populate_scene(self, world: World) -> None:
        """
        Differs from :meth:`RecordingDemonstration.populate_scene` in also adding the
        opened door and the box.
        """
        super().populate_scene(world)
        door = Body(name=PrefixedName(HingedSceneBody.DOOR))
        box = Body(name=PrefixedName(HingedSceneBody.BOX))
        with world.modify_world():
            hinge = RevoluteConnection.create_with_dofs(
                world,
                world.root,
                door,
                axis=Vector3.Z(reference_frame=world.root),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=self.hinge_distance, reference_frame=world.root
                ),
            )
            world.add_connection(hinge)
            world.add_connection(
                Connection6DoF.create_with_dofs(world, world.root, box)
            )
        hinge.position = self.door_opening


# %% world acquisition


def test_simulated_run_still_builds_its_own_world(cylinder_bot_world):
    """
    A simulated run builds its own world rather than fetching one from a controller,
    even though it starts a ROS session of its own to visualize that world.
    """
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot
    )

    acquired_world = demonstration.acquire_world()

    assert acquired_world is cylinder_bot_world
    assert demonstration.ros_session is not None
    assert demonstration.ros_node is not None

    demonstration.tear_down()


def test_simulated_run_puts_the_robot_into_the_scene_it_builds():
    demonstration = RecordingDemonstration(used_robot=MinimalRobot)

    world = demonstration.build_simulated_world()

    assert demonstration.populated_scene_roots == [world.root.id]
    assert demonstration.robot.spawned_into == [world]


# %% scene population


def serve_from_the_controller(monkeypatch: pytest.MonkeyPatch, world: World) -> None:
    """
    Make every fetch of a real run return `world`, as if a controller served it.
    """
    monkeypatch.setattr(
        RobotDemonstrationRosSession,
        "fetch_world",
        lambda session, timeout_seconds=None: world,
    )


def test_real_run_merges_the_scene_into_a_world_that_does_not_have_it(
    cylinder_bot_world, monkeypatch
):
    serve_from_the_controller(monkeypatch, cylinder_bot_world)
    demonstration = RecordingDemonstration(
        used_robot=MinimalRobot, executor_type=RealRobotPlanExecutor
    )
    controller_root = cylinder_bot_world.root

    try:
        acquired_world = demonstration.acquire_world()
    finally:
        demonstration.tear_down()

    [scene_root_id] = demonstration.populated_scene_roots
    scene_root = cylinder_bot_world.get_kinematic_structure_entity_by_id(scene_root_id)
    assert acquired_world is cylinder_bot_world
    assert cylinder_bot_world.root is controller_root
    assert scene_root.parent_connection.parent is controller_root
    assert demonstration.robot.spawned_into == []


def test_real_run_does_not_merge_the_scene_into_a_world_that_has_it(
    cylinder_bot_world, monkeypatch
):
    """
    A run against a controller receives a world that may already hold this
    demonstration's scene, and must not merge in a second copy of it.
    """
    serve_from_the_controller(monkeypatch, cylinder_bot_world)
    demonstration = RecordingDemonstration(
        used_robot=MinimalRobot,
        executor_type=RealRobotPlanExecutor,
        scene_already_populated=True,
    )
    bodies_before = len(cylinder_bot_world.kinematic_structure_entities)

    try:
        demonstration.acquire_world()
    finally:
        demonstration.tear_down()

    assert demonstration.populated_scene_roots == []
    assert len(cylinder_bot_world.kinematic_structure_entities) == bodies_before


# %% scene joints


@pytest.fixture
def scene_with_movable_joints() -> World:
    """
    The scene of a :class:`HingedSceneDemonstration`, with the door on its hinge.
    """
    return HingedSceneDemonstration(used_robot=MinimalRobot).build_scene()


@pytest.fixture
def scene_with_fixed_joints() -> World:
    """
    The scene of a :class:`HingedSceneDemonstration` that does not let its joints move.
    """
    return HingedSceneDemonstration(
        used_robot=MinimalRobot, scene_joints_movable=False
    ).build_scene()


def test_the_joints_of_the_scene_move_by_default(
    scene_with_movable_joints: World,
) -> None:
    """
    A demonstration that does not ask for fixed joints acts on the scene as its
    specification builds it.
    """
    door = scene_with_movable_joints.get_body_by_name(HingedSceneBody.DOOR)

    assert type(door.parent_connection) is RevoluteConnection


def test_the_joints_of_a_scene_that_must_not_move_become_fixed_connections(
    scene_with_fixed_joints: World,
) -> None:
    """
    Neither the plan nor physics can swing the door once its hinge is replaced.
    """
    door = scene_with_fixed_joints.get_body_by_name(HingedSceneBody.DOOR)

    assert type(door.parent_connection) is FixedConnection


def test_a_replaced_joint_holds_its_child_where_the_joint_had_it(
    scene_with_movable_joints: World, scene_with_fixed_joints: World
) -> None:
    """
    The door stays where its opened hinge held it, offset from the root as the hinge
    is, rather than falling back to where a closed hinge would hold it.
    """
    movable_door = scene_with_movable_joints.get_body_by_name(HingedSceneBody.DOOR)
    fixed_door = scene_with_fixed_joints.get_body_by_name(HingedSceneBody.DOOR)

    np.testing.assert_allclose(
        fixed_door.global_transform.to_np(), movable_door.global_transform.to_np()
    )


def test_a_replaced_joint_keeps_its_name(
    scene_with_movable_joints: World, scene_with_fixed_joints: World
) -> None:
    """
    Code that looks a connection up by name finds the fixed connection where the
    joint was.
    """
    movable_door = scene_with_movable_joints.get_body_by_name(HingedSceneBody.DOOR)
    fixed_door = scene_with_fixed_joints.get_body_by_name(HingedSceneBody.DOOR)

    assert fixed_door.parent_connection.name == movable_door.parent_connection.name


def test_a_free_body_stays_free_when_the_joints_of_the_scene_are_replaced(
    scene_with_fixed_joints: World,
) -> None:
    """
    An object attached by a free connection, such as one the plan picks up, can still
    be moved.
    """
    box = scene_with_fixed_joints.get_body_by_name(HingedSceneBody.BOX)

    assert type(box.parent_connection) is Connection6DoF


# %% execution environment


def test_plan_runs_in_the_demonstrations_execution_environment(cylinder_bot_world):
    """
    The plan is what the executor and collision avoidance settings exist for, so they
    have to be in force while it runs.
    """
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world,
        used_robot=MinimalRobot,
        executor_type=KinematicallySimulatedPlanExecutor,
        collision_avoidance=True,
    )

    demonstration.run()

    assert type(demonstration.observed_executor) is KinematicallySimulatedPlanExecutor
    assert demonstration.observed_executor.collision_avoidance is True


def test_run_returns_the_world_it_acted_on(cylinder_bot_world):
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot
    )

    assert demonstration.run() is cylinder_bot_world


def test_a_simulated_plan_is_given_up_on_at_the_time_limit_of_its_demonstration(
    cylinder_bot_world,
) -> None:
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world,
        used_robot=MinimalRobot,
        simulation_time_per_action=timedelta(0),
    )

    with pytest.raises(MotionExceededSimulationTimeLimit) as error:
        demonstration.run()

    assert error.value.time_limit == demonstration.simulation_time_per_action


# %% physical simulation


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_a_physically_simulated_run_performs_its_plan_in_running_physics(
    cylinder_bot_world,
) -> None:
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world,
        used_robot=MinimalRobot,
        executor_type=PhysicallySimulatedPlanExecutor,
        headless=True,
    )

    demonstration.run()

    assert type(demonstration.observed_executor) is PhysicallySimulatedPlanExecutor
    assert demonstration.observed_executor.simulation.world is cylinder_bot_world
    assert demonstration.simulation_running_during_the_plan is True


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_physics_of_a_run_stops_when_its_plan_fails(cylinder_bot_world) -> None:
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world,
        used_robot=MinimalRobot,
        executor_type=PhysicallySimulatedPlanExecutor,
        headless=True,
        fail_the_plan=True,
    )

    with pytest.raises(PlanDeliberatelyFailed):
        demonstration.run()

    assert demonstration.simulation is None


# %% event segmentation


def test_the_events_of_the_plan_are_segmented_while_it_runs(cylinder_bot_world):
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot
    )

    demonstration.run()

    assert demonstration.observed_segmenting_events is True
    assert demonstration.segmenting_events is False


def test_the_events_of_the_plan_are_not_segmented_when_switched_off(
    cylinder_bot_world,
):
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot, event_segmentation=False
    )

    demonstration.run()

    assert demonstration.observed_segmenting_events is False


# %% debugging


def test_a_demonstration_runs_without_debugging_by_default(cylinder_bot_world):
    """
    Debugging publishes every copy of the world a candidate is tried in, which a run
    only pays for when someone is watching it.
    """
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot
    )

    demonstration.run()

    assert not demonstration.observed_executor.debug


def test_a_demonstration_debugs_its_plan_when_asked_to(cylinder_bot_world):
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot, debug=True
    )

    demonstration.run()

    assert demonstration.observed_executor.debug


# %% tear down


def test_tear_down_runs_when_the_plan_fails(cylinder_bot_world):
    """
    A demonstration that dies mid-plan still has to release what it acquired, or a real
    run leaves its ROS session behind.
    """
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot, fail_the_plan=True
    )

    with pytest.raises(PlanDeliberatelyFailed):
        demonstration.run()

    assert demonstration.tear_down_calls == 1


# %% ros context ownership


def test_session_releases_a_context_it_started():
    """
    A demonstration run on its own owns the ROS context and must give it back, so the
    process it ran in is left as it was found.
    """
    assert not rclpy.ok(), "another test left a ROS context running"
    session = RobotDemonstrationRosSession.start("context_ownership_probe")

    assert session.owns_context

    session.stop()
    assert not rclpy.ok()


def test_session_leaves_a_context_somebody_else_started(rclpy_node):
    """
    Inside a process that already has a ROS context -- a test, or a larger application
    embedding the demonstration -- the session must not shut that context down.
    """
    session = RobotDemonstrationRosSession.start("context_borrowing_probe")

    assert not session.owns_context

    session.stop()
    assert rclpy.ok()


# %% callbacks of the session's node


def test_a_callback_of_another_group_runs_while_one_is_busy(rclpy_node) -> None:
    """
    A client announces itself to its controller from a callback group of its own, which
    has to be served while the node is busy choosing what the plan does next.
    """
    session = RobotDemonstrationRosSession.start("busy_callback_probe")
    busy_callback_started = threading.Event()
    other_callback_ran_meanwhile = threading.Event()
    busy_callback_outcomes: list[bool] = []

    def stay_busy_until_the_other_callback_ran() -> None:
        busy_callback_started.set()
        busy_callback_outcomes.append(other_callback_ran_meanwhile.wait(timeout=2.0))

    def run_while_the_other_callback_is_busy() -> None:
        if busy_callback_started.is_set() and not busy_callback_outcomes:
            other_callback_ran_meanwhile.set()

    session.node.create_timer(
        0.01,
        stay_busy_until_the_other_callback_ran,
        callback_group=MutuallyExclusiveCallbackGroup(),
    )
    session.node.create_timer(
        0.01,
        run_while_the_other_callback_is_busy,
        callback_group=MutuallyExclusiveCallbackGroup(),
    )
    busy_callback_started.wait(timeout=5.0)
    other_callback_ran_meanwhile.wait(timeout=5.0)
    session.stop()

    assert busy_callback_outcomes[0]
