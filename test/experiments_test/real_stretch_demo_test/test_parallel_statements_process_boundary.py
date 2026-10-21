"""
A parallel of two underspecified statements, grounded in one process and across the
process boundary to a standalone controller.

The statements of a parallel are grounded together when it starts, see
:class:`~coraplex.plans.underspecified.UnderspecifiedChildChooser`. In one process the
grounding reads the statechart it runs in; against a controller in its own process it
reads the client's copy, which only learns what the running statechart did through the
child choice exchange.
"""

from dataclasses import fields

from krrood.entity_query_language.factories import a, variable_from
from krrood.entity_query_language.query.match import Match
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
)
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    SetGripperAction,
)
from cramph.composites import Parallel, Sequence, StatementNode
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.world import World

from .conftest import StandaloneControllerProcess

TORSO_STATES = (TorsoState.HIGH, TorsoState.LOW)
"""
The heights the torso statement chooses between, the first ranked first.
"""

GRIPPER_STATES = (GripperState.OPEN, GripperState.CLOSE)
"""
The states the gripper statement chooses between, the first ranked first.
"""

JOINT_GOAL_THRESHOLD = next(
    field.default for field in fields(JointPositionList) if field.name == "threshold"
)
"""
How close the torso and gripper goals bring each joint before they count as reached.
"""

# %% the plan


def torso_statement() -> Match:
    """
    :return: A statement moving the torso to one of :data:`TORSO_STATES`.
    """
    return a(MoveTorsoAction)(torso_state=variable_from(list(TORSO_STATES)))


def gripper_statement(stretch: Stretch) -> Match:
    """
    :return: A statement driving the gripper of `stretch` to one of
        :data:`GRIPPER_STATES`.
    """
    [arm] = stretch.all_arms
    return a(SetGripperAction)(
        gripper=arm.end_effector, motion=variable_from(list(GRIPPER_STATES))
    )


def steps_leaving_torso_and_gripper_off_the_first_ranked_states(
    stretch: Stretch,
) -> list[MoveTorsoAction | SetGripperAction]:
    """
    :return: The steps that bring the torso and the gripper into states other than the
        ones the statements rank first, so reaching those afterwards is the statements'
        doing.
    """
    [arm] = stretch.all_arms
    return [
        MoveTorsoAction(torso_state=TorsoState.MID),
        SetGripperAction(gripper=arm.end_effector, motion=GRIPPER_STATES[1]),
    ]


def run(executor_type: type[PlanExecutor], world: World, plan: Sequence, **options):
    """
    Run `plan` against the Stretch in `world` with an executor of `executor_type`,
    avoiding collisions as the demonstrations do.
    """
    stretch = world.get_semantic_annotations_by_type(Stretch)[0]
    executor = executor_type(
        StatechartContext.with_extensions(
            world, [RobotAccess(stretch), StatementGrounding()]
        ),
        collision_avoidance=True,
        **options,
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(plan)
    executor.compile(statechart)
    try:
        executor.execute()
    finally:
        executor.close()


# %% reading the outcome


def assert_ran_one_child(statement: StatementNode) -> None:
    """
    `statement` succeeded, running the one candidate it was grounded to.
    """
    assert isinstance(statement, StatementNode)
    assert statement.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert len(statement.chosen_children) == 1
    [child] = statement.chosen_children
    assert child.life_cycle_state == LifeCycleValues.SUCCEEDED


def assert_at(world: World, joint_state: JointState) -> None:
    """
    Every joint of `joint_state` stands at its target in `world`, which need not be the
    world `joint_state` was read off.
    """
    for connection, target in zip(joint_state.connections, joint_state.target_values):
        position = world.get_connection_by_name(connection.name).position
        assert abs(position - target) < JOINT_GOAL_THRESHOLD, connection.name


def assert_not_at(world: World, joint_state: JointState) -> None:
    """
    Some joint of `joint_state` stands off its target in `world`, so reaching it later
    is not left over from before.
    """
    assert any(
        abs(world.get_connection_by_name(connection.name).position - target)
        >= JOINT_GOAL_THRESHOLD
        for connection, target in zip(
            joint_state.connections, joint_state.target_values
        )
    )


def torso_goal(stretch: Stretch, torso_state: TorsoState) -> JointState:
    """
    :return: The joint state `torso_state` stands for on `stretch`.
    """
    return stretch.get_torso().get_joint_state_by_type(torso_state)


def gripper_goal(stretch: Stretch, gripper_state: GripperState) -> JointState:
    """
    :return: The joint state `gripper_state` stands for on the gripper of `stretch`.
    """
    [arm] = stretch.all_arms
    return arm.end_effector.get_joint_state_by_type(gripper_state)


def assert_grounded_to_the_first_ranked_states(
    torso: StatementNode, gripper: StatementNode
) -> None:
    """
    Nothing keeps the candidates each statement ranks first from succeeding together,
    so grounding together settles on them.
    """
    [moving_the_torso] = torso.grounded_nodes
    [setting_the_gripper] = gripper.grounded_nodes
    assert moving_the_torso.torso_state == TORSO_STATES[0]
    assert setting_the_gripper.motion == GRIPPER_STATES[0]


# %% the controller


def controller_world(controller: StandaloneControllerProcess) -> World:
    """
    :return: The world `controller` serves, kept synchronized with it.
    """
    world = controller.session.fetch_world()
    WorldSynchronizer(_world=world, node=controller.session.node)
    return world


def run_against(controller: StandaloneControllerProcess, world: World, plan: Sequence):
    """
    Run `plan` against `controller`, the world it serves being `world`.
    """
    run(RealRobotPlanExecutor, world, plan, ros_node=controller.session.node)


def world_after_the_run(controller: StandaloneControllerProcess) -> World:
    """
    :return: A fresh copy of the world `controller` serves, so what it holds was reached
        in the controller's process rather than only in the client's copy.
    """
    return controller.session.fetch_world()


# %% in one process


def test_parallel_statements_are_grounded_together_in_one_process(stretch_world_copy):
    """
    Without a process boundary both statements of the parallel are grounded together
    to the candidates each ranks first, and both run to their goals.
    """
    stretch = stretch_world_copy.get_semantic_annotations_by_type(Stretch)[0]
    parallel = Parallel(nodes=[torso_statement(), gripper_statement(stretch)])
    plan = Sequence(
        nodes=[
            *steps_leaving_torso_and_gripper_off_the_first_ranked_states(stretch),
            parallel,
        ]
    )

    run(KinematicallySimulatedPlanExecutor, stretch_world_copy, plan)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    torso, gripper = parallel.nodes
    assert_ran_one_child(torso)
    assert_ran_one_child(gripper)
    assert_grounded_to_the_first_ranked_states(torso, gripper)
    assert_at(stretch_world_copy, torso_goal(stretch, TORSO_STATES[0]))
    assert_at(stretch_world_copy, gripper_goal(stretch, GRIPPER_STATES[0]))


# %% across the process boundary


def test_a_single_statement_is_grounded_across_the_process_boundary(
    stretch_controller_process: StandaloneControllerProcess,
):
    """
    A statement outside any parallel is grounded on its own by the client, and the
    controller runs the child it was sent to its goal.
    """
    world = controller_world(stretch_controller_process)
    stretch = world.get_semantic_annotations_by_type(Stretch)[0]
    plan = Sequence(
        nodes=[MoveTorsoAction(torso_state=TorsoState.MID), torso_statement()]
    )

    run_against(stretch_controller_process, world, plan)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    statement = plan.nodes[1]
    assert_ran_one_child(statement)
    [moving_the_torso] = statement.grounded_nodes
    assert moving_the_torso.torso_state == TORSO_STATES[0]
    assert_at(
        world_after_the_run(stretch_controller_process),
        torso_goal(stretch, TORSO_STATES[0]),
    )


def test_parallel_statements_starting_the_plan_are_grounded_together_across_the_process_boundary(
    stretch_controller_process: StandaloneControllerProcess,
):
    """
    The parallel starts with the statechart, so both statements ask the client for
    their child on the controller's first tick; the client grounds them together on its
    copy of the statechart, and the controller runs both children to their goals.
    """
    world = controller_world(stretch_controller_process)
    stretch = world.get_semantic_annotations_by_type(Stretch)[0]
    assert_not_at(world, torso_goal(stretch, TORSO_STATES[0]))
    assert_not_at(world, gripper_goal(stretch, GRIPPER_STATES[0]))
    parallel = Parallel(nodes=[torso_statement(), gripper_statement(stretch)])
    plan = Sequence(nodes=[parallel])

    run_against(stretch_controller_process, world, plan)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    torso, gripper = parallel.nodes
    assert_ran_one_child(torso)
    assert_ran_one_child(gripper)
    assert_grounded_to_the_first_ranked_states(torso, gripper)
    world_after = world_after_the_run(stretch_controller_process)
    assert_at(world_after, torso_goal(stretch, TORSO_STATES[0]))
    assert_at(world_after, gripper_goal(stretch, GRIPPER_STATES[0]))


def test_parallel_statements_after_earlier_steps_are_grounded_together_across_the_process_boundary(
    stretch_controller_process: StandaloneControllerProcess,
):
    """
    As in a plan, the parallel only starts once the steps before it are done, so the
    statements ask for their child well after the statechart started.
    """
    world = controller_world(stretch_controller_process)
    stretch = world.get_semantic_annotations_by_type(Stretch)[0]
    parallel = Parallel(nodes=[torso_statement(), gripper_statement(stretch)])
    plan = Sequence(
        nodes=[
            *steps_leaving_torso_and_gripper_off_the_first_ranked_states(stretch),
            parallel,
        ]
    )

    run_against(stretch_controller_process, world, plan)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    torso, gripper = parallel.nodes
    assert_ran_one_child(torso)
    assert_ran_one_child(gripper)
    assert_grounded_to_the_first_ranked_states(torso, gripper)
    world_after = world_after_the_run(stretch_controller_process)
    assert_at(world_after, torso_goal(stretch, TORSO_STATES[0]))
    assert_at(world_after, gripper_goal(stretch, GRIPPER_STATES[0]))


def test_a_statement_opening_a_sequence_beside_another_is_grounded_across_the_process_boundary(
    stretch_controller_process: StandaloneControllerProcess,
):
    """
    As in the GARMI demonstration, one statement of the parallel opens a sequence with a
    step after it, beside a statement of its own: both are grounded together, and the
    controller runs the step after the statement once its child is done.
    """
    world = controller_world(stretch_controller_process)
    stretch = world.get_semantic_annotations_by_type(Stretch)[0]
    [arm] = stretch.all_arms
    step_after_the_statement = SetGripperAction(
        gripper=arm.end_effector, motion=GRIPPER_STATES[1]
    )
    gripper_steps = Sequence(
        nodes=[gripper_statement(stretch), step_after_the_statement]
    )
    parallel = Parallel(nodes=[gripper_steps, torso_statement()])
    plan = Sequence(
        nodes=[
            *steps_leaving_torso_and_gripper_off_the_first_ranked_states(stretch),
            parallel,
        ]
    )

    run_against(stretch_controller_process, world, plan)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    gripper = gripper_steps.nodes[0]
    torso = parallel.nodes[1]
    assert_ran_one_child(torso)
    assert_ran_one_child(gripper)
    assert_grounded_to_the_first_ranked_states(torso, gripper)
    assert step_after_the_statement.life_cycle_state == LifeCycleValues.SUCCEEDED
    world_after = world_after_the_run(stretch_controller_process)
    assert_at(world_after, torso_goal(stretch, TORSO_STATES[0]))
    assert_at(world_after, gripper_goal(stretch, GRIPPER_STATES[1]))
