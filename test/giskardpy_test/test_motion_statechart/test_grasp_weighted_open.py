"""
Tests for weighing the Open goal's grip on the grasped part by the grasp belief.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.beliefs.grasp_weighted_tasks import (
    GraspWeightedCartesianOrientation,
    GraspWeightedCartesianPose,
    GraspWeightedCartesianPosition,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.exceptions import NodeNotBuiltError
from giskardpy.motion_statechart.goals.open_close import Close, Open
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from krrood.symbolic_math.symbolic_math import VariableParameters
from semantic_digital_twin.world import World

from .test_grasp_belief import grasp_belief

# %% expanding an Open goal on a world of its own


def expanded_nodes(goal: Open, world: World) -> list:
    """
    Expands a goal on a statechart of its own and returns the children it built.

    :param goal: The goal under test.
    :param world: The world it is expanded against.
    :return: The children the goal built.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(goal)
    goal.expand(MotionStatechartContext(world=world))
    return goal.nodes


def open_goal(world: World, **goal_arguments) -> Open:
    """
    An Open goal over the one mechanism the prismatic world has.

    :param world: The world holding the mechanism.
    :param goal_arguments: Further arguments for the goal, overriding its defaults.
    :return: The goal.
    """
    return Open(
        tip_link=world.root,
        environment_link=world.get_body_by_name("robot"),
        **goal_arguments,
    )


def hold_handle_task(goal: Open, world: World) -> CartesianPose:
    """
    :param goal: The goal under test.
    :param world: The world it is expanded against.
    :return: The child keeping the end effector on the grasped part.
    """
    return next(
        node for node in expanded_nodes(goal, world) if isinstance(node, CartesianPose)
    )


def grip_halves(goal: Open, world: World) -> list:
    """
    :param goal: The goal under test.
    :param world: The world it is expanded against.
    :return: The position and orientation tasks the grip is built from.
    """
    task = hold_handle_task(goal, world)
    task.expand(MotionStatechartContext(world=world))
    return task.nodes


def evaluate(expression, context: MotionStatechartContext) -> float:
    """
    :param expression: The symbolic expression to read.
    :param context: The context holding the values it is evaluated against.
    :return: What that expression comes to this cycle.
    """
    compiled = expression.compile(
        parameters=VariableParameters.from_lists(context.float_variable_data.variables)
    )
    return np.asarray(compiled(context.float_variable_data.data)).item()


# %% what the grip is weighted by


def test_the_grip_is_weighted_by_the_constant_when_no_belief_is_given(prismatic_bot):
    """
    A caller that names no belief gets the grip it always got, weighted by
    :attr:`Open.grasp_weight` alone and by nothing that can change while the motion
    runs.
    """
    goal = open_goal(prismatic_bot)

    assert type(hold_handle_task(goal, prismatic_bot)) is CartesianPose
    assert [half.constraint_weight for half in grip_halves(goal, prismatic_bot)] == [
        DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
    ]


def test_the_grip_follows_the_belief_when_one_is_given(prismatic_bot):
    """
    A caller that names a belief gets a grip whose weight the belief scales, rather than
    a second goal to keep in step with it.
    """
    belief = grasp_belief()

    task = hold_handle_task(
        open_goal(prismatic_bot, grasp_belief=belief), prismatic_bot
    )

    assert type(task) is GraspWeightedCartesianPose
    assert task.grasp_belief is belief


def test_closing_weighs_its_grip_by_the_belief_the_same_way(prismatic_bot):
    """
    Closing is opening against the other limit, so it inherits the grip unchanged.
    """
    belief = grasp_belief()
    goal = Close(
        tip_link=prismatic_bot.root,
        environment_link=prismatic_bot.get_body_by_name("robot"),
        grasp_belief=belief,
    )

    task = hold_handle_task(goal, prismatic_bot)

    assert type(task) is GraspWeightedCartesianPose
    assert task.grasp_belief is belief


def test_the_mechanism_is_not_weighted_by_the_belief(prismatic_bot):
    """
    Only the grip follows the belief.

    Driving the mechanism is a separate goal with a weight of its own, and a caller who
    wants it to yield says so through :attr:`Open.mechanism_weight`.
    """
    goal = open_goal(prismatic_bot, grasp_belief=grasp_belief())

    mechanism_task = next(
        node
        for node in expanded_nodes(goal, prismatic_bot)
        if isinstance(node, JointPositionList)
    )

    assert mechanism_task.constraint_weight == mechanism_task.weight


# %% both halves of the grip follow it


def test_both_halves_of_the_grip_follow_the_same_belief(prismatic_bot):
    """
    A grip is a position and an orientation, and letting go of one while holding the
    other is not a grip at all, so both halves read the same belief.
    """
    belief = grasp_belief()
    goal = open_goal(prismatic_bot, grasp_belief=belief)

    halves = grip_halves(goal, prismatic_bot)

    assert [type(half) for half in halves] == [
        GraspWeightedCartesianPosition,
        GraspWeightedCartesianOrientation,
    ]
    assert [half.grasp_belief for half in halves] == [belief, belief]


# %% the belief is what orders the build


def test_the_grip_declares_the_belief_a_prerequisite(prismatic_bot):
    """
    The probability the weight reads is created while the belief builds, so the belief
    has to build first.

    Declaring it a prerequisite is what says so.
    """
    belief = grasp_belief()
    goal = open_goal(prismatic_bot, grasp_belief=belief)

    halves = grip_halves(goal, prismatic_bot)

    assert [half.prerequisite_nodes for half in halves] == [[belief], [belief]]


def test_the_weight_is_unreadable_before_the_belief_is_built(prismatic_bot):
    """
    Reading the weight before the belief has published anything cannot answer, and says
    so rather than reading a variable that does not exist yet.
    """
    goal = open_goal(prismatic_bot, grasp_belief=grasp_belief())

    halves = grip_halves(goal, prismatic_bot)

    for half in halves:
        with pytest.raises(NodeNotBuiltError):
            half.constraint_weight


# %% what the solver is actually handed


@dataclass
class TickedGrip:
    """
    An Open goal whose grip follows a belief, compiled and ticked once.
    """

    halves: list
    """
    The position and orientation tasks the grip is built from.
    """

    constraint_weights: list
    """
    The quadratic weights each half put into the program, in the order of
    :attr:`halves`.
    """

    context: MotionStatechartContext
    """
    The context holding what the belief published on that cycle.
    """

    @property
    def probability(self) -> float:
        """
        :return: How likely the belief held the grasp to be on that cycle.
        """
        return self.context.float_variable_data.get_value(
            self.halves[0].grasp_belief.probability
        )

    @property
    def task_weights(self) -> list:
        """
        :return: What each half's weight comes to this cycle, before the life cycle
            gates it.
        """
        return [evaluate(half.constraint_weight, self.context) for half in self.halves]

    def variables_read_by(self, half_index: int) -> set:
        """
        :param half_index: Which of :attr:`halves` to look at.
        :return: Every variable the program's weights for that half depend on.
        """
        return {
            variable
            for weight in self.constraint_weights[half_index]
            for variable in weight.free_variables()
        }


def ticked_grip(world: World, **belief_arguments) -> TickedGrip:
    """
    Compiles an Open goal whose grip follows a belief, together with that belief and the
    likelihood it reads, and ticks it once.

    :param world: The world holding the mechanism.
    :param belief_arguments: Arguments for the belief, overriding its defaults.
    :return: The grip, the weights its constraints carry, and the context.
    """
    belief = grasp_belief(**belief_arguments)
    goal = open_goal(world, grasp_belief=belief)
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(belief.likelihood_source)
    motion_statechart.add_node(belief)
    motion_statechart.add_node(goal)
    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(motion_statechart=motion_statechart)
    executor.tick()

    halves = next(node for node in goal.nodes if isinstance(node, CartesianPose)).nodes
    constraints = motion_statechart.combine_constraint_collections_of_nodes()
    return TickedGrip(
        halves=halves,
        constraint_weights=[
            [
                constraint.quadratic_weight
                for constraint in constraints.equality_constraints
                if constraint.name.startswith(half.unique_name)
            ]
            for half in halves
        ],
        context=executor.context,
    )


def test_the_weight_the_solver_gets_is_the_constant_times_the_probability(
    prismatic_bot,
):
    """
    The grip is worth what it would have been worth, discounted by how likely the grasp
    is, so the whole range between held and ruled out is representable rather than only
    the two ends.
    """
    grip = ticked_grip(prismatic_bot)
    expected = grip.halves[0].weight * grip.probability

    assert grip.task_weights == pytest.approx([expected, expected])


def test_a_grasp_ruled_out_leaves_the_grip_carrying_less_than_one_believed_in(
    prismatic_bot,
):
    """
    The point of the whole change: a grip the belief has given up on costs the solver
    less to abandon than one it is still convinced by, so the motion gives way on its
    own.
    """
    no_rays_hit = ticked_grip(prismatic_bot, measured=0.0)
    every_ray_hits = ticked_grip(prismatic_bot, measured=1.0)

    assert max(no_rays_hit.task_weights) < min(every_ray_hits.task_weights)


def test_the_program_weighs_the_grip_by_the_belief_rather_than_the_task_alone(
    prismatic_bot,
):
    """
    A weight the task computes and the program never reads would change nothing, so the
    constraints the grip contributes have to depend on the belief's probability
    themselves — and both halves separately, since half a grip held unconditionally is
    still a grip held unconditionally.
    """
    grip = ticked_grip(prismatic_bot)
    probability = grip.halves[0].grasp_belief.probability

    assert all(weights for weights in grip.constraint_weights)
    for half_index in range(len(grip.halves)):
        assert probability in grip.variables_read_by(half_index)
