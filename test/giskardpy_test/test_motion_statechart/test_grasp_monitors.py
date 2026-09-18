from dataclasses import dataclass

import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import GraspLikelihoodNotBuiltError
from giskardpy.motion_statechart.monitors.grasp_monitors import GraspLikelihood
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper

# %% running a single node


@dataclass
class TickedGraspLikelihood:
    """
    One grasp likelihood node that has been built into a statechart and ticked once.
    """

    node: GraspLikelihood
    """
    The node under test.
    """

    context: MotionStatechartContext
    """
    The context it was ticked with, holding the value it published.
    """

    @property
    def measured_likelihood(self) -> float:
        """
        :return: The likelihood the node published on its last tick.
        """
        return self.context.float_variable_data.get_value(self.node.likelihood)


def tick_once(scenario, false_below: float, true_above: float) -> TickedGraspLikelihood:
    """
    Builds a statechart holding a single grasp likelihood node and ticks it once.

    :param scenario: The world, body and gripper to measure.
    :param false_below: The likelihood under which the node observes false.
    :param true_above: The likelihood over which the node observes true.
    :return: The ticked node and its context.
    """
    node = GraspLikelihood(
        body=scenario.body,
        gripper=scenario.gripper,
        false_below=false_below,
        true_above=true_above,
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(node)
    context = MotionStatechartContext(world=scenario.world)
    executor = Executor(context)
    executor.compile(motion_statechart=motion_statechart)
    executor.tick()
    return TickedGraspLikelihood(node=node, context=context)


# %% the continuous value


def test_the_measured_likelihood_is_published_as_a_variable(body_between_fingers):
    """
    The point of the node: the share of rays that hit is readable as a number, not only
    as a yes or no.
    """
    ticked = tick_once(body_between_fingers, false_below=0.05, true_above=0.9)
    assert ticked.measured_likelihood > 0


def test_an_empty_gripper_measures_exactly_zero(body_between_fingers):
    body_between_fingers.move_body_away()
    ticked = tick_once(body_between_fingers, false_below=0.05, true_above=0.9)
    assert ticked.measured_likelihood == 0


def test_the_likelihood_is_unavailable_before_the_node_is_built(body_between_fingers):
    node = GraspLikelihood(
        body=body_between_fingers.body,
        gripper=body_between_fingers.gripper,
        false_below=0.05,
    )
    with pytest.raises(GraspLikelihoodNotBuiltError) as error:
        node.likelihood
    assert error.value.node_name == node.name


# %% the trinary observation derived from it


def test_a_likelihood_over_the_true_threshold_is_observed_as_true(body_between_fingers):
    """
    The band is placed under the likelihood this scenario actually produces, so the
    observation is decided by the measurement rather than by where the band sits.
    """
    measured = is_body_in_gripper(
        body_between_fingers.body, body_between_fingers.gripper
    )
    assert measured > 0, "the fixture is supposed to put the body between the fingers"
    ticked = tick_once(
        body_between_fingers, false_below=measured / 3, true_above=measured / 2
    )
    assert ticked.node.observation_state == ObservationStateValues.TRUE


def test_a_likelihood_under_the_false_threshold_is_observed_as_false(
    body_between_fingers,
):
    body_between_fingers.move_body_away()
    ticked = tick_once(body_between_fingers, false_below=0.05, true_above=0.9)
    assert ticked.node.observation_state == ObservationStateValues.FALSE


def test_a_likelihood_inside_the_band_is_observed_as_unknown(body_between_fingers):
    """
    A measurement that is neither confident enough to accept nor low enough to rule out
    has no answer yet, which is the state a thresholded boolean cannot express.
    """
    measured = is_body_in_gripper(
        body_between_fingers.body, body_between_fingers.gripper
    )
    assert measured > 0, "the fixture is supposed to put the body between the fingers"
    ticked = tick_once(
        body_between_fingers, false_below=measured / 2, true_above=measured * 2
    )
    assert ticked.node.observation_state == ObservationStateValues.UNKNOWN


def test_the_observation_stays_one_of_the_three_truth_values(body_between_fingers):
    """
    The statechart reads an observation back through ``ObservationStateValues``, and
    every life cycle verdict compares against those three constants exactly.

    A continuous observation would raise here instead, which is why the continuous value
    is carried by a variable and only the trinary view of it is observed.
    """
    ticked = tick_once(body_between_fingers, false_below=0.05, true_above=0.9)
    assert ticked.node.observation_state in set(ObservationStateValues)
