"""
Tests for the node that publishes how uncertain an externally observed pose is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import PoseUncertaintyNotBuiltError
from giskardpy.motion_statechart.monitors.uncertainty_monitors import PoseUncertainty
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.pose_covariance_source import PoseCovarianceSource
from semantic_digital_twin.spatial_types import PoseAxis, PoseCovariance
from semantic_digital_twin.world import World

# %% a source whose covariance the test decides


@dataclass
class RecordedPoseCovariance(PoseCovarianceSource):
    """
    Reports one covariance the test put there, standing in for whatever receives poses
    from outside the process.
    """

    covariance: PoseCovariance | None = field(default=None)
    """
    The covariance handed back to the caller.
    """

    @property
    def pose_covariance(self) -> PoseCovariance | None:
        return self.covariance


def covariance_of_total_variance(total_variance: float) -> PoseCovariance:
    """
    A covariance whose six axes sum to the given total.

    :param total_variance: The total the axes should sum to.
    :return: The covariance with that total.
    """
    values = np.zeros((len(PoseAxis), len(PoseAxis)), dtype=np.float64)
    for axis in PoseAxis:
        values[axis, axis] = total_variance / len(PoseAxis)
    return PoseCovariance(values=values)


# %% running a single node


@dataclass
class TickedPoseUncertainty:
    """
    One pose uncertainty node that has been built into a statechart and ticked.
    """

    node: PoseUncertainty
    """
    The node under test.
    """

    context: MotionStatechartContext
    """
    The context it was ticked with, holding the value it published.
    """

    executor: Executor
    """
    The executor driving it, so a test can tick it again.
    """

    @property
    def published_variance(self) -> float:
        """
        :return: The uncertainty the node published on its last tick.
        """
        return self.context.float_variable_data.get_value(self.node.total_variance)


def tick_once(
    world: World, source: PoseCovarianceSource, **node_arguments
) -> TickedPoseUncertainty:
    """
    Builds a statechart holding a single pose uncertainty node and ticks it once.

    :param world: The world the statechart is executed in.
    :param source: The source of the covariance the node publishes.
    :param node_arguments: Further arguments for the node under test.
    :return: The ticked node, its context and its executor.
    """
    node = PoseUncertainty(source=source, **node_arguments)
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(node)
    context = MotionStatechartContext(world=world)
    executor = Executor(context)
    executor.compile(motion_statechart=motion_statechart)
    executor.tick()
    return TickedPoseUncertainty(node=node, context=context, executor=executor)


# %% the published value


def test_the_uncertainty_of_the_observed_pose_is_published_as_a_variable(mini_world):
    """
    The point of the node: the uncertainty of the base pose is readable as a number that
    a constraint or condition can be written against.
    """
    covariance = covariance_of_total_variance(0.75)

    ticked = tick_once(mini_world, RecordedPoseCovariance(covariance=covariance))

    assert ticked.published_variance == covariance.total_variance


def test_a_pose_that_was_never_observed_is_maximally_uncertain(mini_world):
    """
    Zero would read as perfect certainty, so a condition waiting for the uncertainty to
    fall would pass before anything had been measured.
    """
    ticked = tick_once(mini_world, RecordedPoseCovariance())

    assert ticked.published_variance == np.inf


def test_a_caller_may_say_how_uncertain_an_unobserved_pose_is(mini_world):
    """
    A caller that can bound the uncertainty before the first reading may say so, rather
    than being stuck with a value no comparison can narrow.
    """
    ticked = tick_once(
        mini_world, RecordedPoseCovariance(), uncertainty_without_a_reading=12.5
    )

    assert ticked.published_variance == ticked.node.uncertainty_without_a_reading


def test_a_later_observation_replaces_the_published_value(mini_world):
    """
    The variable tracks the source rather than latching the first reading.
    """
    source = RecordedPoseCovariance(covariance=covariance_of_total_variance(0.75))
    ticked = tick_once(mini_world, source)

    source.covariance = covariance_of_total_variance(0.25)
    ticked.executor.tick()

    assert ticked.published_variance == source.covariance.total_variance


def test_the_uncertainty_is_unavailable_before_the_node_is_built():
    node = PoseUncertainty(source=RecordedPoseCovariance())

    with pytest.raises(PoseUncertaintyNotBuiltError) as error:
        node.total_variance

    assert error.value.node_name == node.name


# %% the observation derived from it


def test_the_node_observes_true_once_a_pose_has_been_observed(mini_world):
    ticked = tick_once(
        mini_world,
        RecordedPoseCovariance(covariance=covariance_of_total_variance(0.75)),
    )

    assert ticked.node.observation_state == ObservationStateValues.TRUE


def test_the_node_observes_false_while_no_pose_has_been_observed(mini_world):
    ticked = tick_once(mini_world, RecordedPoseCovariance())

    assert ticked.node.observation_state == ObservationStateValues.FALSE
