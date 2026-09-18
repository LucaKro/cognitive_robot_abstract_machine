"""
Tests for the recursive grasp belief and the sampled likelihood it is corrected with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest
from random_events.variable import Continuous
from typing_extensions import Optional

from giskardpy.executor import Executor
from giskardpy.motion_statechart.beliefs.context import BeliefContext
from giskardpy.motion_statechart.beliefs.grasp import GraspBelief, SampledLikelihood
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import (
    CertainPriorError,
    NodeNotBuiltError,
)
from giskardpy.motion_statechart.grasp_likelihood_source import GraspLikelihoodSource
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.world import World

# %% a likelihood the test decides, in place of a raycast


@dataclass(eq=False, repr=False)
class RecordedLikelihoodSource(MotionStatechartNode, GraspLikelihoodSource):
    """
    Publishes the share of hits the test put there, standing in for the raycast a real
    measurement samples.
    """

    measured: float = field(kw_only=True)
    """
    The share of rays that are to hit the body on every cycle.
    """

    sample_size: int = field(default=100, kw_only=True)
    """
    How many rays that share is reported out of.
    """

    _likelihood: Optional[FloatVariable] = field(default=None, init=False, repr=False)
    """
    The variable the share is published to, created while building.
    """

    @property
    def likelihood(self) -> FloatVariable:
        return self._likelihood

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        self._likelihood = FloatVariable(f"{self.unique_name}_recorded_likelihood")
        context.float_variable_data.register_expression(self._likelihood)
        return NodeArtifacts()

    def on_start(self, context: MotionStatechartContext) -> None:
        context.float_variable_data.set_value(self._likelihood, self.measured)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        context.float_variable_data.set_value(self._likelihood, self.measured)
        return None


# %% a gripper opening the test decides, in place of a robot


@dataclass
class RecordedGripperOpening(JointState):
    """
    Answers whether the gripper is open with what the test put there, standing in for
    the joint positions a real one is read from.
    """

    reached: bool = True
    """
    Whether the gripper is to count as open.
    """

    def is_achieved(self) -> bool:
        return self.reached


# %% building and ticking one grasp belief


@dataclass
class TickedBelief:
    """
    One grasp belief that has been built into a statechart and ticked.
    """

    node: GraspBelief
    """
    The node under test.
    """

    context: MotionStatechartContext
    """
    The context it was ticked with, holding the values it published.
    """

    @property
    def published_probability(self) -> float:
        """
        :return: The probability of a grasp the node published on its last cycle.
        """
        return self.context.float_variable_data.get_value(self.node.probability)

    @property
    def published_estimate(self) -> float:
        """
        :return: The log-odds the node published on its last cycle.
        """
        return self.context.float_variable_data.get_value(
            self.node.estimate_variable_of(self.node.grasp)
        )

    @property
    def published_uncertainty(self) -> float:
        """
        :return: How uncertain those log-odds are, as the node published it.
        """
        return self.context.float_variable_data.get_value(
            self.node.uncertainty_variable_of(self.node.grasp)
        )


def grasp_belief(
    measured: float = 0.5,
    sample_size: int = 100,
    gripper_is_open: bool = False,
    **node_arguments,
) -> GraspBelief:
    """
    A grasp belief over a likelihood the test decides.

    :param measured: The share of rays hitting the body on every cycle.
    :param sample_size: How many rays that share is reported out of.
    :param gripper_is_open: Whether the gripper counts as open.
    :param node_arguments: Further arguments for the node, overriding its defaults.
    :return: The node.
    """
    arguments = dict(
        grasp=Continuous("grasp"),
        likelihood_source=RecordedLikelihoodSource(
            measured=measured, sample_size=sample_size
        ),
        gripper_open=RecordedGripperOpening(reached=gripper_is_open),
        prior_uncertainty=4.0,
        forgetting_half_life=0.5,
        drift=0.0,
        false_below=0.2,
    )
    arguments.update(node_arguments)
    return GraspBelief(**arguments)


def build(node: GraspBelief, target_frequency: float = 20) -> Executor:
    """
    Compiles a statechart holding a grasp belief and the source it reads, without
    ticking it.

    :param node: The node under test.
    :param target_frequency: The rate the control loop is configured to run at.
    :return: The executor driving them.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(node.likelihood_source)
    motion_statechart.add_node(node)
    executor = Executor(
        MotionStatechartContext(
            world=World(),
            qp_controller_config=QPControllerConfig(target_frequency=target_frequency),
        )
    )
    executor.compile(motion_statechart=motion_statechart)
    return executor


def tick(
    node: GraspBelief, times: int = 1, target_frequency: float = 20
) -> TickedBelief:
    """
    Compiles a statechart holding a grasp belief and ticks it.

    :param node: The node under test.
    :param times: How many control cycles to run.
    :param target_frequency: The rate the control loop is configured to run at.
    :return: The ticked node and the context holding what it published.
    """
    executor = build(node, target_frequency=target_frequency)
    for _ in range(times):
        executor.tick()
    return TickedBelief(node=node, context=executor.context)


# %% turning a sampled share of hits into a reading


def test_every_ray_hitting_is_a_finite_estimate():
    """
    The log-odds of one are infinite, and every ray hitting is a routine outcome rather
    than an error, so the share is read through the half-count correction.
    """
    sampled = SampledLikelihood(hits=100.0, sample_size=100)

    assert math.isfinite(sampled.log_odds)
    assert sampled.log_odds > 0.0


def test_no_ray_hitting_is_a_finite_estimate():
    sampled = SampledLikelihood(hits=0.0, sample_size=100)

    assert math.isfinite(sampled.log_odds)
    assert sampled.log_odds < 0.0


def test_an_even_split_is_no_evidence_either_way():
    sampled = SampledLikelihood(hits=50.0, sample_size=100)

    assert sampled.log_odds == 0.0


def test_the_same_share_read_from_more_rays_scatters_less():
    """
    How far a reading is trusted follows from how many rays it was taken from, which is
    what makes the measurement noise a measurement rather than a tuned number.
    """
    few = SampledLikelihood(hits=8.0, sample_size=10)
    many = SampledLikelihood(hits=80.0, sample_size=100)

    assert many.variance < few.variance


def test_the_two_halves_of_a_reading_agree_on_the_correction():
    """
    The scatter of a log-odds is stated in the same corrected counts the estimate is, so
    a share of zero has a finite variance too.
    """
    sampled = SampledLikelihood(hits=0.0, sample_size=10)

    assert math.isfinite(sampled.variance)
    assert sampled.variance > 0.0


# %% what the belief publishes


def test_the_prior_is_published_before_anything_is_measured():
    node = grasp_belief(prior_probability=0.25)

    executor = build(node)
    node.on_start(executor.context)

    published = executor.context.float_variable_data.get_value(node.probability)
    assert published == pytest.approx(0.25)


def test_a_measurement_of_mostly_hits_raises_the_published_probability():
    node = grasp_belief(measured=0.95, prior_probability=0.5)

    ticked = tick(node)

    assert ticked.published_probability > 0.5


def test_a_measurement_of_mostly_misses_lowers_the_published_probability():
    node = grasp_belief(measured=0.05, prior_probability=0.5)

    ticked = tick(node)

    assert ticked.published_probability < 0.5


def test_the_published_probability_is_what_the_published_log_odds_mean():
    """
    The two variables describe one quantity, so a consumer reading either gets the same
    belief.
    """
    node = grasp_belief(measured=0.9)

    ticked = tick(node)

    assert ticked.published_probability == pytest.approx(
        1.0 / (1.0 + math.exp(-ticked.published_estimate))
    )


def test_a_measurement_makes_the_belief_less_uncertain():
    node = grasp_belief(measured=0.9, prior_uncertainty=4.0)

    ticked = tick(node)

    assert ticked.published_uncertainty < 4.0


# %% the memory, which is what this adds over the measurement itself


def test_the_same_measurement_seen_twice_is_believed_more_firmly():
    """
    The whole point of filtering a stateless raycast: evidence accumulates, which a
    single measurement cannot express however confident it is.
    """
    after_one = tick(grasp_belief(measured=0.9))
    after_two = tick(grasp_belief(measured=0.9), times=2)

    assert after_two.published_probability > after_one.published_probability


# %% forgetting a grasp the gripper can no longer be holding


def test_an_open_gripper_pulls_the_belief_toward_the_prior():
    """
    A closed gripper says nothing about whether the grasp succeeded, so the estimate is
    left to the readings; an open one rules a grasp out however the rays fall.
    """
    open_gripper = tick(
        grasp_belief(measured=0.5, gripper_is_open=True, prior_probability=0.1), times=5
    )
    closed_gripper = tick(
        grasp_belief(measured=0.5, gripper_is_open=False, prior_probability=0.1),
        times=5,
    )

    assert open_gripper.published_estimate < closed_gripper.published_estimate


def test_a_shorter_half_life_forgets_faster():
    quickly, slowly = (
        tick(
            grasp_belief(
                measured=0.5,
                gripper_is_open=True,
                prior_probability=0.1,
                forgetting_half_life=half_life,
            ),
            times=5,
        )
        for half_life in (0.2, 5.0)
    )

    assert quickly.published_estimate < slowly.published_estimate


def test_the_half_life_is_in_seconds_rather_than_in_cycles():
    """
    Configuring a faster control loop must not silently make the robot forget faster, so
    two and a half cycles of a 50 Hz loop have to decay as much as one 20 Hz cycle does.
    """
    slow, fast = (
        grasp_belief(gripper_is_open=True, forgetting_half_life=0.5) for _ in range(2)
    )
    slow_executor, fast_executor = (
        build(slow, target_frequency=20),
        build(fast, target_frequency=50),
    )

    slow_retention = slow.create_prediction(slow_executor.context).transition[
        (slow.grasp, slow.grasp)
    ]
    fast_retention = fast.create_prediction(fast_executor.context).transition[
        (fast.grasp, fast.grasp)
    ]

    assert slow_retention == pytest.approx(fast_retention ** (50 / 20))


def test_a_cycle_still_adds_the_stated_drift():
    """
    A belief nobody is correcting has to grow less certain, which is what keeps an old
    grasp from staying as convincing as a fresh one.
    """
    without_drift = tick(grasp_belief(measured=0.5, drift=0.0))
    with_drift = tick(grasp_belief(measured=0.5, drift=1.0))

    assert with_drift.published_uncertainty > without_drift.published_uncertainty


def test_the_drift_is_in_seconds_rather_than_in_cycles():
    """
    Stated per cycle, the same configuration would make a 50 Hz loop lose certainty two
    and a half times as fast as a 20 Hz one over the same second.
    """
    slow, fast = (grasp_belief(drift=1.0) for _ in range(2))
    slow_context = build(slow, target_frequency=20).context
    fast_context = build(fast, target_frequency=50).context

    slow_drift = slow.create_prediction(slow_context).process_noise[
        (slow.grasp, slow.grasp)
    ]
    fast_drift = fast.create_prediction(fast_context).process_noise[
        (fast.grasp, fast.grasp)
    ]

    assert fast_drift == pytest.approx(slow_drift * (20 / 50))


# %% what the belief observes


def test_a_confident_belief_observes_true():
    node = grasp_belief(measured=0.98, prior_probability=0.9, true_above=0.8)

    ticked = tick(node, times=5)

    assert ticked.node.observation_state == ObservationStateValues.TRUE


def test_a_belief_of_mostly_misses_observes_false():
    node = grasp_belief(measured=0.02, prior_probability=0.1, false_below=0.2)

    ticked = tick(node, times=5)

    assert ticked.node.observation_state == ObservationStateValues.FALSE


def test_a_belief_between_the_thresholds_observes_unknown():
    """
    A grasp that is neither ruled in nor ruled out is the state the trinary observation
    exists to carry, and the one a boolean predicate cannot express at all.
    """
    node = grasp_belief(
        measured=0.5, prior_probability=0.5, false_below=0.2, true_above=0.8
    )

    ticked = tick(node)

    assert ticked.node.observation_state == ObservationStateValues.UNKNOWN


# %% reading the published variables


def test_the_probability_is_unavailable_before_the_node_is_built():
    node = grasp_belief()

    with pytest.raises(NodeNotBuiltError) as error:
        node.probability
    assert error.value.node is node


def test_the_belief_is_reachable_by_the_quantity_it_is_about():
    """
    A goal scaling its weight by the grasp finds the belief through the statechart's
    beliefs, not through the node that maintains it.
    """
    node = grasp_belief(prior_probability=0.5)

    executor = build(node)

    beliefs = BeliefContext.of(executor.context)
    assert beliefs.require(node.grasp).mean_of(node.grasp) == 0.0


def test_a_prior_that_already_rules_a_grasp_in_is_rejected():
    """
    A certain prior has no log-odds to start from, and no reading could move it, so the
    node says so where it is written rather than failing while the statechart compiles.
    """
    with pytest.raises(CertainPriorError) as error:
        grasp_belief(prior_probability=1.0)
    assert error.value.probability == 1.0


def test_a_prior_that_already_rules_a_grasp_out_is_rejected():
    with pytest.raises(CertainPriorError) as error:
        grasp_belief(prior_probability=0.0)
    assert error.value.probability == 0.0
