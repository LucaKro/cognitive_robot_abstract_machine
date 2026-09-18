from __future__ import annotations

import math
from dataclasses import dataclass, field

from random_events.variable import Continuous
from scipy.special import expit
from typing_extensions import List, Optional

from giskardpy.motion_statechart.beliefs.estimator import EstimatorNode, Prediction
from giskardpy.motion_statechart.beliefs.gaussian import GaussianBelief, Reading
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import CertainPriorError, NodeNotBuiltError
from giskardpy.motion_statechart.grasp_likelihood_source import GraspLikelihoodSource
from giskardpy.motion_statechart.graph_node import NodeArtifacts
from krrood.symbolic_math.symbolic_math import (
    FloatVariable,
    trinary_logic_from_continuous,
)
from semantic_digital_twin.datastructures.joint_state import JointState

# %% what one sampled measurement says about a grasp


@dataclass(frozen=True)
class SampledLikelihood:
    """
    A share of rays that hit the body, read as evidence about whether it is held.

    Both halves of a reading follow from the two counts: what the measurement says, and
    how far it can be trusted. Nothing here is tuned — a hundred rays say more than ten
    because they are a hundred, not because a parameter says so.
    """

    hits: float
    """
    How many of the rays hit the body.
    """

    sample_size: int
    """
    How many rays were cast.
    """

    half_count_correction: float = 0.5
    """
    How much of a hit and of a miss to add before taking the log-odds, so that a
    measurement at either extreme stays finite.

    Half of each is the Haldane-Anscombe correction, which is the usual choice.
    """

    @property
    def corrected_hits(self) -> float:
        """
        :return: The hits, adjusted so a measurement without any is still evidence about
            a probability rather than a statement that it is zero.
        """
        return self.hits + self.half_count_correction

    @property
    def corrected_misses(self) -> float:
        """
        :return: The rays that missed, adjusted the same way.
        """
        return self.sample_size - self.hits + self.half_count_correction

    @property
    def log_odds(self) -> float:
        """
        :return: What this measurement says the log-odds of a grasp are.
        """
        return math.log(self.corrected_hits / self.corrected_misses)

    @property
    def variance(self) -> float:
        """
        :return: How far a measurement of this many rays scatters around the truth, in
            the same log-odds.
        """
        return 1.0 / self.corrected_hits + 1.0 / self.corrected_misses


# %% a grasp that is remembered rather than re-measured


@dataclass(eq=False, repr=False)
class GraspBelief(EstimatorNode):
    """
    Keeps a belief about whether a body is held, corrected each control cycle by a
    sampled likelihood and decayed toward the prior whenever the gripper is open.

    The quantity is estimated in log-odds, where a Gaussian is the right shape: a
    probability is bounded and an update could push an estimate of one outside its own
    range. The probability those log-odds stand for is published as well, since that is
    what a goal weighing itself by the grasp reads.

    This is the first quantity in the stack with a memory. The predicate behind the
    likelihood is a stateless raycast against the point-estimate world, so on its own it
    can say how a single sample fell but not how convincing a grasp has become.

    .. warning:: The belief is only as good as the rays behind it. A gripper wedged
        beside a handle can produce a confident posterior, so judge a grasp against
        contact state rather than against this.
    """

    grasp: Continuous = field(kw_only=True)
    """
    The quantity being estimated: the log-odds that the body is held.
    """

    likelihood_source: GraspLikelihoodSource = field(kw_only=True)
    """
    Where the share of rays hitting the body is read from each cycle.
    """

    gripper_open: JointState = field(kw_only=True)
    """
    The gripper state that means nothing can be held, which the belief decays toward the
    prior while it is reached.
    """

    prior_uncertainty: float = field(kw_only=True)
    """
    How uncertain the prior is, in the log-odds the belief is estimated in.
    """

    forgetting_half_life: float = field(kw_only=True)
    """
    How many seconds an open gripper takes to carry the estimate halfway back to the
    prior.

    There is no established value for this: it says how long a grasp stays worth
    believing once it has been ruled out, which depends on what the caller does next.
    """

    drift: float = field(kw_only=True)
    """
    How much uncertainty a second adds on its own, which is what keeps a grasp nobody is
    measuring from staying as convincing as a fresh one.

    There is no established value for this either; it says how fast a grasp can change
    without being seen to.
    """

    false_below: float = field(kw_only=True)
    """
    The probability of a grasp under which the observation is false.

    There is no established value for this: it says how unlikely a grasp must be before
    it counts as ruled out, which depends on what the caller does next.
    """

    prior_probability: float = field(default=0.5, kw_only=True)
    """
    How likely a grasp is before anything has been measured, defaulting to an even
    chance, which is what nothing being known yet means.
    """

    true_above: float = field(default=0.9, kw_only=True)
    """
    The probability of a grasp over which the observation is true, defaulting to the
    same confidence
    :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped` calls a
    grasp.
    """

    _probability: Optional[FloatVariable] = field(default=None, init=False, repr=False)
    """
    The variable the probability of a grasp is written to, created while building.
    """

    def __post_init__(self):
        """
        :raises CertainPriorError: If the prior already rules a grasp in or out, which
            leaves it no log-odds to start from.
        """
        super().__post_init__()
        if not 0.0 < self.prior_probability < 1.0:
            raise CertainPriorError(probability=self.prior_probability)

    @property
    def probability(self) -> FloatVariable:
        """
        :return: The variable carrying how likely a grasp currently is, for use in
            constraints and conditions.
        :raises NodeNotBuiltError: If the node has not been built yet.
        """
        if self._probability is None:
            raise NodeNotBuiltError(node=self)
        return self._probability

    @property
    def prior_log_odds(self) -> float:
        """
        :return: The estimate to start from, and the one an open gripper carries the
            belief back toward.
        """
        return math.log(self.prior_probability / (1.0 - self.prior_probability))

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = super().build_artifacts(context)
        self._probability = FloatVariable(f"{self.unique_name}_grasp_probability")
        context.float_variable_data.register_expression(self._probability)
        artifacts.observation = trinary_logic_from_continuous(
            self._probability,
            false_below=self.false_below,
            true_above=self.true_above,
        )
        return artifacts

    def create_initial_belief(self, context: MotionStatechartContext) -> GaussianBelief:
        return GaussianBelief.of_one_variable(
            self.grasp, mean=self.prior_log_odds, variance=self.prior_uncertainty
        )

    def create_prediction(self, context: MotionStatechartContext) -> Prediction:
        control_period = context.qp_controller_config.control_dt
        drift = {(self.grasp, self.grasp): self.drift * control_period}
        if not self.gripper_open.is_achieved():
            return Prediction(
                transition={(self.grasp, self.grasp): 1.0}, process_noise=drift
            )

        retention = 0.5 ** (control_period / self.forgetting_half_life)
        return Prediction(
            transition={(self.grasp, self.grasp): retention},
            process_noise=drift,
            offset={self.grasp: (1.0 - retention) * self.prior_log_odds},
        )

    def measure(self, context: MotionStatechartContext) -> List[Reading]:
        measured = context.float_variable_data.get_value(
            self.likelihood_source.likelihood
        )
        sampled = SampledLikelihood(
            hits=measured * self.likelihood_source.sample_size,
            sample_size=self.likelihood_source.sample_size,
        )
        return [
            Reading.of_one_variable(
                self.grasp, value=sampled.log_odds, variance=sampled.variance
            )
        ]

    def on_start(self, context: MotionStatechartContext) -> None:
        super().on_start(context)
        self._publish_probability(context)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Runs the cycle the base class defines and publishes the probability alongside
        it.

        :param context: The context holding the float variable data to write to.
        :return: Nothing, so that the threshold on the probability decides what this
            node observes rather than whether it measured at all.
        """
        super().on_tick(context)
        self._publish_probability(context)
        return None

    def _publish_probability(self, context: MotionStatechartContext) -> None:
        """
        Writes how likely a grasp currently is into the variable carrying it, reading
        the estimate the base class has just published so the two can never disagree.

        :param context: The context holding the float variable data to write to.
        """
        estimate = context.float_variable_data.get_value(
            self.estimate_variable_of(self.grasp)
        )
        context.float_variable_data.set_value(self.probability, float(expit(estimate)))
