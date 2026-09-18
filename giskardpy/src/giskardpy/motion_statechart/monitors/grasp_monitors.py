from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import GraspLikelihoodNotBuiltError
from giskardpy.motion_statechart.grasp_likelihood_source import GraspLikelihoodSource
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from krrood.symbolic_math.symbolic_math import (
    FloatVariable,
    trinary_logic_from_continuous,
)
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class GraspLikelihood(MotionStatechartNode, GraspLikelihoodSource):
    """
    Publishes how strongly a body is currently held by a gripper, as a continuous
    quantity rather than a yes or no.

    The share of rays that hit the body is written to :attr:`likelihood` every control
    cycle, so constraints and transition conditions can read the confidence itself. The
    node's own observation is that same variable seen through
    :func:`~krrood.symbolic_math.symbolic_math.trinary_logic_from_continuous`, which
    keeps the observation one of the three truth values the life cycle is built on.

    The observation is evaluated before the nodes of a cycle tick, so it answers for the
    measurement taken on the previous cycle. The first measurement is taken when the
    node starts, so the observation is never read off an unmeasured variable.

    .. warning:: Sampling the rays runs on the control loop. Lower :attr:`sample_size`
        if the cycle time suffers.
    """

    body: Body = field(kw_only=True)
    """
    The body whose presence in the gripper is measured.
    """

    gripper: EndEffector = field(kw_only=True)
    """
    The gripper to measure it in.
    """

    false_below: float = field(kw_only=True)
    """
    The likelihood under which the observation is false.

    There is no established value for this: it says how few rays may hit the body before
    a grasp counts as ruled out, which depends on what the caller does next.
    """

    true_above: float = field(default=0.9, kw_only=True)
    """
    The likelihood over which the observation is true, defaulting to the same share of
    hits :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped` calls
    a grasp.
    """

    sample_size: int = field(default=100, kw_only=True)
    """
    How many rays to cast between the fingers per measurement.
    """

    _likelihood: Optional[FloatVariable] = field(default=None, init=False, repr=False)
    """
    The variable the measured likelihood is written to, created while building.
    """

    @property
    def likelihood(self) -> FloatVariable:
        """
        :return: The variable carrying the measured likelihood, for use in constraints
            and conditions.
        :raises GraspLikelihoodNotBuiltError: If the node has not been built yet.
        """
        if self._likelihood is None:
            raise GraspLikelihoodNotBuiltError(node_name=self.name)
        return self._likelihood

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        self._likelihood = FloatVariable(f"{self.unique_name}_grasp_likelihood")
        context.float_variable_data.register_expression(self._likelihood)
        artifacts.observation = trinary_logic_from_continuous(
            self._likelihood,
            false_below=self.false_below,
            true_above=self.true_above,
        )
        return artifacts

    def on_start(self, context: MotionStatechartContext) -> None:
        self._measure(context)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        self._measure(context)
        return None

    def _measure(self, context: MotionStatechartContext) -> None:
        """
        Writes the current likelihood into :attr:`likelihood`.

        :param context: The context holding the float variable data to write to.
        """
        context.float_variable_data.set_value(
            self.likelihood,
            is_body_in_gripper(self.body, self.gripper, self.sample_size),
        )
