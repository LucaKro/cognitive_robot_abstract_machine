from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import PoseUncertaintyNotBuiltError
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.pose_covariance import PoseCovarianceSource
from krrood.symbolic_math.symbolic_math import FloatVariable

UNCERTAINTY_WITHOUT_A_READING: float = np.inf
"""
The uncertainty published while no pose has been observed yet.

Nothing has been measured, so the pose is as uncertain as it can be. Zero would be the
one wrong answer: it reads as perfect certainty, and a condition that waits for the
uncertainty to fall would pass before the first message arrives.
"""


@dataclass(eq=False, repr=False)
class PoseUncertainty(MotionStatechartNode):
    """
    Publishes how uncertain a pose observed from outside the process currently is,
    typically the pose of a mobile base reported by odometry.

    The summed variance over all six degrees of freedom is written to
    :attr:`total_variance` every control cycle, so constraints and transition conditions
    can read it, as in *do not begin the final approach while the base is less certain
    than this*. The node observes whether a pose has been observed at all.
    """

    source: PoseCovarianceSource = field(kw_only=True)
    """
    Whatever receives the poses whose uncertainty is published.
    """

    _total_variance: Optional[FloatVariable] = field(
        default=None, init=False, repr=False
    )
    """
    The variable the summed variance is written to, created while building.
    """

    @property
    def total_variance(self) -> FloatVariable:
        """
        :return: The variable carrying the summed variance, for use in constraints and
            conditions.
        :raises PoseUncertaintyNotBuiltError: If the node has not been built yet.
        """
        if self._total_variance is None:
            raise PoseUncertaintyNotBuiltError(node_name=self.name)
        return self._total_variance

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        self._total_variance = FloatVariable(f"{self.unique_name}_total_variance")
        context.float_variable_data.register_expression(self._total_variance)
        return NodeArtifacts()

    def on_start(self, context: MotionStatechartContext) -> None:
        self._publish(context)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        self._publish(context)
        if self.source.pose_covariance is None:
            return ObservationStateValues.FALSE
        return ObservationStateValues.TRUE

    def _publish(self, context: MotionStatechartContext) -> None:
        """
        Writes the current uncertainty into :attr:`total_variance`.

        :param context: The context holding the float variable data to write to.
        """
        covariance = self.source.pose_covariance
        context.float_variable_data.set_value(
            self.total_variance,
            (
                UNCERTAINTY_WITHOUT_A_READING
                if covariance is None
                else covariance.total_variance
            ),
        )
