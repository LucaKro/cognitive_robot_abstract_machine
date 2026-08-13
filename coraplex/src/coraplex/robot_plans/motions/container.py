from dataclasses import dataclass

from typing_extensions import Optional

from giskardpy.motion_statechart.goals.open_close import Open, Close
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.robot_plans.motions.base import StandaloneMotion
from coraplex.datastructures.enums import Arms
from coraplex.view_manager import ViewManager

CLOSED_ENOUGH_MARGIN = 0.09
"""
How far short of its goal a container may stop and still count as open or shut.

The last stretch onto a mechanism's own limit is only approached asymptotically, so a
container that has to arrive exactly never reports that it got there.
"""

CONTAINER_STALL_TIME = 1.0
"""
How long, in seconds, a container has to stop moving before it counts as stuck.

Long enough that the slow approach onto a limit is not mistaken for having stopped.
"""


@dataclass
class ContainerMotion(StandaloneMotion):
    """
    Base for the motions that drive a container's own degree of freedom while the hand
    holds its handle.
    """

    object_part: Body
    """
    Object designator for the drawer handle.
    """

    arm: Arms
    """
    Arm that should be used.
    """

    def perform(self):
        return

    def _chart_around(self, goal: Open) -> MotionStatechartNode:
        """
        Wrap a container goal in what every container motion needs around it.

        :param goal: The goal driving the container's degree of freedom.
        :return: That goal, done as soon as the container arrives *or* stops moving, with
            the hand free to touch the handle it is holding.
        """
        return Parallel(
            [
                Parallel(
                    [goal, self._container_stalled()],
                    minimum_success=1,
                    name=goal.name,
                ),
                *self._only_allow_gripper_collision_rules(self.arm),
            ],
            name=goal.name,
        )

    def _container_stalled(self) -> LocalMinimumReached:
        """
        :return: A monitor that turns true once the container has stopped moving, which
            is as far as it goes when its own limit is only approached asymptotically.
        """
        connection = self.object_part.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        return LocalMinimumReached(
            degrees_of_freedom=[connection.raw_dof],
            minimum_time=CONTAINER_STALL_TIME,
            measure_from_own_start=True,
        )


@dataclass
class OpeningMotion(ContainerMotion):
    """
    Designator for opening container.
    """

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        return self._chart_around(
            Open(
                tip_link=tip,
                environment_link=self.object_part,
                goal_joint_state=1.45,
                name="Open",
            )
        )


@dataclass
class ClosingMotion(ContainerMotion):
    """
    Designator for closing a container.
    """

    goal_joint_state: float = 0.01
    """
    How far the container is left open.
    """

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        return self._chart_around(
            Close(
                tip_link=tip,
                environment_link=self.object_part,
                goal_joint_state=self.goal_joint_state,
                name="Close",
            )
        )
