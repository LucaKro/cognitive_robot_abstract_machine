from dataclasses import dataclass

from typing_extensions import Optional

from giskardpy.motion_statechart.goals.open_close import Open, Close
from giskardpy.motion_statechart.goals.templates import Parallel
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.robot_plans.motions.base import StandaloneMotion
from coraplex.datastructures.enums import Arms
from coraplex.view_manager import ViewManager

CLOSED_ENOUGH_MARGIN = 0.09
"""
How far short of its goal a container may stop and still count as open or shut.

A goal close to the mechanism's own limit is only approached asymptotically, so a
container that has to arrive exactly never reports that it got there.
"""


@dataclass
class OpeningMotion(StandaloneMotion):
    """
    Designator for opening container.
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

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        return Parallel(
            [
                Open(
                    tip_link=tip,
                    environment_link=self.object_part,
                    goal_joint_state=1.45,
                    threshold=CLOSED_ENOUGH_MARGIN,
                ),
                *self._only_allow_gripper_collision_rules(self.arm),
            ],
            name="Open",
        )


@dataclass
class ClosingMotion(StandaloneMotion):
    """
    Designator for closing a container.
    """

    object_part: Body
    """
    Object designator for the drawer handle.
    """

    arm: Arms
    """
    Arm that should be used.
    """

    goal_joint_state: float = 0.01
    """
    How far the container is left open.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        return Parallel(
            [
                Close(
                    tip_link=tip,
                    environment_link=self.object_part,
                    goal_joint_state=self.goal_joint_state,
                    threshold=CLOSED_ENOUGH_MARGIN,
                ),
                *self._only_allow_gripper_collision_rules(self.arm),
            ],
            name="Close",
        )
