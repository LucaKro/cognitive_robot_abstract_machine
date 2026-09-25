from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from typing_extensions import Optional, Any

from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import NavigateAction, LookAtAction
from semantic_digital_twin.spatial_types import (
    Quaternion,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class FaceAtAction(ActionDescription):
    """
    Turn the robot chassis such that is faces the ``pose`` and after that perform a look
    at action.
    """

    pose: Pose
    """
    The pose to face 
    """
    standing_position: Optional[Pose] = None
    """
    Where the robot faces :attr:`pose` from, or ``None`` for where it stands when this
    action's plan is built.

    A plan is built before any of it runs, so a step that follows a navigation names
    where that navigation sends the robot rather than reading where it stands now.
    """

    @property
    def _action_plan(self) -> PlanNode:
        robot_position = (
            self.robot.root.global_transform
            if self.standing_position is None
            else self.standing_position.to_homogeneous_matrix()
        )

        # calculate orientation for robot to face the object
        angle = (
            np.arctan2(
                robot_position.y - self.pose.y,
                robot_position.x - self.pose.x,
            )
            + np.pi
        )

        # create new robot pose
        new_robot_pose = Pose(
            robot_position.to_position(),
            Quaternion.from_rpy(0, 0, angle),
            reference_frame=self.world.root,
        )

        return sequential(
            [
                NavigateAction(new_robot_pose),  # turn robot
                LookAtAction(self.pose),  # look at the target
            ]
        )
