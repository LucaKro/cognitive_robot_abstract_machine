from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from typing_extensions import Any, Dict

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    and_,
    variable_from,
    ConditionType,
)
from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.locations.pose_validator import IsReachableBy
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.querying.predicates import GripperIsFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from coraplex.robot_plans.motions.base import BaseMotion
from coraplex.robot_plans.motions.container import OpeningMotion, ClosingMotion
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ContainerAction(ActionDescription):
    """
    Moves a container like object by its handle.

    Taking hold of the handle, letting go of it again and clearing it afterwards is the
    same whichever way the container is moved; only the motion working the mechanism
    differs.
    """

    object_designator: Handle
    """
    The handle of the container that should be moved.
    """

    arm: Arms
    """
    Arm that should be used.
    """

    approach_clearance: float = ActionConfig.approach_clearance
    """
    The gap in meters between the handle and the gripper before it closes on it.
    """

    release_clearance: float = ActionConfig.release_clearance
    """
    The gap in meters between the handle and the gripper once it has let go.
    """

    @property
    @abstractmethod
    def _mechanism_motion(self) -> BaseMotion:
        """
        :return: The motion that moves the container while its handle is held.
        """

    def back_off_pose(self, grasp_pose: Pose, end_effector: EndEffector) -> Pose:
        """
        The tool frame goal that clears the released handle.

        The gripper leaves the way it came in, so that an open gripper still straddling
        the handle is drawn off it rather than across it.

        :param grasp_pose: The grasp frame the handle was held by.
        :param end_effector: The end effector that held it.
        :return: The pose the tool frame backs off to, in ``grasp_pose``'s frame.
        """
        return HasApproachesGraspPoses.standoff_pose(
            end_effector.tool_frame_goal(grasp_pose),
            end_effector,
            self.release_clearance,
        )

    @property
    def _action_plan(self) -> PlanNode:
        handle_grasp = Pose(reference_frame=self.object_designator.root)
        return sequential(
            [
                GraspingAction(
                    self.object_designator,
                    self.arm,
                    handle_grasp,
                    approach_clearance=self.approach_clearance,
                ),
                self._mechanism_motion,
                MoveGripperMotion(
                    GripperState.OPEN, self.arm, allow_gripper_collision=True
                ),
                MoveToolCenterPointMotion(
                    self.back_off_pose(
                        handle_grasp,
                        ViewManager.get_end_effector_view(self.arm, self.robot),
                    ),
                    self.arm,
                    allow_gripper_collision=True,
                ),
            ]
        )


@dataclass
class OpenAction(ContainerAction):
    """
    Opens a container like object.
    """

    @property
    def _mechanism_motion(self) -> BaseMotion:
        return OpeningMotion(self.object_designator.root, self.arm)

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper with which to open the container has to be free and the handle has
        to be reachable.
        """
        end_effector = ViewManager.get_end_effector_view(
            variables["arm"], context.robot
        )
        return and_(
            GripperIsFree(end_effector),
            IsReachableBy(
                context=Context(
                    robot=context.robot,
                    world=context.world,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                pose=end_effector.tool_frame_goal(
                    Pose(reference_frame=kwargs["object_designator"].root)
                ),
                tip_link=end_effector.tool_frame,
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The container has to be open.

        The gripper is clear of the handle by then, so what it holds says nothing about
        whether the container was opened.
        """
        open_connection = kwargs[
            "object_designator"
        ].root.get_first_parent_connection_of_type(ActiveConnection1DOF)

        return variable_from(open_connection).position > 0.3


@dataclass
class CloseAction(ContainerAction):
    """
    Closes a container like object.
    """

    @property
    def _mechanism_motion(self) -> BaseMotion:
        return ClosingMotion(self.object_designator.root, self.arm)

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression | bool:
        """
        The container has to be closed.
        """
        close_connection = kwargs[
            "object_designator"
        ].root.get_first_parent_connection_of_type(ActiveConnection1DOF)

        return variable_from(close_connection).position < 0.1
