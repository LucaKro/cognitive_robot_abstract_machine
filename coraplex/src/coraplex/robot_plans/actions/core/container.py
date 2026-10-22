from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


from krrood.entity_query_language.factories import (
    a,
    flat_variable,
    variable,
)
from krrood.entity_query_language.query.match import Match
from coraplex.querying.grasp_closeness import GraspsByCloseness
from coraplex.robot_plans.actions.base import GoalAction, SequentialAction
from cramph.node import StatechartNode
from coraplex.robot_plans.actions.core.pick_up import (
    GraspingAction,
    GripperContactAction,
    HasGraspChoice,
)
from coraplex.robot_plans.mixins import HasApproachesGraspPoses, MovesToolCenterPoint
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from giskardpy.motion_statechart.goals.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

# %% taking a container by its handle


@dataclass(eq=False, repr=False, kw_only=True)
class GraspsAndMovesContainerByItsHandle(
    MovesToolCenterPoint, SequentialAction, HasGraspChoice, ABC
):
    """
    An action that takes hold of a container's handle by :attr:`grasp`, moves the
    container with it, lets go and clears the handle.
    """

    approach_clearance: float = HasApproachesGraspPoses.approach_clearance
    """
    The gap in meters between the handle and the gripper before it closes on it.
    """

    release_clearance: float = 0.05
    """
    The gap in meters between the handle and the gripper once it has let go.
    """

    @classmethod
    def from_handle(cls, handle: Handle, arm: Arm) -> Match:
        """
        :param handle: The handle of the container.
        :param arm: The arm that takes hold of the handle.
        :return: This action on `handle`, with the grasp left open among the grasps
            the handle offers, the closest to where the robot stands when it is grounded
            first.
        """
        # A variable, so the grasps are ranked when the action is grounded, from where
        # the robot stands then.
        where_the_robot_stands = variable(
            Pose, domain=[Pose(reference_frame=arm._robot.root)]
        )
        return a(cls)(
            grasp=flat_variable(
                GraspsByCloseness(handle, where_the_robot_stands, number_of_grasps=None)
            ),
            arm=arm,
        )

    @abstractmethod
    def _create_mechanism_goal(self) -> MotionStatechartNode:
        """
        :return: A new goal moving the container while its handle is held.
        """

    def back_off_pose(self, grasp_pose: Pose, end_effector: EndEffector) -> Pose:
        """
        The tool frame goal that clears the released handle, back along the way the
        gripper came in, so an open gripper still straddling the handle is drawn off it
        rather than across it.

        :param grasp_pose: The grasp frame the handle was held by.
        :param end_effector: The end effector that held it.
        :return: The pose the tool frame backs off to, in `grasp_pose`'s frame.
        """
        return HasApproachesGraspPoses.pre_grasp_pose(
            grasp_pose, end_effector, self.release_clearance
        )

    def create_steps(self) -> list[StatechartNode]:
        """
        The gripper on the handle touches its surroundings from taking hold of the
        handle until it has backed off it, so only those steps let it, see
        :class:`~coraplex.robot_plans.actions.core.pick_up.GripperContactAction`.
        """
        end_effector = self.arm.end_effector
        return [
            GraspingAction(
                grasp=self.grasp,
                arm=self.arm,
                approach_clearance=self.approach_clearance,
            ),
            GripperContactAction(
                end_effector=end_effector,
                steps=[
                    self._create_mechanism_goal(),
                    MoveGripper(
                        end_effector=end_effector,
                        state=GripperState.OPEN,
                    ),
                    self.tool_center_point_goal(
                        self.back_off_pose(self.grasp.grasp_pose, end_effector),
                        self.arm,
                    ),
                ],
            ),
        ]


# %% opening and closing


@dataclass(eq=False, repr=False, kw_only=True)
class GraspOpenAndRetractAction(GraspsAndMovesContainerByItsHandle):
    """
    Takes hold of the handle of a container like object, opens it, driving its mechanism
    to its upper limit, then lets go of the handle and backs off it.

    :class:`OpenAction` opens a container whose handle is already held, and keeps
    holding it.
    """

    def _create_mechanism_goal(self) -> MotionStatechartNode:
        handle = self.grasp.graspable.root
        mechanism = handle.get_first_parent_connection_of_type(ActiveConnection1DOF)
        return ArticulateGraspedMechanism(
            tip_link=self.arm.end_effector.tool_frame,
            environment_link=handle,
            goal_joint_state=mechanism.dof.limits.upper.position * 0.8,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class GraspCloseAndRetractAction(GraspsAndMovesContainerByItsHandle):
    """
    Takes hold of the handle of a container like object, closes it, then lets go of the
    handle and backs off it.

    :class:`CloseAction` closes a container whose handle is already held, and keeps
    holding it.
    """

    closed_joint_position: float = 0.01
    """
    The joint position, in meters or radians, the container's mechanism is driven to, or
    its lower limit if that lies above.
    """

    def _create_mechanism_goal(self) -> MotionStatechartNode:
        return ArticulateGraspedMechanism(
            tip_link=self.arm.end_effector.tool_frame,
            environment_link=self.grasp.graspable.root,
            goal_joint_state=self.closed_joint_position,
        )


# %% opening and closing a container whose handle is held


@dataclass(eq=False, repr=False, kw_only=True)
class MovesHeldContainer(GoalAction, ABC):
    """
    An action that moves a container like object by its handle, which the gripper of
    :attr:`arm` already holds, and keeps holding the handle.
    """

    arm: Arm
    """
    The arm whose gripper holds the handle.
    """

    handle: Handle
    """
    The handle of the container.
    """

    @abstractmethod
    def goal_joint_position(self) -> float:
        """
        :return: The joint position, in meters or radians, the container's mechanism
            is driven to.
        """

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the gripper on the handle to touch its surroundings
            while it moves the container.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_goal(self) -> StatechartNode:
        return ArticulateGraspedMechanism(
            tip_link=self.arm.end_effector.tool_frame,
            environment_link=self.handle.root,
            goal_joint_state=self.goal_joint_position(),
        )


@dataclass(eq=False, repr=False, kw_only=True)
class OpenAction(MovesHeldContainer):
    """
    Opens a container like object whose handle is already held, driving its mechanism to
    its upper limit.
    """

    def goal_joint_position(self) -> float:
        mechanism = self.handle.root.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        return mechanism.dof.limits.upper.position


@dataclass(eq=False, repr=False, kw_only=True)
class CloseAction(MovesHeldContainer):
    """
    Closes a container like object whose handle is already held.
    """

    closed_joint_position: float = 0.01
    """
    The joint position, in meters or radians, the container's mechanism is driven to, or
    its lower limit if that lies above.
    """

    def goal_joint_position(self) -> float:
        return self.closed_joint_position
