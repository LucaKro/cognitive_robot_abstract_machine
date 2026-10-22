"""
Plan transformations that let particular robots carry out an action with a motion of
their own.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Generic

from coraplex.exceptions import ClosingGoalNotFound, ToolPathNotFound
from coraplex.plans.plan_transformation import (
    ConfigurationTransformation,
    MatchedType,
    ParallelInsertion,
    PlanTransformation,
)
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.composite.tool_based import ToolMotionAction
from coraplex.robot_plans.actions.core.container import (
    GraspCloseAndRetractAction,
    GraspsAndMovesContainerByItsHandle,
)
from coraplex.robot_plans.actions.core.navigation import DrivesBase
from coraplex.robot_plans.actions.core.pick_up import GripperContactAction
from coraplex.robot_plans.actions.core.robot_body import MoveFingersToWidthAction
from coraplex.robot_plans.mixins import MovesToolCenterPoint
from cramph.composites import ControlStructureNode, Parallel
from cramph.node import StatechartNode
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.goals.base_pose_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.goals.cartesian_goals import CartesianPoseStraight
from giskardpy.motion_statechart.goals.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    GripperGraspTask,
    GripperMoveTask,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPositionTrajectory,
    HoldOrientation,
)
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.garmi import GarmiLeftGripper, GarmiRightGripper
from semantic_digital_twin.robots.justin import Justin
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.spatial_types.spatial_types import Pose, Vector3

# %% keeping the torso upright while using a tool


@dataclass
class KeepTheTorsoUprightWhileUsingATool(ParallelInsertion[ToolMotionAction]):
    """
    Keeps Justin's torso upright while it moves a tool, which no other robot needs: its
    torso would otherwise lean into the motion.
    """

    def is_applicable(self, plan_node: ToolMotionAction) -> bool:
        return isinstance(plan_node.robot, Justin)

    def anchor(self, plan_node: ToolMotionAction) -> StatechartNode:
        """
        :return: The goal moving the tool along its path, alongside which the torso is
            held.
        :raises ToolPathNotFound: If the action holds no goal moving the tool along a
            path.
        """
        for node in plan_node.descendants:
            if isinstance(node, CartesianPositionTrajectory):
                return node
        raise ToolPathNotFound(plan_node)

    def nodes_to_insert(self, plan_node: ToolMotionAction) -> list[StatechartNode]:
        root = plan_node.controlled_root
        torso_tip = plan_node.robot.mobile_base.torso.tip
        return [
            AlignPlanes(
                tip_link=torso_tip,
                root_link=root,
                tip_normal=Vector3.X(torso_tip),
                goal_normal=Vector3.Z(root),
                weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE.value,
            )
        ]


# %% driving a differential drive


@dataclass
class DriveWithTheDifferentialDrive(ConfigurationTransformation[DrivesBase], ABC):
    """
    Moves the base of a robot with a goal for its differential drive, which turns
    towards a pose, drives there and turns again, instead of a goal that asks the base
    to move sideways.
    """

    def apply(self, plan_node: DrivesBase) -> None:
        plan_node.base_goal_type = DifferentialDriveBaseGoal


@dataclass
class DriveStretchWithItsDifferentialDrive(DriveWithTheDifferentialDrive):
    """
    Moves the base of a Stretch with a goal for its differential drive, simulated or
    not.
    """

    def is_applicable(self, plan_node: DrivesBase) -> bool:
        return isinstance(plan_node.robot, Stretch)


@dataclass
class DriveTiagoWithItsDifferentialDrive(DriveWithTheDifferentialDrive):
    """
    Moves the base of a Tiago with a goal for its differential drive, simulated or not.
    """

    def is_applicable(self, plan_node: DrivesBase) -> bool:
        return isinstance(plan_node.robot, Tiago)


# %% keeping the base from turning while moving the tool


@dataclass
class KeepStretchFromTurningWhileMovingItsTool(ParallelInsertion[MovesToolCenterPoint]):
    """
    Holds the orientation of a Stretch's base while its tool moves, so that the base
    only drives along its heading, as the tool's sixth degree of freedom.

    The base does not turn to bring a target into reach, which would sweep it through
    wide arcs: a target out of reach fails, and where the robot stands is up to the
    plan. Left as they are: actions driving the base, actions moving a container by its
    handle, and actions below one that holds the base already.
    """

    def is_applicable(self, plan_node: MovesToolCenterPoint) -> bool:
        robot = plan_node.robot
        return (
            isinstance(robot, Stretch)
            and robot.mobile_base.full_body_controlled
            and not any(
                isinstance(node, GraspsAndMovesContainerByItsHandle)
                for node in [plan_node, *plan_node.path]
            )
            and not any(isinstance(node, DrivesBase) for node in plan_node.descendants)
            and not any(
                self._holds_the_base(ancestor, robot) for ancestor in plan_node.path
            )
        )

    def anchor(self, plan_node: MovesToolCenterPoint) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: MovesToolCenterPoint) -> list[StatechartNode]:
        return [
            HoldOrientation(
                root_link=plan_node.world.root, tip_link=plan_node.robot.root
            )
        ]

    @staticmethod
    def _holds_the_base(node: StatechartNode, robot: Stretch) -> bool:
        """
        :return: Whether `node` holds the orientation of the base of `robot` beside its
            other children, as this transformation makes an action do. A hold further
            down, such as that of an earlier action, does not cover the others.
        """
        return isinstance(node, Parallel) and any(
            type(child) is HoldOrientation and child.tip_link is robot.root
            for child in node.nodes
        )


# %% moving the tool center point in a straight line


@dataclass
class MoveStretchToolCenterPointInAStraightLine(PlanTransformation[CartesianPose]):
    """
    Moves the tool center point of a Stretch to each of its targets along a straight
    line, by giving every pose goal that an action moving it sets relative to its
    controlled root the tasks of a
    :class:`~giskardpy.motion_statechart.goals.cartesian_goals.CartesianPoseStraight`
    reaching the same pose. Pose goals inside another goal are left as they are.
    """

    def is_applicable(self, plan_node: CartesianPose) -> bool:
        if type(plan_node) is not CartesianPose:
            return False
        owner = next(
            (
                node
                for node in plan_node.path
                # a gripper contact action runs steps of the action around it
                if not isinstance(node, GripperContactAction)
                and (
                    isinstance(node, Action)
                    or not isinstance(node, ControlStructureNode)
                )
            ),
            None,
        )
        return (
            isinstance(owner, MovesToolCenterPoint)
            and isinstance(owner.robot, Stretch)
            and plan_node.root_link is owner.controlled_root
        )

    def apply(self, plan_node: CartesianPose) -> None:
        straight = CartesianPoseStraight(
            name=plan_node.name,
            root_link=plan_node.root_link,
            tip_link=plan_node.tip_link,
            goal_pose=plan_node.goal_pose,
            translation_threshold=plan_node.translation_threshold,
            orientation_threshold=plan_node.orientation_threshold,
            binding_policy=plan_node.binding_policy,
        )
        # the tasks of a pose goal come before the nodes accompanying it
        for task, straight_task in zip(list(plan_node.nodes), straight.create_tasks()):
            plan_node.statechart.replace(task, straight_task)


# %% closing a container


@dataclass
class KeepStretchInFrontOfTheContainerItCloses(
    PlanTransformation[GraspCloseAndRetractAction]
):
    """
    Keeps a kinematically simulated Stretch standing in front of the container it closes
    and holding on to the handle, so that it closes the container by pushing its arm
    forward.
    """

    def is_applicable(self, plan_node: GraspCloseAndRetractAction) -> bool:
        return isinstance(plan_node.robot, Stretch)

    @staticmethod
    def closing_goal(
        plan_node: GraspCloseAndRetractAction,
    ) -> ArticulateGraspedMechanism:
        """
        :param plan_node: The close action.
        :return: The goal closing the container.
        :raises ClosingGoalNotFound: If the action holds no goal closing a container,
            which it has none of before it is expanded.
        """
        for node in plan_node.descendants:
            if isinstance(node, ArticulateGraspedMechanism):
                return node
        raise ClosingGoalNotFound(plan_node)

    def apply(self, plan_node: GraspCloseAndRetractAction) -> None:
        closing_goal = self.closing_goal(plan_node)
        handle = closing_goal.environment_link
        tool_frame = closing_goal.tip_link
        robot_root = plan_node.robot.root
        plan_node.statechart.replace(
            closing_goal,
            Parallel(
                nodes=[
                    CartesianPose(
                        name="Keep holding handle",
                        root_link=handle,
                        tip_link=tool_frame,
                        goal_pose=Pose(reference_frame=tool_frame),
                    ),
                    AlignPlanes(
                        root_link=plan_node.world.root,
                        tip_link=robot_root,
                        goal_normal=Vector3.X(handle),
                        tip_normal=Vector3.NEGATIVE_Y(robot_root),
                    ),
                    ArticulateGraspedMechanism(
                        tip_link=tool_frame,
                        environment_link=handle,
                        goal_joint_state=closing_goal.goal_joint_state,
                        grasping_end_effector=closing_goal.grasping_end_effector,
                    ),
                ]
            ),
        )


# %% moving GARMI's fingers through the Franka gripper driver


class FrankaGripperActionName(StrEnum):
    """
    The names the Franka gripper driver serves its actions under, inside its namespace.
    """

    MOVE = "move"
    GRASP = "grasp"


@dataclass
class DriveGarmiGrippersWithTheirFrankaDriver(
    PlanTransformation[MatchedType], Generic[MatchedType], SubClassSafeGeneric, ABC
):
    """
    Moves the fingers of GARMI's real grippers through the actions of their Franka
    gripper driver instead of with a Giskard goal.
    """

    left_gripper_namespace: str = "/garmi/arms/left/franka_gripper"
    """
    The ROS namespace the driver of the left gripper serves its actions in.
    """

    right_gripper_namespace: str = "/garmi/arms/right/franka_gripper"
    """
    The ROS namespace the driver of the right gripper serves its actions in.
    """

    finger_speed: float = 0.1
    """
    The speed the fingers move at, in metres per second.
    """

    grasp_force: float = 0.0
    """
    The force the fingers hold an object with, in newtons.
    """

    grasp_width_tolerance: float = 0.005
    """
    How far the grasped width may fall below or exceed the expected width of the object,
    in metres.
    """

    def action_name(
        self,
        gripper: GarmiLeftGripper | GarmiRightGripper,
        action: FrankaGripperActionName,
    ) -> str:
        """
        :return: The name the driver of `gripper` serves `action` under.
        """
        if isinstance(gripper, GarmiLeftGripper):
            return f"{self.left_gripper_namespace}/{action}"
        return f"{self.right_gripper_namespace}/{action}"

    def grasp(
        self, gripper: GarmiLeftGripper | GarmiRightGripper, width: float | None = None
    ) -> GripperGraspTask:
        """
        :param gripper: The gripper to shut.
        :param width: The expected width of the object between the fingers, in metres,
            or None if it is unknown.
        :return: A new task shutting `gripper` on whatever lies between its fingers,
            which succeeds where the fingers stop within :attr:`grasp_width_tolerance`
            of `width`, or wherever they stop if `width` is None.

        NOTE: width is not needed anymore, so the commit it added is not needed and can be reverted.
        also the tolerances must be quite high, since the gripper expects its goal to be at exact width, and if thats
        not true then it just aborts. same for force
        """
        action_topic = self.action_name(gripper, FrankaGripperActionName.GRASP)
        if width is None:
            return GripperGraspTask(
                action_topic=action_topic,
                width=0.0,
                speed=self.finger_speed,
                force=self.grasp_force,
                inner_tolerance=0.1,
                outer_tolerance=0.1,
            )
        return GripperGraspTask(
            action_topic=action_topic,
            width=0.0,
            speed=self.finger_speed,
            force=self.grasp_force,
            inner_tolerance=0.1,
            outer_tolerance=0.1,
        )


@dataclass
class DriveGarmiGripperStatesWithTheirFrankaDriver(
    DriveGarmiGrippersWithTheirFrankaDriver[MoveGripper]
):
    """
    Opens a gripper of GARMI fully, or shuts it on whatever lies between its fingers,
    through its Franka gripper driver.
    """

    def is_applicable(self, plan_node: MoveGripper) -> bool:
        return isinstance(
            plan_node.end_effector, (GarmiLeftGripper, GarmiRightGripper)
        ) and plan_node.state in (GripperState.OPEN, GripperState.CLOSE)

    def apply(self, plan_node: MoveGripper) -> None:
        """
        Puts the task in place of the step that drives the fingers: `plan_node`, or the
        action whose goal it is.
        """
        gripper = plan_node.end_effector
        if plan_node.state is GripperState.CLOSE:
            task = self.grasp(gripper)
        else:
            task = GripperMoveTask(
                action_topic=self.action_name(gripper, FrankaGripperActionName.MOVE),
                width=gripper.separation_when_open,
                speed=self.finger_speed,
            )
        plan_node.statechart.replace(self._step_holding(plan_node), task)


@dataclass
class CloseGarmiFingersToWidthWithTheirFrankaDriver(
    DriveGarmiGrippersWithTheirFrankaDriver[MoveFingersToWidthAction]
):
    """
    Shuts a gripper of GARMI on the object it is to meet at its width through its Franka
    gripper driver, whose grasp stops the fingers where the object pushes back.
    """

    def is_applicable(self, plan_node: MoveFingersToWidthAction) -> bool:
        return isinstance(plan_node.gripper, (GarmiLeftGripper, GarmiRightGripper))

    def apply(self, plan_node: MoveFingersToWidthAction) -> None:
        plan_node.statechart.replace(
            plan_node, self.grasp(plan_node.gripper, plan_node.width)
        )
