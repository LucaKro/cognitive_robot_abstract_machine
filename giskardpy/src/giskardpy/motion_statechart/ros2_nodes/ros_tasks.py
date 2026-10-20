from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from franka_msgs.action import Grasp, Homing, Move
from franka_msgs.msg import GraspEpsilon
from geometry_msgs.msg import (
    PoseStamped as ROSPoseStamped,
    Pose as ROSPose,
    Point as ROSPoint,
    Quaternion as ROSQuaternion,
)

try:
    from nav2_msgs.action import NavigateToPose
except ModuleNotFoundError:
    NavigateToPose = None
from rclpy.action import ActionClient
from rclpy.task import Future
from std_msgs.msg import Header
from typing_extensions import Type, TypeVar, Generic

import krrood.symbolic_math.symbolic_math as sm
from krrood.ormatic.utils import classproperty
from cramph.node import EndedByOwner
from cramph.context import ContextExtension
from cramph.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from cramph.node import NodeArtifacts
from giskardpy.motion_statechart.ros_context import RosContextExtension
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


Action = TypeVar("Action")
ActionGoal = TypeVar("ActionGoal")
ActionResult = TypeVar("ActionResult")
ActionFeedback = TypeVar("ActionFeedback")


@dataclass(eq=False, repr=False, kw_only=True)
class ActionServerTask(
    EndedByOwner,
    MotionStatechartNode,
    ABC,
    Generic[Action, ActionGoal, ActionResult, ActionFeedback],
):
    """
    Abstract base class for tasks that call a ROS2 action server.
    """

    action_topic: str
    """
    Topic name for the action server.
    """

    message_type: Type[Action]
    """
    Fully specified goal message that can be send out.
    """

    _action_client: ActionClient = field(init=False)
    """
    ROS action client, is created in `set_up`.
    """

    _msg: ActionGoal = field(init=False, default=None)
    """
    ROS message to send to the action server.
    """

    _result: ActionResult = field(init=False, default=None)
    """
    ROS action server result.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        return super().required_context_extensions + (RosContextExtension,)

    @abstractmethod
    def build_msg(self) -> None:
        """
        Build the goal message this node sends to the action server.
        """
        ...

    def set_up(self) -> None:
        """
        Creates the action client and the message to send.
        """
        super().set_up()
        ros_context_extension = self.context.require_extension(RosContextExtension)
        self._action_client = ros_context_extension.get_or_create_action_client(
            self.message_type, self.action_topic
        )
        self.build_msg()
        logger.info(f"Waiting for action server {self.action_topic}")
        self._action_client.wait_for_server()

    def on_start(self) -> None:
        """
        Creates a goal and sends it to the action server asynchronously.
        """
        future = self._action_client.send_goal_async(self._msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handles the server's response to the goal submission.

        On rejection a failure sentinel is stored so that :meth:`on_tick` can return
        :attr:`~ObservationStateValues.FALSE` immediately.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.error("Goal rejected by action server")
            return

        logger.info("Sent query to action server ")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        self._result = future.result()
        logger.info(f"Action server {self.action_topic} done.")


@dataclass(eq=False, repr=False, kw_only=True)
class NavigateActionServerTask(
    ActionServerTask[
        NavigateToPose,
        NavigateToPose.Goal,
        NavigateToPose.Result,
        NavigateToPose.Feedback,
    ]
):
    """
    Node for calling a Navigation2 ROS2 action server to navigate to a given pose.
    """

    target_pose: Pose
    """
    Target pose to which the robot should navigate.
    """

    base_link: Body
    """
    Base link of the robot, used for estimating the distance to the goal.
    """

    def build_msg(self) -> None:
        root_p_goal = self.context.world.transform(
            target_frame=self.context.world.root, spatial_object=self.target_pose
        )
        position = root_p_goal.position.to_np()
        orientation = root_p_goal.quaternion.to_np()
        pose_stamped = ROSPoseStamped(
            header=Header(frame_id="map"),
            pose=ROSPose(
                position=ROSPoint(x=position[0], y=position[1], z=position[2]),
                orientation=ROSQuaternion(
                    x=orientation[0],
                    y=orientation[1],
                    z=orientation[2],
                    w=orientation[3],
                ),
            ),
        )
        self._msg = NavigateToPose.Goal(pose=pose_stamped)

    def build_artifacts(self) -> NodeArtifacts:
        """
        Observes whether the robot is within 1cm of the target pose.
        """
        artifacts = NodeArtifacts()
        root_T_goal = self.context.world.transform(
            target_frame=self.context.world.root, spatial_object=self.target_pose
        )
        root_T_current = self.context.world.compose_forward_kinematics_expression(
            self.context.world.root, self.base_link
        )

        position_error = root_T_goal.position.euclidean_distance(
            root_T_current.position
        )
        rotation_error = root_T_goal.rotation_matrix.rotational_distance(
            root_T_current.rotation_matrix
        )

        artifacts.observation = sm.trinary_logic_and(
            position_error < 0.01, sm.abs(rotation_error) < 0.01
        )

        return artifacts

    def result_callback(self, future):
        """
        Stores the navigation result returned by the action server.
        """
        # TODO: Check the ._result.result calls against a running action server because I'm not sure if all of them are correct
        super().result_callback(future)
        logger.info(
            f"Finished navigation with response status: {self._result.result.status} and result code: {self._result.error_code}"
        )

    def on_tick(self) -> ObservationStateValues:
        if self._result.result:
            return (
                ObservationStateValues.TRUE
                if self._result.error_code == NavigateToPose.Result.NONE
                else ObservationStateValues.FALSE
            )
        return ObservationStateValues.UNKNOWN


@dataclass(eq=False, repr=False, kw_only=True)
class FrankaGripperTask(
    ActionServerTask[Action, ActionGoal, ActionResult, ActionFeedback], ABC
):
    """
    Calls an action server of a Franka gripper, whose results report ``success`` and
    ``error``.
    """

    def result_callback(self, future: Future) -> None:
        super().result_callback(future)
        if not self._result.result.success:
            logger.error(
                f"Gripper action {self.action_topic} failed: {self._result.result.error}"
            )

    def on_tick(self) -> ObservationStateValues:
        """
        :return: Whether the gripper reported success, unknown until it reports.
        """
        if self._result is None:
            return ObservationStateValues.UNKNOWN
        if self._result.result.success:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


@dataclass(eq=False, repr=False, kw_only=True)
class GripperMoveTask(FrankaGripperTask[Move, Move.Goal, Move.Result, Move.Feedback]):
    """
    Moves the fingers of a Franka gripper to a width without checking for a grasped
    object.
    """

    message_type: type[Move] = field(default=Move, init=False)

    width: float
    """
    Distance between the fingers to move to, in meters.
    """

    speed: float
    """
    Speed of the fingers, in meters per second.
    """

    def build_msg(self) -> None:
        self._msg = Move.Goal(width=self.width, speed=self.speed)


@dataclass(eq=False, repr=False, kw_only=True)
class GripperGraspTask(
    FrankaGripperTask[Grasp, Grasp.Goal, Grasp.Result, Grasp.Feedback]
):
    """
    Closes a Franka gripper on an object and succeeds if the grasped width lies within
    the tolerance of :attr:`width`.
    """

    message_type: type[Grasp] = field(default=Grasp, init=False)

    width: float
    """
    Expected width of the grasped object, in meters.
    """

    speed: float
    """
    Speed of the fingers, in meters per second.
    """

    force: float
    """
    Grasping force, in newtons.
    """

    inner_tolerance: float
    """
    Largest accepted amount by which the grasped width falls below :attr:`width`, in
    meters.
    """

    outer_tolerance: float
    """
    Largest accepted amount by which the grasped width exceeds :attr:`width`, in meters.
    """

    def build_msg(self) -> None:
        self._msg = Grasp.Goal(
            width=self.width,
            epsilon=GraspEpsilon(
                inner=self.inner_tolerance, outer=self.outer_tolerance
            ),
            speed=self.speed,
            force=self.force,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class GripperHomingTask(
    FrankaGripperTask[Homing, Homing.Goal, Homing.Result, Homing.Feedback]
):
    """
    Homes a Franka gripper, which calibrates its maximum width.
    """

    message_type: type[Homing] = field(default=Homing, init=False)

    def build_msg(self) -> None:
        self._msg = Homing.Goal()
