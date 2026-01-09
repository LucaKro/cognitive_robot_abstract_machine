from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rclpy.node import MsgType
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription
from typing_extensions import Generic, Type

import krrood.symbolic_math.symbolic_math as sm
from ..context import ExecutionContext, BuildContext
from ..data_types import ObservationStateValues
from ..graph_node import MotionStatechartNode, NodeArtifacts
from ..ros_context import RosContextExtension


@dataclass
class TopicSubscriberNode(MotionStatechartNode, Generic[MsgType]):
    topic_name: str = field(kw_only=True)
    msg_type: Type[MsgType] = field(kw_only=True)
    qos_profile: QoSProfile | int = field(kw_only=True, default=10)
    _subscriber: Subscription = field(init=False)
    current_msg: MsgType | None = field(init=False, default=None)
    __last_msg: MsgType | None = field(init=False, default=None)

    def build(self, context: BuildContext) -> NodeArtifacts:
        ros_context_extension = context.require_extension(RosContextExtension)
        self._subscriber = ros_context_extension.ros_node.create_subscription(
            msg_type=self.msg_type,
            topic=self.topic_name,
            callback=self.callback,
            qos_profile=self.qos_profile,
        )
        return NodeArtifacts()

    def callback(self, msg: MsgType):
        self.__last_msg = msg

    def has_msg(self) -> bool:
        return self.__last_msg is not None

    def clear_msg(self):
        self.__last_msg = None

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        self.current_msg = self.__last_msg


@dataclass(eq=False, repr=False)
class TopicPublisherNode(MotionStatechartNode, Generic[MsgType]):
    topic_name: str = field(kw_only=True)
    msg_type: Type[MsgType] = field(kw_only=True)
    qos_profile: QoSProfile | int = field(kw_only=True, default=10)
    publisher: Publisher = field(init=False)

    def build(self, context: BuildContext) -> NodeArtifacts:
        ros_context_extension = context.require_extension(RosContextExtension)
        self.publisher = ros_context_extension.ros_node.create_publisher(
            msg_type=self.msg_type,
            topic=self.topic_name,
            qos_profile=self.qos_profile,
        )
        return NodeArtifacts()


@dataclass(eq=False, repr=False)
class WaitForMessage(TopicSubscriberNode[MsgType]):
    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        super().on_tick(context)
        if self.has_msg():
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


@dataclass(eq=False, repr=False)
class PublishOnStart(TopicPublisherNode[MsgType]):
    msg: MsgType = field(kw_only=True)
    msg_type: Type[MsgType] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.msg_type = type(self.msg)

    def build(self, context: BuildContext) -> NodeArtifacts:
        node_artifacts = super().build(context)
        node_artifacts.observation = sm.Scalar.const_true()
        return node_artifacts

    def on_start(self, context: ExecutionContext):
        self.publisher.publish(self.msg)
