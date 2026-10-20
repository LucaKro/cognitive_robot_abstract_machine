from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from geometry_msgs.msg import Pose as RosPose
from geometry_msgs.msg import Quaternion as RosQuaternion
from geometry_msgs.msg import Vector3 as RosVector3
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from typing_extensions import Generic, TypeVar
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import recursive_subclasses
from semantic_digital_twin.adapters.ros.ros2_to_semdt_converters import (
    PoseToSemDTConverter,
)
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    ColorToRos2Converter,
    PoseToRos2Converter,
)
from semantic_digital_twin.adapters.ros.tf_publisher import (
    PositionedInTfTree,
    TfFrameNames,
)
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.spatial_types import RotationMatrix
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    PrismaticConnection,
    RevoluteConnection,
    ScrewConnection,
    WheeledDrive,
)
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Connection

ConnectionType = TypeVar("ConnectionType", bound=Connection)
"""
The kind of connection a marker sets the state of.
"""

SingleAxisConnectionType = TypeVar(
    "SingleAxisConnectionType", bound=ActiveConnection1DOF
)
"""
The kind of connection with one degree of freedom a marker sets the state of.
"""

# %% marker controls


class ControlName(StrEnum):
    """
    The names of the controls a marker can be dragged by.
    """

    MOVE_X = "move_x"
    """
    Moves along the x-axis of the child.
    """

    MOVE_Y = "move_y"
    """
    Moves along the y-axis of the child.
    """

    MOVE_Z = "move_z"
    """
    Moves along the z-axis of the child.
    """

    ROTATE_X = "rotate_x"
    """
    Rotates about the x-axis of the child.
    """

    ROTATE_Y = "rotate_y"
    """
    Rotates about the y-axis of the child.
    """

    ROTATE_Z = "rotate_z"
    """
    Rotates about the z-axis of the child.
    """

    DRIVE = "drive"
    """
    Moves within the floor plane.
    """

    TURN = "turn"
    """
    Rotates about the vertical axis.
    """

    MOVE_ALONG_AXIS = "move_along_axis"
    """
    Moves along the axis of a connection with one degree of freedom.
    """

    ROTATE_ABOUT_AXIS = "rotate_about_axis"
    """
    Rotates about the axis of a connection with one degree of freedom.
    """


# %% one marker per connection


@dataclass
class ConnectionMarker(Generic[ConnectionType], SubClassSafeGeneric, ABC):
    """
    An RViz interactive marker that sets the state of one connection when dragged.

    The marker is shown in the frame of the parent at the origin of the connection, so
    its controls are expressed in the frame of the child.
    """

    connection: ConnectionType
    """
    The connection whose state the marker sets.
    """

    scale: float = 0.25
    """
    The size of the controls in meters.
    """

    box_size: float = 0.175
    """
    The edge length of the box shown at the origin of the connection, in meters.
    """

    box_color: Color = field(default_factory=lambda: Color(0.5, 0.5, 0.5, 0.5))
    """
    The color of the box shown at the origin of the connection.
    """

    @classmethod
    def marker_class_for(cls, connection: Connection) -> type[ConnectionMarker] | None:
        """
        :param connection: The connection to find a marker for.
        :return: the marker handling the most specific type of the connection, or
            ``None`` if no marker handles it.
        """
        handled_type_per_marker_class = {
            marker_class: marker_class.get_generic_type_parameters()[0]
            for marker_class in recursive_subclasses(cls)
        }
        marker_class_per_connection_type = {
            handled_type: marker_class
            for marker_class, handled_type in handled_type_per_marker_class.items()
            if not isinstance(handled_type, TypeVar)
        }
        return next(
            (
                marker_class_per_connection_type[connection_type]
                for connection_type in type(connection).__mro__
                if connection_type in marker_class_per_connection_type
            ),
            None,
        )

    @property
    def name(self) -> str:
        """
        The name the marker is served under, unique among the connections of a world.
        """
        return self.connection.id.hex

    @property
    def pose(self) -> RosPose:
        """
        The origin of the connection, in the frame of its parent.
        """
        return PoseToRos2Converter.convert(self.connection.origin.pose)

    def interactive_marker(self, frame_names: TfFrameNames) -> InteractiveMarker:
        """
        .. note:: The header keeps a zero stamp, so RViz locks the marker to the frame
            of the parent and reports where it was dragged to in that frame.

        :param frame_names: The tf frame names the entities of the world are published
            under.
        :return: the message showing the box and the :meth:`controls` at the origin of
            the connection.
        """
        interactive_marker = InteractiveMarker(
            name=self.name,
            description=str(self.connection.name),
            scale=self.scale,
            pose=self.pose,
            controls=[self._box_control(), *self.controls()],
        )
        interactive_marker.header.frame_id = frame_names.assign(self.connection.parent)
        return interactive_marker

    def _box_control(self) -> InteractiveMarkerControl:
        """
        :return: a control that only shows a box at the origin of the connection.
        """
        box = Marker(
            type=Marker.CUBE,
            scale=RosVector3(x=self.box_size, y=self.box_size, z=self.box_size),
            color=ColorToRos2Converter.convert(self.box_color),
        )
        return InteractiveMarkerControl(
            always_visible=True,
            markers=[box],
            interaction_mode=InteractiveMarkerControl.NONE,
        )

    @staticmethod
    def _axis_control(
        name: ControlName, interaction_mode: int, axis: Vector3
    ) -> InteractiveMarkerControl:
        """
        :param name: The name of the control.
        :param interaction_mode: The :class:`InteractiveMarkerControl` interaction mode
            of the control.
        :param axis: The axis, in the frame of the child, the control moves along,
            rotates about, or moves in the plane orthogonal to.
        :return: a control acting along the axis.
        """
        orientation = RotationMatrix.from_x_axis(axis).quaternion.to_np()
        return InteractiveMarkerControl(
            name=name,
            interaction_mode=interaction_mode,
            orientation=RosQuaternion(
                x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]
            ),
        )

    @abstractmethod
    def controls(self) -> list[InteractiveMarkerControl]:
        """
        :return: the controls the marker can be dragged by, oriented in the frame of the
            child.
        """

    @abstractmethod
    def move_to(self, parent_T_goal: HomogeneousTransformationMatrix) -> None:
        """
        Set the state of the connection that brings its child closest to a goal.

        :param parent_T_goal: The pose the marker was dragged to, in the frame of the
            parent.
        """


# %% connections whose origin can be set


@dataclass
class SettableOriginMarker(ConnectionMarker[ConnectionType], ABC):
    """
    A marker for a connection that reaches a goal by having its origin set to it.
    """

    def move_to(self, parent_T_goal: HomogeneousTransformationMatrix) -> None:
        self.connection.origin = parent_T_goal


@dataclass
class Connection6DoFMarker(SettableOriginMarker[Connection6DoF]):
    """
    A marker that moves and rotates a free connection along every axis of its child.
    """

    def controls(self) -> list[InteractiveMarkerControl]:
        move, rotate = (
            InteractiveMarkerControl.MOVE_AXIS,
            InteractiveMarkerControl.ROTATE_AXIS,
        )
        return [
            self._axis_control(ControlName.MOVE_X, move, Vector3.X()),
            self._axis_control(ControlName.MOVE_Y, move, Vector3.Y()),
            self._axis_control(ControlName.MOVE_Z, move, Vector3.Z()),
            self._axis_control(ControlName.ROTATE_X, rotate, Vector3.X()),
            self._axis_control(ControlName.ROTATE_Y, rotate, Vector3.Y()),
            self._axis_control(ControlName.ROTATE_Z, rotate, Vector3.Z()),
        ]


@dataclass
class WheeledDriveMarker(SettableOriginMarker[WheeledDrive]):
    """
    A marker that drives a mobile base within the floor plane and turns it about the
    vertical axis.
    """

    def controls(self) -> list[InteractiveMarkerControl]:
        return [
            self._axis_control(
                ControlName.DRIVE, InteractiveMarkerControl.MOVE_PLANE, Vector3.Z()
            ),
            self._axis_control(
                ControlName.TURN, InteractiveMarkerControl.ROTATE_AXIS, Vector3.Z()
            ),
        ]


# %% connections with one degree of freedom


@dataclass
class SingleAxisConnectionMarker(ConnectionMarker[SingleAxisConnectionType], ABC):
    """
    A marker for a connection with one degree of freedom, which reaches a goal by
    changing its position by as much as the goal lies along its axis, within the limits
    of its degree of freedom.
    """

    @property
    def child_axis(self) -> Vector3:
        """
        The axis of the connection, in the frame of its child.
        """
        connection_T_child = self.connection.connection_T_child_expression
        return connection_T_child.inverse().rotation_matrix.dot(self.connection.axis)

    def move_to(self, parent_T_goal: HomogeneousTransformationMatrix) -> None:
        child_T_goal = self.connection.origin.inverse() @ parent_T_goal
        position = self.connection.position + self.position_change(child_T_goal)
        limits = self.connection.dof.limits
        if limits.lower.position is not None:
            position = max(position, limits.lower.position)
        if limits.upper.position is not None:
            position = min(position, limits.upper.position)
        self.connection.position = position

    @abstractmethod
    def position_change(self, child_T_goal: HomogeneousTransformationMatrix) -> float:
        """
        :param child_T_goal: The pose the marker was dragged to, in the frame of the
            child.
        :return: by how much the position of the connection has to change for the child
            to reach the goal along the axis.
        """


@dataclass
class PrismaticConnectionMarker(SingleAxisConnectionMarker[PrismaticConnection]):
    """
    A marker that slides a prismatic connection along its axis.
    """

    def controls(self) -> list[InteractiveMarkerControl]:
        return [
            self._axis_control(
                ControlName.MOVE_ALONG_AXIS,
                InteractiveMarkerControl.MOVE_AXIS,
                self.child_axis,
            )
        ]

    def position_change(self, child_T_goal: HomogeneousTransformationMatrix) -> float:
        return float(
            np.dot(self.child_axis.to_np()[:3], child_T_goal.position.to_np()[:3])
        )


@dataclass
class AxisRotationMarker(SingleAxisConnectionMarker[SingleAxisConnectionType], ABC):
    """
    A marker that turns a connection whose position is an angle about its axis.
    """

    def controls(self) -> list[InteractiveMarkerControl]:
        return [
            self._axis_control(
                ControlName.ROTATE_ABOUT_AXIS,
                InteractiveMarkerControl.ROTATE_AXIS,
                self.child_axis,
            )
        ]

    def position_change(self, child_T_goal: HomogeneousTransformationMatrix) -> float:
        """
        :return: the signed angle the goal is rotated by about the axis, within
            ``(-pi, pi]``.
        """
        quaternion = child_T_goal.quaternion.to_np()
        sine_part = np.dot(self.child_axis.to_np()[:3], quaternion[:3])
        angle = 2 * np.arctan2(sine_part, quaternion[3])
        return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass
class RevoluteConnectionMarker(AxisRotationMarker[RevoluteConnection]):
    """
    A marker that turns a revolute connection about its axis.
    """


@dataclass
class ScrewConnectionMarker(AxisRotationMarker[ScrewConnection]):
    """
    A marker that turns a screw about its axis, which also advances it along the axis.
    """


# %% serving the markers of a world


@dataclass(eq=False)
class ConnectionMarkerServer(StateChangeCallback, PositionedInTfTree):
    """
    Serves an RViz interactive marker for every connection of the world a
    :class:`ConnectionMarker` handles, and sets the state of a connection while its
    marker is dragged.

    Connections no marker handles, such as fixed connections and soft connections, get
    no marker.

    .. warning:: To drag the markers in RViz, add an InteractiveMarkers display on
        :attr:`topic_namespace` and set the fixed frame to the tf root.
    """

    topic_namespace: str = "connection_markers"
    """
    The namespace the interactive marker topics are served under.
    """

    interactive_marker_server: InteractiveMarkerServer = field(init=False)
    """
    The server RViz receives the markers from and reports interactions to.
    """

    markers: dict[str, ConnectionMarker] = field(init=False, default_factory=dict)
    """
    The served markers, by the name they are served under.
    """

    model_callback: ConnectionMarkerModelCallback = field(init=False)
    """
    Shows the markers anew whenever the model of the world changes.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._adopt_tf_publisher()
        self.interactive_marker_server = InteractiveMarkerServer(
            self.node, self.topic_namespace
        )
        self.model_callback = ConnectionMarkerModelCallback(
            _world=self._world, connection_marker_server=self
        )
        self.model_callback.notify_model_change()

    def show_markers(self) -> None:
        """
        Replace the served markers with one for every connection of the world a
        :class:`ConnectionMarker` handles.
        """
        self.interactive_marker_server.clear()
        self.markers = {}
        for connection in self._world.connections:
            marker_class = ConnectionMarker.marker_class_for(connection)
            if marker_class is None:
                continue
            marker = marker_class(connection=connection)
            self.markers[marker.name] = marker
            self.interactive_marker_server.insert(
                marker.interactive_marker(self.frame_names),
                feedback_callback=self.process_feedback,
            )
        self.interactive_marker_server.applyChanges()

    def process_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        """
        Move the connection of a marker that is being dragged to where it was dragged.

        :param feedback: The interaction RViz reports, posed in the frame of the parent
            of the connection.
        """
        if feedback.event_type != InteractiveMarkerFeedback.POSE_UPDATE:
            return
        marker = self.markers[feedback.marker_name]
        parent_P_goal = PoseToSemDTConverter.convert(feedback.pose, self._world)
        parent_P_goal.reference_frame = marker.connection.parent
        marker.move_to(parent_P_goal.homogeneous_matrix)

    def on_state_change(self, **kwargs) -> None:
        for name, marker in self.markers.items():
            self.interactive_marker_server.setPose(name, marker.pose)
        self.interactive_marker_server.applyChanges()

    def stop(self) -> None:
        """
        Deregister this server and the model callback it owns, and take its markers
        down.
        """
        self.model_callback.stop()
        self.interactive_marker_server.shutdown()
        super().stop()


@dataclass(eq=False)
class ConnectionMarkerModelCallback(ModelChangeCallback):
    """
    Shows the markers of a :class:`ConnectionMarkerServer` anew whenever the model of
    its world changes.
    """

    connection_marker_server: ConnectionMarkerServer = field(kw_only=True)
    """
    The server whose markers are shown anew.
    """

    def on_model_change(self, **kwargs) -> None:
        self.connection_marker_server.show_markers()
