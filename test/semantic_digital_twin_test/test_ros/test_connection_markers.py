from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from visualization_msgs.msg import InteractiveMarkerFeedback

from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)
from semantic_digital_twin.adapters.ros.visualization.connection_markers import (
    Connection6DoFMarker,
    ConnectionMarker,
    ConnectionMarkerModelCallback,
    ConnectionMarkerServer,
    PrismaticConnectionMarker,
    RevoluteConnectionMarker,
    ScrewConnectionMarker,
    WheeledDriveMarker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
    ScrewConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.soft_connections import (
    PiecewiseConstantCurvatureConnection,
)
from semantic_digital_twin.world_description.world_entity import Body, Connection

# %% a world with every kind of connection


@dataclass
class ConnectionsOfEveryKind:
    """
    A world whose root carries one child body per kind of connection.
    """

    world: World
    """
    The world the connections belong to.
    """

    fixed: FixedConnection
    """
    A connection without degrees of freedom.
    """

    free: Connection6DoF
    """
    A connection that moves freely in all six degrees of freedom.
    """

    drive: OmniDrive
    """
    A mobile base driving on the floor plane.
    """

    slider: PrismaticConnection
    """
    A slider that scales and offsets its degree of freedom, with limited travel.
    """

    hinge: RevoluteConnection
    """
    A limited hinge whose connection frame is rotated against both its parent and its
    child.
    """

    screw: ScrewConnection
    """
    An unlimited screw.
    """

    soft: PiecewiseConstantCurvatureConnection
    """
    A soft connection, which has degrees of freedom but no pose that can be set.
    """

    @classmethod
    def build(cls) -> ConnectionsOfEveryKind:
        """
        :return: A fresh world with one connection of every kind below its root.
        """
        world = World.create_with_root_body("map")
        limits = DegreeOfFreedomLimits.from_position_range_and_speed
        kappa = DegreeOfFreedom(name=PrefixedName("kappa"), limits=limits(-10, 10, 10))
        phi = DegreeOfFreedom(name=PrefixedName("phi"), limits=limits(-10, 10, 10))
        with world.modify_world():
            bodies = {
                name: Body(name=PrefixedName(name))
                for name in (
                    "fixed_box",
                    "free_box",
                    "base",
                    "drawer",
                    "door",
                    "lid",
                    "trunk",
                )
            }
            for body in bodies.values():
                world.add_body(body)
            world.add_degree_of_freedom(kappa)
            world.add_degree_of_freedom(phi)
            fixed = FixedConnection(parent=world.root, child=bodies["fixed_box"])
            free = Connection6DoF.create_with_dofs(
                world, world.root, bodies["free_box"]
            )
            drive = OmniDrive.create_with_dofs(world, world.root, bodies["base"])
            slider = PrismaticConnection.create_with_dofs(
                world,
                world.root,
                bodies["drawer"],
                axis=Vector3.X(),
                multiplier=2.0,
                offset=0.1,
                dof_limits=limits(0.0, 0.5, 1.0),
            )
            hinge = RevoluteConnection.create_with_dofs(
                world,
                world.root,
                bodies["door"],
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.0, yaw=0.5
                ),
                connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=0.2, roll=0.3
                ),
                axis=Vector3.Z(),
                dof_limits=limits(-1.0, 1.0, 1.0),
            )
            screw = ScrewConnection.create_with_dofs(
                world, world.root, bodies["lid"], axis=Vector3.Z(), screw_pitch=0.01
            )
            soft = PiecewiseConstantCurvatureConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=bodies["trunk"],
                kappa_dof_id=kappa.id,
                phi_dof_id=phi.id,
                segment_length=1.0,
            )
            for connection in (fixed, free, drive, slider, hinge, screw, soft):
                world.add_connection(connection)
        return cls(
            world=world,
            fixed=fixed,
            free=free,
            drive=drive,
            slider=slider,
            hinge=hinge,
            screw=screw,
            soft=soft,
        )

    @property
    def movable(self) -> list[Connection]:
        """
        The connections whose pose a marker can set.
        """
        return [self.free, self.drive, self.slider, self.hinge, self.screw]


def origin_at(
    connection: ActiveConnection1DOF, position: float
) -> HomogeneousTransformationMatrix:
    """
    :param connection: The connection to look at.
    :param position: The position to look at it in.
    :return: the origin the connection has in the given position, leaving its state
        as it was.
    """
    original_position = connection.position
    connection.position = position
    origin = connection.origin
    connection.position = original_position
    return origin


def drag(
    server: ConnectionMarkerServer,
    connection: Connection,
    parent_T_goal: HomogeneousTransformationMatrix,
    event_type: int = InteractiveMarkerFeedback.POSE_UPDATE,
) -> None:
    """
    Report to the server that the marker of a connection was dragged to a goal.

    :param server: The server owning the marker.
    :param connection: The connection whose marker is dragged.
    :param parent_T_goal: Where the marker was dragged to, in the parent frame.
    :param event_type: The kind of interaction reported.
    """
    server.process_feedback(
        InteractiveMarkerFeedback(
            marker_name=connection.id.hex,
            event_type=event_type,
            pose=PoseToRos2Converter.convert(parent_T_goal.pose),
        )
    )


# %% which connections get a marker


def test_every_movable_connection_gets_a_marker(rclpy_node):
    connections = ConnectionsOfEveryKind.build()

    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)

    assert set(server.markers) == {
        connection.id.hex for connection in connections.movable
    }


def test_each_kind_of_connection_is_handled_by_its_own_marker():
    connections = ConnectionsOfEveryKind.build()

    marker_classes = {
        type(connection): ConnectionMarker.marker_class_for(connection)
        for connection in connections.world.connections
    }

    assert marker_classes == {
        FixedConnection: None,
        Connection6DoF: Connection6DoFMarker,
        OmniDrive: WheeledDriveMarker,
        PrismaticConnection: PrismaticConnectionMarker,
        RevoluteConnection: RevoluteConnectionMarker,
        ScrewConnection: ScrewConnectionMarker,
        PiecewiseConstantCurvatureConnection: None,
    }


# %% where a marker is shown


def test_a_marker_is_shown_in_the_tf_frame_of_the_parent(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)
    hinge = connections.hinge

    marker = server.interactive_marker_server.get(hinge.id.hex)

    frame_names = server.tf_publisher.tf_model_callback.frame_names
    assert marker.header.frame_id == frame_names.assign(hinge.parent)


def test_a_marker_is_shown_at_the_origin_of_its_connection(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    hinge = connections.hinge
    hinge.position = 0.3
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)

    marker = server.interactive_marker_server.get(hinge.id.hex)

    assert marker.pose == PoseToRos2Converter.convert(hinge.origin.pose)


def test_a_marker_follows_a_state_change_made_elsewhere(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)
    slider = connections.slider

    slider.position = 0.4

    marker = server.interactive_marker_server.get(slider.id.hex)
    assert marker.pose == PoseToRos2Converter.convert(slider.origin.pose)


def test_a_connection_added_later_gets_a_marker(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    world = connections.world
    server = ConnectionMarkerServer(node=rclpy_node, _world=world)
    handle = Body(name=PrefixedName("handle"))

    with world.modify_world():
        world.add_body(handle)
        connection = RevoluteConnection.create_with_dofs(
            world, world.root, handle, axis=Vector3.Y()
        )
        world.add_connection(connection)

    assert connection.id.hex in server.markers


# %% dragging a marker


def test_dragging_a_free_connection_moves_it_to_the_goal(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)
    free = connections.free
    goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.3, y=-0.2, z=0.5, roll=0.1, pitch=0.2, yaw=0.3, reference_frame=free.parent
    )

    drag(server, free, goal)

    np.testing.assert_allclose(free.origin.to_np(), goal.to_np(), atol=1e-9)


def test_dragging_a_drive_moves_it_on_the_floor_plane_only(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)
    drive = connections.drive
    goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.3, y=0.2, z=0.5, roll=0.1, yaw=0.4, reference_frame=drive.parent
    )

    drag(server, drive, goal)

    expected = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, y=0.2, yaw=0.4)
    np.testing.assert_allclose(drive.origin.to_np(), expected.to_np(), atol=1e-9)


def test_dragging_a_scaled_slider_reaches_the_dragged_position(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)
    slider = connections.slider

    drag(server, slider, origin_at(slider, 0.6))

    assert np.isclose(slider.position, 0.6)


def test_dragging_a_rotated_hinge_reaches_the_dragged_position(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)
    hinge = connections.hinge

    drag(server, hinge, origin_at(hinge, -0.7))

    assert np.isclose(hinge.position, -0.7)


def test_dragging_a_screw_reaches_the_dragged_position(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)
    screw = connections.screw

    drag(server, screw, origin_at(screw, 2.0))

    assert np.isclose(screw.position, 2.0)


def test_dragging_a_hinge_beyond_its_limit_stops_at_the_limit(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)
    hinge = connections.hinge
    upper_limit = hinge.dof.limits.upper.position

    drag(server, hinge, origin_at(hinge, upper_limit + 0.5))

    assert np.isclose(hinge.position, upper_limit)


def test_a_marker_dragged_beyond_the_limit_returns_to_the_connection(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)
    hinge = connections.hinge

    drag(server, hinge, origin_at(hinge, hinge.dof.limits.upper.position + 0.5))

    marker = server.interactive_marker_server.get(hinge.id.hex)
    assert marker.pose == PoseToRos2Converter.convert(hinge.origin.pose)


def test_pressing_a_marker_leaves_the_world_state_alone(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    server = ConnectionMarkerServer(node=rclpy_node, _world=connections.world)
    hinge = connections.hinge
    position_before = hinge.position

    drag(
        server,
        hinge,
        origin_at(hinge, 0.5),
        event_type=InteractiveMarkerFeedback.MOUSE_DOWN,
    )

    assert hinge.position == position_before


# %% server lifetime


def test_a_stopped_server_is_no_longer_notified(rclpy_node):
    connections = ConnectionsOfEveryKind.build()
    world = connections.world
    server = ConnectionMarkerServer(node=rclpy_node, _world=world)

    server.stop()

    assert ConnectionMarkerServer.all_callbacks_of_this_type_from_world(world) == []
    assert (
        ConnectionMarkerModelCallback.all_callbacks_of_this_type_from_world(world) == []
    )
