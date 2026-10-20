"""
Calibrate the poses of the bodies of an MJCF scene by dragging them in RViz, and write
them back into the scene file.

Every movable connection of the scene gets an interactive marker. Bodies named with
``--free`` are detached from the joint or fixed connection to their parent, so they can
be dragged freely relative to it. Press Enter to write the pose of every freely movable
body into the scene file; enter ``q`` or press Ctrl-D to quit.

In RViz, set the fixed frame to ``map`` and add an InteractiveMarkers display on
``connection_markers``.

Example, placing the handles of three drawers::

    python calibrate_mjcf_body_poses.py scene-bodies.xml \\
        --free drawer_1_handle drawer_2_handle drawer_3_handle
"""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass, field
from enum import StrEnum

import rclpy
from rclpy.executors import SingleThreadedExecutor

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.mjcf_body_pose_writer import MJCFBodyPoseWriter
from semantic_digital_twin.adapters.ros.visualization.connection_markers import (
    ConnectionMarkerServer,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% command line


class CalibrationCommand(StrEnum):
    """
    What can be entered while calibrating, besides an empty line to write the poses.
    """

    QUIT = "q"
    """
    Stop calibrating without writing the poses again.
    """


@dataclass
class MJCFBodyPoseCalibration:
    """
    Shows an MJCF scene in RViz with an interactive marker on every movable connection,
    and writes the poses of its freely movable bodies back into the scene file on
    request.
    """

    file_path: str
    """
    The MJCF scene file to calibrate.
    """

    free_body_names: list[str] = field(default_factory=list)
    """
    The names of the bodies that are freed from their joint or fixed connection to their
    parent for the calibration.
    """

    tf_root_name: str = "map"
    """
    The name of the frame the scene is published under, which RViz takes as fixed frame.
    """

    node_name: str = "mjcf_body_pose_calibration"
    """
    The name of the ROS node serving the markers.
    """

    world: World = field(init=False)
    """
    The world parsed from the scene file.
    """

    pose_writer: MJCFBodyPoseWriter = field(init=False)
    """
    Writes the poses of the freely movable bodies into the scene file.
    """

    def __post_init__(self) -> None:
        self.world = MJCFParser(self.file_path).parse()
        with self.world.modify_world():
            self.world.add_connection(
                FixedConnection(
                    parent=Body(name=PrefixedName(self.tf_root_name)),
                    child=self.world.root,
                )
            )
        for name in self.free_body_names:
            self.world.free_branch(self.world.get_body_by_name(name))
        self.pose_writer = MJCFBodyPoseWriter(
            file_path=self.file_path, world=self.world
        )

    @classmethod
    def from_command_line(cls) -> MJCFBodyPoseCalibration:
        """
        :return: the calibration the command line arguments describe.
        """
        parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
        parser.add_argument("file_path", help="the MJCF scene file to calibrate")
        parser.add_argument(
            "--free",
            nargs="+",
            default=[],
            dest="free_body_names",
            metavar="BODY",
            help="bodies to free from their connection to their parent",
        )
        arguments = parser.parse_args()
        return cls(
            file_path=arguments.file_path, free_body_names=arguments.free_body_names
        )

    @property
    def freely_movable_bodies(self) -> list[Body]:
        """
        The bodies connected to their parent by a :class:`Connection6DoF`.
        """
        return [
            connection.child
            for connection in self.world.connections
            if isinstance(connection, Connection6DoF)
        ]

    def run(self) -> None:
        """
        Serve the markers and write the poses whenever an empty line is entered, until
        :attr:`CalibrationCommand.QUIT` is entered or the input ends.
        """
        rclpy.init()
        node = rclpy.create_node(self.node_name)
        viz_marker_publisher = VizMarkerPublisher(_world=self.world, node=node)
        connection_marker_server = ConnectionMarkerServer(_world=self.world, node=node)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin)
        spin_thread.start()

        print(
            f"Drag the markers in RViz. Enter: write the poses into {self.file_path}, "
            f"{CalibrationCommand.QUIT}: quit."
        )
        for line in sys.stdin:
            if line.strip() == CalibrationCommand.QUIT:
                break
            written_bodies = self.pose_writer.write(self.freely_movable_bodies)
            print(f"wrote {[body.name.name for body in written_bodies]}")
        connection_marker_server.stop()
        viz_marker_publisher.stop()
        executor.shutdown()
        spin_thread.join()
        node.destroy_node()
        rclpy.shutdown()


# %% entry point

if __name__ == "__main__":
    MJCFBodyPoseCalibration.from_command_line().run()
