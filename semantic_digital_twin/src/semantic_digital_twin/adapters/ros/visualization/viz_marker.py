from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import MarkerArray

from semantic_digital_twin.adapters.ros.msg_converter import SemDTToRos2Converter
from semantic_digital_twin.adapters.ros.tf_publisher import (
    PositionedInTfTree,
    TfFrameNames,
)
from semantic_digital_twin.adapters.ros.visualization.collision_viz_marker import (
    CollisionVisualizationMarkerPublisher,
)
from semantic_digital_twin.callbacks.callback import ModelChangeCallback
from semantic_digital_twin.world_description.geometry import Shape
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


class ShapeSource(Enum):
    """
    Enum to specify which shapes to use for visualization.
    """

    VISUAL_ONLY = "visual_only"
    """
    The shapes to use for visualization are visual shapes only.
    """

    COLLISION_ONLY = "collision_only"
    """
    The shapes to use for visualization are collision shapes only.
    """

    VISUAL_WITH_COLLISION_BACKUP = "visual_with_collision_backup"
    """
    The shapes to use for visualization are visual shapes, but if there are no visual shapes, use collision shapes as a backup.
    """


@dataclass(eq=False)
class VizMarkerPublisher(ModelChangeCallback, PositionedInTfTree):
    """
    Publishes the world model as a visualization marker.

    Markers are stamped with the frame names the world's tf publisher hands out, so
    equally named bodies are told apart the same way in both.
    .. warning:: Relies on the tf tree to correctly position the markers.
    .. warning:: To see something in Rviz you must:
        1. add a MarkerArray plugin,
        2. set the current topic name,
        3. set DurabilityPolicy.TRANSIENT_LOCAL,
        4. make sure that the fixed frame is the tf root.
    """

    topic_name: str = "/semworld/viz_marker"
    """
    The name of the topic to which the Visualization Marker should be published.
    """

    shape_source: ShapeSource = field(
        kw_only=True, default=ShapeSource.VISUAL_WITH_COLLISION_BACKUP
    )
    """
    Which shapes to use for each body
    """

    alpha: float = field(kw_only=True, default=1.0)
    """
    Marker transparency in [0.0, 1.0]. 0.0 is fully transparent.
    """

    region_alpha: float = field(kw_only=True, default=0.2)
    """
    Marker transparency forced onto every Region marker, in [0.0, 1.0]. Regions are
    always rendered at this alpha, regardless of the color their shapes were assigned.
    """

    markers: MarkerArray = field(init=False, default_factory=MarkerArray)
    """Maker message to be published."""
    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )
    """QoS profile for the publisher."""

    _collision_publisher: CollisionVisualizationMarkerPublisher | None = field(
        init=False, default=None
    )
    """
    Reference to a collision marker publisher created by this class.
    """

    _publisher: Publisher = field(init=False)
    """
    The ROS publisher for the marker.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self._adopt_tf_publisher()
        self.publisher = self.node.create_publisher(
            MarkerArray, self.topic_name, self.qos_profile
        )
        time.sleep(0.2)
        self.notify_model_change()
        logger.info(
            f"VizMarkerPublisher started. Fixed frame is {self._world.root.name}"
        )
        time.sleep(0.2)

    def stop(self) -> None:
        """
        Deregister this publisher and stop the publishers it started.

        Anything left running publishes on a node that may already be gone, and a world
        outliving this publisher would keep notifying it.
        """
        if self._collision_publisher is not None:
            self._collision_publisher.stop()
        super().stop()

    def with_collision_visualization(self, **kwargs) -> None:
        """
        Launches a publisher for closest-points collision results alongside the VizMarkerPublisher.

        :param kwargs: Forwarded to :class:`CollisionVisualizationMarkerPublisher`.
        """
        self._collision_publisher = CollisionVisualizationMarkerPublisher(
            node=self.node, world=self._world, **kwargs
        )

    def _select_shapes(self, body: Body) -> list[Shape]:
        """
        :param body: The body to show.
        :return: the shapes of the body that :attr:`shape_source` selects for showing.
        """
        if self.shape_source is ShapeSource.VISUAL_ONLY:
            return body.visual.shapes
        if self.shape_source is ShapeSource.COLLISION_ONLY:
            return body.collision.shapes
        if self.shape_source is ShapeSource.VISUAL_WITH_COLLISION_BACKUP:
            return body.visual.shapes if body.visual.shapes else body.collision.shapes
        raise ValueError(f"Unsupported shape_source: {self.shape_source!r}")

    def on_model_change(self, **kwargs) -> None:
        self.markers = MarkerArray()
        frame_names = self.frame_names
        for body in self._world.bodies:
            shapes = self._select_shapes(body)
            self._add_markers_for_shapes(shapes, str(body.name), frame_names)

        for region in self._world.regions:
            self._add_markers_for_shapes(
                region.area.shapes,
                str(region.name),
                frame_names,
                force_alpha=self.region_alpha,
            )

        self.publisher.publish(self.markers)

    def _add_markers_for_shapes(
        self,
        shapes: list[Shape],
        marker_ns: str,
        frame_names: TfFrameNames,
        force_alpha: float | None = None,
    ) -> None:
        """
        Add a marker per shape to :attr:`markers`.

        :param shapes: The shapes to show.
        :param marker_ns: The namespace the markers are grouped under.
        :param frame_names: The tf frame names the entities of the world are published
            under.
        :param force_alpha: The transparency every marker is shown with, overriding the
            colors of the shapes and :attr:`alpha`.
        """
        if not shapes:
            return
        for i, shape in enumerate(shapes):
            marker = SemDTToRos2Converter.convert(shape)
            marker.header.frame_id = frame_names.assign(shape.origin.reference_frame)
            if force_alpha is not None:
                marker.color.a = force_alpha
            elif not marker.mesh_use_embedded_materials:
                marker.color.a *= self.alpha
            marker.frame_locked = True
            marker.id = i
            marker.ns = marker_ns
            self.markers.markers.append(marker)
