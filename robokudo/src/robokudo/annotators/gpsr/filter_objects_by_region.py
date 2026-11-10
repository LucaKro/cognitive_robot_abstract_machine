"""
Marking of the hypotheses that stand in the queried room.
"""

from __future__ import annotations

# %% Imports
import cv2
from py_trees.common import Status
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import (
    LocationAnnotation,
    PoseAnnotation,
    PositionAnnotation,
)
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

# %% room filter


class FilterObjectsByRegion(BaseAnnotator):
    """
    Marks every hypothesis that stands in the queried room with a
    :class:`LocationAnnotation` naming the room.

    The rooms come from the shared world and only the position on the floor decides. A
    query without a known room or a CAS without a query marks nothing.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the room filter.
        """

        class Parameters:
            """
            Parameters of the room filter.
            """

            def __init__(self) -> None:
                """
                Start with the world frame ``map`` and no active region.
                """
                self.world_frame_name: str = "map"
                """
                Name of the world frame.
                """
                self.active_region: str = ""
                """
                Name of the active region.
                """

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self,
        name: str = "FilterObjectsByRegion",
        descriptor: FilterObjectsByRegion.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        """
        super().__init__(name=name, descriptor=descriptor)
        self.world_frame_name: str = self.descriptor.parameters.world_frame_name
        """
        The world frame name of the parameters.
        """
        self.active_region: str = self.descriptor.parameters.active_region
        """
        The active region of the parameters.
        """
        self.world: World | None = None
        """
        The shared world with the rooms, fetched on the first query that names a room.
        """

    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to mark every hypothesis that stands in
        the queried room.

        :return: :attr:`Status.SUCCESS`.
        """
        if not self.get_cas().contains(CASViews.QUERY):
            return Status.SUCCESS
        queried_location = self.get_cas().get(CASViews.QUERY).obj.location
        if queried_location == "":
            return Status.SUCCESS

        # The root frame of the shared world is assumed to be the robot's map frame.
        if self.world is None:
            node = ROSNodeRegistry().get()
            self.world = fetch_world_from_service(node)
            # keeps the fetched copy up to date, for example with rooms added after the fetch
            WorldSynchronizer(_world=self.world, node=node)
        rooms = [
            room
            for room in self.world.get_semantic_annotations_by_type(Room)
            if room.name.name == queried_location
        ]
        if not rooms:
            # an unknown room gets an empty answer, raising would leave the query open
            self.rk_logger.warning(
                f"The shared world has no room '{queried_location}', no hypothesis is marked"
            )
            self._draw_room_overlay(queried_location)
            return Status.SUCCESS
        floor_boxes = list(
            rooms[0].floor.root.collision.as_bounding_box_collection_in_frame(
                self.world.root
            )
        )

        location_annotation = LocationAnnotation()
        location_annotation.name = queried_location
        for hypothesis in self.get_cas().filter_annotations_by_type(ObjectHypothesis):
            for annotation in hypothesis.annotations:
                if isinstance(annotation, PoseAnnotation):
                    pose_to_transform = annotation
                elif isinstance(annotation, PositionAnnotation):
                    pose_to_transform = PoseAnnotation()
                    pose_to_transform.rotation = [0.0, 0.0, 0.0, 1.0]
                    pose_to_transform.translation = annotation.translation
                    pose_to_transform.source = annotation.source
                else:
                    continue
                world_pose = transform_pose_from_camera_to_world(
                    self.get_cas(), pose_to_transform
                )
                x, y = world_pose.translation[0], world_pose.translation[1]
                if any(
                    box.min_x <= x <= box.max_x and box.min_y <= y <= box.max_y
                    for box in floor_boxes
                ):
                    hypothesis.annotations.append(location_annotation)
                    break

        self._draw_room_overlay(queried_location)
        return Status.SUCCESS

    def _draw_room_overlay(self, queried_location: str) -> None:
        """
        Draw every hypothesis of the CAS into the output image: a green frame for one
        marked with the queried room, a red frame over a red veil for any other.

        :param queried_location: The room the hypotheses are marked with.
        """
        image = self.get_cas().get(CASViews.COLOR_IMAGE)
        if image is None:
            return
        image = image.copy()
        for hypothesis in self.get_cas().filter_annotations_by_type(ObjectHypothesis):
            try:
                region = hypothesis.roi.roi
                x, y = int(region.pos.x), int(region.pos.y)
                width, height = int(region.width), int(region.height)
            except AttributeError:
                continue
            in_room = any(
                isinstance(annotation, LocationAnnotation)
                and annotation.name == queried_location
                for annotation in hypothesis.annotations
            )
            color = (0, 200, 0) if in_room else (0, 0, 255)
            if not in_room:
                overlay = image.copy()
                cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 3)
            cv2.putText(
                image,
                "in room" if in_room else "wrong room",
                (x, max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        self.get_annotator_output_struct().set_image(image)
