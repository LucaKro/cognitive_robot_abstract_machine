"""
A camera facing a door square-on, with a handle standing out of it, measured by a depth
image.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

import numpy as np
from krrood.ormatic.utils import classproperty
from robokudo.annotators.gpsr.door_handle_pose_annotator import (
    DoorObjectClass,
    DoorHandlePoseAnnotator,
)
from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import Classification
from robokudo.types.cv import Point2D, Rect
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from typing_extensions import TYPE_CHECKING

from ._gpsr_scene import GpsrScene

if TYPE_CHECKING:
    from robokudo.annotators.core import BaseAnnotator

# %% door scene


@dataclass
class DoorScene(GpsrScene):
    """
    A door filling the whole image at a fixed distance, and image regions standing out
    of it by chosen distances.
    """

    resolution: CameraResolution = field(
        default_factory=lambda: CameraResolution(width=160, height=120)
    )
    """
    Resolution of the color and depth images.
    """

    door_distance: float = 1.5
    """
    Distance in meters between the camera and the door.
    """

    depth: np.ndarray | None = None
    """
    The depth image in millimeters, built when the scene is placed in a pipeline.
    """

    @classproperty
    def depth_image_units_per_meter(cls) -> float:
        """
        :return: Depth image value per meter of measured distance, since RoboKudo stores
            depth images in millimeters.
        """
        return 1000.0

    def whole_image(self) -> Rect:
        """
        :return: The image region covering the whole image.
        """
        return Rect(
            pos=Point2D(x=0, y=0),
            width=self.resolution.width,
            height=self.resolution.height,
        )

    @staticmethod
    def detection_box_around(region: Rect, margin: int = 5) -> Rect:
        """
        :param region: The region to enclose.
        :param margin: How far the box reaches beyond the region on every side, in
            pixels.
        :return: A detection box around the region, with some of its surroundings.
        """
        return Rect(
            pos=Point2D(x=region.pos.x - margin, y=region.pos.y - margin),
            width=region.width + 2 * margin,
            height=region.height + 2 * margin,
        )

    def camera_position_of(self, region: Rect, distance: float) -> np.ndarray:
        """
        :param region: The image region.
        :param distance: Distance of the region from the camera, in meters.
        :return: The camera frame position of the middle of an image region.
        """
        column = region.pos.x + (region.width - 1) / 2.0
        row = region.pos.y + (region.height - 1) / 2.0
        return distance * np.linalg.solve(
            self.camera_model().intrinsic_matrix, np.array([column, row, 1.0])
        )

    def place_in_pipeline(self, annotator: BaseAnnotator) -> CAS:
        """
        Like :meth:`GpsrScene.place_in_pipeline`, and also put the depth image of the
        door into the CAS.

        :param annotator: The annotator to set up.
        :return: The CAS the annotator reads and writes.
        """
        cas = super().place_in_pipeline(annotator)
        self.depth = np.full(
            (self.resolution.height, self.resolution.width),
            round(self.door_distance * self.depth_image_units_per_meter),
            dtype=np.uint16,
        )
        cas.set_ref(CASViews.DEPTH_IMAGE, self.depth)
        return cas

    def stand_out(self, region: Rect, distances: np.ndarray | float) -> None:
        """
        Let an image region stand out of the door toward the camera.

        :param region: The image region.
        :param distances: How far the region stands out in meters, for every column of
            the region or for the whole region.
        """
        x, y = region.pos.x, region.pos.y
        self.depth[y : y + region.height, x : x + region.width] = np.round(
            (self.door_distance - np.asarray(distances))
            * self.depth_image_units_per_meter
        )

    def clear(self, region: Rect) -> None:
        """
        Remove the depth measurements of an image region.

        :param region: The image region.
        """
        x, y = region.pos.x, region.pos.y
        self.depth[y : y + region.height, x : x + region.width] = 0

    def add_detection(self, classname: str, region: Rect) -> ObjectHypothesis:
        """
        Add an object detected as the given class in the given image region.

        :param classname: The class of the object.
        :param region: The image region of the object.
        :return: The added detection.
        """
        hypothesis = ObjectHypothesis()
        hypothesis.roi.roi = region
        hypothesis.annotations.append(Classification(classname=classname))
        self.cas.annotations.append(hypothesis)
        return hypothesis

    def grasp_points(self) -> list[ObjectHypothesis]:
        """
        :return: The grasp point hypotheses the door handle annotator added to the CAS.
        """
        grasp_point = Classification(
            classname=DoorObjectClass.GRASP_POINT.value,
            source=DoorHandlePoseAnnotator.__name__,
        )
        return [
            hypothesis
            for hypothesis in self.cas.filter_annotations_by_type(ObjectHypothesis)
            if grasp_point in hypothesis.annotations
        ]
