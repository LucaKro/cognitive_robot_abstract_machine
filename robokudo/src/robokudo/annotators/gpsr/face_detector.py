"""
Detection of faces in the color image, each turned into a human hypothesis.
"""

from __future__ import annotations

# %% Imports
import copy
from timeit import default_timer

import cv2
import face_recognition
import numpy as np
import open3d as o3d
import py_trees

import robokudo.annotators.core
import robokudo.types.human
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.cv_helper
import robokudo.utils.error_handling
import robokudo.utils.o3d_helper
from robokudo.cas import CASViews
from robokudo.types.annotation import Encoding
from robokudo.annotators.gpsr.exceptions import (
    FaceImageScalingFailed,
    UnequalColorDepthRatios,
)


# %% face detection
class FaceDetector(robokudo.annotators.core.ThreadedAnnotator):
    """
    Detects faces in the color image with the face_recognition library, without identifying them.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """
        Configuration of face detection.
        """

        class Parameters:
            """
            Image scaling and face detection options.
            """

            def __init__(self) -> None:
                """
                Detect at the original image scale and generate points and encodings.
                """
                self.scale_factor: float = 1.0
                """Image scale applied before face detection."""
                self.with_point_generation: bool = True
                """Whether to create a depth point cloud for each face."""
                self.generate_encodings: bool = True
                """Whether to generate face encodings."""

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self,
        name: str = "FaceDetector",
        descriptor: FaceDetector.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.get_class_name())

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self) -> py_trees.common.Status:
        """
        Overrides :meth:`ThreadedAnnotator.compute` to add a human hypothesis to the CAS
        for each face in the color image.

        :return: :attr:`Status.SUCCESS`.
        :raises FaceImageScalingFailed: If the color image cannot be scaled to the depth
            image.
        :raises UnequalColorDepthRatios: If the ratios differ between image axes.
        """
        start_timer = default_timer()

        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)

        if self.descriptor.parameters.with_point_generation:
            self.depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
            resized_color = None
            try:
                resized_color = (
                    robokudo.utils.cv_helper.get_scaled_color_image_for_depth_image(
                        self.get_cas(), self.color
                    )
                )
                robokudo.utils.annotator_helper.scale_camera_intrinsics(self)
            except RuntimeError as error:
                self.rk_logger.error(
                    "No color to depth ratio set by your camera driver! Can't scale image for Point Cloud creation."
                )
                raise FaceImageScalingFailed() from error
        else:
            resized_color = self.color

        color_downsampled = cv2.resize(
            resized_color,
            (0, 0),
            fx=self.descriptor.parameters.scale_factor,
            fy=self.descriptor.parameters.scale_factor,
        )

        face_locations = face_recognition.face_locations(color_downsampled)

        visualization_image = copy.deepcopy(self.color)
        visualization_clouds = []
        human_hypotheses = []

        face_index = 0
        for top, right, bottom, left in face_locations:
            original_face_location = (
                top,
                right,
                bottom,
                left,
            )

            # If the color image has been resized to match it with depth data, respect this scaling
            if self.descriptor.parameters.with_point_generation:
                color_to_depth_ratio = self.get_cas().color_to_depth_ratio
                horizontal_color_to_depth_ratio = color_to_depth_ratio.horizontal
                vertical_color_to_depth_ratio = color_to_depth_ratio.vertical
                if horizontal_color_to_depth_ratio != vertical_color_to_depth_ratio:
                    raise UnequalColorDepthRatios()
                top *= 1 / horizontal_color_to_depth_ratio
                right *= 1 / horizontal_color_to_depth_ratio
                bottom *= 1 / horizontal_color_to_depth_ratio
                left *= 1 / horizontal_color_to_depth_ratio

            # Revert the downscaling so that the bounding boxes match the original input image
            top *= 1 / self.descriptor.parameters.scale_factor
            right *= 1 / self.descriptor.parameters.scale_factor
            bottom *= 1 / self.descriptor.parameters.scale_factor
            left *= 1 / self.descriptor.parameters.scale_factor

            # TODO This should be a Parameter in the future to allow other human detectors to generate
            #      the first hypothesis or generate two hypotheses and merge them later.
            human_hypothesis = robokudo.types.scene.HumanHypothesis()
            human_hypothesis.id = face_index
            human_hypothesis.source = self.name
            face_annotation = robokudo.types.human.FaceAnnotation()

            if self.descriptor.parameters.with_point_generation:
                face_mask = np.zeros_like(self.depth, dtype=np.uint8)
                cv2.rectangle(
                    face_mask,
                    (
                        int(left * horizontal_color_to_depth_ratio),
                        int(top * horizontal_color_to_depth_ratio),
                    ),
                    (
                        int(right * horizontal_color_to_depth_ratio),
                        int(bottom * horizontal_color_to_depth_ratio),
                    ),
                    255,
                    -1,
                )
                color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
                depth_masked = copy.deepcopy(self.depth)
                depth_masked = np.where(face_mask == 255, depth_masked, 0)

                color_image = o3d.geometry.Image(color_rgb)
                depth_image = o3d.geometry.Image(
                    depth_masked
                )  # depth values in millimeters
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_image,
                    depth_image,
                    convert_rgb_to_intensity=False,
                    depth_trunc=9.0,
                )

                cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, self.camera_intrinsics
                )
                human_hypothesis.points = cloud
                face_annotation.points = cloud
                visualization_clouds.append(cloud)

            human_hypothesis.roi.roi.pos.x = int(left)
            human_hypothesis.roi.roi.pos.y = int(top)
            human_hypothesis.roi.roi.width = int(right - left)
            human_hypothesis.roi.roi.height = int(bottom - top)

            face_annotation.roi = human_hypothesis.roi
            human_hypothesis.annotations.append(face_annotation)

            if self.descriptor.parameters.generate_encodings:
                face_encoding = face_recognition.face_encodings(
                    color_downsampled, known_face_locations=[original_face_location]
                )
                encoding_annotation = Encoding()
                encoding_annotation.encoding = face_encoding
                encoding_annotation.source = self.name
                human_hypothesis.annotations.append(encoding_annotation)

            human_hypotheses.append(human_hypothesis)
            self.get_cas().annotations.append(human_hypothesis)
            face_index += 1

        robokudo.utils.annotator_helper.draw_bounding_boxes_from_object_hypotheses(
            visualization_image,
            human_hypotheses,
            lambda human_hypothesis: f"Face-{human_hypothesis.id}",
        )
        self.get_annotator_output_struct().set_image(visualization_image)

        if self.descriptor.parameters.with_point_generation:
            if len(visualization_clouds) > 0:
                visualization_cloud = robokudo.utils.o3d_helper.concatenate_clouds(
                    visualization_clouds
                )
                self.get_annotator_output_struct().set_geometries(visualization_cloud)
            else:
                self.get_annotator_output_struct().set_geometries([])

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS
