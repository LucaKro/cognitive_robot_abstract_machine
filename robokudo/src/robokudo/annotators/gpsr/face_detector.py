import copy
from timeit import default_timer

import cv2
import face_recognition
import numpy
import numpy as np
import open3d as o3d
import py_trees

import robokudo
import robokudo.annotators.core
import robokudo.types.annotation
import robokudo.types.human
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.cv_helper
import robokudo.utils.error_handling
import robokudo.utils.o3d_helper
from robokudo.cas import CASViews


class FaceDetector(robokudo.annotators.core.ThreadedAnnotator):
    """
    Detect faces in 2D images.

    This annotator is not provided labels/classes/identification. Based on
    face_recognition library.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.scale_factor = 1.0  # Downscale image before processing.
                # Will be applied after rgb image is scaled to depth image when with_point_generation is true

                self.with_point_generation = True  # Requires RGB data
                self.generate_encodings = True

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="FaceDetector", descriptor=Descriptor()):
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self):
        start_timer = default_timer()

        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)

        if self.descriptor.parameters.with_point_generation:
            # Scale the color image down so that it matches the depth image size
            self.depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
            resized_color = (
                robokudo.utils.cv_helper.get_scaled_color_image_for_depth_image(
                    self.get_cas(), self.color
                )
            )
            robokudo.utils.annotator_helper.scale_camera_intrinsics(self)
        else:
            resized_color = self.color

        # Downsample camera image before face recognition, if necessary
        color_downsampled = cv2.resize(
            resized_color,
            (0, 0),
            fx=self.descriptor.parameters.scale_factor,
            fy=self.descriptor.parameters.scale_factor,
        )

        # Actual face detection
        face_locations = face_recognition.face_locations(color_downsampled)

        # Create visualization Output and ObjectHypotheses
        #
        visualization_img = copy.deepcopy(self.color)
        visualization_clouds = []
        object_hypotheses = []

        # Each detected face will be considered as an Object Detection for now
        idx = 0
        for top, right, bottom, left in face_locations:
            original_face_location = (
                top,
                right,
                bottom,
                left,
            )  # store for face encoding computation later on

            # If the color image has been resized to match it with depth data, respect this scaling
            if self.descriptor.parameters.with_point_generation:
                color2depth_ratio = self.get_cas().color_to_depth_ratio
                c2d_ratio_x = color2depth_ratio.horizontal
                c2d_ratio_y = color2depth_ratio.vertical
                if c2d_ratio_x != c2d_ratio_y:
                    raise RuntimeError(
                        "color to depth ratio x&y is not equal. Can't process."
                    )
                top *= 1 / c2d_ratio_x
                right *= 1 / c2d_ratio_x
                bottom *= 1 / c2d_ratio_x
                left *= 1 / c2d_ratio_x

            # Respect/Revert downscaling factor to have BBs relative to the original input image
            top *= 1 / self.descriptor.parameters.scale_factor
            right *= 1 / self.descriptor.parameters.scale_factor
            bottom *= 1 / self.descriptor.parameters.scale_factor
            left *= 1 / self.descriptor.parameters.scale_factor

            # Create object hypothesis
            # Add each cluster to CAS
            # TODO This should be a Parameter in the future to allow other human detectors to generate
            #      the first hypothesis OR generate two hypotheses and merge them later.
            human_hypothesis = robokudo.types.scene.HumanHypothesis()
            human_hypothesis.id = idx
            human_hypothesis.source = self.name
            face_annotation = robokudo.types.human.FaceAnnotation()

            # Actual 3D Point generation for face
            if self.descriptor.parameters.with_point_generation:
                face_mask = np.zeros_like(self.depth, dtype=numpy.uint8)
                cv2.rectangle(
                    face_mask,
                    (int(left * c2d_ratio_x), int(top * c2d_ratio_x)),
                    (int(right * c2d_ratio_x), int(bottom * c2d_ratio_x)),
                    255,
                    -1,
                )
                color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
                depth_masked = copy.deepcopy(self.depth)
                depth_masked = numpy.where(
                    face_mask == 255, depth_masked, 0
                )  # mask all depth values

                o3d_color = o3d.geometry.Image(color_rgb)
                o3d_depth = o3d.geometry.Image(
                    depth_masked
                )  # Please note that depth values should be in mm
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color,
                    o3d_depth,
                    convert_rgb_to_intensity=False,
                    depth_trunc=9.0,
                )

                cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, self.camera_intrinsics
                )
                # visualization_img = face_mask
                human_hypothesis.points = cloud
                face_annotation.points = cloud
                visualization_clouds.append(cloud)

            human_hypothesis.roi.roi.pos.x = int(left)
            human_hypothesis.roi.roi.pos.y = int(top)
            human_hypothesis.roi.roi.width = int(right - left)
            human_hypothesis.roi.roi.height = int(bottom - top)

            face_annotation.roi = human_hypothesis.roi
            human_hypothesis.annotations.append(face_annotation)

            # Generate an encoding for the detected face if desired
            # The encoding will be used by a classifier to recognize pre-trained people
            if self.descriptor.parameters.generate_encodings:
                person_encoding = face_recognition.face_encodings(
                    color_downsampled, known_face_locations=[original_face_location]
                )
                encoding_annotation = robokudo.types.annotation.Encoding()
                encoding_annotation.encoding = person_encoding
                encoding_annotation.source = self.name
                human_hypothesis.annotations.append(encoding_annotation)

            object_hypotheses.append(human_hypothesis)
            self.get_cas().annotations.append(human_hypothesis)
            idx += 1

        # 2D Visualization
        robokudo.utils.annotator_helper.draw_bounding_boxes_from_object_hypotheses(
            visualization_img, object_hypotheses, lambda oh: f"Face-{oh.id}"
        )
        self.get_annotator_output_struct().set_image(visualization_img)

        # 3D Visualization
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
