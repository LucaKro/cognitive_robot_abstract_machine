from __future__ import annotations

# %% imports
from enum import Enum
from timeit import default_timer

import geometry_msgs.msg
import numpy
import numpy as np
import open3d as o3d
import py_trees
import py_trees.common
import visualization_msgs.msg
from rclpy.node import Node
from typing_extensions import Any
from visualization_msgs.msg import Marker

import robokudo
import robokudo.annotators.core
import robokudo.cas
import robokudo.types.human
import robokudo.types.scene
import robokudo.types.tf
import robokudo.utils.cv_helper
import robokudo.utils.decorators
import robokudo.utils.error_handling
import robokudo.utils.math_helper
import robokudo.utils.o3d_helper
import robokudo.utils.transform
from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.types.annotation import PoseAnnotation
from robokudo.utils.annotator_helper import (
    transform_pose_from_camera_to_world,
    transform_pose_from_world_to_camera,
)
from robokudo.utils.math_helper import compute_direction_vector_angle
from robokudo.utils.transform import get_rotation_matrix_from_q


# %% pointing direction
class Arm(Enum):
    """Side of a person's pointing arm."""

    LEFT = 0
    RIGHT = 1

    def __str__(self) -> str:
        return self.name.lower()


def azimuth_angle(direction_vector: np.ndarray) -> float:
    """
    Computes the azimuth angle (the direction) of a direction vector in degrees.

    :param direction_vector: A 3D vector (numpy array) representing the direction.
    :return: Azimuth angle in degrees.
    """
    x, y, z = direction_vector
    rotated_x = y
    rotated_y = -x

    return np.degrees(np.arctan2(rotated_y, rotated_x))


# %% pointing annotator
class HumanPointingAnnotator(robokudo.annotators.core.BaseAnnotator):
    """Annotate pointing activities and detected pointing targets."""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration for pointing detection."""

        class Parameters:
            """Thresholds and target structures for pointing detection."""

            def __init__(self) -> None:
                # A list of spheres defined by triple (name, center, radius). In Cam coordinates.
                # These spheres will be used to check if something is 'pointed at'
                self.pointing_target_structures = []
                self.pointing_target_structures_type = "box"  # or "sphere"
                self.pointing_length_meters = 5.0
                """Length of the pointing line in metres."""
                self.raise_on_failure = True
                """Not read by this annotator."""
                # TODO: fine-tune value based on real pointing vs waving observations
                self.max_pointing_angle = 45.0
                """Maximum angle below horizontal for a pointing direction."""
                self.min_arm_extension = 0.0
                """Minimum wrist distance from the torso for an extended arm."""

                self.known = False
                """Process only people already marked as pointing or no gesture."""

        parameters = Parameters()
        """Default parameters on the class; each descriptor instance gets its own."""

    def __init__(
        self,
        name: str = "HumanPointingAnnotator",
        descriptor: HumanPointingAnnotator.Descriptor = Descriptor(),
    ) -> None:
        super().__init__(name, descriptor)
        self.node = Node("human_pointing_annotator")
        """ROS node used for marker publication."""
        self.pointing_marker_publisher = self.node.create_publisher(
            visualization_msgs.msg.MarkerArray,
            "pointing_marker",  # relativ: <node_name>/pointing_marker
            10,
        )
        """Publisher for pointing direction markers."""
        self.threshold_marker_publisher = self.node.create_publisher(
            visualization_msgs.msg.MarkerArray, "pointing_threshold_marker", 10
        )
        """Publisher for pointing threshold markers."""
        self.possible_structures = {
            "box": o3d.geometry.TriangleMesh.create_box,
            "sphere": o3d.geometry.TriangleMesh.create_sphere,
        }
        if (
            descriptor.parameters.pointing_target_structures_type
            not in self.possible_structures
        ):
            raise ValueError(
                f"Unknown pointing target structure type: {descriptor.parameters.pointing_target_structures_type}. "
                f"Possible values are: {list(self.possible_structures.keys())}"
            )
        self.pointing_target_structures_type = (
            descriptor.parameters.pointing_target_structures_type
        )
        self.color: np.ndarray | None = None
        """Color image of the current observation."""

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
        start_timer = default_timer()

        self.color = self.get_cas().get(robokudo.cas.CASViews.COLOR_IMAGE)
        cloud = self.get_cas().get(robokudo.cas.CASViews.CLOUD)
        all_vis_geometries = [cloud]

        human_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.HumanHypothesis
        )

        if self.descriptor.parameters.known:
            # only process humans the HumanActivityAnnotator already tagged as pointing or no gesture
            human_hypotheses = [
                hh
                for hh in human_hypotheses
                if any(
                    isinstance(a, robokudo.types.human.ActivityAnnotation)
                    and a.source == "HumanActivityAnnotator"
                    and a.type in ("Pointing", "No gesture")
                    for a in hh.annotations
                )
            ]
            if len(human_hypotheses) == 0:
                self.feedback_message = (
                    "known=True but no HH tagged as Pointing, skipping"
                )
                return py_trees.common.Status.SUCCESS
        elif len(human_hypotheses) == 0:
            # nobody in view is an answer, not an error: the result annotator answers with nobody
            self.feedback_message = "No humans detected, nobody is pointing"
            return py_trees.common.Status.SUCCESS

        one_human_handled = False
        for human_hypothesis in human_hypotheses:
            keypoint_annotations = robokudo.cas.CAS.filter_by_type(
                robokudo.types.human.KeypointAnnotation, human_hypothesis.annotations
            )
            keypoint_3d_annotations = [
                a
                for a in keypoint_annotations
                if a.kind == robokudo.types.human.KeypointKind.POINT_CLOUD
            ]
            if len(keypoint_3d_annotations) != 1:
                self.rk_logger.warning("Couldn't find 3D Keypoints in the CAS")
                continue

            keypoint_3d_annotation = keypoint_3d_annotations[0]

            # Calculate the centroid for each of the keypoints of interest
            left_elbow_cloud = keypoint_3d_annotation.keypoints[
                PoseKeypoint.LEFT_ELBOW
            ].points
            left_wrist_cloud = keypoint_3d_annotation.keypoints[
                PoseKeypoint.LEFT_WRIST
            ].points

            right_elbow_cloud = keypoint_3d_annotation.keypoints[
                PoseKeypoint.RIGHT_ELBOW
            ].points
            right_wrist_cloud = keypoint_3d_annotation.keypoints[
                PoseKeypoint.RIGHT_WRIST
            ].points

            if len(left_elbow_cloud.points) == 0 or len(left_wrist_cloud.points) == 0:
                self.rk_logger.warning(
                    "One of the relevant left keypoint clouds was empty - Can't estimate pointing"
                )
                continue

            if len(right_elbow_cloud.points) == 0 or len(right_wrist_cloud.points) == 0:
                self.rk_logger.warning(
                    "One of the relevant right keypoint clouds was empty - Can't estimate pointing"
                )
                continue

            one_human_handled = True

            le_centroid, _ = left_elbow_cloud.compute_mean_and_covariance()
            lw_centroid, _ = left_wrist_cloud.compute_mean_and_covariance()
            re_centroid, _ = right_elbow_cloud.compute_mean_and_covariance()
            rw_centroid, _ = right_wrist_cloud.compute_mean_and_covariance()

            # if both wrists are close to the torso center the person is not extending an arm
            if self.descriptor.parameters.min_arm_extension > 0.0:
                torso_center = self.torso_center(keypoint_3d_annotation)
                if torso_center is not None:
                    lw_dist = numpy.linalg.norm(lw_centroid - torso_center)
                    rw_dist = numpy.linalg.norm(rw_centroid - torso_center)
                    if (
                        lw_dist < self.descriptor.parameters.min_arm_extension
                        and rw_dist < self.descriptor.parameters.min_arm_extension
                    ):
                        self.rk_logger.debug(
                            f"HH {human_hypothesis.id}: both wrists within "
                            f"{self.descriptor.parameters.min_arm_extension}m of torso "
                            f"(lw={lw_dist:.2f}m, rw={rw_dist:.2f}m) skipping"
                        )
                        continue

            # Calculate the line used for visualization and intersection computation
            pointing_direction_left = numpy.subtract(lw_centroid, le_centroid)
            pointing_direction_right = numpy.subtract(rw_centroid, re_centroid)
            unit_pointing_direction_left = pointing_direction_left / numpy.linalg.norm(
                pointing_direction_left
            )
            unit_pointing_direction_right = (
                pointing_direction_right / numpy.linalg.norm(pointing_direction_right)
            )

            # Scale direction by defined length
            pointing_line_left = (
                unit_pointing_direction_left
                * self.descriptor.parameters.pointing_length_meters
            )
            pointing_line_right = (
                unit_pointing_direction_right
                * self.descriptor.parameters.pointing_length_meters
            )

            pointing_end_left = numpy.add(pointing_line_left, le_centroid)
            pointing_end_right = numpy.add(pointing_line_right, re_centroid)

            # Transform all structures from world to cam coordinates
            updated_structures = {}
            for structure in self.descriptor.parameters.pointing_target_structures:
                pose = PoseAnnotation()
                pose.translation = list(structure[1])  # center of the structure
                if self.pointing_target_structures_type == "sphere":
                    pose.rotation = (1.0, 0.0, 0.0, 0.0)  # No rotation for spheres
                else:
                    pose.rotation = structure[2]

                transformed_pose = transform_pose_from_world_to_camera(
                    self.get_cas(), pose
                )

                new_structure_pose = (
                    structure[0],
                    (
                        transformed_pose.translation[0],
                        transformed_pose.translation[1],
                        transformed_pose.translation[2],
                    ),
                    (
                        transformed_pose.rotation[0],
                        transformed_pose.rotation[1],
                        transformed_pose.rotation[2],
                        transformed_pose.rotation[3],
                    ),
                    (structure[-1]),
                )

                updated_structures[structure[0]] = new_structure_pose

            # Get all intersections between defined structures and pointing line
            intersections_left = robokudo.utils.math_helper.intersecting_cuboids(
                lw_centroid,
                pointing_end_left,
                updated_structures.values(),
            )
            intersections_right = robokudo.utils.math_helper.intersecting_cuboids(
                rw_centroid,
                pointing_end_right,
                updated_structures.values(),
            )

            intersections, arm = self.choose_arm(
                intersections_left,
                intersections_right,
                pointing_direction_left,
                pointing_direction_right,
            )

            # wrist above shoulder means raising or waving (camera Y points down so smaller Y = higher)
            left_shoulder, right_shoulder = self.shoulder_centroids(
                keypoint_3d_annotation
            )
            if left_shoulder is not None and right_shoulder is not None:
                chosen_shoulder = left_shoulder if arm == Arm.LEFT else right_shoulder
                chosen_wrist = lw_centroid if arm == Arm.LEFT else rw_centroid
                if chosen_wrist[1] < chosen_shoulder[1]:
                    self.rk_logger.debug(
                        f"HH {human_hypothesis.id}: wrist above shoulder on {arm} arm not pointing"
                    )
                    continue

            # set relevant variables based on the chosen arm
            pointing_direction = (
                pointing_direction_left if arm == Arm.LEFT else pointing_direction_right
            )
            pointing_end = pointing_end_left if arm == Arm.LEFT else pointing_end_right
            elbow_centroid = le_centroid if arm == Arm.LEFT else re_centroid
            wrist_centroid = lw_centroid if arm == Arm.LEFT else rw_centroid

            # Create a pose annotation so you can transform the cam pose to world pose
            pose_elbow = PoseAnnotation()
            pose_elbow.translation = list(elbow_centroid)

            rotation_matrix = (
                robokudo.utils.transform.get_rotation_matrix_from_direction_vector(
                    pointing_direction
                )
            )
            pose_elbow.rotation = list(
                robokudo.utils.transform.get_quaternion_from_rotation_matrix(
                    rotation_matrix
                )
            )

            transform_elbow = transform_pose_from_camera_to_world(
                self.get_cas(), pose_elbow
            )

            pointed_at_objects = []
            object_hypotheses = []
            if intersections:
                self.rk_logger.debug(f"Pointing intersected {intersections}")
                for _, structure_name, _, _, _ in intersections:
                    pointed_at_position = PoseAnnotation()
                    pointed_at_position.source = type(self).__name__
                    pointed_at_position.translation = updated_structures[
                        structure_name
                    ][1]
                    pointed_at_position.rotation = updated_structures[structure_name][2]

                    object_hypothesis = robokudo.types.scene.ObjectHypothesis()
                    object_hypothesis.type = structure_name
                    object_hypothesis.source = type(self).__name__
                    object_hypothesis.annotations.append(pointed_at_position)
                    object_hypotheses.append(object_hypothesis)

                    self.rk_logger.debug(f"Pointing hit {structure_name}")
                    pointed_at_objects.append(structure_name)

                    activity = robokudo.types.human.ActivityAnnotation()
                    activity.type = "pointing"
                    activity.source = type(self).__name__

                    activity.interaction_with = object_hypothesis
                    human_hypothesis.annotations.append(activity)
            else:
                self.rk_logger.debug(f"No intersections found")
                # If no intersections were found, we still want to create an activity annotation if an arm suggests pointing
                self.check_pointing_without_intersections(
                    human_hypothesis,
                    pointing_direction,
                    self.descriptor.parameters.max_pointing_angle,
                    transform_elbow,
                )

            self.get_cas().annotations.extend(object_hypotheses)

            all_vis_geometries = self.create_annotator_output_struct(
                all_vis_geometries,
                elbow_centroid,
                wrist_centroid,
                pointing_end,
                pointed_at_objects,
                updated_structures,
            )

            # Publish pointing vector marker
            marker_msg = self.generate_marker(
                transform_elbow.translation, transform_elbow.rotation, "map"
            )
            marker_array_msg = visualization_msgs.msg.MarkerArray()
            marker_array_msg.markers.append(marker_msg)
            self.pointing_marker_publisher.publish(marker_array_msg)

            # Legacy single-human mode (carry-my-luggage): stop after first handled person.
            # Multi-human mode (known=True, GPSR) processes every tagged HH so each gets a direction.
            if one_human_handled and not self.descriptor.parameters.known:
                break

        if not one_human_handled:
            if self.descriptor.parameters.known:
                self.feedback_message = "No pointing detected"
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.SUCCESS

        self.get_annotator_output_struct().set_geometries(all_vis_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def torso_center(
        self, keypoint_3d_annotation: robokudo.types.human.KeypointAnnotation
    ) -> np.ndarray | None:
        """
        Torso center from shoulders + hips.

        Falls back to shoulders only if hips unavailable.
        """
        try:
            left_shoulder_cloud = keypoint_3d_annotation.keypoints[
                PoseKeypoint.LEFT_SHOULDER
            ].points
            right_shoulder_cloud = keypoint_3d_annotation.keypoints[
                PoseKeypoint.RIGHT_SHOULDER
            ].points
            if (
                len(left_shoulder_cloud.points) == 0
                or len(right_shoulder_cloud.points) == 0
            ):
                return None
            left_shoulder, _ = left_shoulder_cloud.compute_mean_and_covariance()
            right_shoulder, _ = right_shoulder_cloud.compute_mean_and_covariance()
            points = [left_shoulder, right_shoulder]
            try:
                left_hip_cloud = keypoint_3d_annotation.keypoints[
                    PoseKeypoint.LEFT_HIP
                ].points
                right_hip_cloud = keypoint_3d_annotation.keypoints[
                    PoseKeypoint.RIGHT_HIP
                ].points
                if len(left_hip_cloud.points) > 0 and len(right_hip_cloud.points) > 0:
                    left_hip, _ = left_hip_cloud.compute_mean_and_covariance()
                    right_hip, _ = right_hip_cloud.compute_mean_and_covariance()
                    points.extend([left_hip, right_hip])
            except (KeyError, IndexError):
                pass
            return sum(points) / len(points)
        except (KeyError, IndexError):
            return None

    def shoulder_centroids(
        self, keypoint_3d_annotation: robokudo.types.human.KeypointAnnotation
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Returns (left_shoulder, right_shoulder) centroids, or (None, None) if
        unavailable.
        """
        try:
            left_shoulder_cloud = keypoint_3d_annotation.keypoints[
                PoseKeypoint.LEFT_SHOULDER
            ].points
            right_shoulder_cloud = keypoint_3d_annotation.keypoints[
                PoseKeypoint.RIGHT_SHOULDER
            ].points
            if (
                len(left_shoulder_cloud.points) == 0
                or len(right_shoulder_cloud.points) == 0
            ):
                return None, None
            left_shoulder, _ = left_shoulder_cloud.compute_mean_and_covariance()
            right_shoulder, _ = right_shoulder_cloud.compute_mean_and_covariance()
            return left_shoulder, right_shoulder
        except (KeyError, IndexError):
            return None, None

    def check_pointing_without_intersections(
        self,
        human_hypothesis: robokudo.types.scene.HumanHypothesis,
        pointing_direction: np.ndarray,
        maximum_pointing_angle: float,
        transform_elbow: PoseAnnotation,
    ) -> None:
        """
        Annotate a pointing direction below the angle threshold and publish its marker.
        """
        pointing_angle = compute_direction_vector_angle(
            direction_vector=pointing_direction
        )
        if pointing_angle < maximum_pointing_angle:
            activity = robokudo.types.human.ActivityAnnotation()
            azimuth = azimuth_angle(pointing_direction)
            if azimuth < 0:
                activity.type = "pointing left"
            elif azimuth >= 0:
                activity.type = "pointing right"
            activity.source = type(self).__name__
            activity.interaction_with = None
            human_hypothesis.annotations.append(activity)

        threshold_markers = self.generate_pointing_threshold_markers(
            transform_elbow.translation,
            transform_elbow.rotation,
            "map",
            self.descriptor.parameters.max_pointing_angle,
        )
        threshold_marker_array = visualization_msgs.msg.MarkerArray()
        threshold_marker_array.markers.extend(threshold_markers)
        self.threshold_marker_publisher.publish(threshold_marker_array)

    def choose_arm(
        self,
        intersections_left: list[Any],
        intersections_right: list[Any],
        direction_vector_left: np.ndarray,
        direction_vector_right: np.ndarray,
    ) -> tuple[list[Any], Arm]:
        """
        Choose the arm based on the intersections found.

        If both arms have intersections, choose the one which points higher. If no
        intersections, choose the arm with the higher wrist centroid.

        :param intersections_left: List of intersections for the left arm
        :param intersections_right: List of intersections for the right arm
        :param direction_vector_left: The pointing vector of the left arm
        :param direction_vector_right: The pointing vector of the right arm
        :return: Tuple of intersections and the chosen arm
        """
        pointing_angle_left = compute_direction_vector_angle(
            direction_vector=direction_vector_left
        )
        pointing_angle_right = compute_direction_vector_angle(
            direction_vector=direction_vector_right
        )

        if not intersections_left and not intersections_right:
            self.rk_logger.warning("No pointing intersections found for either arm")
            if pointing_angle_left > pointing_angle_right:
                self.rk_logger.debug(
                    "Right arm points higher than left arm (no intersections)"
                )
                return [], Arm.RIGHT
            else:
                self.rk_logger.debug(
                    "Left arm points higher than right arm (no intersections)"
                )
                return [], Arm.LEFT

        if intersections_left and not intersections_right:
            self.rk_logger.debug("Only left arm has pointing intersections")
            return intersections_left, Arm.LEFT

        if not intersections_left and intersections_right:
            self.rk_logger.debug("Only right arm has pointing intersections")
            return intersections_right, Arm.RIGHT

        # If both arms have intersections, choose the one which points higher
        if pointing_angle_left > pointing_angle_right:
            # Left arm points higher
            self.rk_logger.debug("Left arm points higher than right arm")
            return intersections_left, Arm.LEFT
        else:
            # Right arm points higher
            self.rk_logger.debug("Right arm points higher than left arm")
            return intersections_right, Arm.RIGHT

    def create_annotator_output_struct(
        self,
        all_vis_geometries: list[Any],
        elbow_centroid: np.ndarray,
        wrist_centroid: np.ndarray,
        pointing_end: np.ndarray,
        pointed_at_objects: list[str],
        updated_structures: dict[str, Any],
    ) -> list[Any]:
        """
        Create the geometries for visualization in Open3D and return them.

        :param all_vis_geometries: List of geometries to visualize
        :param elbow_centroid: Centroid of the elbow
        :param wrist_centroid: Centroid of the wrist
        :param pointing_end: End point of the pointing line
        :param pointed_at_objects: List of objects that were pointed at
        :param updated_structures: List of updated structures with their poses
        :return: List of geometries for visualization
        """
        # Left Elbow
        centroid_structure = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        centroid_structure.paint_uniform_color([1, 0, 0])
        centroid_structure.translate(elbow_centroid)
        all_vis_geometries.append(centroid_structure)

        # Left Wrist
        centroid_structure = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.04
        )  # in meters
        centroid_structure.paint_uniform_color([0, 1, 0])
        centroid_structure.translate(wrist_centroid)
        all_vis_geometries.append(centroid_structure)

        # Pointing line
        all_vis_geometries.append(
            robokudo.utils.o3d_helper.create_line_for_visualization(
                elbow_centroid, pointing_end, [1, 0, 0]
            )
        )

        # End of pointing line
        centroid_structure = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.04
        )  # in meters
        centroid_structure.paint_uniform_color([1, 0, 0])
        centroid_structure.translate(pointing_end)
        all_vis_geometries.append(centroid_structure)

        # All configured spheres
        for structure in updated_structures.values():
            if self.pointing_target_structures_type == "sphere":
                name, center, _, radius = structure
                centroid_structure = o3d.geometry.TriangleMesh.create_sphere(
                    radius=radius
                )  # in meters
            else:
                name, center, orientation, size = structure
                centroid_structure = o3d.geometry.TriangleMesh.create_box(
                    width=size[0], height=size[1], depth=size[2]
                )
                rotation = get_rotation_matrix_from_q(orientation)
                centroid_structure.rotate(rotation)

            if name in pointed_at_objects:
                centroid_structure.paint_uniform_color([0, 1, 1])
            else:
                centroid_structure.paint_uniform_color([0, 0, 1])
            centroid_structure.translate(center)
            all_vis_geometries.append(centroid_structure)

        return all_vis_geometries

    def generate_marker(
        self, origin: Any, target: Any, frame: str
    ) -> visualization_msgs.msg.Marker:
        """Return a marker for a pointing direction in the given frame."""
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = frame
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.position.x = origin[0]
        marker.pose.position.y = origin[1]
        marker.pose.position.z = origin[2]
        marker.pose.orientation.x = target[0]
        marker.pose.orientation.y = target[1]
        marker.pose.orientation.z = target[2]
        marker.pose.orientation.w = target[3]

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        marker.scale.x = float(self.descriptor.parameters.pointing_length_meters)
        marker.scale.y = float(0.1)
        marker.scale.z = float(0.1)
        marker.ns = f"Pointing Marker-{self.node.get_name()}"
        return marker

    def generate_pointing_threshold_markers(
        self, origin: Any, orientation_q: Any, frame: str, max_pointing_angle: float
    ) -> list[visualization_msgs.msg.Marker]:
        """
        Return markers for the configured pointing threshold in the world frame.
        """
        length = self.descriptor.parameters.pointing_length_meters
        radius = length * numpy.tan(numpy.deg2rad(max_pointing_angle))
        num_segments = 32

        up = numpy.array([0.0, 0.0, 1.0], dtype=float)

        u = numpy.array([1.0, 0.0, 0.0], dtype=float)
        v = numpy.array([0.0, 1.0, 0.0], dtype=float)

        origin_np = numpy.array(origin, dtype=float)
        circle_center = origin_np - up * length

        points = []
        for i in range(num_segments + 1):
            theta = 2.0 * numpy.pi * i / num_segments

            point = circle_center + radius * (
                numpy.cos(theta) * u + numpy.sin(theta) * v
            )
            p = geometry_msgs.msg.Point(
                x=float(point[0]), y=float(point[1]), z=float(point[2])
            )
            points.append(p)

        stamp = self.node.get_clock().now().to_msg()
        ns = f"Pointing Threshold-{self.node.get_name()}"

        circle_marker = visualization_msgs.msg.Marker()
        circle_marker.header.frame_id = frame
        circle_marker.header.stamp = stamp
        circle_marker.ns = ns
        circle_marker.id = 2
        circle_marker.type = visualization_msgs.msg.Marker.LINE_STRIP
        circle_marker.action = visualization_msgs.msg.Marker.ADD
        circle_marker.scale.x = 0.02
        circle_marker.color.r = 1.0
        circle_marker.color.g = 0.65
        circle_marker.color.b = 0.0
        circle_marker.color.a = 0.9
        circle_marker.points = points

        rays_marker = visualization_msgs.msg.Marker()
        rays_marker.header.frame_id = frame
        rays_marker.header.stamp = stamp
        rays_marker.ns = ns
        rays_marker.id = 3
        rays_marker.type = visualization_msgs.msg.Marker.LINE_LIST
        rays_marker.action = visualization_msgs.msg.Marker.ADD
        rays_marker.scale.x = 0.02
        rays_marker.color.r = 1.0
        rays_marker.color.g = 0.0
        rays_marker.color.b = 0.0
        rays_marker.color.a = 0.9
        origin_point = geometry_msgs.msg.Point(
            x=float(origin_np[0]), y=float(origin_np[1]), z=float(origin_np[2])
        )
        rays_marker.points = []
        for p in points:
            rays_marker.points.append(origin_point)
            rays_marker.points.append(p)

        return [circle_marker, rays_marker]
