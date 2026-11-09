"""
Detection of pointing persons, their pointing direction and the targets they point at.
"""

from __future__ import annotations

# %% Imports
from enum import Enum
from timeit import default_timer

import geometry_msgs.msg
import numpy as np
import open3d as o3d
import py_trees.common
import visualization_msgs.msg
from krrood_spatial_math.spatial_types import Point3, Quaternion
from rclpy.node import Node
from typing_extensions import Any
from visualization_msgs.msg import Marker

import robokudo.annotators.core
import robokudo.cas
import robokudo.types.human
import robokudo.types.scene
import robokudo.utils.error_handling
from robokudo.exceptions import UnknownMode
import robokudo.utils.math_helper
import robokudo.utils.o3d_helper
import robokudo.utils.transform
from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.annotators.gpsr.human_activity_annotator import (
    HumanActivityAnnotator,
    HumanGesture,
)
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
    """
    The left arm of the person.
    """
    RIGHT = 1
    """
    The right arm of the person.
    """

    def __str__(self) -> str:
        """
        :return: The lowercase name of the side.
        """
        return self.name.lower()


def compute_azimuth_angle(direction_vector: np.ndarray) -> float:
    """
    Computes the azimuth angle (the direction) of a direction vector in degrees.

    :param direction_vector: The direction as a 3D vector.
    :return: Azimuth angle in degrees.
    """
    x, y, z = direction_vector
    rotated_x = y
    rotated_y = -x

    return np.degrees(np.arctan2(rotated_y, rotated_x))


# %% pointing annotator
class HumanPointingAnnotator(robokudo.annotators.core.BaseAnnotator):
    """Annotates pointing activities and detected pointing targets."""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration for pointing detection."""

        class Parameters:
            """Thresholds and target structures for pointing detection."""

            def __init__(self) -> None:
                """
                Start with no target structures and the default pointing thresholds.
                """
                self.pointing_target_structures = []
                """
                Target structures in the world frame. A box has a name, center, rotation
                and size. A sphere has a name, center and radius.
                """
                self.pointing_target_structures_type = "box"
                """
                Shape of the target structures, ``box`` or ``sphere``.
                """
                self.pointing_length_meters = 5.0
                """Length of the pointing line."""
                self.raise_on_failure = True
                """Whether a failure raises an error."""
                # TODO: fine-tune value based on real pointing versus waving observations
                self.maximum_pointing_angle = 45.0
                """Maximum angle below horizontal for a pointing direction, in degrees."""
                self.minimum_arm_extension = 0.0
                """Minimum wrist distance from the torso for an extended arm, in meters."""

                self.known = False
                """Process only persons already marked as pointing or no gesture."""

        parameters = Parameters()
        """Default parameters on the class. Each descriptor instance gets its own copy."""

    def __init__(
        self,
        name: str = "HumanPointingAnnotator",
        descriptor: HumanPointingAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        :raises UnknownMode: If the shape of the target structures is unknown.
        """
        super().__init__(name, descriptor)
        self.node = Node("human_pointing_annotator")
        """ROS node used for marker publication."""
        self.pointing_marker_publisher = self.node.create_publisher(
            visualization_msgs.msg.MarkerArray,
            "pointing_marker",  # relative: <node_name>/pointing_marker
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
        """
        The mesh factory of every target structure shape.
        """
        if (
            descriptor.parameters.pointing_target_structures_type
            not in self.possible_structures
        ):
            raise UnknownMode(
                mode=descriptor.parameters.pointing_target_structures_type,
                context="Pointing target structure",
            )
        self.pointing_target_structures_type = (
            descriptor.parameters.pointing_target_structures_type
        )
        """
        Shape of the target structures.
        """
        self.color: np.ndarray | None = None
        """Color image of the current observation."""

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
        """
        Overrides :meth:`BaseAnnotator.update` to annotate the pointing direction and
        the pointed-at targets of the persons.

        :return: :attr:`Status.SUCCESS`.
        """
        start_timer = default_timer()

        self.color = self.get_cas().get(robokudo.cas.CASViews.COLOR_IMAGE)
        cloud = self.get_cas().get(robokudo.cas.CASViews.CLOUD)
        all_visualization_geometries = [cloud]

        human_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.HumanHypothesis
        )

        if self.descriptor.parameters.known:
            human_hypotheses = [
                human_hypothesis
                for human_hypothesis in human_hypotheses
                if any(
                    isinstance(annotation, robokudo.types.human.ActivityAnnotation)
                    and annotation.source == HumanActivityAnnotator.__name__
                    and annotation.type in (HumanGesture.POINTING, HumanGesture.NONE)
                    for annotation in human_hypothesis.annotations
                )
            ]
            if len(human_hypotheses) == 0:
                self.feedback_message = (
                    "known=True but no human hypothesis tagged as Pointing, skipping"
                )
                return py_trees.common.Status.SUCCESS
        elif len(human_hypotheses) == 0:
            self.feedback_message = "No humans detected, nobody is pointing"
            return py_trees.common.Status.SUCCESS

        one_human_handled = False
        for human_hypothesis in human_hypotheses:
            keypoint_annotations = robokudo.cas.CAS.filter_by_type(
                robokudo.types.human.KeypointAnnotation, human_hypothesis.annotations
            )
            cloud_keypoint_annotations = [
                annotation
                for annotation in keypoint_annotations
                if annotation.kind == robokudo.types.human.KeypointKind.POINT_CLOUD
            ]
            if len(cloud_keypoint_annotations) != 1:
                self.rk_logger.warning("Couldn't find 3D Keypoints in the CAS")
                continue

            cloud_keypoints = cloud_keypoint_annotations[0]

            left_elbow_cloud = cloud_keypoints.keypoints[PoseKeypoint.LEFT_ELBOW].points
            left_wrist_cloud = cloud_keypoints.keypoints[PoseKeypoint.LEFT_WRIST].points

            right_elbow_cloud = cloud_keypoints.keypoints[
                PoseKeypoint.RIGHT_ELBOW
            ].points
            right_wrist_cloud = cloud_keypoints.keypoints[
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

            left_elbow_centroid, _ = left_elbow_cloud.compute_mean_and_covariance()
            left_wrist_centroid, _ = left_wrist_cloud.compute_mean_and_covariance()
            right_elbow_centroid, _ = right_elbow_cloud.compute_mean_and_covariance()
            right_wrist_centroid, _ = right_wrist_cloud.compute_mean_and_covariance()

            if self.descriptor.parameters.minimum_arm_extension > 0.0:
                torso_center = self.compute_torso_center(cloud_keypoints)
                if torso_center is not None:
                    left_wrist_distance = np.linalg.norm(
                        left_wrist_centroid - torso_center
                    )
                    right_wrist_distance = np.linalg.norm(
                        right_wrist_centroid - torso_center
                    )
                    if (
                        left_wrist_distance
                        < self.descriptor.parameters.minimum_arm_extension
                        and right_wrist_distance
                        < self.descriptor.parameters.minimum_arm_extension
                    ):
                        self.rk_logger.debug(
                            f"human hypothesis {human_hypothesis.id}: both wrists within "
                            f"{self.descriptor.parameters.minimum_arm_extension}m of torso "
                            f"(left wrist={left_wrist_distance:.2f}m, right wrist={right_wrist_distance:.2f}m) skipping"
                        )
                        continue

            pointing_direction_left = np.subtract(
                left_wrist_centroid, left_elbow_centroid
            )
            pointing_direction_right = np.subtract(
                right_wrist_centroid, right_elbow_centroid
            )
            unit_pointing_direction_left = pointing_direction_left / np.linalg.norm(
                pointing_direction_left
            )
            unit_pointing_direction_right = pointing_direction_right / np.linalg.norm(
                pointing_direction_right
            )

            pointing_line_left = (
                unit_pointing_direction_left
                * self.descriptor.parameters.pointing_length_meters
            )
            pointing_line_right = (
                unit_pointing_direction_right
                * self.descriptor.parameters.pointing_length_meters
            )

            pointing_end_left = np.add(pointing_line_left, left_elbow_centroid)
            pointing_end_right = np.add(pointing_line_right, right_elbow_centroid)

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

            intersections_left = robokudo.utils.math_helper.intersecting_cuboids(
                left_wrist_centroid,
                pointing_end_left,
                updated_structures.values(),
            )
            intersections_right = robokudo.utils.math_helper.intersecting_cuboids(
                right_wrist_centroid,
                pointing_end_right,
                updated_structures.values(),
            )

            intersections, arm = self.choose_arm(
                intersections_left,
                intersections_right,
                pointing_direction_left,
                pointing_direction_right,
            )

            # a wrist above the shoulder is raising or waving, the camera y axis points down
            left_shoulder, right_shoulder = self.compute_shoulder_centroids(
                cloud_keypoints
            )
            if left_shoulder is not None and right_shoulder is not None:
                chosen_shoulder = left_shoulder if arm == Arm.LEFT else right_shoulder
                chosen_wrist = (
                    left_wrist_centroid if arm == Arm.LEFT else right_wrist_centroid
                )
                if chosen_wrist[1] < chosen_shoulder[1]:
                    self.rk_logger.debug(
                        f"human hypothesis {human_hypothesis.id}: wrist above shoulder on {arm} arm not pointing"
                    )
                    continue

            pointing_direction = (
                pointing_direction_left if arm == Arm.LEFT else pointing_direction_right
            )
            pointing_end = pointing_end_left if arm == Arm.LEFT else pointing_end_right
            elbow_centroid = (
                left_elbow_centroid if arm == Arm.LEFT else right_elbow_centroid
            )
            wrist_centroid = (
                left_wrist_centroid if arm == Arm.LEFT else right_wrist_centroid
            )

            elbow_pose_in_camera = PoseAnnotation()
            elbow_pose_in_camera.translation = list(elbow_centroid)

            rotation_matrix = (
                robokudo.utils.transform.get_rotation_matrix_from_direction_vector(
                    pointing_direction
                )
            )
            elbow_pose_in_camera.rotation = list(
                robokudo.utils.transform.get_quaternion_from_rotation_matrix(
                    rotation_matrix
                )
            )

            elbow_pose_in_world = transform_pose_from_camera_to_world(
                self.get_cas(), elbow_pose_in_camera
            )

            pointed_at_objects = []
            object_hypotheses = []
            if intersections:
                self.rk_logger.debug(f"Pointing intersected {intersections}")
                for _, structure_name, _, _, _ in intersections:
                    pointed_at_pose = PoseAnnotation()
                    pointed_at_pose.source = self.get_class_name()
                    pointed_at_pose.translation = updated_structures[structure_name][1]
                    pointed_at_pose.rotation = updated_structures[structure_name][2]

                    object_hypothesis = robokudo.types.scene.ObjectHypothesis()
                    object_hypothesis.type = structure_name
                    object_hypothesis.source = self.get_class_name()
                    object_hypothesis.annotations.append(pointed_at_pose)
                    object_hypotheses.append(object_hypothesis)

                    self.rk_logger.debug(f"Pointing hit {structure_name}")
                    pointed_at_objects.append(structure_name)

                    activity = robokudo.types.human.ActivityAnnotation()
                    activity.type = "pointing"
                    activity.source = self.get_class_name()

                    activity.interaction_with = object_hypothesis
                    human_hypothesis.annotations.append(activity)
            else:
                self.rk_logger.debug("No intersections found")
                # without a hit, the arm direction alone can still count as pointing
                self.annotate_pointing_direction(
                    human_hypothesis,
                    pointing_direction,
                    self.descriptor.parameters.maximum_pointing_angle,
                    elbow_pose_in_world,
                )

            self.get_cas().annotations.extend(object_hypotheses)

            all_visualization_geometries = self.create_visualization_geometries(
                all_visualization_geometries,
                elbow_centroid,
                wrist_centroid,
                pointing_end,
                pointed_at_objects,
                updated_structures,
            )

            marker_message = self.generate_marker(
                Point3.from_iterable(elbow_pose_in_world.translation),
                Quaternion.from_iterable(elbow_pose_in_world.rotation),
                "map",
            )
            marker_array_message = visualization_msgs.msg.MarkerArray()
            marker_array_message.markers.append(marker_message)
            self.pointing_marker_publisher.publish(marker_array_message)

            # without known gestures only the first handled person is decided
            if one_human_handled and not self.descriptor.parameters.known:
                break

        if not one_human_handled:
            if self.descriptor.parameters.known:
                self.feedback_message = "No pointing detected"
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.SUCCESS

        self.get_annotator_output_struct().set_geometries(all_visualization_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def compute_torso_center(
        self, cloud_keypoints: robokudo.types.human.KeypointAnnotation
    ) -> np.ndarray | None:
        """
        Compute the torso center from the shoulders and, if available, the hips.

        :param cloud_keypoints: The 3D keypoints of the person.
        :return: The torso center, or *None* if it cannot be determined.
        """
        try:
            left_shoulder_cloud = cloud_keypoints.keypoints[
                PoseKeypoint.LEFT_SHOULDER
            ].points
            right_shoulder_cloud = cloud_keypoints.keypoints[
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
                left_hip_cloud = cloud_keypoints.keypoints[PoseKeypoint.LEFT_HIP].points
                right_hip_cloud = cloud_keypoints.keypoints[
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

    def compute_shoulder_centroids(
        self, cloud_keypoints: robokudo.types.human.KeypointAnnotation
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        :param cloud_keypoints: The 3D keypoints of the person.
        :return: The left and right shoulder centroids, or *None* for both if
            unavailable.
        """
        try:
            left_shoulder_cloud = cloud_keypoints.keypoints[
                PoseKeypoint.LEFT_SHOULDER
            ].points
            right_shoulder_cloud = cloud_keypoints.keypoints[
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

    def annotate_pointing_direction(
        self,
        human_hypothesis: robokudo.types.scene.HumanHypothesis,
        pointing_direction: np.ndarray,
        maximum_pointing_angle: float,
        elbow_pose_in_world: PoseAnnotation,
    ) -> None:
        """
        Annotate a pointing direction below the angle threshold and publish its marker.

        :param human_hypothesis: The person.
        :param pointing_direction: The direction of the forearm.
        :param maximum_pointing_angle: The steepest angle that counts as pointing, in
            degrees.
        :param elbow_pose_in_world: The elbow pose the threshold markers start at.
        """
        pointing_angle = compute_direction_vector_angle(
            direction_vector=pointing_direction
        )
        if pointing_angle < maximum_pointing_angle:
            activity = robokudo.types.human.ActivityAnnotation()
            azimuth = compute_azimuth_angle(pointing_direction)
            if azimuth < 0:
                activity.type = "pointing left"
            elif azimuth >= 0:
                activity.type = "pointing right"
            activity.source = self.get_class_name()
            activity.interaction_with = None
            human_hypothesis.annotations.append(activity)

        threshold_markers = self.generate_pointing_threshold_markers(
            Point3.from_iterable(elbow_pose_in_world.translation),
            Quaternion.from_iterable(elbow_pose_in_world.rotation),
            "map",
            self.descriptor.parameters.maximum_pointing_angle,
        )
        threshold_marker_array = visualization_msgs.msg.MarkerArray()
        threshold_marker_array.markers.extend(threshold_markers)
        self.threshold_marker_publisher.publish(threshold_marker_array)

    def choose_arm(
        self,
        intersections_left: list[Any],
        intersections_right: list[Any],
        pointing_direction_left: np.ndarray,
        pointing_direction_right: np.ndarray,
    ) -> tuple[list[Any], Arm]:
        """
        Choose the arm based on the intersections found.

        With intersections on both arms, choose the larger pointing angle, without any
        the smaller one. With intersections on one arm, choose that arm.

        :param intersections_left: The intersections of the left arm.
        :param intersections_right: The intersections of the right arm.
        :param pointing_direction_left: The pointing vector of the left arm.
        :param pointing_direction_right: The pointing vector of the right arm.
        :return: The intersections and the chosen arm.
        """
        pointing_angle_left = compute_direction_vector_angle(
            direction_vector=pointing_direction_left
        )
        pointing_angle_right = compute_direction_vector_angle(
            direction_vector=pointing_direction_right
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

        if pointing_angle_left > pointing_angle_right:
            self.rk_logger.debug("Left arm points higher than right arm")
            return intersections_left, Arm.LEFT
        else:
            self.rk_logger.debug("Right arm points higher than left arm")
            return intersections_right, Arm.RIGHT

    def create_visualization_geometries(
        self,
        all_visualization_geometries: list[Any],
        elbow_centroid: np.ndarray,
        wrist_centroid: np.ndarray,
        pointing_end: np.ndarray,
        pointed_at_objects: list[str],
        updated_structures: dict[str, Any],
    ) -> list[Any]:
        """
        Create the geometries for the Open3D visualization.

        :param all_visualization_geometries: The geometries to visualize.
        :param elbow_centroid: The centroid of the elbow.
        :param wrist_centroid: The centroid of the wrist.
        :param pointing_end: The end point of the pointing line.
        :param pointed_at_objects: The objects that were pointed at.
        :param updated_structures: The updated structures with their poses.
        :return: The geometries for the visualization.
        """
        centroid_structure = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        centroid_structure.paint_uniform_color([1, 0, 0])
        centroid_structure.translate(elbow_centroid)
        all_visualization_geometries.append(centroid_structure)

        centroid_structure = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.04
        )  # in meters
        centroid_structure.paint_uniform_color([0, 1, 0])
        centroid_structure.translate(wrist_centroid)
        all_visualization_geometries.append(centroid_structure)

        all_visualization_geometries.append(
            robokudo.utils.o3d_helper.create_line_for_visualization(
                elbow_centroid, pointing_end, [1, 0, 0]
            )
        )

        centroid_structure = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.04
        )  # in meters
        centroid_structure.paint_uniform_color([1, 0, 0])
        centroid_structure.translate(pointing_end)
        all_visualization_geometries.append(centroid_structure)

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
            all_visualization_geometries.append(centroid_structure)

        return all_visualization_geometries

    def generate_marker(
        self, origin: Point3, orientation: Quaternion, frame: str
    ) -> visualization_msgs.msg.Marker:
        """
        :param origin: Start of the marker.
        :param orientation: Orientation of the marker.
        :param frame: Frame of the marker.
        :return: The arrow marker of the pointing direction.
        """
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = frame
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.position.x = float(origin.x)
        marker.pose.position.y = float(origin.y)
        marker.pose.position.z = float(origin.z)
        marker.pose.orientation.x = float(orientation.x)
        marker.pose.orientation.y = float(orientation.y)
        marker.pose.orientation.z = float(orientation.z)
        marker.pose.orientation.w = float(orientation.w)

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
        self,
        origin: Point3,
        orientation: Quaternion,
        frame: str,
        maximum_pointing_angle: float,
    ) -> list[visualization_msgs.msg.Marker]:
        """
        :param origin: Tip of the threshold cone.
        :param orientation: Orientation of the pointing direction.
        :param frame: Frame of the markers.
        :param maximum_pointing_angle: The steepest angle that counts as pointing, in
            degrees.
        :return: The circle and the rays of the pointing threshold.
        """
        length = self.descriptor.parameters.pointing_length_meters
        radius = length * np.tan(np.deg2rad(maximum_pointing_angle))
        segment_count = 32

        up = np.array([0.0, 0.0, 1.0], dtype=float)

        first_circle_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        second_circle_axis = np.array([0.0, 1.0, 0.0], dtype=float)

        origin_array = origin.to_np()[:3]
        circle_center = origin_array - up * length

        points = []
        for segment_index in range(segment_count + 1):
            theta = 2.0 * np.pi * segment_index / segment_count

            point = circle_center + radius * (
                np.cos(theta) * first_circle_axis + np.sin(theta) * second_circle_axis
            )
            circle_point = geometry_msgs.msg.Point(
                x=float(point[0]), y=float(point[1]), z=float(point[2])
            )
            points.append(circle_point)

        stamp = self.node.get_clock().now().to_msg()
        marker_namespace = f"Pointing Threshold-{self.node.get_name()}"

        circle_marker = visualization_msgs.msg.Marker()
        circle_marker.header.frame_id = frame
        circle_marker.header.stamp = stamp
        circle_marker.ns = marker_namespace
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
        rays_marker.ns = marker_namespace
        rays_marker.id = 3
        rays_marker.type = visualization_msgs.msg.Marker.LINE_LIST
        rays_marker.action = visualization_msgs.msg.Marker.ADD
        rays_marker.scale.x = 0.02
        rays_marker.color.r = 1.0
        rays_marker.color.g = 0.0
        rays_marker.color.b = 0.0
        rays_marker.color.a = 0.9
        origin_point = geometry_msgs.msg.Point(
            x=float(origin_array[0]), y=float(origin_array[1]), z=float(origin_array[2])
        )
        rays_marker.points = []
        for circle_point in points:
            rays_marker.points.append(origin_point)
            rays_marker.points.append(circle_point)

        return [circle_marker, rays_marker]
