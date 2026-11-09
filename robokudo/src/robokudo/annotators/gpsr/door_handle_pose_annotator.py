"""
Grasp poses of door handles, estimated from the door plane and the points in front of it.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from timeit import default_timer
from enum import StrEnum

import cv2
import numpy as np
import open3d as o3d
import py_trees
from typing_extensions import Any

import robokudo.annotators.core
import robokudo.utils.error_handling
import robokudo.utils.transform
from robokudo.cas import CAS, CASViews
from robokudo.io.camera_model_adapters import Open3DCameraModelAdapter
from robokudo.types.annotation import (
    Classification,
    PoseAnnotation,
)
from robokudo.types.cv import Point2D, Rect
from robokudo.types.scene import ObjectHypothesis


# %% door object classes
class DoorObjectClass(StrEnum):
    """Classifications of doors, handles and grasp points."""

    DOOR = "door"
    """
    A room door.
    """
    CABINET_DOOR = "cabinet door"
    """
    The door of a cabinet.
    """
    REFRIGERATOR_DOOR = "refrigerator door"
    """
    The door of a refrigerator.
    """
    HANDLE = "handle"
    """
    The handle of a door.
    """
    GRASP_POINT = "grasp_point"
    """
    The spot a gripper closes on, answered as its own hypothesis.
    """


# %% door and handle geometry


@dataclass
class DoorPlane:
    """
    The plane of a door, facing the camera.
    """

    normal: np.ndarray
    """
    Unit normal of the plane, pointing toward the camera.
    """
    offset: float
    """
    Offset of the plane in meters. A point on the plane has ``normal · point + offset``
    equal to zero.
    """
    inlier_count: int
    """
    Number of points that lie on the plane.
    """


@dataclass
class PlanePose:
    """
    A pose on a plane, facing it, in camera coordinates.
    """

    translation: np.ndarray
    """
    Position on the plane.
    """
    rotation: np.ndarray
    """
    Rotation matrix whose x axis approaches the plane and whose y axis points up in the
    image.
    """


@dataclass
class HandleGrasp:
    """
    A grasp of a door handle, in camera coordinates.
    """

    centroid: np.ndarray
    """
    Center of the handle points.
    """
    translation: np.ndarray
    """
    Where the gripper closes on the handle.
    """
    rotation: np.ndarray
    """
    Rotation matrix whose x axis approaches the door and whose y axis is :attr:`axis`.
    """
    axis: np.ndarray
    """
    Unit direction of the handle, pointing at its free end.
    """
    spread: float
    """
    Largest distance of a handle point from the centroid along the closing direction of
    the jaws, in meters.
    """
    length: float
    """
    Extent of the handle along its axis, in meters.
    """
    offset: float
    """
    Distance of the grasp from the centroid toward the free end, in meters.
    """
    free_end_reason: str
    """
    What decided the direction of the free end.
    """


@dataclass
class HandleResult:
    """
    The outcome for one detected handle, for the log and the debug image.
    """

    region: Rect
    """
    Image region of the handle, clipped to the image.
    """
    note: str
    """
    Why the handle was rejected, or how it is grasped.
    """
    handle_pixels: np.ndarray | None = None
    """
    Pixels of the handle points in front of the door, *None* for a rejected handle.
    """
    grasp_translation: np.ndarray | None = None
    """
    Where the gripper closes, in camera coordinates, *None* for a rejected handle.
    """


# %% door handle pose
class DoorHandlePoseAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Estimates a grasp pose for door handles.

    The door is a plane and the handle is what sticks out of it. The plane gives the
    approach direction, the points in front of it the axis of the handle.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration for door and handle pose estimation."""

        class Parameters:
            """Thresholds and options for door and handle pose estimation."""

            def __init__(self) -> None:
                """
                Start with the default thresholds for the door plane and the handle points.
                """
                self.depth_scale: float = 0.001
                """Factor from depth image values to meters."""
                self.plane_distance_threshold: float = 0.02
                """Maximum distance of a plane inlier from the plane, in meters."""
                self.ransac_iterations: int = 200
                """Number of plane fitting iterations."""
                self.minimum_protrusion: float = 0.02
                """Minimum distance of a handle point in front of the plane, in meters."""
                self.maximum_protrusion: float = 0.12
                """Maximum distance of a handle point in front of the plane, in meters."""
                self.minimum_handle_points: int = 40
                """Minimum number of handle points required for a grasp pose."""
                self.valid_depth_range: tuple[float, float] = (0.3, 4.0)
                """Exclusive bounds for valid depth values, in meters."""
                self.jaw_clearance: float = 0.015
                """Clearance added to each side of the handle, in meters."""
                self.jaw_pad_width: float = 0.04
                """Width of a finger pad along the handle, in meters."""
                self.jaw_pad_depth: float = 0.035
                """Depth of a finger pad toward the door, in meters."""
                self.tip_clearance: float = 0.01
                """Clearance between the finger pad and the handle tip, in meters."""
                self.minimum_protrusion_gradient: float = 0.005
                """Minimum protrusion difference for choosing a free end, in meters."""
                self.debug_view: bool = False
                """Draw the plane region and rejected detections."""
                self.drop_handles_without_pose: bool = True
                """Remove handle hypotheses without a grasp pose."""
                self.minimum_plane_points: int = 100
                """Minimum number of points required to fit a plane."""
                self.plane_sample_points: int = 3
                """Number of points sampled for each plane fit."""
                self.end_fraction: float = 0.1
                """Fraction of handle points used at each end."""
                self.minimum_end_points: int = 5
                """Minimum number of points used at each handle end."""
                self.tip_percentile: int = 98
                """Percentile of handle points used to locate the tip."""

        parameters = Parameters()
        """Default parameters on the class. Each descriptor instance gets its own copy."""

    def __init__(
        self,
        name: str = "DoorHandlePoseAnnotator",
        descriptor: DoorHandlePoseAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        """
        super(DoorHandlePoseAnnotator, self).__init__(name, descriptor)
        self.depth: np.ndarray | None = None
        """Depth image of the current observation, in meters."""
        self.focal_length_x: float | None = None
        """Horizontal focal length, in pixels."""
        self.focal_length_y: float | None = None
        """Vertical focal length, in pixels."""
        self.principal_point_x: float | None = None
        """Horizontal principal point, in pixels."""
        self.principal_point_y: float | None = None
        """Vertical principal point, in pixels."""

    def get_hypotheses_by_classname(
        self, classnames: tuple[str, ...] | list[str]
    ) -> list[tuple[ObjectHypothesis, Classification]]:
        """
        :param classnames: The classes to look for.
        :return: Every hypothesis of such a class with its first matching
            classification.
        """
        result = []
        for object_hypothesis in self.get_cas().filter_annotations_by_type(
            ObjectHypothesis
        ):
            classifications = CAS.filter_by_type(
                Classification, object_hypothesis.annotations
            )
            for classification in classifications:
                if classification.classname in classnames:
                    result.append((object_hypothesis, classification))
                    break
        return result

    def clip_region_to_image(
        self, object_hypothesis: ObjectHypothesis, width: int, height: int
    ) -> Rect:
        """
        :param object_hypothesis: The hypothesis with the region.
        :param width: Width of the image in pixels.
        :param height: Height of the image in pixels.
        :return: The region clipped to the image.
        """
        region = object_hypothesis.roi.roi
        x = int(max(region.pos.x, 0))
        y = int(max(region.pos.y, 0))
        return Rect(
            pos=Point2D(x=x, y=y),
            width=int(min(region.width, width - x)),
            height=int(min(region.height, height - y)),
        )

    def find_points_in_box(self, box: Rect) -> tuple[np.ndarray, np.ndarray]:
        """
        Backproject the valid depth pixels of one image region into camera coordinates.

        :param box: The image region.
        :return: The points in camera coordinates and the pixels they come from.
        """
        x, y, box_width, box_height = box.pos.x, box.pos.y, box.width, box.height
        if box_width <= 0 or box_height <= 0:
            return np.empty((0, 3)), np.empty((0, 2), dtype=int)

        depth = self.depth[y : y + box_height, x : x + box_width]
        minimum_depth, maximum_depth = self.descriptor.parameters.valid_depth_range
        valid = np.isfinite(depth) & (depth > minimum_depth) & (depth < maximum_depth)
        if not valid.any():
            return np.empty((0, 3)), np.empty((0, 2), dtype=int)

        rows, columns = np.nonzero(valid)
        columns = columns + x
        rows = rows + y
        distances = depth[valid]
        points = np.stack(
            [
                (columns - self.principal_point_x) * distances / self.focal_length_x,
                (rows - self.principal_point_y) * distances / self.focal_length_y,
                distances,
            ],
            axis=1,
        )
        return points, np.stack([columns, rows], axis=1)

    def fit_plane(self, points: np.ndarray) -> DoorPlane | None:
        """
        :param points: The points in camera coordinates.
        :return: The camera-facing plane, or *None* without a plane.
        """
        if len(points) < self.descriptor.parameters.minimum_plane_points:
            return None
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        model, inliers = cloud.segment_plane(
            distance_threshold=self.descriptor.parameters.plane_distance_threshold,
            ransac_n=self.descriptor.parameters.plane_sample_points,
            num_iterations=self.descriptor.parameters.ransac_iterations,
        )
        normal_x, normal_y, normal_z, plane_offset = model
        normal = np.array([normal_x, normal_y, normal_z], dtype=float)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            return None
        normal, plane_offset = normal / length, plane_offset / length
        if normal[2] > 0:  # let the normal point back toward the camera
            normal, plane_offset = -normal, -plane_offset
        return DoorPlane(normal=normal, offset=plane_offset, inlier_count=len(inliers))

    def find_free_end_sign(
        self,
        along: np.ndarray,
        protrusion: np.ndarray,
        axis: np.ndarray,
        centroid: np.ndarray,
        door_center: np.ndarray | None,
    ) -> tuple[float, str]:
        """
        Decide which way along the axis the handle ends free.

        A handle drops toward its rosette. With alike ends it points to the door center.

        :param along: Position of each handle point along the axis.
        :param protrusion: Distance of each handle point in front of the door, in meters.
        :param axis: The handle axis.
        :param centroid: Center of the handle points.
        :param door_center: Center of the door, if known.
        :return: The direction along the axis as 1 or -1, and what decided it.
        """
        edge = max(
            int(self.descriptor.parameters.end_fraction * len(along)),
            self.descriptor.parameters.minimum_end_points,
        )
        order = np.argsort(along)
        low = float(protrusion[order[:edge]].mean())
        high = float(protrusion[order[-edge:]].mean())
        if abs(high - low) >= self.descriptor.parameters.minimum_protrusion_gradient:
            return (1.0 if high > low else -1.0), "protrusion"
        if door_center is None:
            return 1.0, "no hint"
        return (
            1.0 if np.dot(door_center - centroid, axis) > 0 else -1.0
        ), "door center"

    def compute_grasp_pose(
        self,
        points: np.ndarray,
        protrusion: np.ndarray,
        normal: np.ndarray,
        door_center: np.ndarray | None,
    ) -> HandleGrasp | None:
        """
        Build a right handed frame: x approaches the door, y follows the handle axis.

        :param points: The handle points in camera coordinates.
        :param protrusion: Distance of each handle point in front of the door, in meters.
        :param normal: Normal of the door plane.
        :param door_center: Center of the door, if known.
        :return: The grasp, or *None* if the axis has no direction across the approach.
        """
        centroid = points.mean(axis=0)
        centered = points - centroid
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        axis = eigenvectors[:, eigenvalues.argmax()]

        approach = -normal
        axis = axis - np.dot(axis, approach) * approach
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return None
        axis = axis / norm

        along = centered @ axis
        sign, reason = self.find_free_end_sign(
            along, protrusion, axis, centroid, door_center
        )
        axis = sign * axis  # from here on the axis points at the free end

        third = np.cross(approach, axis)
        rotation = np.column_stack([approach, axis, third])

        # Sit one pad half plus some air inside the tip, never past the centroid.
        margin = (
            0.5 * self.descriptor.parameters.jaw_pad_width
            + self.descriptor.parameters.tip_clearance
        )
        tip = float(
            np.percentile(sign * along, self.descriptor.parameters.tip_percentile)
        )  # a percentile ignores single stray points
        offset = max(tip - margin, 0.0)
        translation = centroid + offset * axis
        length = float(along.max() - along.min())

        # How thick the handle is across the closing direction. The jaws have to clear that.
        spread = float(np.abs(centered @ third).max())
        return HandleGrasp(
            centroid=centroid,
            translation=translation,
            rotation=rotation,
            axis=axis,
            spread=spread,
            length=length,
            offset=offset,
            free_end_reason=reason,
        )

    def compute_frame_from_normal(self, normal: np.ndarray) -> np.ndarray | None:
        """
        Frame of a flat surface: x approaches it, y points up in the image.

        :param normal: Normal of the surface.
        :return: The rotation matrix, or *None* for a surface facing straight up or down
            in the image.
        """
        approach = -normal
        up = np.array([0.0, -1.0, 0.0])
        axis = up - np.dot(up, approach) * approach
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return None
        axis = axis / norm
        return np.column_stack([approach, axis, np.cross(approach, axis)])

    def compute_pose_on_plane(
        self, points: np.ndarray, plane: DoorPlane
    ) -> PlanePose | None:
        """
        Center of the plane region, snapped exactly onto the plane.

        :param points: The points of the plane region.
        :param plane: The plane.
        :return: The pose, or *None* if no point lies on the plane.
        """
        signed = points @ plane.normal + plane.offset
        inliers = points[
            np.abs(signed) < self.descriptor.parameters.plane_distance_threshold
        ]
        if len(inliers) == 0:
            return None
        rotation = self.compute_frame_from_normal(plane.normal)
        if rotation is None:
            return None
        centroid = inliers.mean(axis=0)
        return PlanePose(
            translation=centroid
            - (np.dot(centroid, plane.normal) + plane.offset) * plane.normal,
            rotation=rotation,
        )

    def create_grasp_hypothesis(
        self, translation: np.ndarray, rotation: np.ndarray
    ) -> ObjectHypothesis:
        """
        Build a hypothesis of its own for the grasp point, answered as a separate entry.

        :param translation: Position of the grasp point.
        :param rotation: Rotation of the grasp point.
        :return: The hypothesis of the grasp point.
        """
        object_hypothesis = ObjectHypothesis()
        object_hypothesis.type = DoorObjectClass.GRASP_POINT.value
        classification = Classification()
        classification.classname = DoorObjectClass.GRASP_POINT.value
        classification.source = self.get_class_name()
        object_hypothesis.annotations.append(classification)
        self.add_pose(
            object_hypothesis, translation, rotation, DoorObjectClass.GRASP_POINT.value
        )
        return object_hypothesis

    def add_pose(
        self,
        object_hypothesis: ObjectHypothesis,
        translation: np.ndarray,
        rotation: np.ndarray,
        source: str,
    ) -> None:
        """
        Attach one pose to a hypothesis.

        :param object_hypothesis: The hypothesis that receives the pose.
        :param translation: Position of the pose.
        :param rotation: Rotation of the pose.
        :param source: Label of what the pose belongs to, answered as pose source.
        """
        pose_annotation = PoseAnnotation()
        pose_annotation.translation = list(translation)
        pose_annotation.rotation = list(
            robokudo.utils.transform.get_quaternion_from_rotation_matrix(rotation)
        )
        pose_annotation.source = source
        object_hypothesis.annotations.append(pose_annotation)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
        """
        Overrides :meth:`BaseAnnotator.update` to fit the door plane and add the poses
        of the largest door and of the handles on it.

        :return: :attr:`Status.SUCCESS`.
        """
        start_timer = default_timer()

        cas = self.get_cas()
        color = cas.get_copy(CASViews.COLOR_IMAGE)
        depth = cas.get(CASViews.DEPTH_IMAGE)
        intrinsics = Open3DCameraModelAdapter.to_intrinsic(
            cas.require_camera_observation().effective_camera_model
        )

        self.depth = (
            np.asarray(depth, dtype=np.float32) * self.descriptor.parameters.depth_scale
        )
        intrinsic_matrix = intrinsics.intrinsic_matrix
        self.focal_length_x = intrinsic_matrix[0, 0]
        self.focal_length_y = intrinsic_matrix[1, 1]
        self.principal_point_x = intrinsic_matrix[0, 2]
        self.principal_point_y = intrinsic_matrix[1, 2]

        height, width = self.depth.shape[:2]
        doors = self.get_hypotheses_by_classname(
            [
                DoorObjectClass.DOOR.value,
                DoorObjectClass.CABINET_DOOR.value,
                DoorObjectClass.REFRIGERATOR_DOOR.value,
            ]
        )
        handles = self.get_hypotheses_by_classname([DoorObjectClass.HANDLE.value])

        geometries = []
        messages = []

        plane = None
        plane_box = None
        door_hypothesis = None
        door_points = np.empty((0, 3))
        if doors:
            door_hypothesis = max(
                doors, key=lambda door: door[0].roi.roi.width * door[0].roi.roi.height
            )[0]
            plane_box = self.clip_region_to_image(door_hypothesis, width, height)
            door_points, _ = self.find_points_in_box(plane_box)
            plane = self.fit_plane(door_points)
        if plane is None:
            plane_box = Rect(pos=Point2D(x=0, y=0), width=width, height=height)
            door_points, _ = self.find_points_in_box(plane_box)
            plane = self.fit_plane(door_points)
            if doors:
                messages.append("door plane fit fell back to the full image")

        if plane is None:
            messages.append("no plane found")
            self.draw(color, plane_box, [], messages)
            self.finish(start_timer, geometries, messages)
            return py_trees.common.Status.SUCCESS

        tilt = np.degrees(np.arccos(min(abs(plane.normal[2]), 1.0)))
        messages.append(f"plane: {plane.inlier_count} inliers, tilt {tilt:.1f} degrees")

        door_center = None
        if door_hypothesis is not None:
            door_pose = self.compute_pose_on_plane(door_points, plane)
            if door_pose is None:
                messages.append("no door pose")
            else:
                door_center = door_pose.translation
                self.add_pose(
                    door_hypothesis,
                    door_pose.translation,
                    door_pose.rotation,
                    DoorObjectClass.DOOR.value,
                )

        results = []
        grasped = []
        grasp_hypotheses = []
        for object_hypothesis, classification in handles:
            box = self.clip_region_to_image(object_hypothesis, width, height)
            points, pixels = self.find_points_in_box(box)
            if len(points) == 0:
                results.append(
                    HandleResult(region=box, note="no depth data on the handle")
                )
                continue

            signed = points @ plane.normal + plane.offset
            selected = (signed > self.descriptor.parameters.minimum_protrusion) & (
                signed < self.descriptor.parameters.maximum_protrusion
            )
            count = int(selected.sum())
            if count < self.descriptor.parameters.minimum_handle_points:
                results.append(
                    HandleResult(
                        region=box,
                        note=f"only {count} points in front of the plane "
                        f"({100.0 * count / max(len(points), 1):.0f} % of the box)",
                    )
                )
                continue

            grasp = self.compute_grasp_pose(
                points[selected], signed[selected], plane.normal, door_center
            )
            if grasp is None:
                results.append(
                    HandleResult(region=box, note="handle axis is degenerate")
                )
                continue

            half_gap = grasp.spread + self.descriptor.parameters.jaw_clearance
            self.add_pose(
                object_hypothesis,
                grasp.centroid,
                grasp.rotation,
                DoorObjectClass.HANDLE.value,
            )
            grasp_hypotheses.append(
                self.create_grasp_hypothesis(grasp.translation, grasp.rotation)
            )

            transform = np.eye(4)
            transform[:3, :3] = grasp.rotation
            transform[:3, 3] = grasp.translation
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame.transform(transform)
            geometries.append(frame)

            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            marker.translate(grasp.translation)
            marker.paint_uniform_color([0.0, 0.35, 1.0])
            marker.compute_vertex_normals()
            geometries.append(marker)

            protrusion = float(np.median(signed[selected]))
            horizontal = np.degrees(np.arccos(min(abs(grasp.axis[0]), 1.0)))
            orientation = "horizontal" if horizontal < 45 else "vertical"

            reach = (
                self.descriptor.parameters.jaw_pad_depth
                + self.descriptor.parameters.tip_clearance
            )
            if protrusion < reach:
                messages.append(
                    f"fingers reach {reach * 100:.1f} cm but the handle only "
                    f"stands {protrusion * 100:.1f} cm out"
                )

            grasped.append(object_hypothesis)
            results.append(
                HandleResult(
                    region=box,
                    note=f"{orientation}, {protrusion * 100:.1f} cm out, "
                    f"{grasp.translation[2]:.2f} m away, "
                    f"jaws {2 * half_gap * 100:.1f} cm apart, "
                    f"handle {grasp.length * 100:.1f} cm, "
                    f"grasp {grasp.offset * 100:.1f} cm "
                    f"off center toward the free end ({grasp.free_end_reason})",
                    handle_pixels=pixels[selected],
                    grasp_translation=grasp.translation,
                )
            )

        if not handles:
            messages.append("no handle detected")
        messages.append(f"{len(grasped)} of {len(handles)} handles with grasp pose")

        for result in results:
            outcome = "rejected" if result.grasp_translation is None else "grasp"
            self.rk_logger.info(f"{outcome}: {result.note}")

        if self.descriptor.parameters.drop_handles_without_pose:
            rejected = {id(handle) for handle, _ in handles} - {
                id(handle) for handle in grasped
            }
            cas.annotations = [
                annotation
                for annotation in cas.annotations
                if id(annotation) not in rejected
            ]
        cas.annotations.extend(grasp_hypotheses)

        self.draw(color, plane_box, results, messages)
        self.finish(start_timer, geometries, messages)
        return py_trees.common.Status.SUCCESS

    def project(self, point: np.ndarray) -> tuple[int, int] | None:
        """
        Project a point in front of the camera into the image.

        :param point: The point in camera coordinates.
        :return: The pixel of the point, or *None* for a point behind the camera.
        """
        if point[2] <= 1e-6:
            return None
        return (
            int(
                round(
                    point[0] * self.focal_length_x / point[2] + self.principal_point_x
                )
            ),
            int(
                round(
                    point[1] * self.focal_length_y / point[2] + self.principal_point_y
                )
            ),
        )

    def draw_grasp_point(self, image: np.ndarray, translation: np.ndarray) -> None:
        """
        Draw a dot on the spot the gripper closes on.

        :param image: The image to draw on.
        :param translation: The grasp position in camera coordinates.
        """
        origin = self.project(translation)
        if origin is None:
            return
        cv2.circle(image, origin, 10, (255, 255, 255), -1)
        cv2.circle(image, origin, 9, (255, 90, 0), -1)

    def draw(
        self,
        image: np.ndarray,
        plane_box: Rect | None,
        results: list[HandleResult],
        messages: list[str],
    ) -> None:
        """
        Draw grasp results and optional diagnostics on the image.

        :param image: The image to draw on.
        :param plane_box: The region of the plane fit, if any.
        :param results: The result of every handle.
        :param messages: The diagnostic messages.
        """
        debug = self.descriptor.parameters.debug_view
        if debug and plane_box is not None:
            x, y = plane_box.pos.x, plane_box.pos.y
            cv2.rectangle(
                image,
                (x, y),
                (x + plane_box.width, y + plane_box.height),
                (255, 160, 0),
                2,
            )

        for result in results:
            x, y = result.region.pos.x, result.region.pos.y
            corner = (x + result.region.width, y + result.region.height)
            if result.grasp_translation is None:
                if debug:
                    cv2.rectangle(image, (x, y), corner, (0, 0, 255), 1)
                continue
            pixels = result.handle_pixels
            image[pixels[:, 1], pixels[:, 0]] = (0, 255, 255)
            cv2.rectangle(image, (x, y), corner, (0, 200, 0), 2)
            self.draw_grasp_point(image, result.grasp_translation)

        self.get_annotator_output_struct().set_image(image)

    def finish(
        self, start_timer: float, geometries: list[Any], messages: list[str]
    ) -> None:
        """
        Publish the geometries and feedback for the observation.

        :param start_timer: Time at which the update started.
        :param geometries: The geometries to visualize.
        :param messages: The messages for the feedback.
        """
        if geometries and self.get_cas().contains(CASViews.CLOUD):
            cloud = self.get_cas().get(CASViews.CLOUD)
            self.get_annotator_output_struct().set_geometries([cloud] + geometries)
        self.feedback_message = (
            f"{'; '.join(messages)} | "
            f"Processing took {(default_timer() - start_timer):.4f}s"
        )
