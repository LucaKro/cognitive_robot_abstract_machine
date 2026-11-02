from __future__ import annotations

# %% imports
from timeit import default_timer

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
from robokudo.types.scene import ObjectHypothesis

DOOR_CLASSES = ("door", "cabinet door", "refrigerator door")
HANDLE_CLASS = "handle"
GRASP_CLASS = "grasp_point"


# %% door handle pose
class DoorHandlePoseAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Estimate a grasp pose for door handles.

    The door is a plane, the handle is what sticks out of it. A RANSAC plane fit on the
    door region gives the approach direction, the points in front of that plane give the
    handle geometry, and a PCA on those points gives its axis.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration for door and handle pose estimation."""

        class Parameters:
            """Thresholds and options for door and handle pose estimation."""

            def __init__(self) -> None:
                self.depth_scale = 0.001
                """Scale applied to depth values."""
                self.plane_distance_threshold = 0.02
                """Maximum distance of a plane inlier from the plane."""
                self.ransac_iterations = 200
                """Number of plane fitting iterations."""
                self.min_protrusion = 0.02
                """Minimum distance of a handle point in front of the plane."""
                self.max_protrusion = 0.12
                """Maximum distance of a handle point in front of the plane."""
                self.min_handle_points = 40
                """Minimum number of handle points required for a grasp pose."""
                self.valid_depth_range = (0.3, 4.0)
                """Exclusive bounds for valid depth values."""
                self.jaw_clearance = 0.015
                """Clearance added to each side of the handle."""
                self.jaw_pad_width = 0.04
                """Width of a finger pad along the handle."""
                self.jaw_pad_depth = 0.035
                """Depth of a finger pad toward the door."""
                self.tip_clearance = 0.01
                """Clearance between the finger pad and the handle tip."""
                self.min_protrusion_gradient = 0.005
                """Minimum protrusion difference for choosing a free end."""
                self.debug_view = False
                """Draw the plane region and rejected detections."""
                self.drop_handles_without_pose = True
                """Remove handle hypotheses without a grasp pose."""
                self.min_plane_points = 100
                """Minimum number of points required to fit a plane."""
                self.plane_sample_points = 3
                """Number of points sampled for each plane fit."""
                self.end_fraction = 0.1
                """Fraction of handle points used at each end."""
                self.min_end_points = 5
                """Minimum number of points used at each handle end."""
                self.tip_percentile = 98
                """Percentile of handle points used to locate the tip."""

        parameters = Parameters()
        """Default parameters on the class; each descriptor instance gets its own."""

    def __init__(
        self,
        name: str = "DoorHandlePoseAnnotator",
        descriptor: DoorHandlePoseAnnotator.Descriptor = Descriptor(),
    ) -> None:
        super(DoorHandlePoseAnnotator, self).__init__(name, descriptor)
        self.depth: np.ndarray | None = None
        """Scaled depth image of the current observation."""
        self.fx: float | None = None
        """Horizontal focal length."""
        self.fy: float | None = None
        """Vertical focal length."""
        self.cx: float | None = None
        """Horizontal principal point."""
        self.cy: float | None = None
        """Vertical principal point."""

    def get_hypotheses_by_classname(
        self, classnames: tuple[str, ...] | list[str]
    ) -> list[tuple[ObjectHypothesis, Classification]]:
        """Return hypotheses and their first matching classification."""
        result = []
        for hypothesis in self.get_cas().filter_annotations_by_type(ObjectHypothesis):
            classifications = CAS.filter_by_type(Classification, hypothesis.annotations)
            for classification in classifications:
                if classification.classname in classnames:
                    result.append((hypothesis, classification))
                    break
        return result

    def roi_box(
        self, hypothesis: ObjectHypothesis, width: int, height: int
    ) -> tuple[int, int, int, int]:
        """Return the hypothesis region clipped to the image dimensions."""
        roi = hypothesis.roi.roi
        x = int(max(roi.pos.x, 0))
        y = int(max(roi.pos.y, 0))
        box_width = int(min(roi.width, width - x))
        box_height = int(min(roi.height, height - y))
        return x, y, box_width, box_height

    def points_in_box(
        self, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Backproject the valid depth pixels of one image region into camera coordinates.
        """
        x, y, box_width, box_height = box
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
                (columns - self.cx) * distances / self.fx,
                (rows - self.cy) * distances / self.fy,
                distances,
            ],
            axis=1,
        )
        return points, np.stack([columns, rows], axis=1)

    def fit_plane(self, points: np.ndarray) -> tuple[np.ndarray, float, int] | None:
        """Return a camera-facing plane fitted to the points, if possible."""
        if len(points) < self.descriptor.parameters.min_plane_points:
            return None
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        model, inliers = cloud.segment_plane(
            distance_threshold=self.descriptor.parameters.plane_distance_threshold,
            ransac_n=self.descriptor.parameters.plane_sample_points,
            num_iterations=self.descriptor.parameters.ransac_iterations,
        )
        a, b, c, d = model
        normal = np.array([a, b, c], dtype=float)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            return None
        normal, d = normal / length, d / length
        if normal[2] > 0:  # let the normal point back towards the camera
            normal, d = -normal, -d
        return normal, d, len(inliers)

    def free_end_sign(
        self,
        along: np.ndarray,
        protrusion: np.ndarray,
        axis: np.ndarray,
        centroid: np.ndarray,
        door_center: np.ndarray | None,
    ) -> tuple[float, str]:
        """
        Which way along the axis the handle ends free.

        Towards the rosette the handle curves back into the door, so its protrusion
        drops. When both ends look alike, fall back on the door: a lever points to the
        door centre.
        """
        edge = max(
            int(self.descriptor.parameters.end_fraction * len(along)),
            self.descriptor.parameters.min_end_points,
        )
        order = np.argsort(along)
        low = float(protrusion[order[:edge]].mean())
        high = float(protrusion[order[-edge:]].mean())
        if abs(high - low) >= self.descriptor.parameters.min_protrusion_gradient:
            return (1.0 if high > low else -1.0), "protrusion"
        if door_center is None:
            return 1.0, "no hint"
        return (
            1.0 if np.dot(door_center - centroid, axis) > 0 else -1.0
        ), "door centre"

    def grasp_pose_from_points(
        self,
        points: np.ndarray,
        protrusion: np.ndarray,
        normal: np.ndarray,
        door_center: np.ndarray | None,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, str]
        | None
    ):
        """
        Build a right handed frame: x approaches the door, y follows the handle axis.
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
        sign, reason = self.free_end_sign(
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
        )  # robust against single stray points
        offset = max(tip - margin, 0.0)
        translation = centroid + offset * axis
        length = float(along.max() - along.min())

        # How thick the handle is across the closing direction. The jaws have to clear that.
        spread = float(np.abs(centered @ third).max())
        return centroid, translation, rotation, axis, spread, length, offset, reason

    def frame_from_normal(self, normal: np.ndarray) -> np.ndarray | None:
        """
        Frame of a flat surface: x approaches it, y points up in the image.
        """
        approach = -normal
        up = np.array([0.0, -1.0, 0.0])
        axis = up - np.dot(up, approach) * approach
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return None
        axis = axis / norm
        return np.column_stack([approach, axis, np.cross(approach, axis)])

    def pose_on_plane(
        self, points: np.ndarray, normal: np.ndarray, plane_d: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Centre of the plane region, snapped exactly onto the plane.
        """
        signed = points @ normal + plane_d
        inliers = points[
            np.abs(signed) < self.descriptor.parameters.plane_distance_threshold
        ]
        if len(inliers) == 0:
            return None
        rotation = self.frame_from_normal(normal)
        if rotation is None:
            return None
        centroid = inliers.mean(axis=0)
        return centroid - (np.dot(centroid, normal) + plane_d) * normal, rotation

    def grasp_hypothesis(
        self, translation: np.ndarray, rotation: np.ndarray
    ) -> ObjectHypothesis:
        """
        Own hypothesis, so the grasp point becomes a separate entry in the query result.
        """
        hypothesis = ObjectHypothesis()
        hypothesis.type = GRASP_CLASS
        classification = Classification()
        classification.classname = GRASP_CLASS
        classification.source = type(self).__name__
        hypothesis.annotations.append(classification)
        self.add_pose(hypothesis, translation, rotation, GRASP_CLASS)
        return hypothesis

    def add_pose(
        self,
        hypothesis: ObjectHypothesis,
        translation: np.ndarray,
        rotation: np.ndarray,
        source: str,
    ) -> None:
        """
        Attach one pose to a hypothesis.

        The source label survives into pose_source.
        """
        pose_annotation = PoseAnnotation()
        pose_annotation.translation = list(translation)
        pose_annotation.rotation = list(
            robokudo.utils.transform.get_quaternion_from_rotation_matrix(rotation)
        )
        pose_annotation.source = source
        hypothesis.annotations.append(pose_annotation)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
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
        k = intrinsics.intrinsic_matrix
        self.fx, self.fy, self.cx, self.cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

        height, width = self.depth.shape[:2]
        doors = self.get_hypotheses_by_classname(DOOR_CLASSES)
        handles = self.get_hypotheses_by_classname([HANDLE_CLASS])

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
            plane_box = self.roi_box(door_hypothesis, width, height)
            door_points, _ = self.points_in_box(plane_box)
            plane = self.fit_plane(door_points)
        if plane is None:
            plane_box = (0, 0, width, height)
            door_points, _ = self.points_in_box(plane_box)
            plane = self.fit_plane(door_points)
            if doors:
                messages.append("door plane fit fell back to the full image")

        if plane is None:
            messages.append("no plane found")
            self.draw(color, plane_box, [], messages)
            self.finish(start_timer, geometries, messages)
            return py_trees.common.Status.SUCCESS

        normal, plane_d, inlier_count = plane
        tilt = np.degrees(np.arccos(min(abs(normal[2]), 1.0)))
        messages.append(f"plane: {inlier_count} inliers, tilt {tilt:.1f} deg")

        door_center = None
        if door_hypothesis is not None:
            door_pose = self.pose_on_plane(door_points, normal, plane_d)
            if door_pose is None:
                messages.append("no door pose")
            else:
                door_center = door_pose[0]
                self.add_pose(door_hypothesis, door_center, door_pose[1], "door")

        results = []
        grasped = []
        grasp_hypotheses = []
        for hypothesis, classification in handles:
            box = self.roi_box(hypothesis, width, height)
            points, pixels = self.points_in_box(box)
            if len(points) == 0:
                results.append((box, None, "no depth data on the handle"))
                continue

            signed = points @ normal + plane_d
            selected = (signed > self.descriptor.parameters.min_protrusion) & (
                signed < self.descriptor.parameters.max_protrusion
            )
            count = int(selected.sum())
            if count < self.descriptor.parameters.min_handle_points:
                results.append(
                    (
                        box,
                        None,
                        f"only {count} points in front of the plane "
                        f"({100.0 * count / max(len(points), 1):.0f} % of the box)",
                    )
                )
                continue

            pose = self.grasp_pose_from_points(
                points[selected], signed[selected], normal, door_center
            )
            if pose is None:
                results.append((box, None, "handle axis is degenerate"))
                continue

            centroid, translation, rotation, axis, spread, length, offset, reason = pose
            half_gap = spread + self.descriptor.parameters.jaw_clearance
            self.add_pose(hypothesis, centroid, rotation, "handle")
            grasp_hypotheses.append(self.grasp_hypothesis(translation, rotation))

            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame.transform(transform)
            geometries.append(frame)

            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            marker.translate(translation)
            marker.paint_uniform_color([0.0, 0.35, 1.0])
            marker.compute_vertex_normals()
            geometries.append(marker)

            protrusion = float(np.median(signed[selected]))
            horizontal = np.degrees(np.arccos(min(abs(axis[0]), 1.0)))
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

            grasped.append(hypothesis)
            results.append(
                (
                    box,
                    (pixels[selected], translation),
                    f"{orientation}, {protrusion * 100:.1f} cm out, "
                    f"{translation[2]:.2f} m away, "
                    f"jaws {2 * half_gap * 100:.1f} cm apart, "
                    f"handle {length * 100:.1f} cm, grasp {offset * 100:.1f} cm "
                    f"off centre towards the free end ({reason})",
                )
            )

        if not handles:
            messages.append("no handle detected")
        messages.append(f"{len(grasped)} of {len(handles)} handles with grasp pose")

        # The image only carries the dot, the numbers go to the terminal.
        for _, payload, note in results:
            self.rk_logger.info(f"{'grasp' if payload else 'rejected'}: {note}")

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
        """Project a point in front of the camera into the image."""
        if point[2] <= 1e-6:
            return None
        return (
            int(round(point[0] * self.fx / point[2] + self.cx)),
            int(round(point[1] * self.fy / point[2] + self.cy)),
        )

    def draw_grasp_point(self, image: np.ndarray, translation: np.ndarray) -> None:
        """
        One dot on the spot the gripper should close on.

        Nothing else.
        """
        origin = self.project(translation)
        if origin is None:
            return
        cv2.circle(image, origin, 10, (255, 255, 255), -1)
        cv2.circle(image, origin, 9, (255, 90, 0), -1)

    def draw(
        self,
        image: np.ndarray,
        plane_box: tuple[int, int, int, int] | None,
        results: list[Any],
        messages: list[str],
    ) -> None:
        """Draw grasp results and optional diagnostics on the image."""
        debug = self.descriptor.parameters.debug_view
        if debug and plane_box is not None:
            x, y, box_width, box_height = plane_box
            cv2.rectangle(
                image, (x, y), (x + box_width, y + box_height), (255, 160, 0), 2
            )

        for box, payload, note in results:
            x, y, box_width, box_height = box
            if payload is None:
                if debug:
                    cv2.rectangle(
                        image, (x, y), (x + box_width, y + box_height), (0, 0, 255), 1
                    )
                continue
            pixels, translation = payload
            image[pixels[:, 1], pixels[:, 0]] = (0, 255, 255)
            cv2.rectangle(
                image, (x, y), (x + box_width, y + box_height), (0, 200, 0), 2
            )
            self.draw_grasp_point(image, translation)

        self.get_annotator_output_struct().set_image(image)

    def finish(
        self, start_timer: float, geometries: list[Any], messages: list[str]
    ) -> None:
        """Publish the geometries and feedback for the observation."""
        if geometries and self.get_cas().contains(CASViews.CLOUD):
            cloud = self.get_cas().get(CASViews.CLOUD)
            self.get_annotator_output_struct().set_geometries([cloud] + geometries)
        self.feedback_message = (
            f"{'; '.join(messages)} | "
            f"Processing took {(default_timer() - start_timer):.4f}s"
        )
