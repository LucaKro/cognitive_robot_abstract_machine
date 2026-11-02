from __future__ import annotations

# %% Imports
import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.types.annotation import (
    CloudAnnotation,
    LocationAnnotation,
    PoseAnnotation,
    PositionAnnotation,
)
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world

if TYPE_CHECKING:
    from robokudo.cas import CAS

# %% furniture filter


class FilterHumansByObjectAnnotator(BaseAnnotator):
    """
    Removes every person from the CAS who is not near a piece of the queried furniture,
    and marks the others with a :class:`LocationAnnotation` naming the furniture.

    Every object hypothesis with a position that is no person counts as a piece of the
    furniture. Without a direction, a person is near when they stand in a box around
    the furniture. A direction (``left``, ``right``, ``front`` or ``back``, seen from the
    robot) moves the box next to the furniture on that side, with the sides taken from
    the camera axes projected onto the floor of the world. Without a furniture type
    nobody is removed; without a piece of furniture every person is removed.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the furniture filter.
        """

        class Parameters:
            """
            Parameters of the furniture filter.
            """

            def __init__(self) -> None:
                self.box_x: float = 1.5
                """
                Extent of the search box along the world *x* axis, or along the robot's
                right when a direction is set, in meters.
                """
                self.box_y: float = 1.5
                """
                Extent of the search box along the world *y* axis, or along the robot's
                viewing direction when a direction is set, in meters.
                """
                self.box_z: float = 2.5
                """
                Height of the search box, centred on the furniture, in meters.
                """
                self.direction: str | None = None
                """
                Side of the furniture, seen from the robot, that the search box is
                placed on: ``left``, ``right``, ``front`` or ``back``. *None* searches
                around the furniture.
                """
                self.target_type: str | None = None
                """
                Name of the furniture the persons are searched near, written into their
                location; without one nobody is removed.
                """

        parameters = Parameters()
        """
        Default parameters on the class; each descriptor instance gets its own.
        """

    def __init__(
        self,
        name: str = "FilterHumansByObjectAnnotator",
        descriptor: FilterHumansByObjectAnnotator.Descriptor = Descriptor(),
    ) -> None:
        super().__init__(name, descriptor)

    def inside_box(
        self,
        human_pos: list[float],
        furniture_pos: list[float],
        box_size: list[float],
        direction: str | None = None,
        right: np.ndarray | None = None,
        forward: np.ndarray | None = None,
    ) -> bool | np.bool_:
        """
        :param human_pos: Position of the person in the world.
        :param furniture_pos: Position of the furniture in the world.
        :param box_size: Extent of the search box along its three axes.
        :param direction: Side of the furniture the box is placed on, or *None* for a
            box around the furniture.
        :param right: The robot's right on the floor, needed with a direction.
        :param forward: The robot's viewing direction on the floor, needed with a
            direction; positive is behind the furniture.
        :return: Whether the person stands in the search box. An unknown direction
            searches around the furniture.
        """
        offset = np.array(human_pos) - np.array(furniture_pos)
        if not direction:
            return np.all(np.abs(offset) <= np.array(box_size) / 2.0)

        box_x, box_y, box_z = box_size
        if abs(offset[2]) > box_z / 2.0:
            return False
        lateral = float(np.dot(offset, right))
        longitudinal = float(np.dot(offset, forward))
        if direction == "right":
            return 0.0 <= lateral <= box_x and abs(longitudinal) <= box_y / 2.0
        if direction == "left":
            return -box_x <= lateral <= 0.0 and abs(longitudinal) <= box_y / 2.0
        if direction == "back":
            return 0.0 <= longitudinal <= box_y and abs(lateral) <= box_x / 2.0
        if direction == "front":
            return -box_y <= longitudinal <= 0.0 and abs(lateral) <= box_x / 2.0
        return np.all(np.abs(offset) <= np.array(box_size) / 2.0)

    def _robot_ground_basis(
        self, cas: CAS
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        :param cas: The CAS whose camera pose is used.
        :return: The unit vectors along the camera's right and along its viewing
            direction, projected onto the floor of the world; each is *None* when its
            axis is (nearly) vertical.
        """

        def to_world(point: list[float]) -> np.ndarray:
            """
            :return: The camera frame point in the world.
            """
            pose = PoseAnnotation()
            pose.translation = list(point)
            pose.rotation = [0.0, 0.0, 0.0, 1.0]
            return np.array(transform_pose_from_camera_to_world(cas, pose).translation)

        origin = to_world([0.0, 0.0, 0.0])
        right = to_world([1.0, 0.0, 0.0]) - origin  # camera image-right axis
        forward = to_world([0.0, 0.0, 1.0]) - origin  # camera optical (forward) axis

        def project_ground(vector: np.ndarray) -> np.ndarray | None:
            """
            :return: The vector projected onto the floor and normalized, or *None* if
                it is (nearly) vertical.
            """
            flat = np.array([vector[0], vector[1], 0.0])
            norm = np.linalg.norm(flat)
            return flat / norm if norm > 1e-6 else None

        return project_ground(right), project_ground(forward)

    def update(self) -> Status:
        cas = self.get_cas()
        parameters = self.descriptor.parameters
        target_type = parameters.target_type
        box_size = [parameters.box_x, parameters.box_y, parameters.box_z]
        direction = parameters.direction
        furniture_world_positions = []
        furniture_camera_positions = []
        furniture_clouds = []
        human_visualization_data = []
        kept_count = 0

        if not target_type:
            self.rk_logger.warning("No target type set")
            return Status.SUCCESS

        # getting every human and every furniture of the given type (yolo filtered it already)
        all_hypotheses = cas.filter_annotations_by_type(ObjectHypothesis)
        humans = [
            hypothesis
            for hypothesis in all_hypotheses
            if isinstance(hypothesis, HumanHypothesis)
        ]
        furniture = [
            hypothesis
            for hypothesis in all_hypotheses
            if not isinstance(hypothesis, HumanHypothesis)
        ]

        right, forward = None, None
        if direction:
            right, forward = self._robot_ground_basis(cas)
            if right is None or forward is None:
                self.rk_logger.warning(
                    "Degenerate camera orientation, ignoring direction"
                )
                direction = None

        for piece_of_furniture in furniture:
            positions = cas.filter_by_type(
                PositionAnnotation, piece_of_furniture.annotations
            )
            if not positions:
                continue
            pose = PoseAnnotation()
            pose.translation = positions[0].translation
            pose.rotation = [0.0, 0.0, 0.0, 1.0]
            furniture_camera_positions.append(pose.translation)
            furniture_clouds.append(self._cluster_cloud(piece_of_furniture))
            furniture_world_positions.append(
                transform_pose_from_camera_to_world(cas, pose).translation
            )

        if not furniture_world_positions:
            self.rk_logger.info(
                f"No '{target_type}' detected, removing all {len(humans)} humans"
            )
            for human in humans:
                cas.annotations.remove(human)
            return Status.SUCCESS

        for human in humans:
            positions = cas.filter_by_type(PositionAnnotation, human.annotations)
            if not positions:
                cas.annotations.remove(human)
                continue

            pose = PoseAnnotation()
            pose.translation = positions[0].translation
            pose.rotation = [0.0, 0.0, 0.0, 1.0]
            world_position = transform_pose_from_camera_to_world(cas, pose).translation

            near_furniture = False
            for furniture_position in furniture_world_positions:
                if self.inside_box(
                    world_position,
                    furniture_position,
                    box_size,
                    direction,
                    right,
                    forward,
                ):
                    near_furniture = True
                    break

            if near_furniture:
                location = LocationAnnotation()
                location.name = target_type
                human.annotations.append(location)
                kept_count += 1
                human_visualization_data.append(
                    (pose.translation, True, self._cluster_cloud(human))
                )
            else:
                cas.annotations.remove(human)
                human_visualization_data.append(
                    (pose.translation, False, self._cluster_cloud(human))
                )

        side = direction if direction else "any side"
        self.rk_logger.info(
            f"Kept {kept_count}/{len(humans)} humans near '{target_type}' ({side})"
        )
        self._visualize(
            furniture_camera_positions,
            furniture_clouds,
            human_visualization_data,
            box_size,
            direction,
        )
        return Status.SUCCESS

    def _visualize(
        self,
        furniture_camera_positions: list[list[float]],
        furniture_clouds: list[o3d.geometry.PointCloud | None],
        human_visualization_data: list[
            tuple[list[float], bool, o3d.geometry.PointCloud | None]
        ],
        box_size: list[float],
        direction: str | None,
    ) -> None:
        """
        Show the search box of every piece of furniture in green, the furniture with its
        cluster in orange and every person with their cluster in blue when kept and in
        red when removed, all in the camera frame.

        :param furniture_camera_positions: Position of every piece of furniture.
        :param furniture_clouds: Cluster of every piece of furniture, if it has one.
        :param human_visualization_data: Position, whether the person is kept, and
            cluster of every person with a position.
        :param box_size: Extent of the search box along its three axes.
        :param direction: Side of the furniture the search box is placed on, or *None*.
        """
        box_x, box_y, box_z = box_size
        geometries = []
        for camera_position, furniture_cloud in zip(
            furniture_camera_positions, furniture_clouds
        ):
            center = np.array(camera_position)
            if direction:
                # camera axes: x=right (box_x), y=down (box_z), z=forward (box_y)
                half_size = np.array([box_x / 2.0, box_z / 2.0, box_y / 2.0])
                min_corner, max_corner = center - half_size, center + half_size
                if direction == "right":
                    min_corner[0], max_corner[0] = center[0], center[0] + box_x
                elif direction == "left":
                    min_corner[0], max_corner[0] = center[0] - box_x, center[0]
                elif direction == "back":
                    min_corner[2], max_corner[2] = center[2], center[2] + box_y
                elif direction == "front":
                    min_corner[2], max_corner[2] = center[2] - box_y, center[2]
            else:
                half_size = np.array(box_size) / 2.0
                min_corner, max_corner = center - half_size, center + half_size
            search_box = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
            search_box.color = [0.0, 1.0, 0.0]
            geometries.append(search_box)
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            marker.translate(center)
            marker.paint_uniform_color([1.0, 0.6, 0.0])
            geometries.append(marker)
            self._append_cloud(geometries, furniture_cloud, [1.0, 0.6, 0.0])
        for camera_position, was_kept, human_cloud in human_visualization_data:
            if camera_position is None:
                continue
            color = [0.0, 0.5, 1.0] if was_kept else [1.0, 0.2, 0.2]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            sphere.translate(np.array(camera_position))
            sphere.paint_uniform_color(color)
            geometries.append(sphere)
            self._append_cloud(geometries, human_cloud, color)
        self.get_annotator_output_struct().set_geometries(geometries)

    def _cluster_cloud(
        self, hypothesis: ObjectHypothesis
    ) -> o3d.geometry.PointCloud | None:
        """
        :return: The points of the hypothesis, or else the cloud of its first
            :class:`CloudAnnotation`, or *None* if it has neither.
        """
        if getattr(hypothesis, "points", None) is not None:
            return hypothesis.points
        clouds = self.get_cas().filter_by_type(CloudAnnotation, hypothesis.annotations)
        return clouds[0].points if clouds else None

    def _append_cloud(
        self,
        geometries: list[o3d.geometry.Geometry],
        cloud: o3d.geometry.PointCloud | None,
        color: list[float],
    ) -> None:
        """
        Add a copy of the cloud painted in the color to the geometries, unless the
        cloud is missing or empty.
        """
        if cloud is None or not cloud.has_points():
            return
        painted = o3d.geometry.PointCloud()
        painted.points = o3d.utility.Vector3dVector(np.asarray(cloud.points))
        painted.paint_uniform_color(color)
        geometries.append(painted)
