"""
Selection of the persons near a piece of the queried furniture.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from krrood_spatial_math.spatial_types import Point3
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.gpsr.furniture_direction import FurnitureDirection
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

# %% person in the visualization


@dataclass
class HumanVisualization:
    """
    A person as the furniture filter draws them, in the camera frame.
    """

    camera_position: Point3
    """
    Position of the person in the camera frame.
    """
    is_kept: bool
    """
    Whether the person stands near the furniture and stays in the CAS.
    """
    cloud: o3d.geometry.PointCloud | None
    """
    Cluster of the person, if they have one.
    """


# %% furniture filter


class FilterHumansByObjectAnnotator(BaseAnnotator):
    """
    Keeps the persons near the queried furniture, marked with a
    :class:`LocationAnnotation` naming it.

    A person is near inside a box around the furniture, or next to it on the queried
    side. Without a furniture type nobody is removed, without furniture every person is.
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
                """
                Start with the default search box around the furniture and no queried furniture.
                """
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
                Height of the search box, centered on the furniture, in meters.
                """
                self.direction: str | None = None
                """
                Side of the furniture the search box is placed on, as seen from the
                robot. One of ``left``, ``right``, ``front`` or ``back``. *None*
                searches around the furniture.
                """
                self.target_type: str | None = None
                """
                Name of the furniture the persons are searched near. It is written into
                their location. Without a name, nobody is removed.
                """

        parameters = Parameters()
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self,
        name: str = "FilterHumansByObjectAnnotator",
        descriptor: FilterHumansByObjectAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        """
        super().__init__(name, descriptor)

    def is_inside_box(
        self,
        human_position: Point3,
        furniture_position: Point3,
        box_size: list[float],
        direction: str | None = None,
        right: np.ndarray | None = None,
        forward: np.ndarray | None = None,
    ) -> bool | np.bool_:
        """
        :param human_position: Position of the person in the world.
        :param furniture_position: Position of the furniture in the world.
        :param box_size: Extent of the search box along its three axes.
        :param direction: Side of the furniture the box is placed on, or *None* for a
            box around the furniture.
        :param right: The robot's right on the floor, needed with a direction.
        :param forward: The robot's viewing direction on the floor, needed with a
            direction. Positive offsets along it lie behind the furniture.
        :return: Whether the person stands in the search box. An unknown direction
            searches around the furniture.
        """
        offset = (human_position - furniture_position).to_np()[:3]
        if not direction:
            return np.all(np.abs(offset) <= np.array(box_size) / 2.0)

        box_x, box_y, box_z = box_size
        if abs(offset[2]) > box_z / 2.0:
            return False
        lateral = float(np.dot(offset, right))
        longitudinal = float(np.dot(offset, forward))
        if direction == FurnitureDirection.RIGHT.value:
            return 0.0 <= lateral <= box_x and abs(longitudinal) <= box_y / 2.0
        if direction == FurnitureDirection.LEFT.value:
            return -box_x <= lateral <= 0.0 and abs(longitudinal) <= box_y / 2.0
        if direction == FurnitureDirection.BACK.value:
            return 0.0 <= longitudinal <= box_y and abs(lateral) <= box_x / 2.0
        if direction == FurnitureDirection.FRONT.value:
            return -box_y <= longitudinal <= 0.0 and abs(lateral) <= box_x / 2.0
        return np.all(np.abs(offset) <= np.array(box_size) / 2.0)

    def _robot_ground_basis(
        self, cas: CAS
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        :param cas: The CAS whose camera pose is used.
        :return: The unit vectors along the camera's right and along its viewing
            direction, projected onto the floor of the world. Each is *None* when its
            axis is (nearly) vertical.
        """

        def to_world(point: Point3) -> np.ndarray:
            """
            :param point: The point in the camera frame.
            :return: The camera frame point in the world.
            """
            pose = PoseAnnotation()
            pose.translation = point.to_np()[:3].tolist()
            pose.rotation = [0.0, 0.0, 0.0, 1.0]
            return np.array(transform_pose_from_camera_to_world(cas, pose).translation)

        origin = to_world(Point3(0.0, 0.0, 0.0))
        right = to_world(Point3(1.0, 0.0, 0.0)) - origin  # camera image-right axis
        forward = (
            to_world(Point3(0.0, 0.0, 1.0)) - origin
        )  # camera optical (forward) axis

        def project_ground(vector: np.ndarray) -> np.ndarray | None:
            """
            :param vector: The vector in the world.
            :return: The vector projected onto the floor and normalized, or *None* if it
                is (nearly) vertical.
            """
            flat = np.array([vector[0], vector[1], 0.0])
            norm = np.linalg.norm(flat)
            return flat / norm if norm > 1e-6 else None

        return project_ground(right), project_ground(forward)

    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to remove the persons away from the
        queried furniture and mark the others.

        :return: :attr:`Status.SUCCESS`.
        """
        cas = self.get_cas()
        parameters = self.descriptor.parameters
        target_type = parameters.target_type
        box_size = [parameters.box_x, parameters.box_y, parameters.box_z]
        direction = parameters.direction
        furniture_world_positions = []
        furniture_camera_positions = []
        furniture_clouds = []
        human_visualizations = []
        kept_count = 0

        if not target_type:
            self.rk_logger.warning("No target type set")
            return Status.SUCCESS

        # the detector only reports furniture of the given type
        all_hypotheses = cas.filter_annotations_by_type(ObjectHypothesis)
        human_hypotheses = [
            hypothesis
            for hypothesis in all_hypotheses
            if isinstance(hypothesis, HumanHypothesis)
        ]
        furniture_hypotheses = [
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

        for furniture_hypothesis in furniture_hypotheses:
            positions = cas.filter_by_type(
                PositionAnnotation, furniture_hypothesis.annotations
            )
            if not positions:
                continue
            pose = PoseAnnotation()
            pose.translation = positions[0].translation
            pose.rotation = [0.0, 0.0, 0.0, 1.0]
            furniture_camera_positions.append(Point3.from_iterable(pose.translation))
            furniture_clouds.append(self._get_cluster_cloud(furniture_hypothesis))
            furniture_world_positions.append(
                Point3.from_iterable(
                    transform_pose_from_camera_to_world(cas, pose).translation
                )
            )

        if not furniture_world_positions:
            self.rk_logger.info(
                f"No '{target_type}' detected, removing all {len(human_hypotheses)} humans"
            )
            for human_hypothesis in human_hypotheses:
                cas.annotations.remove(human_hypothesis)
            return Status.SUCCESS

        for human_hypothesis in human_hypotheses:
            positions = cas.filter_by_type(
                PositionAnnotation, human_hypothesis.annotations
            )
            if not positions:
                cas.annotations.remove(human_hypothesis)
                continue

            pose = PoseAnnotation()
            pose.translation = positions[0].translation
            pose.rotation = [0.0, 0.0, 0.0, 1.0]
            world_position = Point3.from_iterable(
                transform_pose_from_camera_to_world(cas, pose).translation
            )
            camera_position = Point3.from_iterable(pose.translation)

            near_furniture = False
            for furniture_position in furniture_world_positions:
                if self.is_inside_box(
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
                human_hypothesis.annotations.append(location)
                kept_count += 1
                human_visualizations.append(
                    HumanVisualization(
                        camera_position=camera_position,
                        is_kept=True,
                        cloud=self._get_cluster_cloud(human_hypothesis),
                    )
                )
            else:
                cas.annotations.remove(human_hypothesis)
                human_visualizations.append(
                    HumanVisualization(
                        camera_position=camera_position,
                        is_kept=False,
                        cloud=self._get_cluster_cloud(human_hypothesis),
                    )
                )

        side = direction if direction else "any side"
        self.rk_logger.info(
            f"Kept {kept_count}/{len(human_hypotheses)} humans near '{target_type}' ({side})"
        )
        self._visualize(
            furniture_camera_positions,
            furniture_clouds,
            human_visualizations,
            box_size,
            direction,
        )
        return Status.SUCCESS

    def _visualize(
        self,
        furniture_camera_positions: list[Point3],
        furniture_clouds: list[o3d.geometry.PointCloud | None],
        human_visualizations: list[HumanVisualization],
        box_size: list[float],
        direction: str | None,
    ) -> None:
        """
        Show the search box of every piece of furniture in green, the furniture with its
        cluster in orange and every person with their cluster in blue when kept and in
        red when removed, all in the camera frame.

        :param furniture_camera_positions: Position of every piece of furniture.
        :param furniture_clouds: Cluster of every piece of furniture, if it has one.
        :param human_visualizations: Every person with a position.
        :param box_size: Extent of the search box along its three axes.
        :param direction: Side of the furniture the search box is placed on, or *None*.
        """
        box_x, box_y, box_z = box_size
        geometries = []
        for camera_position, furniture_cloud in zip(
            furniture_camera_positions, furniture_clouds
        ):
            center = camera_position.to_np()[:3]
            if direction:
                # camera axes: x=right (box_x), y=down (box_z), z=forward (box_y)
                half_size = np.array([box_x / 2.0, box_z / 2.0, box_y / 2.0])
                minimum_corner, maximum_corner = center - half_size, center + half_size
                if direction == FurnitureDirection.RIGHT.value:
                    minimum_corner[0], maximum_corner[0] = center[0], center[0] + box_x
                elif direction == FurnitureDirection.LEFT.value:
                    minimum_corner[0], maximum_corner[0] = center[0] - box_x, center[0]
                elif direction == FurnitureDirection.BACK.value:
                    minimum_corner[2], maximum_corner[2] = center[2], center[2] + box_y
                elif direction == FurnitureDirection.FRONT.value:
                    minimum_corner[2], maximum_corner[2] = center[2] - box_y, center[2]
            else:
                half_size = np.array(box_size) / 2.0
                minimum_corner, maximum_corner = center - half_size, center + half_size
            search_box = o3d.geometry.AxisAlignedBoundingBox(
                minimum_corner, maximum_corner
            )
            search_box.color = [0.0, 1.0, 0.0]
            geometries.append(search_box)
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            marker.translate(center)
            marker.paint_uniform_color([1.0, 0.6, 0.0])
            geometries.append(marker)
            self._append_cloud(geometries, furniture_cloud, [1.0, 0.6, 0.0])
        for human in human_visualizations:
            if human.camera_position is None:
                continue
            color = [0.0, 0.5, 1.0] if human.is_kept else [1.0, 0.2, 0.2]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            sphere.translate(human.camera_position.to_np()[:3])
            sphere.paint_uniform_color(color)
            geometries.append(sphere)
            self._append_cloud(geometries, human.cloud, color)
        self.get_annotator_output_struct().set_geometries(geometries)

    def _get_cluster_cloud(
        self, hypothesis: ObjectHypothesis
    ) -> o3d.geometry.PointCloud | None:
        """
        :param hypothesis: The hypothesis with the cloud.
        :return: The points of the hypothesis, or else the cloud of its first
            :class:`CloudAnnotation`, or *None* if it has neither.
        """
        if hypothesis.points is not None:
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

        :param geometries: The geometries that receive the copy.
        :param cloud: The cloud to paint, if any.
        :param color: The red, green and blue share of the color.
        """
        if cloud is None or not cloud.has_points():
            return
        painted = o3d.geometry.PointCloud()
        painted.points = o3d.utility.Vector3dVector(np.asarray(cloud.points))
        painted.paint_uniform_color(color)
        geometries.append(painted)
