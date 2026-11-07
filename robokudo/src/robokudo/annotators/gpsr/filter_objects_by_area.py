"""
Removal of hypotheses outside of a configured area.
"""

from __future__ import annotations

# %% Imports
import py_trees
from robokudo.annotators.gpsr.exceptions import InvalidAreaPose
from robokudo.types.annotation import PoseAnnotation, PositionAnnotation
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world

import robokudo.types.scene
import robokudo.annotators.core


# %% area filter
class FilterObjectsByArea(robokudo.annotators.core.BaseAnnotator):
    """
    Removes every hypothesis with a pose or position outside of the configured area.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """
        Configuration of the area filter.
        """

        class Parameters:
            """
            Bounds of the area.
            """

            def __init__(self) -> None:
                """
                Start with the default area in the map frame.
                """
                self.minimum_x: float = -1.0
                """Lower bound of the area along x, in meters."""
                self.minimum_y: float = -3.0
                """Lower bound of the area along y, in meters."""
                self.minimum_z: float = -float("inf")
                """Lower bound of the area along z, in meters."""
                self.maximum_x: float = 11.0
                """Upper bound of the area along x, in meters."""
                self.maximum_y: float = 10.0
                """Upper bound of the area along y, in meters."""
                self.maximum_z: float = float("inf")
                """Upper bound of the area along z, in meters."""
                self.frame: str = "map"
                """Frame of the bounds, poses are transformed to the world for ``map``."""

        parameters = Parameters()
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self,
        name: str = "FilterObjectsByArea",
        descriptor: FilterObjectsByArea.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.get_class_name())

    def update(self) -> py_trees.common.Status:
        """
        Overrides :meth:`BaseAnnotator.update` to remove each hypothesis with a pose or
        position outside of the area.

        :return: :attr:`Status.SUCCESS`.
        """
        cas = self.get_cas()
        hypotheses = cas.filter_annotations_by_type(
            robokudo.types.scene.ObjectHypothesis
        )

        for hypothesis in hypotheses:
            for annotation in hypothesis.annotations:
                if not isinstance(annotation, (PoseAnnotation, PositionAnnotation)):
                    continue
                pose_map = self._as_pose(annotation)

                if self.descriptor.parameters.frame == "map":
                    pose_map = transform_pose_from_camera_to_world(cas, pose_map)

                if not self._in_area(pose_map):
                    cas.annotations.remove(hypothesis)
                    break
        return py_trees.common.Status.SUCCESS

    def _in_area(self, pose: PoseAnnotation) -> bool:
        """
        :param pose: The pose to check.
        :return: Whether the pose lies inside of the area.
        """
        parameters = self.descriptor.parameters
        x, y, z = pose.translation[:3]
        return not (
            parameters.minimum_x > x
            or x > parameters.maximum_x
            or parameters.minimum_y > y
            or y > parameters.maximum_y
            or parameters.minimum_z > z
            or z > parameters.maximum_z
        )

    def _as_pose(self, pose: object) -> PoseAnnotation:
        """
        :param pose: A pose or position annotation.
        :return: The pose, or a pose at the given position.
        :raises InvalidAreaPose: If the annotation is neither a pose nor a position.
        """
        if isinstance(pose, PoseAnnotation):
            return pose
        if not isinstance(pose, PositionAnnotation):
            raise InvalidAreaPose(type(pose))

        return PoseAnnotation(
            translation=list(pose.translation[:3]), rotation=[0.0, 0.0, 0.0, 1.0]
        )
