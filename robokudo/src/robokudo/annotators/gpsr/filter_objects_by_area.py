from __future__ import annotations

# %% imports
import py_trees
from robokudo.annotators.gpsr.exceptions import InvalidAreaPose
from robokudo.types.annotation import PoseAnnotation, PositionAnnotation
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world

import robokudo.types.scene
import robokudo.annotators.core
import robokudo.defs


# %% area filter
class FilterObjectsByArea(robokudo.annotators.core.BaseAnnotator):
    """
    This Annotator removes all Annotations that are not in a submitted area.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """
        Descriptor for FilterObjectsByArea.
        """

        class Parameters:
            """
            Parameters for the Descriptor.
            """

            def __init__(self) -> None:
                # Submitted area coordinates
                self.min_x = -1  # -float('inf')
                self.min_y = -3
                self.min_z = -float("inf")
                self.max_x = 11
                self.max_y = 10
                self.max_z = float("inf")

                # In which frame are the coordinates?
                # If \map the coordinates will be transformed first
                self.frame = "map"

        parameters = Parameters()

    def __init__(
        self,
        name: str = "FilterObjectsByArea",
        descriptor: FilterObjectsByArea.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self) -> py_trees.common.Status:
        cas = self.get_cas()
        object_hypothesis = cas.filter_annotations_by_type(
            robokudo.types.scene.ObjectHypothesis
        )

        for hypothesis in object_hypothesis:
            for annotation in hypothesis.annotations:
                if not isinstance(
                    annotation, robokudo.types.annotation.PoseAnnotation
                ) and not isinstance(
                    annotation, robokudo.types.annotation.PositionAnnotation
                ):
                    continue
                # Transform to get unique type
                pose_map = self.__transform(annotation)

                # If submitted ares coordinates are in map coordinates, transform PoseAnnotation to map
                if self.descriptor.parameters.frame == "map":
                    pose_map = transform_pose_from_camera_to_world(
                        self.get_cas(), pose_map
                    )

                self.__transform(pose_map)

                # If position is not in area, remove Hypothesis out of cas.
                if not self.__in_area(pose_map):
                    cas.annotations.remove(hypothesis)
                    break
        return py_trees.common.Status.SUCCESS

    def __in_area(self, pose: robokudo.types.annotation.PoseAnnotation) -> bool:
        """
        Checks if a submitted PoseAnnotation is in a predefined area.
        """
        parameter = self.descriptor.parameters

        # Check x coordinate
        if (
            parameter.min_x > pose.translation[0]
            or pose.translation[0] > parameter.max_x
        ):
            return False
        # Check Y coordinate
        if (
            parameter.min_y > pose.translation[1]
            or pose.translation[1] > parameter.max_y
        ):
            return False
        # Check Z coordinate
        if (
            parameter.min_z > pose.translation[2]
            or pose.translation[2] > parameter.max_z
        ):
            return False
        return True

    def __transform(self, pose: object) -> robokudo.types.annotation.PoseAnnotation:
        """
        Convert a position annotation to a pose, or return an existing pose.
        """
        if isinstance(pose, robokudo.types.annotation.PoseAnnotation):
            return pose
        if not isinstance(pose, robokudo.types.annotation.PositionAnnotation):
            raise InvalidAreaPose(type(pose))

        pos = robokudo.types.annotation.PoseAnnotation()
        pos.translation.insert(0, pose.translation[0])
        pos.translation.insert(1, pose.translation[1])
        pos.translation.insert(2, pose.translation[2])

        pos.rotation.insert(0, 0)
        pos.rotation.insert(1, 0)
        pos.rotation.insert(2, 0)
        pos.rotation.insert(3, 1)
        return pos
