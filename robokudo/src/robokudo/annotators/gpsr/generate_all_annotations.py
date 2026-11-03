from __future__ import annotations

# %% Imports
from geometry_msgs.msg import Vector3
from py_trees.common import Status
from robokudo_msgs.action import Query
from robokudo_msgs.msg import ObjectDesignator, ShapeSize

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefEntry
from robokudo.annotators.gpsr.gpsr_results import camera_header
from robokudo.cas import CASViews
from robokudo.types.annotation import (
    Classification,
    LocationAnnotation,
    PoseAnnotation,
    PositionAnnotation,
    SemanticColor,
)
from robokudo.types.cv import BoundingBox3D
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world
from robokudo.utils.query import QueryHandler
from robokudo.utils.type_conversion import (
    get_geometry_msgs_pose_stamped_from_pose_annotation,
)

# %% object answer


class GenerateAllQueryResult(BaseAnnotator):
    """
    Answers the query with one object designator per object hypothesis, built from the
    annotations of the hypothesis.

    With a room in the query, only the hypotheses marked with that room are answered.
    Classifications of the CLIP annotator become attributes and the last other
    classification becomes the type; semantic colors, the 3D bounding box, the belief id
    and every pose and position are taken over, poses and positions in the world frame
    ``map``.
    """

    def __init__(self, name: str = "GenerateQueryResult") -> None:
        super().__init__(name=name)

    def _append_pose(
        self, object_designator: ObjectDesignator, pose: PoseAnnotation
    ) -> None:
        """
        Append a camera-frame pose in the world frame to an answer.
        """
        world_pose = transform_pose_from_camera_to_world(self.get_cas(), pose)
        header = camera_header(self.get_cas())
        header.frame_id = "map"
        pose_stamped = get_geometry_msgs_pose_stamped_from_pose_annotation(
            world_pose, header
        )
        object_designator.pose.append(pose_stamped)
        object_designator.pose_source.append(pose.source)

    def update(self) -> Status:

        query = self.get_cas().get(CASViews.QUERY)
        queried_location = query.obj.location
        if queried_location == "":
            filter_by_region = False
        else:
            filter_by_region = True

        annotations = self.get_cas().annotations
        object_hypotheses_count = 0
        query_result = []
        result = Query.Result()

        for hypothesis in annotations:
            if not isinstance(hypothesis, ObjectHypothesis):
                continue

            object_hypotheses_count += 1
            in_queried_room = False
            if filter_by_region:
                for annotation in hypothesis.annotations:
                    if isinstance(annotation, LocationAnnotation):
                        if annotation.name == queried_location:
                            in_queried_room = True

            object_designator = ObjectDesignator()
            belief_ids = [
                annotation
                for annotation in hypothesis.annotations
                if isinstance(annotation, BeliefEntry)
            ]
            if belief_ids:
                object_designator.uid = str(belief_ids[0].id)
            if filter_by_region == in_queried_room:
                for annotation in hypothesis.annotations:
                    # TODO Rewrite with a generic conversion module for each of the types

                    if isinstance(annotation, Classification):
                        if annotation.source == "CLIPAnnotator":
                            object_designator.attribute.append(annotation.classname)
                        else:
                            object_designator.type = annotation.classname
                    if isinstance(annotation, SemanticColor):
                        object_designator.color.append(annotation.color)

                    if isinstance(annotation, BoundingBox3D):
                        object_designator.size = f"x: {annotation.x_length}, y: {annotation.y_length}, z: {annotation.z_length}"
                        shape_size = ShapeSize()
                        dimensions = Vector3()
                        dimensions.x = annotation.x_length
                        dimensions.y = annotation.y_length
                        dimensions.z = annotation.z_length
                        shape_size.dimensions = dimensions
                        shape_size.radius = 0
                        object_designator.shape_size.append(shape_size)

                    if isinstance(annotation, PoseAnnotation):
                        self._append_pose(object_designator, annotation)

                    if isinstance(annotation, PositionAnnotation):
                        pose = PoseAnnotation()
                        pose.source = annotation.source
                        pose.translation.insert(0, annotation.translation[0])
                        pose.translation.insert(1, annotation.translation[1])
                        pose.translation.insert(2, annotation.translation[2])

                        pose.rotation.insert(0, 0)
                        pose.rotation.insert(1, 0)
                        pose.rotation.insert(2, 0)
                        pose.rotation.insert(3, 1)

                        self._append_pose(object_designator, pose)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)
        self.feedback_message = (
            f"Send result for {object_hypotheses_count} object hypotheses"
        )
        return Status.SUCCESS
