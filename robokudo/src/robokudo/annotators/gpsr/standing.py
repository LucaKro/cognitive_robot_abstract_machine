"""
Keypoint rule for the standing posture.
"""

from __future__ import annotations

# %% Imports
import copy
from timeit import default_timer

from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.gpsr.sitting import HumanPosture
from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.cas import CAS, CASViews
from robokudo.types.human import ActivityAnnotation, KeypointAnnotation, KeypointKind
from robokudo.types.scene import HumanHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    import numpy as np
    import open3d as o3d

# %% standing rule


class HumanStandingAnnotator(BaseAnnotator):
    """
    Annotates every person whose image keypoints show a standing posture.

    The rule needs reliable shoulders, nose and one leg, the left leg if it is reliable.
    The keypoints have to be clearly taller than wide. Nose, shoulders, hip and knee
    have to follow each other from top to bottom at clear distances.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the standing rule.
        """

        class Parameters:
            """
            Thresholds of the standing rule.
            """

            def __init__(self) -> None:
                """
                Start with the default thresholds of the standing rule.
                """
                self.raise_on_failure: bool = False
                """
                Whether a failure raises an error.
                """

                self.keypoint_confidence_threshold: float = 0.5
                """
                Smallest confidence of a keypoint for the rule to trust it.
                """

                self.minimum_relative_height: float = 0.6
                """
                Smallest share of the vertical extent of the keypoints in the sum of
                their vertical and horizontal extent for which the person stands.
                """
                self.maximum_relative_width: float = 0.45
                """
                Largest share of the horizontal extent of the keypoints in the sum of
                their vertical and horizontal extent for which the person stands.
                """
                self.minimum_shoulder_over_hip: float = 0.25
                """
                Smallest height of the shoulders above the hip, relative to the vertical
                extent of the keypoints, for which the person stands.
                """
                self.minimum_hip_over_knee: float = 0.30
                """
                Smallest height of the hip above the knee, relative to the vertical
                extent of the keypoints, for which the person stands.
                """
                self.minimum_nose_over_shoulder: float = 0.10
                """
                Smallest height of the nose above the shoulders, relative to the
                vertical extent of the keypoints, for which the person stands.
                """
                self.minimum_body_height: float = 20
                """
                Smallest vertical extent of the keypoints in the image, in pixels, for
                which the posture is decided.
                """

        parameters = Parameters()
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self,
        name: str = "HumanStandingAnnotator",
        descriptor: HumanStandingAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        """
        super().__init__(name, descriptor)
        self.color: np.ndarray | None = None
        """
        The color image of the last update.
        """
        self.cloud: o3d.geometry.PointCloud | None = None
        """
        The point cloud of the last update.
        """

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to annotate every person whose image
        keypoints show a standing posture.

        :return: :attr:`Status.SUCCESS`.
        """
        start_timer = default_timer()

        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        self.cloud = self.get_cas().get(CASViews.CLOUD)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)

        visualization_geometries = [self.cloud]

        for human_hypothesis in human_hypotheses:
            keypoint_annotations = CAS.filter_by_type(
                KeypointAnnotation, human_hypothesis.annotations
            )
            image_keypoint_annotations = [
                annotation
                for annotation in keypoint_annotations
                if annotation.kind == KeypointKind.IMAGE_POSITION
            ]

            if len(image_keypoint_annotations) == 0:
                continue

            image_keypoints = image_keypoint_annotations[0]

            if self.keypoints_suggest_standing(image_keypoints):
                activity = ActivityAnnotation()
                activity.type = HumanPosture.STANDING.value
                activity.source = self.get_class_name()
                human_hypothesis.annotations.append(activity)

        self.get_annotator_output_struct().set_geometries(visualization_geometries)
        self.get_annotator_output_struct().set_image(copy.deepcopy(self.color))

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def keypoints_suggest_standing(self, image_keypoints: KeypointAnnotation) -> bool:
        """
        :param image_keypoints: The image keypoints of one person.
        :return: Whether the keypoints show a standing person.
        """
        parameters = self.descriptor.parameters
        confidence_threshold = parameters.keypoint_confidence_threshold

        left_hip = image_keypoints.keypoints[PoseKeypoint.LEFT_HIP]
        right_hip = image_keypoints.keypoints[PoseKeypoint.RIGHT_HIP]
        left_knee = image_keypoints.keypoints[PoseKeypoint.LEFT_KNEE]
        right_knee = image_keypoints.keypoints[PoseKeypoint.RIGHT_KNEE]
        left_shoulder = image_keypoints.keypoints[PoseKeypoint.LEFT_SHOULDER]
        right_shoulder = image_keypoints.keypoints[PoseKeypoint.RIGHT_SHOULDER]
        nose = image_keypoints.keypoints[PoseKeypoint.NOSE]

        upper_body_reliable = (
            min(left_shoulder.confidence, right_shoulder.confidence, nose.confidence)
            >= confidence_threshold
        )

        left_leg_reliable = (
            min(left_hip.confidence, left_knee.confidence) >= confidence_threshold
        )

        right_leg_reliable = (
            min(right_hip.confidence, right_knee.confidence) >= confidence_threshold
        )

        if not upper_body_reliable or not (left_leg_reliable or right_leg_reliable):
            return False

        image_columns = [
            left_hip.x,
            right_hip.x,
            left_knee.x,
            right_knee.x,
            left_shoulder.x,
            right_shoulder.x,
            nose.x,
        ]

        image_rows = [
            left_hip.y,
            right_hip.y,
            left_knee.y,
            right_knee.y,
            left_shoulder.y,
            right_shoulder.y,
            nose.y,
        ]
        vertical_extent = max(image_rows) - min(image_rows)
        horizontal_extent = max(image_columns) - min(image_columns)

        if vertical_extent < parameters.minimum_body_height:
            return False

        relative_height = vertical_extent / (vertical_extent + horizontal_extent)
        relative_width = horizontal_extent / (vertical_extent + horizontal_extent)

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        if left_leg_reliable:
            hip_y = left_hip.y
            knee_y = left_knee.y
        elif right_leg_reliable:
            hip_y = right_hip.y
            knee_y = right_knee.y
        else:
            return False  # Shouldn't happen due to earlier check

        shoulder_hip_offset = (shoulder_y - hip_y) / vertical_extent
        hip_knee_offset = (hip_y - knee_y) / vertical_extent
        nose_shoulder_offset = (nose.y - shoulder_y) / vertical_extent

        standing = (
            relative_height >= parameters.minimum_relative_height
            and relative_width <= parameters.maximum_relative_width
            and shoulder_hip_offset <= -parameters.minimum_shoulder_over_hip
            and hip_knee_offset <= -parameters.minimum_hip_over_knee
            and nose_shoulder_offset <= -parameters.minimum_nose_over_shoulder
        )

        return standing
