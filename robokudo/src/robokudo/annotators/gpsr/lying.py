"""
Keypoint rule for the lying posture.
"""

from __future__ import annotations

# %% Imports
import copy
from timeit import default_timer

from krrood.ormatic.utils import classproperty
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

# %% lying rule


class HumanLyingAnnotator(BaseAnnotator):
    """
    Annotates every person whose image keypoints show a lying posture.

    The rule needs reliable hips, knees and ankles on both sides. The keypoints have to
    be clearly wider than tall, with one knee about level with its hip and nose and
    shoulders about level with each other.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the lying rule.
        """

        class Parameters:
            """
            Thresholds of the lying rule.
            """

            def __init__(self) -> None:
                """
                Start with the default thresholds of the lying rule.
                """
                self.raise_on_failure: bool = False
                """
                Whether a failure raises an error.
                """

                self.keypoint_confidence_threshold: float = 0.5
                """
                Smallest confidence of a keypoint for the rule to trust it.
                """

                # TODO: Adjust lying thresholds.
                self.maximum_relative_height: float = 0.35
                """
                Largest share of the vertical extent of the keypoints in the sum of
                their vertical and horizontal extent for which the person lies.
                """
                self.minimum_relative_width: float = 0.4
                """
                Smallest share of the horizontal extent of the keypoints in the sum of
                their vertical and horizontal extent for which the person lies.
                """
                self.maximum_knee_hip_height_difference: float = 0.2
                """
                Largest vertical distance between knee and hip of one side, relative to
                the vertical extent of the keypoints, for which the person lies. One
                side is enough.
                """
                self.maximum_shoulder_height_difference: float = 0.35
                """
                Largest vertical distance between the two shoulders, relative to the
                vertical extent of the keypoints, for which the person lies.
                """
                self.maximum_nose_shoulder_height_difference: float = 0.4
                """
                Largest vertical distance between the nose and the middle of the
                shoulders, relative to the vertical extent of the keypoints, for which
                the person lies.
                """
                self.minimum_body_height: float = 20
                """
                Smallest vertical extent of the keypoints in the image, in pixels, for
                which the posture is decided.
                """

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self,
        name: str = "HumanLyingAnnotator",
        descriptor: HumanLyingAnnotator.Descriptor | None = None,
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

    @classproperty
    def posture(cls) -> str:
        """
        :return: The activity this rule annotates a lying person with.
        """
        return "lying"

    @classproperty
    def query_word(cls) -> str:
        """
        :return: The word GPSR queries use for a lying person.
        """
        return HumanPosture.LAYING.value

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to annotate every person whose image
        keypoints show a lying posture.

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

            if self.keypoints_suggest_lying(image_keypoints):
                activity = ActivityAnnotation()
                activity.type = HumanLyingAnnotator.posture
                activity.source = self.get_class_name()
                human_hypothesis.annotations.append(activity)

        self.get_annotator_output_struct().set_geometries(visualization_geometries)
        self.get_annotator_output_struct().set_image(copy.deepcopy(self.color))

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def keypoints_suggest_lying(self, image_keypoints: KeypointAnnotation) -> bool:
        """
        :param image_keypoints: The image keypoints of one person.
        :return: Whether the keypoints show a lying person.
        """
        parameters = self.descriptor.parameters
        confidence_threshold = parameters.keypoint_confidence_threshold

        left_hip = image_keypoints.keypoints[PoseKeypoint.LEFT_HIP]
        left_knee = image_keypoints.keypoints[PoseKeypoint.LEFT_KNEE]
        left_ankle = image_keypoints.keypoints[PoseKeypoint.LEFT_ANKLE]

        right_hip = image_keypoints.keypoints[PoseKeypoint.RIGHT_HIP]
        right_knee = image_keypoints.keypoints[PoseKeypoint.RIGHT_KNEE]
        right_ankle = image_keypoints.keypoints[PoseKeypoint.RIGHT_ANKLE]

        left_shoulder = image_keypoints.keypoints[PoseKeypoint.LEFT_SHOULDER]
        right_shoulder = image_keypoints.keypoints[PoseKeypoint.RIGHT_SHOULDER]
        nose = image_keypoints.keypoints[PoseKeypoint.NOSE]

        lower_body_reliable = (
            min(
                left_hip.confidence,
                left_knee.confidence,
                left_ankle.confidence,
                right_hip.confidence,
                right_knee.confidence,
                right_ankle.confidence,
            )
            >= confidence_threshold
        )

        if not lower_body_reliable:
            return False

        image_columns = [
            left_hip.x,
            left_knee.x,
            left_ankle.x,
            right_hip.x,
            right_knee.x,
            right_ankle.x,
            left_shoulder.x,
            right_shoulder.x,
            nose.x,
        ]

        image_rows = [
            left_hip.y,
            left_knee.y,
            left_ankle.y,
            right_hip.y,
            right_knee.y,
            right_ankle.y,
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

        left_knee_hip_height = abs(left_knee.y - left_hip.y) / vertical_extent
        right_knee_hip_height = abs(right_knee.y - right_hip.y) / vertical_extent

        shoulder_height_difference = (
            abs(left_shoulder.y - right_shoulder.y) / vertical_extent
        )

        nose_shoulder_height_difference = (
            abs(nose.y - (left_shoulder.y + right_shoulder.y) / 2) / vertical_extent
        )

        lying = (
            relative_height <= parameters.maximum_relative_height
            and relative_width >= parameters.minimum_relative_width
            and (
                left_knee_hip_height <= parameters.maximum_knee_hip_height_difference
                or right_knee_hip_height
                <= parameters.maximum_knee_hip_height_difference
            )
            and shoulder_height_difference
            <= parameters.maximum_shoulder_height_difference
            and nose_shoulder_height_difference
            <= parameters.maximum_nose_shoulder_height_difference
        )

        return lying
