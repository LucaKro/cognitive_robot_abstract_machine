from __future__ import annotations

# %% Imports
import copy
from timeit import default_timer

from krrood.ormatic.utils import classproperty
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
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
    Annotates every person whose image keypoints show a lying posture with the activity
    :attr:`posture`.

    The rule needs reliable hips, knees and ankles on both sides. The person lies when
    the keypoints are clearly wider than tall, the knee of at least one side is about
    level with its hip, the shoulders are about level with each other and the nose is
    about level with the shoulders, all relative to the vertical extent of the
    keypoints. Persons without image keypoints are skipped.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the lying rule.
        """

        class Parameters:
            """
            Parameters of the lying rule.
            """

            def __init__(self) -> None:
                self.raise_on_failure: bool = False
                """
                Not read by this annotator.
                """

                self.keypoint_confidence_threshold: float = 0.5
                """
                Smallest confidence of a keypoint for the rule to trust it.
                """

                # TODO: Adjust lying thresholds.
                self.max_vertical_rel_height: float = 0.35
                """
                Largest share of the vertical extent of the keypoints in the sum of
                their vertical and horizontal extent for which the person lies.
                """
                self.min_horizontal_rel_width: float = 0.4
                """
                Smallest share of the horizontal extent of the keypoints in the sum of
                their vertical and horizontal extent for which the person lies.
                """
                self.max_knee_hip_vertical_diff: float = 0.2
                """
                Largest vertical distance between the knee and the hip of one side,
                relative to the vertical extent of the keypoints, for which the person
                lies; one side is enough.
                """
                self.max_shoulder_height_difference: float = 0.35
                """
                Largest vertical distance between the two shoulders, relative to the
                vertical extent of the keypoints, for which the person lies.
                """
                self.max_head_shoulder_height_difference: float = 0.4
                """
                Largest vertical distance between the nose and the middle of the
                shoulders, relative to the vertical extent of the keypoints, for which
                the person lies.
                """
                self.min_body_height: float = 20
                """
                Smallest vertical extent of the keypoints in the image, in pixels, for
                which the posture is decided.
                """

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion
        """
        Default parameters on the class; each descriptor instance gets its own.
        """

    def __init__(
        self,
        name: str = "HumanLyingAnnotator",
        descriptor: HumanLyingAnnotator.Descriptor | None = None,
    ) -> None:
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
        The activity this rule annotates a lying person with.
        """
        return "lying"

    @classproperty
    def query_word(cls) -> str:
        """
        The word GPSR queries use for a lying person.
        """
        return "laying"

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
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

            if self.keypoints_lying(image_keypoints):
                activity = ActivityAnnotation()
                activity.type = HumanLyingAnnotator.posture
                activity.source = type(self).__name__
                human_hypothesis.annotations.append(activity)

        self.get_annotator_output_struct().set_geometries(visualization_geometries)
        self.get_annotator_output_struct().set_image(copy.deepcopy(self.color))

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def keypoints_lying(self, keypoint_2d_annotation: KeypointAnnotation) -> bool:
        """
        :param keypoint_2d_annotation: The image keypoints of one person.
        :return: Whether the keypoints show a lying person.
        """
        parameters = self.descriptor.parameters
        confidence_threshold = parameters.keypoint_confidence_threshold

        left_hip = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_HIP]
        left_knee = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_KNEE]
        left_ankle = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_ANKLE]

        right_hip = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_HIP]
        right_knee = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_KNEE]
        right_ankle = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_ANKLE]

        left_shoulder = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_SHOULDER]
        right_shoulder = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_SHOULDER]
        nose = keypoint_2d_annotation.keypoints[PoseKeypoint.NOSE]

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

        if vertical_extent < parameters.min_body_height:
            return False

        relative_height = vertical_extent / (vertical_extent + horizontal_extent)
        relative_width = horizontal_extent / (vertical_extent + horizontal_extent)

        left_knee_hip_height = abs(left_knee.y - left_hip.y) / vertical_extent
        right_knee_hip_height = abs(right_knee.y - right_hip.y) / vertical_extent

        shoulder_height_difference = (
            abs(left_shoulder.y - right_shoulder.y) / vertical_extent
        )

        head_shoulder_height_difference = (
            abs(nose.y - (left_shoulder.y + right_shoulder.y) / 2) / vertical_extent
        )

        # TODO: Adjust thresholds for lying detection.
        lying = (
            relative_height <= parameters.max_vertical_rel_height
            and relative_width >= parameters.min_horizontal_rel_width
            and (
                left_knee_hip_height <= parameters.max_knee_hip_vertical_diff
                or right_knee_hip_height <= parameters.max_knee_hip_vertical_diff
            )
            and shoulder_height_difference <= parameters.max_shoulder_height_difference
            and head_shoulder_height_difference
            <= parameters.max_head_shoulder_height_difference
        )

        return lying
