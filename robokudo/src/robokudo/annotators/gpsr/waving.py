"""
Keypoint rules for raised arms and waving hands.
"""

from __future__ import annotations

# %% Imports
import copy
from timeit import default_timer

from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.gpsr.human_activity_annotator import (
    HumanActivityAnnotator,
    HumanGesture,
)
from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.cas import CAS, CASViews
from robokudo.types.human import ActivityAnnotation, KeypointAnnotation, KeypointKind
from robokudo.types.scene import HumanHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    import numpy as np
    import open3d as o3d

# %% waving and arm raising rule


class HumanWavingAnnotator(BaseAnnotator):
    """
    Annotates every person whose image keypoints show a raised arm or a waving hand.

    A trusted wrist above shoulder level raises the arm and one above eye level waves,
    both by a margin that grows with the distance between eye and shoulder. The left wrist
    of the person is checked first.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the waving and arm raising rule.
        """

        class Parameters:
            """
            Thresholds of the waving and arm raising rule.
            """

            def __init__(self) -> None:
                """
                Start with the default thresholds of the raising and waving rules.
                """
                self.raise_on_failure: bool = False
                """
                Whether a failure raises an error.
                """
                self.keypoint_confidence_threshold: float = 0.7
                """
                Confidence a wrist keypoint has to exceed for the rule to trust it.
                """
                self.check_activity: bool = False
                """
                Whether only the persons the CLIP activity annotator found without
                gesture are decided.
                """

                self.waving_decision_factor: float = 0.2
                """
                Share of the vertical distance between eye and shoulder by which a wrist
                has to be above eye level to wave, or above shoulder level to raise the
                arm.
                """

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self,
        name: str = "HumanWavingAnnotator",
        descriptor: HumanWavingAnnotator.Descriptor | None = None,
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
        keypoints show a raised arm or a waving hand.

        :return: :attr:`Status.SUCCESS`.
        """
        start_timer = default_timer()

        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        self.cloud = self.get_cas().get(CASViews.CLOUD)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)

        if self.descriptor.parameters.check_activity:
            human_hypotheses = [
                human_hypothesis
                for human_hypothesis in human_hypotheses
                if any(
                    isinstance(annotation, ActivityAnnotation)
                    and annotation.source == HumanActivityAnnotator.__name__
                    and annotation.type == HumanGesture.NONE
                    for annotation in human_hypothesis.annotations
                )
            ]
            if not human_hypotheses:
                self.feedback_message = (
                    f"No '{HumanGesture.NONE}' human hypothesis found, skipping"
                )
                return Status.SUCCESS

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

            raised_side = self.keypoints_suggest_arm_raising(image_keypoints)
            if raised_side == "left":
                activity = ActivityAnnotation()
                activity.type = "raising right arm"
                activity.source = self.get_class_name()
                human_hypothesis.annotations.append(activity)
            elif raised_side == "right":
                activity = ActivityAnnotation()
                activity.type = "raising left arm"
                activity.source = self.get_class_name()
                human_hypothesis.annotations.append(activity)
            is_waving = self.keypoints_suggest_waving(image_keypoints)
            if is_waving:
                activity = ActivityAnnotation()
                activity.type = "waving"
                activity.source = self.get_class_name()
                human_hypothesis.annotations.append(activity)

        self.get_annotator_output_struct().set_geometries(visualization_geometries)
        self.get_annotator_output_struct().set_image(copy.deepcopy(self.color))

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def keypoints_suggest_arm_raising(
        self, image_keypoints: KeypointAnnotation
    ) -> str | None:
        """
        :param image_keypoints: The image keypoints of one person.
        :return: The image side of the raised wrist of a person facing the camera.
            ``right`` stands for their left wrist, which is checked first. ``left``
            stands for their right wrist. *None* if neither wrist is raised.
        """
        left_wrist = image_keypoints.keypoints[PoseKeypoint.LEFT_WRIST]
        right_wrist = image_keypoints.keypoints[PoseKeypoint.RIGHT_WRIST]
        left_shoulder = image_keypoints.keypoints[PoseKeypoint.LEFT_SHOULDER]
        right_shoulder = image_keypoints.keypoints[PoseKeypoint.RIGHT_SHOULDER]
        left_eye = image_keypoints.keypoints[PoseKeypoint.LEFT_EYE]
        right_eye = image_keypoints.keypoints[PoseKeypoint.RIGHT_EYE]

        left_eye_shoulder_distance_pixel = left_shoulder.y - left_eye.y
        right_eye_shoulder_distance_pixel = right_shoulder.y - right_eye.y

        left_wrist_margin_pixel = (
            left_eye_shoulder_distance_pixel
            * self.descriptor.parameters.waving_decision_factor
        )
        right_wrist_margin_pixel = (
            right_eye_shoulder_distance_pixel
            * self.descriptor.parameters.waving_decision_factor
        )

        if (
            left_wrist.y < (left_shoulder.y - left_wrist_margin_pixel)
            and left_wrist.confidence
            > self.descriptor.parameters.keypoint_confidence_threshold
        ):
            return "right"
        if (
            right_wrist.y < (right_shoulder.y - right_wrist_margin_pixel)
            and right_wrist.confidence
            > self.descriptor.parameters.keypoint_confidence_threshold
        ):
            return "left"
        return None

    def keypoints_suggest_waving(self, image_keypoints: KeypointAnnotation) -> bool:
        """
        :param image_keypoints: The image keypoints of one person.
        :return: Whether a trusted wrist of the person is above eye level by the
            margin.
        """
        left_wrist = image_keypoints.keypoints[PoseKeypoint.LEFT_WRIST]
        right_wrist = image_keypoints.keypoints[PoseKeypoint.RIGHT_WRIST]
        left_shoulder = image_keypoints.keypoints[PoseKeypoint.LEFT_SHOULDER]
        right_shoulder = image_keypoints.keypoints[PoseKeypoint.RIGHT_SHOULDER]
        left_eye = image_keypoints.keypoints[PoseKeypoint.LEFT_EYE]
        right_eye = image_keypoints.keypoints[PoseKeypoint.RIGHT_EYE]

        # 2D coordinates need a size estimate for thresholds at the pixel level. The
        # distance between shoulder and eyes on each side is a rough one.
        left_eye_shoulder_distance_pixel = left_shoulder.y - left_eye.y
        right_eye_shoulder_distance_pixel = right_shoulder.y - right_eye.y

        left_wrist_margin_pixel = (
            left_eye_shoulder_distance_pixel
            * self.descriptor.parameters.waving_decision_factor
        )
        right_wrist_margin_pixel = (
            right_eye_shoulder_distance_pixel
            * self.descriptor.parameters.waving_decision_factor
        )

        # y coordinates start with 0 from the top of the image
        if (
            left_wrist.y < (left_eye.y - left_wrist_margin_pixel)
            and left_wrist.confidence
            > self.descriptor.parameters.keypoint_confidence_threshold
        ):
            return True
        if (
            right_wrist.y < (right_eye.y - right_wrist_margin_pixel)
            and right_wrist.confidence
            > self.descriptor.parameters.keypoint_confidence_threshold
        ):
            return True

        return False
