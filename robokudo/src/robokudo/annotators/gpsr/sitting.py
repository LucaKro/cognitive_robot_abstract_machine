"""
Keypoint rule for the sitting posture.
"""

from __future__ import annotations

# %% Imports
import copy
from enum import StrEnum
from timeit import default_timer

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

    from robokudo.types.human import BodyKeypoint

# %% person postures


class HumanPosture(StrEnum):
    """Postures accepted by person queries."""

    SITTING = "sitting"
    """
    The sitting posture.
    """
    STANDING = "standing"
    """
    The standing posture.
    """
    LAYING = "laying"
    """
    The lying posture, with the word GPSR queries use for it.
    """


# %% sitting rule


class HumanSittingAnnotator(BaseAnnotator):
    """
    Annotates every person whose image keypoints show a sitting posture.

    A body side sits when its keypoints are reliable, knee below hip and ankle below knee,
    with a short hip to knee drop and a long knee to ankle drop relative to the vertical
    extent of the side. One sitting side is enough, and so is :meth:`knee_is_at_hip_height`
    if :attr:`Descriptor.Parameters.knee_at_hip_height_is_sitting` is set.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the sitting rule.
        """

        class Parameters:
            """
            Thresholds of the sitting rule.
            """

            def __init__(self) -> None:
                """
                Start with the default thresholds of the sitting rule.
                """
                self.raise_on_failure: bool = False
                """
                Whether a failure raises an error.
                """

                self.keypoint_confidence_threshold: float = 0.5
                """
                Smallest confidence of a keypoint for the rule to trust it.
                """

                # TODO: Adjust sitting thresholds.
                self.maximum_relative_hip_knee_height: float = 0.5
                """
                Largest vertical distance from hip to knee, relative to the vertical
                extent of the body side, for which the side sits.
                """
                self.maximum_relative_knee_knee_height: float = 0.5
                """
                Largest vertical distance between the two knees, relative to the
                vertical extent of both legs, for which both sides are checked again.
                """
                self.minimum_relative_knee_ankle_height: float = 0.2
                """
                Smallest vertical distance from knee to ankle, relative to the vertical
                extent of the body side, for which the side sits.
                """
                self.minimum_leg_height: float = 20
                """
                Smallest vertical extent of hip, knee and ankle in the image for which a
                body side is decided, in pixels.
                """
                self.knee_at_hip_height_is_sitting: bool = False
                """
                Whether a person also sits when :meth:`knee_is_at_hip_height`, which
                needs no visible ankle.
                """
                self.minimum_torso_height: float = 20
                """
                Smallest vertical distance of a hip below the shoulders, in pixels, for
                which :meth:`knee_is_at_hip_height` judges its leg.
                """
                self.maximum_knee_drop_ratio: float = 0.55
                """
                Largest vertical distance of a knee below its hip, relative to the
                vertical distance of the hip below the shoulders, of a sitting person.
                """

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self,
        name: str = "HumanSittingAnnotator",
        descriptor: HumanSittingAnnotator.Descriptor | None = None,
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
        keypoints show a sitting posture.

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

            if self.keypoints_suggest_sitting(image_keypoints):
                activity = ActivityAnnotation()
                activity.type = HumanPosture.SITTING.value
                activity.source = self.get_class_name()
                human_hypothesis.annotations.append(activity)

        self.get_annotator_output_struct().set_geometries(visualization_geometries)
        self.get_annotator_output_struct().set_image(copy.deepcopy(self.color))

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def keypoints_suggest_sitting(self, image_keypoints: KeypointAnnotation) -> bool:
        """
        :param image_keypoints: The image keypoints of one person.
        :return: Whether the keypoints show a sitting person.
        """
        parameters = self.descriptor.parameters
        confidence_threshold = parameters.keypoint_confidence_threshold

        if parameters.knee_at_hip_height_is_sitting and self.knee_is_at_hip_height(
            image_keypoints
        ):
            return True

        left_hip = image_keypoints.keypoints[PoseKeypoint.LEFT_HIP]
        left_knee = image_keypoints.keypoints[PoseKeypoint.LEFT_KNEE]
        left_ankle = image_keypoints.keypoints[PoseKeypoint.LEFT_ANKLE]

        right_hip = image_keypoints.keypoints[PoseKeypoint.RIGHT_HIP]
        right_knee = image_keypoints.keypoints[PoseKeypoint.RIGHT_KNEE]
        right_ankle = image_keypoints.keypoints[PoseKeypoint.RIGHT_ANKLE]

        left_side_reliable = (
            min(left_hip.confidence, left_knee.confidence, left_ankle.confidence)
            >= confidence_threshold
        )
        right_side_reliable = (
            min(right_hip.confidence, right_knee.confidence, right_ankle.confidence)
            >= confidence_threshold
        )

        if not left_side_reliable and not right_side_reliable:
            return False

        def side_is_sitting(
            hip: BodyKeypoint, knee: BodyKeypoint, ankle: BodyKeypoint
        ) -> bool:
            """
            :param hip: The hip keypoint of the side.
            :param knee: The knee keypoint of the side.
            :param ankle: The ankle keypoint of the side.
            :return: Whether the hip, knee and ankle of one body side show a sitting
                posture.
            """
            image_rows = [hip.y, knee.y, ankle.y]
            side_height = max(image_rows) - min(image_rows)

            if side_height < parameters.minimum_leg_height:
                return False

            relative_hip_knee_height = abs(hip.y - knee.y) / side_height
            relative_knee_ankle_height = abs(knee.y - ankle.y) / side_height

            knee_below_hip = knee.y > hip.y
            ankle_below_knee = ankle.y > knee.y

            return (
                knee_below_hip
                and ankle_below_knee
                and relative_hip_knee_height
                <= parameters.maximum_relative_hip_knee_height
                and relative_knee_ankle_height
                >= parameters.minimum_relative_knee_ankle_height
            )

        if left_side_reliable and side_is_sitting(left_hip, left_knee, left_ankle):
            return True
        if right_side_reliable and side_is_sitting(right_hip, right_knee, right_ankle):
            return True

        # Optional: if both sides are visible, check whether knees are at similar height.
        if left_side_reliable and right_side_reliable:
            legs_height = max(
                left_hip.y,
                left_knee.y,
                left_ankle.y,
                right_hip.y,
                right_knee.y,
                right_ankle.y,
            ) - min(
                left_hip.y,
                left_knee.y,
                left_ankle.y,
                right_hip.y,
                right_knee.y,
                right_ankle.y,
            )
            if legs_height > parameters.minimum_leg_height:
                relative_knee_knee_height = (
                    abs(left_knee.y - right_knee.y) / legs_height
                )
                if (
                    relative_knee_knee_height
                    <= parameters.maximum_relative_knee_knee_height
                ):
                    return side_is_sitting(
                        left_hip, left_knee, left_ankle
                    ) or side_is_sitting(right_hip, right_knee, right_ankle)

        return False

    def knee_is_at_hip_height(self, image_keypoints: KeypointAnnotation) -> bool:
        """
        :param image_keypoints: The image keypoints of one person.
        :return: Whether a shoulder and a hip with its knee are reliable and the knee is
            not farther below the hip than
            :attr:`Descriptor.Parameters.maximum_knee_drop_ratio` of the torso height.
        """
        parameters = self.descriptor.parameters
        confidence_threshold = parameters.keypoint_confidence_threshold
        keypoints = image_keypoints.keypoints

        reliable_shoulders = [
            shoulder
            for shoulder in (
                keypoints[PoseKeypoint.LEFT_SHOULDER],
                keypoints[PoseKeypoint.RIGHT_SHOULDER],
            )
            if shoulder.confidence >= confidence_threshold
        ]
        if not reliable_shoulders:
            return False
        shoulder_y = sum(shoulder.y for shoulder in reliable_shoulders) / len(
            reliable_shoulders
        )

        for hip, knee in (
            (keypoints[PoseKeypoint.LEFT_HIP], keypoints[PoseKeypoint.LEFT_KNEE]),
            (keypoints[PoseKeypoint.RIGHT_HIP], keypoints[PoseKeypoint.RIGHT_KNEE]),
        ):
            torso_height = hip.y - shoulder_y
            if (
                min(hip.confidence, knee.confidence) < confidence_threshold
                or torso_height < parameters.minimum_torso_height
            ):
                continue
            if (knee.y - hip.y) / torso_height <= parameters.maximum_knee_drop_ratio:
                return True
        return False
