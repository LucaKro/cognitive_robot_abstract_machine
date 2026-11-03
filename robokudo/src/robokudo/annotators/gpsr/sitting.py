from __future__ import annotations

# %% Imports
import copy
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

# %% sitting rule


class HumanSittingAnnotator(BaseAnnotator):
    """
    Annotates every person whose image keypoints show a sitting posture with the
    activity ``sitting``.

    A body side sits when its keypoints are reliable, its knee is below the hip and its
    ankle below the knee, the drop from hip to knee is short and the drop from knee to
    ankle is long, both relative to the vertical extent of the side. One sitting side is
    enough. Persons without image keypoints are skipped.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the sitting rule.
        """

        class Parameters:
            """
            Parameters of the sitting rule.
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

                # TODO: Adjust sitting thresholds.
                self.max_hip_knee_rel_bbox: float = 0.5
                """
                Largest vertical distance from hip to knee, relative to the vertical
                extent of the body side, for which the side sits.
                """
                self.max_knee_knee_rel_bbox: float = 0.5
                """
                Largest vertical distance between the two knees, relative to the
                vertical extent of both legs, for which both sides are checked again.
                """
                self.min_ankle_knee_rel_bbox: float = 0.2
                """
                Smallest vertical distance from knee to ankle, relative to the vertical
                extent of the body side, for which the side sits.
                """
                self.min_leg_height: float = 20
                """
                Smallest vertical extent of hip, knee and ankle in the image, in pixels,
                for which a body side is decided.
                """

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion
        """
        Default parameters on the class; each descriptor instance gets its own.
        """

    def __init__(
        self,
        name: str = "HumanSittingAnnotator",
        descriptor: HumanSittingAnnotator.Descriptor | None = None,
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

            if self.keypoints_sitting(image_keypoints):
                activity = ActivityAnnotation()
                activity.type = "sitting"
                activity.source = type(self).__name__
                human_hypothesis.annotations.append(activity)

        self.get_annotator_output_struct().set_geometries(visualization_geometries)
        self.get_annotator_output_struct().set_image(copy.deepcopy(self.color))

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def keypoints_sitting(self, keypoint_2d_annotation: KeypointAnnotation) -> bool:
        """
        :param keypoint_2d_annotation: The image keypoints of one person.
        :return: Whether the keypoints show a sitting person.
        """
        parameters = self.descriptor.parameters
        confidence_threshold = parameters.keypoint_confidence_threshold

        left_hip = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_HIP]
        left_knee = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_KNEE]
        left_ankle = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_ANKLE]

        right_hip = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_HIP]
        right_knee = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_KNEE]
        right_ankle = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_ANKLE]

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
            :return: Whether the hip, knee and ankle of one body side show a sitting
                posture.
            """
            image_rows = [hip.y, knee.y, ankle.y]
            side_height = max(image_rows) - min(image_rows)

            if side_height < parameters.min_leg_height:
                return False

            relative_hip_knee_height = abs(hip.y - knee.y) / side_height
            relative_knee_ankle_height = abs(knee.y - ankle.y) / side_height

            knee_below_hip = knee.y > hip.y
            ankle_below_knee = ankle.y > knee.y

            return (
                knee_below_hip
                and ankle_below_knee
                and relative_hip_knee_height <= parameters.max_hip_knee_rel_bbox
                and relative_knee_ankle_height >= parameters.min_ankle_knee_rel_bbox
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
            if legs_height > parameters.min_leg_height:
                relative_knee_knee_height = (
                    abs(left_knee.y - right_knee.y) / legs_height
                )
                if relative_knee_knee_height <= parameters.max_knee_knee_rel_bbox:
                    return side_is_sitting(
                        left_hip, left_knee, left_ankle
                    ) or side_is_sitting(right_hip, right_knee, right_ankle)

        return False
