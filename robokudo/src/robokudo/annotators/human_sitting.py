"""
Sitting posture recognition from 2D body keypoints.

Annotates every human hypothesis whose body keypoints show a sitting posture with a
sitting activity. Humans whose posture is unclear, for example because their legs are
hidden, are left without one.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass

from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.annotators.human_standing import HumanActivity, Leg
from robokudo.cas import CAS
from robokudo.types.human import (
    ActivityAnnotation,
    BodyKeypoint,
    KeypointAnnotation,
    KeypointKind,
)
from robokudo.types.scene import HumanHypothesis

# %% leg keypoints


@dataclass
class WholeLeg:
    """
    The hip, knee and ankle keypoints of one leg, in color image pixels.
    """

    hip: BodyKeypoint
    """
    The hip keypoint of the leg.
    """

    knee: BodyKeypoint
    """
    The knee keypoint of the leg.
    """

    ankle: BodyKeypoint
    """
    The ankle keypoint of the leg.
    """

    @classmethod
    def both_from_keypoints(cls, keypoints: list[BodyKeypoint]) -> list[WholeLeg]:
        """
        :param keypoints: The body keypoints of one human, indexed by
            :class:`~robokudo.annotators.human_and_pose.PoseKeypoint`.
        :return: The left and the right leg.
        """
        return [
            cls(
                hip=keypoints[PoseKeypoint.LEFT_HIP],
                knee=keypoints[PoseKeypoint.LEFT_KNEE],
                ankle=keypoints[PoseKeypoint.LEFT_ANKLE],
            ),
            cls(
                hip=keypoints[PoseKeypoint.RIGHT_HIP],
                knee=keypoints[PoseKeypoint.RIGHT_KNEE],
                ankle=keypoints[PoseKeypoint.RIGHT_ANKLE],
            ),
        ]

    def is_visible(self, confidence_threshold: float) -> bool:
        """
        :param confidence_threshold: Minimum confidence of a visible keypoint.
        :return: Whether the hip, the knee and the ankle are visible.
        """
        return (
            min(self.hip.confidence, self.knee.confidence, self.ankle.confidence)
            >= confidence_threshold
        )

    @property
    def is_ordered_downwards(self) -> bool:
        """
        :return: Whether the knee is below the hip and the ankle below the knee in the
            image.
        """
        return self.hip.y < self.knee.y < self.ankle.y

    @property
    def height(self) -> float:
        """
        :return: Vertical extent of the hip, the knee and the ankle.
        """
        heights = [self.hip.y, self.knee.y, self.ankle.y]
        return max(heights) - min(heights)

    @property
    def thigh_share(self) -> float:
        """
        :return: Vertical distance between hip and knee, relative to :attr:`height`.
        """
        return abs(self.knee.y - self.hip.y) / self.height


# %% annotator


class HumanSittingAnnotator(BaseAnnotator):
    """
    Annotates every sitting human hypothesis with a :attr:`HumanActivity.SITTING`
    activity: one with a leg visible from hip to ankle that has a short thigh in the
    image, see :meth:`is_sitting_leg`, or one whose knee is close to the height of the
    hip, see :meth:`has_knee_at_hip_height`.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Descriptor for the HumanSittingAnnotator.
        """

        class Parameters:
            """
            Parameters for the HumanSittingAnnotator.
            """

            def __init__(self) -> None:
                """
                Set every parameter to its default.
                """
                self.keypoint_confidence_threshold: float = 0.5
                """
                Minimum confidence of a keypoint to count as visible.
                """
                self.minimum_leg_height: float = 20.0
                """
                Minimum :attr:`WholeLeg.height` in pixels for a leg to be judged.
                """
                self.maximum_thigh_share: float = 0.5
                """
                Maximum :attr:`WholeLeg.thigh_share` of a sitting human.
                """
                self.minimum_torso_height: float = 20.0
                """
                Minimum vertical distance in pixels of a hip below the shoulders for a
                leg to be judged against the torso.
                """
                self.maximum_knee_drop_ratio: float = 0.55
                """
                Maximum vertical distance of a knee below its hip, relative to the
                vertical distance of the hip below the shoulders, of a sitting human.
                """

    def __init__(
        self,
        name: str = "HumanSittingAnnotator",
        descriptor: HumanSittingAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of the annotator in the pipeline.
        :param descriptor: Configuration of the annotator, the default one if None.
        """
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """
        Annotate every sitting human hypothesis with a sitting activity.

        :return: SUCCESS after processing.
        """
        humans = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        sitting_humans = [human for human in humans if self.is_sitting(human)]
        for human in sitting_humans:
            human.annotations.append(
                ActivityAnnotation(source=self.name, type=HumanActivity.SITTING)
            )
        self.feedback_message = f"{len(sitting_humans)} sitting humans"
        return Status.SUCCESS

    def is_sitting(self, human: HumanHypothesis) -> bool:
        """
        :param human: The human hypothesis to judge.
        :return: Whether the image keypoints of the human show a sitting posture.
        """
        image_keypoints = [
            annotation
            for annotation in CAS.filter_by_type(KeypointAnnotation, human.annotations)
            if annotation.kind == KeypointKind.IMAGE_POSITION
        ]
        if not image_keypoints:
            return False

        keypoints = image_keypoints[0].keypoints
        return any(
            self.is_sitting_leg(leg) for leg in WholeLeg.both_from_keypoints(keypoints)
        ) or self.has_knee_at_hip_height(keypoints)

    def is_sitting_leg(self, leg: WholeLeg) -> bool:
        """
        :param leg: The leg to judge.
        :return: Whether the leg is visible and bent like the leg of a sitting human.
        """
        parameters: HumanSittingAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        if (
            not leg.is_visible(parameters.keypoint_confidence_threshold)
            or leg.height < parameters.minimum_leg_height
        ):
            return False

        return (
            leg.is_ordered_downwards
            and leg.thigh_share <= parameters.maximum_thigh_share
        )

    def has_knee_at_hip_height(self, keypoints: list[BodyKeypoint]) -> bool:
        """
        :param keypoints: The body keypoints of one human, indexed by
            :class:`~robokudo.annotators.human_and_pose.PoseKeypoint`.
        :return: Whether a shoulder is visible and a visible leg has its knee not
            farther below the hip than
            :attr:`Descriptor.Parameters.maximum_knee_drop_ratio` of the torso height.
        """
        parameters: HumanSittingAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        threshold = parameters.keypoint_confidence_threshold
        shoulders = [
            keypoint
            for keypoint in (
                keypoints[PoseKeypoint.LEFT_SHOULDER],
                keypoints[PoseKeypoint.RIGHT_SHOULDER],
            )
            if keypoint.confidence >= threshold
        ]
        if not shoulders:
            return False

        shoulder_height = sum(shoulder.y for shoulder in shoulders) / len(shoulders)
        for leg in Leg.both_from_keypoints(keypoints):
            torso_height = leg.hip.y - shoulder_height
            if (
                not leg.is_visible(threshold)
                or torso_height < parameters.minimum_torso_height
            ):
                continue
            if (leg.knee.y - leg.hip.y) / torso_height <= (
                parameters.maximum_knee_drop_ratio
            ):
                return True
        return False
