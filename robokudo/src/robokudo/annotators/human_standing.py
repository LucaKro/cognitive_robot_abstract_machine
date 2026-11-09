"""
Standing posture recognition from 2D body keypoints.

Annotates every human hypothesis whose body keypoints show an upright posture with a
standing activity. Humans whose posture is unclear, for example because their legs are
hidden, are left without one.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum, auto

from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.cas import CAS, CASViews
from robokudo.types.human import (
    ActivityAnnotation,
    BodyKeypoint,
    KeypointAnnotation,
    KeypointKind,
)
from robokudo.types.scene import HumanHypothesis
from robokudo.vis.box_marking import BoxMarking

# %% activities


class HumanActivity(StrEnum):
    """
    Activities a human can be annotated with by an :class:`ActivityAnnotation`.
    """

    STANDING = auto()
    """
    The human stands upright.
    """

    SITTING = auto()
    """
    The human sits.
    """


# %% visualization


class PostureMarking(Enum):
    """
    How a human is marked in the visualization image.
    """

    STANDING = BoxMarking(label=HumanActivity.STANDING, color=(0, 200, 0))
    """
    The human stands.
    """

    NOT_STANDING = BoxMarking(label="not standing", color=(0, 140, 255))
    """
    The human does not stand or their posture is unclear.
    """


# %% posture keypoints


@dataclass
class Leg:
    """
    The hip and knee keypoints of one leg.
    """

    hip: BodyKeypoint
    """
    The hip keypoint of the leg.
    """

    knee: BodyKeypoint
    """
    The knee keypoint of the leg.
    """

    @classmethod
    def both_from_keypoints(cls, keypoints: list[BodyKeypoint]) -> list[Leg]:
        """
        :param keypoints: The body keypoints of one human, indexed by
            :class:`~robokudo.annotators.human_and_pose.PoseKeypoint`.
        :return: The left and the right leg.
        """
        return [
            cls(
                hip=keypoints[PoseKeypoint.LEFT_HIP],
                knee=keypoints[PoseKeypoint.LEFT_KNEE],
            ),
            cls(
                hip=keypoints[PoseKeypoint.RIGHT_HIP],
                knee=keypoints[PoseKeypoint.RIGHT_KNEE],
            ),
        ]

    def is_visible(self, confidence_threshold: float) -> bool:
        """
        :param confidence_threshold: Minimum confidence of a visible keypoint.
        :return: Whether both the hip and the knee are visible.
        """
        return min(self.hip.confidence, self.knee.confidence) >= confidence_threshold


@dataclass
class PostureKeypoints:
    """
    The visible body keypoints of one human that a standing posture is judged on, in
    color image pixels.
    """

    nose: BodyKeypoint
    """
    The nose keypoint.
    """

    left_shoulder: BodyKeypoint
    """
    The left shoulder keypoint.
    """

    right_shoulder: BodyKeypoint
    """
    The right shoulder keypoint.
    """

    leg: Leg
    """
    The visible leg the posture is judged on, the left one if both are visible.
    """

    visible_keypoints: list[BodyKeypoint]
    """
    All visible keypoints among the nose, the shoulders, the hips and the knees.
    """

    @classmethod
    def from_keypoints(
        cls, keypoints: list[BodyKeypoint], confidence_threshold: float
    ) -> PostureKeypoints | None:
        """
        :param keypoints: The body keypoints of one human, indexed by
            :class:`~robokudo.annotators.human_and_pose.PoseKeypoint`.
        :param confidence_threshold: Minimum confidence of a visible keypoint.
        :return: The posture keypoints, or None if the nose, a shoulder or both legs
            are not visible.
        """
        upper_body = [
            keypoints[PoseKeypoint.NOSE],
            keypoints[PoseKeypoint.LEFT_SHOULDER],
            keypoints[PoseKeypoint.RIGHT_SHOULDER],
        ]
        visible_legs = [
            leg
            for leg in Leg.both_from_keypoints(keypoints)
            if leg.is_visible(confidence_threshold)
        ]
        upper_body_visible = (
            min(keypoint.confidence for keypoint in upper_body) >= confidence_threshold
        )
        if not upper_body_visible or not visible_legs:
            return None

        nose, left_shoulder, right_shoulder = upper_body
        return cls(
            nose=nose,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            leg=visible_legs[0],
            visible_keypoints=upper_body
            + [keypoint for leg in visible_legs for keypoint in (leg.hip, leg.knee)],
        )

    @property
    def height(self) -> float:
        """
        :return: Vertical extent of the visible keypoints.
        """
        heights = [keypoint.y for keypoint in self.visible_keypoints]
        return max(heights) - min(heights)

    @property
    def width(self) -> float:
        """
        :return: Horizontal extent of the visible keypoints.
        """
        widths = [keypoint.x for keypoint in self.visible_keypoints]
        return max(widths) - min(widths)

    @property
    def shoulder_height(self) -> float:
        """
        :return: Mean vertical image position of both shoulders.
        """
        return (self.left_shoulder.y + self.right_shoulder.y) / 2

    @property
    def upright_ratio(self) -> float:
        """
        :return: Share of :attr:`height` in the sum of :attr:`height` and
            :attr:`width`.
        """
        return self.height / (self.height + self.width)

    @property
    def shoulder_over_hip(self) -> float:
        """
        :return: How far the shoulders are above the hip, relative to :attr:`height`.
        """
        return (self.leg.hip.y - self.shoulder_height) / self.height

    @property
    def hip_over_knee(self) -> float:
        """
        :return: How far the hip is above the knee, relative to :attr:`height`.
        """
        return (self.leg.knee.y - self.leg.hip.y) / self.height

    @property
    def nose_over_shoulder(self) -> float:
        """
        :return: How far the nose is above the shoulders, relative to :attr:`height`.
        """
        return (self.shoulder_height - self.nose.y) / self.height


# %% annotator


class HumanStandingAnnotator(BaseAnnotator):
    """
    Annotates every human hypothesis that stands upright with a
    :attr:`HumanActivity.STANDING` activity.

    A human is standing when the nose, both shoulders and at least one leg are visible
    and the visible keypoints are upright and ordered from nose over shoulders and hip
    down to the knee. The visualization image marks every human as
    :class:`PostureMarking`.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Descriptor for the HumanStandingAnnotator.
        """

        class Parameters:
            """
            Parameters for the HumanStandingAnnotator.
            """

            def __init__(self) -> None:
                """
                Set every parameter to its default.
                """
                self.keypoint_confidence_threshold: float = 0.5
                """
                Minimum confidence of a keypoint to count as visible.
                """
                self.minimum_body_height: float = 20.0
                """
                Minimum vertical extent in pixels of the visible keypoints for a
                posture to be judged.
                """
                self.minimum_upright_ratio: float = 0.6
                """
                Minimum :attr:`PostureKeypoints.upright_ratio` of a standing human.
                """
                self.minimum_shoulder_over_hip: float = 0.25
                """
                Minimum :attr:`PostureKeypoints.shoulder_over_hip` of a standing human.
                """
                self.minimum_hip_over_knee: float = 0.30
                """
                Minimum :attr:`PostureKeypoints.hip_over_knee` of a standing human.
                """
                self.minimum_nose_over_shoulder: float = 0.10
                """
                Minimum :attr:`PostureKeypoints.nose_over_shoulder` of a standing
                human.
                """

    def __init__(
        self,
        name: str = "HumanStandingAnnotator",
        descriptor: HumanStandingAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of the annotator in the pipeline.
        :param descriptor: Configuration of the annotator, the default one if None.
        """
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """
        Annotate every standing human hypothesis with a standing activity.

        :return: SUCCESS after processing.
        """
        humans = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        standing = [self.is_standing(human) for human in humans]
        for human, stands in zip(humans, standing):
            if stands:
                human.annotations.append(
                    ActivityAnnotation(source=self.name, type=HumanActivity.STANDING)
                )

        if self.descriptor.parameters.global_with_visualization:
            self._visualize(humans, standing)
        self.feedback_message = f"{sum(standing)} standing humans"
        return Status.SUCCESS

    def _visualize(self, humans: list[HumanHypothesis], standing: list[bool]) -> None:
        """
        Mark every human in a copy of the color image by their posture and output the
        image.

        :param humans: The human hypotheses judged in this update.
        :param standing: Whether each of the humans stands, in the same order.
        """
        image = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        for human, stands in zip(humans, standing):
            marking = PostureMarking.STANDING if stands else PostureMarking.NOT_STANDING
            marking.value.draw(image, human.roi.roi)
        self.get_annotator_output_struct().set_image(image)

    def is_standing(self, human: HumanHypothesis) -> bool:
        """
        :param human: The human hypothesis to judge.
        :return: Whether the image keypoints of the human show a standing posture.
        """
        parameters: HumanStandingAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        image_keypoints = [
            annotation
            for annotation in CAS.filter_by_type(KeypointAnnotation, human.annotations)
            if annotation.kind == KeypointKind.IMAGE_POSITION
        ]
        if not image_keypoints:
            return False

        posture = PostureKeypoints.from_keypoints(
            image_keypoints[0].keypoints, parameters.keypoint_confidence_threshold
        )
        if (
            posture is None
            or posture.height == 0
            or posture.height < parameters.minimum_body_height
        ):
            return False

        return (
            posture.upright_ratio >= parameters.minimum_upright_ratio
            and posture.shoulder_over_hip >= parameters.minimum_shoulder_over_hip
            and posture.hip_over_knee >= parameters.minimum_hip_over_knee
            and posture.nose_over_shoulder >= parameters.minimum_nose_over_shoulder
        )
