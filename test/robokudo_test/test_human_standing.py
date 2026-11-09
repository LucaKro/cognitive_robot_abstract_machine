"""
Tests for recognizing standing humans from their 2D body keypoints.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from py_trees.common import Status

from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.annotators.human_standing import (
    HumanActivity,
    HumanStandingAnnotator,
    PostureMarking,
)
from robokudo.annotators.outputs import AnnotatorOutputStruct
from robokudo.cas import CAS, CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.human import (
    ActivityAnnotation,
    BodyKeypoint,
    KeypointAnnotation,
    KeypointKind,
)
from robokudo.types.scene import HumanHypothesis
from ._marked_image import MarkedRegionImage

pytestmark = pytest.mark.usefixtures("annotator_output_buffer")

# %% keypoint layouts


@dataclass
class BodyLayout:
    """
    Image positions of the body keypoints of one human; every keypoint not given is
    hidden.
    """

    visible_keypoints: dict[PoseKeypoint, BodyKeypoint]
    """
    The keypoints the pose model sees clearly.
    """

    hidden_keypoint: BodyKeypoint = field(
        default_factory=lambda: BodyKeypoint(x=0.0, y=0.0, confidence=0.1)
    )
    """
    The position and confidence a pose model reports for a hidden keypoint.
    """

    def keypoints(self) -> list[BodyKeypoint]:
        """
        :return: All body keypoints, indexed by :class:`PoseKeypoint`.
        """
        return [
            self.visible_keypoints.get(keypoint, self.hidden_keypoint)
            for keypoint in PoseKeypoint
        ]

    def without(self, *hidden: PoseKeypoint) -> BodyLayout:
        """
        :param hidden: The keypoints to hide.
        :return: A copy of this layout in which the given keypoints are hidden.
        """
        return BodyLayout(
            visible_keypoints={
                keypoint: position
                for keypoint, position in self.visible_keypoints.items()
                if keypoint not in hidden
            },
            hidden_keypoint=self.hidden_keypoint,
        )

    def human(self) -> HumanHypothesis:
        """
        :return: A human hypothesis carrying the keypoints of this layout.
        """
        human = HumanHypothesis()
        human.roi.roi = MarkedRegionImage().region
        human.annotations.append(
            KeypointAnnotation(
                keypoints=self.keypoints(), kind=KeypointKind.IMAGE_POSITION
            )
        )
        return human


def visible(x: float, y: float) -> BodyKeypoint:
    """
    :return: A keypoint the pose model sees clearly at the given image position.
    """
    return BodyKeypoint(x=x, y=y, confidence=0.9)


@pytest.fixture
def standing_layout() -> BodyLayout:
    """
    A human standing upright, facing the camera.
    """
    return BodyLayout(
        visible_keypoints={
            PoseKeypoint.NOSE: visible(320, 100),
            PoseKeypoint.LEFT_SHOULDER: visible(350, 150),
            PoseKeypoint.RIGHT_SHOULDER: visible(290, 150),
            PoseKeypoint.LEFT_HIP: visible(340, 280),
            PoseKeypoint.RIGHT_HIP: visible(300, 280),
            PoseKeypoint.LEFT_KNEE: visible(340, 380),
            PoseKeypoint.RIGHT_KNEE: visible(300, 380),
        }
    )


@pytest.fixture
def sitting_layout() -> BodyLayout:
    """
    A human sitting, facing the camera, with the knees at the height of the hips.
    """
    return BodyLayout(
        visible_keypoints={
            PoseKeypoint.NOSE: visible(320, 100),
            PoseKeypoint.LEFT_SHOULDER: visible(350, 150),
            PoseKeypoint.RIGHT_SHOULDER: visible(290, 150),
            PoseKeypoint.LEFT_HIP: visible(340, 280),
            PoseKeypoint.RIGHT_HIP: visible(300, 280),
            PoseKeypoint.LEFT_KNEE: visible(345, 290),
            PoseKeypoint.RIGHT_KNEE: visible(295, 290),
        }
    )


# %% annotation


def annotate(
    *humans: HumanHypothesis, annotator: HumanStandingAnnotator | None = None
) -> HumanStandingAnnotator:
    """
    Run a :class:`HumanStandingAnnotator` on a CAS holding the given humans and the
    color image of :class:`MarkedRegionImage`.

    :param annotator: The annotator to run, a default one if None.
    :return: The annotator after its update.
    """
    annotator = annotator if annotator is not None else HumanStandingAnnotator()
    pipeline = Pipeline()
    pipeline.add_child(annotator)
    pipeline.setup()
    pipeline.cas.set(CASViews.COLOR_IMAGE, MarkedRegionImage().color_image())
    pipeline.cas.annotations.extend(humans)
    assert annotator.update() == Status.SUCCESS
    return annotator


def activity_types(human: HumanHypothesis) -> list[str]:
    """
    :return: The activity types the human is annotated with.
    """
    return [
        activity.type
        for activity in CAS.filter_by_type(ActivityAnnotation, human.annotations)
    ]


# %% standing posture


def test_upright_human_is_standing(standing_layout: BodyLayout) -> None:
    """
    A human with an upright posture gets exactly one standing activity.
    """
    human = standing_layout.human()

    annotate(human)

    assert activity_types(human) == [HumanActivity.STANDING]


def test_standing_activity_names_its_source(standing_layout: BodyLayout) -> None:
    """
    The standing activity names the annotator that created it.
    """
    human = standing_layout.human()

    annotator = annotate(human)

    activity = CAS.filter_by_type(ActivityAnnotation, human.annotations)[0]
    assert activity.source == annotator.name


def test_sitting_human_is_not_standing(sitting_layout: BodyLayout) -> None:
    """
    A human whose knees are at the height of the hips gets no standing activity.
    """
    human = sitting_layout.human()

    annotate(human)

    assert activity_types(human) == []


def test_one_hidden_leg_still_allows_standing(standing_layout: BodyLayout) -> None:
    """
    A standing human with one hidden leg is judged on the other leg, and the hidden
    keypoints do not distort the extent of the body.
    """
    human = standing_layout.without(
        PoseKeypoint.LEFT_HIP, PoseKeypoint.LEFT_KNEE
    ).human()

    annotate(human)

    assert activity_types(human) == [HumanActivity.STANDING]


@pytest.mark.parametrize(
    "hidden",
    [
        (PoseKeypoint.NOSE,),
        (PoseKeypoint.LEFT_SHOULDER,),
        (PoseKeypoint.RIGHT_SHOULDER,),
        (PoseKeypoint.LEFT_KNEE, PoseKeypoint.RIGHT_KNEE),
        (PoseKeypoint.LEFT_HIP, PoseKeypoint.RIGHT_HIP),
    ],
)
def test_unclear_posture_is_not_standing(
    standing_layout: BodyLayout, hidden: tuple[PoseKeypoint, ...]
) -> None:
    """
    A human whose nose, a shoulder or both legs are hidden gets no standing activity.
    """
    human = standing_layout.without(*hidden).human()

    annotate(human)

    assert activity_types(human) == []


def test_too_small_human_is_not_standing() -> None:
    """
    A human whose visible keypoints are lower than the minimum body height gets no
    standing activity, even when upright.
    """
    height = HumanStandingAnnotator.Descriptor.Parameters().minimum_body_height / 2
    human = BodyLayout(
        visible_keypoints={
            PoseKeypoint.NOSE: visible(320, 100),
            PoseKeypoint.LEFT_SHOULDER: visible(320, 100 + height * 0.2),
            PoseKeypoint.RIGHT_SHOULDER: visible(320, 100 + height * 0.2),
            PoseKeypoint.LEFT_HIP: visible(320, 100 + height * 0.6),
            PoseKeypoint.LEFT_KNEE: visible(320, 100 + height),
        }
    ).human()

    annotate(human)

    assert activity_types(human) == []


def test_lying_human_is_not_standing(standing_layout: BodyLayout) -> None:
    """
    A human whose keypoints spread wider than high gets no standing activity, even when
    ordered from head to knee.
    """
    human = standing_layout.human()
    keypoints = human.annotations[0].keypoints
    for keypoint in keypoints:
        keypoint.x = (keypoint.x - 320) * 20 + 320

    annotate(human)

    assert activity_types(human) == []


def test_human_without_image_keypoints_is_not_standing() -> None:
    """
    A human without image keypoints gets no standing activity.
    """
    human = HumanHypothesis()

    annotate(human)

    assert activity_types(human) == []


def test_only_standing_humans_are_counted(
    standing_layout: BodyLayout, sitting_layout: BodyLayout
) -> None:
    """
    Of several humans, only the standing ones get a standing activity.
    """
    standing_human = standing_layout.human()
    sitting_human = sitting_layout.human()

    annotate(standing_human, sitting_human)

    assert activity_types(standing_human) == [HumanActivity.STANDING]
    assert activity_types(sitting_human) == []


# %% visualization


def test_standing_human_is_marked_as_standing(standing_layout: BodyLayout) -> None:
    """
    The visualization image frames a standing human in the standing color.
    """
    annotator = annotate(standing_layout.human())

    image = annotator.get_annotator_output_struct().image
    assert MarkedRegionImage().frame_color(image) == PostureMarking.STANDING.value.color


def test_sitting_human_is_marked_as_not_standing(sitting_layout: BodyLayout) -> None:
    """
    The visualization image frames a human who does not stand in the not standing color.
    """
    annotator = annotate(sitting_layout.human())

    image = annotator.get_annotator_output_struct().image
    assert (
        MarkedRegionImage().frame_color(image)
        == PostureMarking.NOT_STANDING.value.color
    )


def test_no_image_without_visualization(standing_layout: BodyLayout) -> None:
    """
    With visualization turned off, the annotator outputs no image.
    """
    annotator = HumanStandingAnnotator()
    annotator.descriptor.parameters.global_with_visualization = False

    annotate(standing_layout.human(), annotator=annotator)

    np.testing.assert_array_equal(
        annotator.get_annotator_output_struct().image, AnnotatorOutputStruct().image
    )
