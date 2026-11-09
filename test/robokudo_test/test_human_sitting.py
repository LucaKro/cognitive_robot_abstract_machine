"""
Tests for recognizing sitting humans from their 2D body keypoints.
"""

# %% imports

from __future__ import annotations

import pytest
from py_trees.common import Status

from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.annotators.human_sitting import HumanSittingAnnotator
from robokudo.annotators.human_standing import HumanActivity
from robokudo.cas import CAS
from robokudo.pipeline import Pipeline
from robokudo.types.human import ActivityAnnotation
from robokudo.types.scene import HumanHypothesis
from .test_human_standing import BodyLayout, activity_types, visible

pytestmark = pytest.mark.usefixtures("annotator_output_buffer")

# %% keypoint layouts


@pytest.fixture
def sitting_layout() -> BodyLayout:
    """
    A human sitting, facing the camera, with short thighs and long shins in the image.
    """
    return BodyLayout(
        visible_keypoints={
            PoseKeypoint.LEFT_HIP: visible(340, 280),
            PoseKeypoint.RIGHT_HIP: visible(300, 280),
            PoseKeypoint.LEFT_KNEE: visible(345, 300),
            PoseKeypoint.RIGHT_KNEE: visible(295, 300),
            PoseKeypoint.LEFT_ANKLE: visible(345, 420),
            PoseKeypoint.RIGHT_ANKLE: visible(295, 420),
        }
    )


@pytest.fixture
def standing_layout() -> BodyLayout:
    """
    A human standing upright, facing the camera, with thighs as long as the shins.
    """
    return BodyLayout(
        visible_keypoints={
            PoseKeypoint.LEFT_HIP: visible(340, 280),
            PoseKeypoint.RIGHT_HIP: visible(300, 280),
            PoseKeypoint.LEFT_KNEE: visible(340, 380),
            PoseKeypoint.RIGHT_KNEE: visible(300, 380),
            PoseKeypoint.LEFT_ANKLE: visible(340, 470),
            PoseKeypoint.RIGHT_ANKLE: visible(300, 470),
        }
    )


# %% annotation


def annotate(*humans: HumanHypothesis) -> HumanSittingAnnotator:
    """
    Run a :class:`HumanSittingAnnotator` on a CAS holding the given humans.

    :return: The annotator after its update.
    """
    annotator = HumanSittingAnnotator()
    pipeline = Pipeline()
    pipeline.add_child(annotator)
    pipeline.setup()
    pipeline.cas.annotations.extend(humans)
    assert annotator.update() == Status.SUCCESS
    return annotator


# %% sitting posture


def test_bent_legs_are_sitting(sitting_layout: BodyLayout) -> None:
    """
    A human with short thighs and long shins gets exactly one sitting activity.
    """
    human = sitting_layout.human()

    annotate(human)

    assert activity_types(human) == [HumanActivity.SITTING]


def test_sitting_activity_names_its_source(sitting_layout: BodyLayout) -> None:
    """
    The sitting activity names the annotator that created it.
    """
    human = sitting_layout.human()

    annotator = annotate(human)

    activity = CAS.filter_by_type(ActivityAnnotation, human.annotations)[0]
    assert activity.source == annotator.name


def test_standing_human_is_not_sitting(standing_layout: BodyLayout) -> None:
    """
    A human whose thighs are as long as the shins gets no sitting activity.
    """
    human = standing_layout.human()

    annotate(human)

    assert activity_types(human) == []


def test_one_hidden_leg_still_allows_sitting(sitting_layout: BodyLayout) -> None:
    """
    A sitting human with one hidden leg is judged on the other leg.
    """
    human = sitting_layout.without(
        PoseKeypoint.LEFT_HIP, PoseKeypoint.LEFT_KNEE, PoseKeypoint.LEFT_ANKLE
    ).human()

    annotate(human)

    assert activity_types(human) == [HumanActivity.SITTING]


@pytest.mark.parametrize(
    "hidden",
    [
        (PoseKeypoint.LEFT_ANKLE, PoseKeypoint.RIGHT_ANKLE),
        (PoseKeypoint.LEFT_KNEE, PoseKeypoint.RIGHT_KNEE),
        (PoseKeypoint.LEFT_HIP, PoseKeypoint.RIGHT_HIP),
    ],
)
def test_hidden_legs_are_not_sitting(
    sitting_layout: BodyLayout, hidden: tuple[PoseKeypoint, ...]
) -> None:
    """
    A human whose ankles, knees or hips are hidden gets no sitting activity.
    """
    human = sitting_layout.without(*hidden).human()

    annotate(human)

    assert activity_types(human) == []


def test_too_small_leg_is_not_sitting() -> None:
    """
    A human whose legs are lower than the minimum leg height gets no sitting activity,
    even when bent.
    """
    height = HumanSittingAnnotator.Descriptor.Parameters().minimum_leg_height / 2
    human = BodyLayout(
        visible_keypoints={
            PoseKeypoint.LEFT_HIP: visible(320, 300),
            PoseKeypoint.LEFT_KNEE: visible(320, 300 + height * 0.1),
            PoseKeypoint.LEFT_ANKLE: visible(320, 300 + height),
        }
    ).human()

    annotate(human)

    assert activity_types(human) == []


def test_leg_not_ordered_downwards_is_not_sitting() -> None:
    """
    A leg whose knee is above the hip gets no sitting activity, even with a short thigh
    and a long shin.
    """
    human = BodyLayout(
        visible_keypoints={
            PoseKeypoint.LEFT_HIP: visible(320, 300),
            PoseKeypoint.LEFT_KNEE: visible(320, 280),
            PoseKeypoint.LEFT_ANKLE: visible(320, 420),
        }
    ).human()

    annotate(human)

    assert activity_types(human) == []


def test_human_without_image_keypoints_is_not_sitting() -> None:
    """
    A human without image keypoints gets no sitting activity.
    """
    human = HumanHypothesis()

    annotate(human)

    assert activity_types(human) == []


# %% knee at hip height


@pytest.fixture
def upper_body_layout() -> BodyLayout:
    """
    A human sitting, facing the camera, with the ankles out of view and the knees a
    little below the hips.
    """
    return BodyLayout(
        visible_keypoints={
            PoseKeypoint.LEFT_SHOULDER: visible(350, 150),
            PoseKeypoint.RIGHT_SHOULDER: visible(290, 150),
            PoseKeypoint.LEFT_HIP: visible(340, 280),
            PoseKeypoint.RIGHT_HIP: visible(300, 280),
            PoseKeypoint.LEFT_KNEE: visible(345, 300),
            PoseKeypoint.RIGHT_KNEE: visible(295, 300),
        }
    )


def test_knee_at_hip_height_is_sitting(upper_body_layout: BodyLayout) -> None:
    """
    A human whose knees are close to the height of the hips sits, even with the ankles
    hidden.
    """
    human = upper_body_layout.human()

    annotate(human)

    assert activity_types(human) == [HumanActivity.SITTING]


def test_knee_above_hip_is_sitting(upper_body_layout: BodyLayout) -> None:
    """
    A human whose knees are slightly above the hips in the image sits.
    """
    human = upper_body_layout.human()
    keypoints = human.annotations[0].keypoints
    for knee in (PoseKeypoint.LEFT_KNEE, PoseKeypoint.RIGHT_KNEE):
        keypoints[knee].y = 270

    annotate(human)

    assert activity_types(human) == [HumanActivity.SITTING]


def test_knee_far_below_hip_is_not_sitting(upper_body_layout: BodyLayout) -> None:
    """
    A human whose knees are farther below the hips than the maximum knee drop ratio of
    the torso height does not sit.
    """
    human = upper_body_layout.human()
    keypoints = human.annotations[0].keypoints
    maximum_knee_drop_ratio = (
        HumanSittingAnnotator.Descriptor.Parameters().maximum_knee_drop_ratio
    )
    torso_height = 280 - 150
    for knee in (PoseKeypoint.LEFT_KNEE, PoseKeypoint.RIGHT_KNEE):
        keypoints[knee].y = 280 + torso_height * (maximum_knee_drop_ratio + 0.1)

    annotate(human)

    assert activity_types(human) == []


def test_one_shoulder_is_enough(upper_body_layout: BodyLayout) -> None:
    """
    A human with one hidden shoulder is judged on the other shoulder.
    """
    human = upper_body_layout.without(PoseKeypoint.LEFT_SHOULDER).human()

    annotate(human)

    assert activity_types(human) == [HumanActivity.SITTING]


def test_hidden_shoulders_without_ankles_are_not_sitting(
    upper_body_layout: BodyLayout,
) -> None:
    """
    A human whose shoulders and ankles are hidden does not sit, even with the knees at
    the height of the hips.
    """
    human = upper_body_layout.without(
        PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.RIGHT_SHOULDER
    ).human()

    annotate(human)

    assert activity_types(human) == []


def test_too_short_torso_is_not_judged(upper_body_layout: BodyLayout) -> None:
    """
    A leg whose hip is less than the minimum torso height below the shoulders is not
    judged against the torso.
    """
    human = upper_body_layout.human()
    keypoints = human.annotations[0].keypoints
    minimum_torso_height = (
        HumanSittingAnnotator.Descriptor.Parameters().minimum_torso_height
    )
    for shoulder in (PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.RIGHT_SHOULDER):
        keypoints[shoulder].y = 280 - minimum_torso_height / 2

    annotate(human)

    assert activity_types(human) == []
