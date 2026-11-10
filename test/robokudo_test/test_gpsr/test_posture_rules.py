"""
Tests for deciding from the image keypoints of a person whether they stand, sit or lie.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.lying import HumanLyingAnnotator
from robokudo.annotators.gpsr.sitting import HumanSittingAnnotator
from robokudo.annotators.gpsr.standing import HumanStandingAnnotator
from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.cas import CAS
from robokudo.types.human import ActivityAnnotation
from typing_extensions import TYPE_CHECKING, Callable

from ._body_postures import BodyInSpace, BodyPosture

if TYPE_CHECKING:
    from robokudo.annotators.core import BaseAnnotator

    from ._gpsr_scene import GpsrScene

# %% keypoints the posture rules do not trust


def unreliable_confidence(
    annotator: HumanStandingAnnotator | HumanSittingAnnotator | HumanLyingAnnotator,
) -> float:
    """
    :param annotator: The posture rule.
    :return: A keypoint confidence just below the one the posture rule trusts.
    """
    return annotator.descriptor.parameters.keypoint_confidence_threshold - 0.01


# %% standing


def test_standing_person_is_standing(scene: GpsrScene) -> None:
    """
    A person in a standing body pose is decided as standing.
    """
    assert scene.decide_activities(
        HumanStandingAnnotator(), BodyPosture.standing().keypoint_annotation()
    ) == ["standing"]


@pytest.mark.parametrize(
    "posture", [BodyPosture.sitting, BodyPosture.lying], ids=["sitting", "lying"]
)
def test_person_in_another_posture_is_not_standing(
    scene: GpsrScene, posture: Callable[[], BodyPosture]
) -> None:
    """
    Person in another posture is not standing.
    """
    assert (
        scene.decide_activities(
            HumanStandingAnnotator(), posture().keypoint_annotation()
        )
        == []
    )


@pytest.mark.parametrize(
    "posture",
    [
        BodyPosture.standing().moved(PoseKeypoint.RIGHT_SHOULDER, x=481, y=150),
        BodyPosture.standing().moved_to_height(
            210, PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.RIGHT_SHOULDER
        ),
        BodyPosture.standing().moved_to_height(320, PoseKeypoint.LEFT_KNEE),
        BodyPosture.standing().moved_to_height(140, PoseKeypoint.NOSE),
    ],
    ids=[
        "wider than tall",
        "shoulders barely above the hips",
        "hips barely above the knees",
        "head barely above the shoulders",
    ],
)
def test_standing_needs_every_body_proportion(
    scene: GpsrScene, posture: BodyPosture
) -> None:
    """
    Standing needs every body proportion.
    """
    assert (
        scene.decide_activities(HumanStandingAnnotator(), posture.keypoint_annotation())
        == []
    )


def test_upper_body_at_the_confidence_threshold_counts(scene: GpsrScene) -> None:
    """
    Upper body at the confidence threshold counts.
    """
    annotator = HumanStandingAnnotator(descriptor=HumanStandingAnnotator.Descriptor())
    threshold = annotator.descriptor.parameters.keypoint_confidence_threshold
    posture = BodyPosture.standing().with_confidence(threshold, PoseKeypoint.NOSE)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        "standing"
    ]


def test_upper_body_below_the_confidence_threshold_is_not_decided(
    scene: GpsrScene,
) -> None:
    """
    Upper body below the confidence threshold is not decided.
    """
    annotator = HumanStandingAnnotator(descriptor=HumanStandingAnnotator.Descriptor())
    posture = BodyPosture.standing().with_confidence(
        unreliable_confidence(annotator), PoseKeypoint.NOSE
    )

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == []


def test_one_visible_leg_is_enough_to_stand(scene: GpsrScene) -> None:
    """
    One visible leg is enough to stand.
    """
    annotator = HumanStandingAnnotator(descriptor=HumanStandingAnnotator.Descriptor())
    posture = BodyPosture.standing().with_confidence(
        unreliable_confidence(annotator), PoseKeypoint.LEFT_HIP, PoseKeypoint.LEFT_KNEE
    )

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        "standing"
    ]


def test_standing_without_a_visible_leg_is_not_decided(scene: GpsrScene) -> None:
    """
    Standing without a visible leg is not decided.
    """
    annotator = HumanStandingAnnotator(descriptor=HumanStandingAnnotator.Descriptor())
    posture = BodyPosture.standing().with_confidence(
        unreliable_confidence(annotator),
        PoseKeypoint.LEFT_HIP,
        PoseKeypoint.LEFT_KNEE,
        PoseKeypoint.RIGHT_HIP,
        PoseKeypoint.RIGHT_KNEE,
    )

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == []


def test_person_too_small_in_the_image_is_not_standing(scene: GpsrScene) -> None:
    """
    Person too small in the image is not standing.
    """
    posture = BodyPosture.standing().scaled(0.05)

    assert (
        scene.decide_activities(HumanStandingAnnotator(), posture.keypoint_annotation())
        == []
    )


# %% sitting


def test_sitting_person_is_sitting(scene: GpsrScene) -> None:
    """
    A person in a sitting body pose is decided as sitting.
    """
    assert scene.decide_activities(
        HumanSittingAnnotator(), BodyPosture.sitting().keypoint_annotation()
    ) == ["sitting"]


@pytest.mark.parametrize(
    "posture", [BodyPosture.standing, BodyPosture.lying], ids=["standing", "lying"]
)
def test_person_in_another_posture_is_not_sitting(
    scene: GpsrScene, posture: Callable[[], BodyPosture]
) -> None:
    """
    Person in another posture is not sitting.
    """
    assert (
        scene.decide_activities(
            HumanSittingAnnotator(), posture().keypoint_annotation()
        )
        == []
    )


def test_one_visible_side_is_enough_to_sit(scene: GpsrScene) -> None:
    """
    One visible side is enough to sit.
    """
    annotator = HumanSittingAnnotator(descriptor=HumanSittingAnnotator.Descriptor())
    posture = BodyPosture.sitting().with_confidence(
        unreliable_confidence(annotator),
        PoseKeypoint.LEFT_HIP,
        PoseKeypoint.LEFT_KNEE,
        PoseKeypoint.LEFT_ANKLE,
    )

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        "sitting"
    ]


def test_sitting_without_a_visible_side_is_not_decided(scene: GpsrScene) -> None:
    """
    Sitting without a visible side is not decided.
    """
    annotator = HumanSittingAnnotator(descriptor=HumanSittingAnnotator.Descriptor())
    posture = BodyPosture.sitting().with_confidence(
        unreliable_confidence(annotator),
        PoseKeypoint.LEFT_HIP,
        PoseKeypoint.LEFT_KNEE,
        PoseKeypoint.LEFT_ANKLE,
        PoseKeypoint.RIGHT_HIP,
        PoseKeypoint.RIGHT_KNEE,
        PoseKeypoint.RIGHT_ANKLE,
    )

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == []


def test_knees_above_the_hips_are_not_sitting(scene: GpsrScene) -> None:
    """
    Knees above the hips are not sitting.
    """
    sitting = BodyPosture.sitting()
    above_the_hips = sitting.keypoints[PoseKeypoint.LEFT_HIP].y - 10
    posture = sitting.moved_to_height(
        above_the_hips, PoseKeypoint.LEFT_KNEE, PoseKeypoint.RIGHT_KNEE
    )

    assert (
        scene.decide_activities(HumanSittingAnnotator(), posture.keypoint_annotation())
        == []
    )


def test_person_too_small_in_the_image_is_not_sitting(scene: GpsrScene) -> None:
    """
    Person too small in the image is not sitting.
    """
    posture = BodyPosture.sitting().scaled(0.1)

    assert (
        scene.decide_activities(HumanSittingAnnotator(), posture.keypoint_annotation())
        == []
    )


def sitting_with_knees_lowered_to(fraction_of_leg: float) -> BodyPosture:
    """
    :param fraction_of_leg: The fraction of the way from the hips to the ankles.
    :return: The sitting person with both knees lowered to the given fraction of the way
        from the hips down to the ankles.
    """
    sitting = BodyPosture.sitting()
    hip_y = sitting.keypoints[PoseKeypoint.LEFT_HIP].y
    ankle_y = sitting.keypoints[PoseKeypoint.LEFT_ANKLE].y
    knee_y = hip_y + fraction_of_leg * (ankle_y - hip_y)
    return sitting.moved_to_height(
        knee_y, PoseKeypoint.LEFT_KNEE, PoseKeypoint.RIGHT_KNEE
    )


def test_knees_at_the_largest_hip_to_knee_drop_are_sitting(scene: GpsrScene) -> None:
    """
    Knees at the largest hip to knee drop are sitting.
    """
    annotator = HumanSittingAnnotator(descriptor=HumanSittingAnnotator.Descriptor())
    largest_drop = annotator.descriptor.parameters.maximum_relative_hip_knee_height

    posture = sitting_with_knees_lowered_to(largest_drop)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        "sitting"
    ]


def test_knees_beyond_the_largest_hip_to_knee_drop_are_not_sitting(
    scene: GpsrScene,
) -> None:
    """
    Knees beyond the largest hip to knee drop are not sitting.
    """
    annotator = HumanSittingAnnotator(descriptor=HumanSittingAnnotator.Descriptor())
    largest_drop = annotator.descriptor.parameters.maximum_relative_hip_knee_height

    posture = sitting_with_knees_lowered_to(largest_drop + 0.01)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == []


def test_ankles_above_the_knees_are_not_sitting(scene: GpsrScene) -> None:
    """
    Ankles above the knees are not sitting.
    """
    sitting = BodyPosture.sitting()
    above_the_hips = sitting.keypoints[PoseKeypoint.LEFT_HIP].y - 20
    posture = sitting.moved_to_height(
        above_the_hips, PoseKeypoint.LEFT_ANKLE, PoseKeypoint.RIGHT_ANKLE
    )

    assert (
        scene.decide_activities(HumanSittingAnnotator(), posture.keypoint_annotation())
        == []
    )


def test_shins_just_longer_than_the_least_knee_to_ankle_drop_are_sitting(
    scene: GpsrScene,
) -> None:
    """
    Shins just longer than the least knee to ankle drop are sitting.
    """
    annotator = HumanSittingAnnotator(descriptor=HumanSittingAnnotator.Descriptor())
    annotator.descriptor.parameters.minimum_relative_knee_ankle_height = 0.7
    least_drop = annotator.descriptor.parameters.minimum_relative_knee_ankle_height

    posture = sitting_with_knees_lowered_to(1.0 - least_drop - 0.01)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        "sitting"
    ]


def test_shins_shorter_than_the_least_knee_to_ankle_drop_are_not_sitting(
    scene: GpsrScene,
) -> None:
    """
    Shins shorter than the least knee to ankle drop are not sitting.
    """
    annotator = HumanSittingAnnotator(descriptor=HumanSittingAnnotator.Descriptor())
    annotator.descriptor.parameters.minimum_relative_knee_ankle_height = 0.7
    least_drop = annotator.descriptor.parameters.minimum_relative_knee_ankle_height

    posture = sitting_with_knees_lowered_to(1.0 - least_drop + 0.01)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == []


# %% sitting with the knee at hip height


@pytest.fixture
def knee_rule_annotator() -> HumanSittingAnnotator:
    """
    :return: A sitting rule that also counts a knee at hip height as sitting.
    """
    annotator = HumanSittingAnnotator(descriptor=HumanSittingAnnotator.Descriptor())
    annotator.descriptor.parameters.knee_at_hip_height_is_sitting = True
    return annotator


def sitting_without_ankles(annotator: HumanSittingAnnotator) -> BodyPosture:
    """
    :param annotator: The sitting rule.
    :return: The sitting person with both ankles below the trusted confidence.
    """
    return BodyPosture.sitting().with_confidence(
        unreliable_confidence(annotator),
        PoseKeypoint.LEFT_ANKLE,
        PoseKeypoint.RIGHT_ANKLE,
    )


def sitting_without_ankles_with_knee_drop(
    annotator: HumanSittingAnnotator, knee_drop_ratio: float
) -> BodyPosture:
    """
    :param annotator: The sitting rule.
    :param knee_drop_ratio: How far the knees are below the hips, relative to the
        vertical distance of the hips below the shoulders.
    :return: The sitting person without ankles and with both knees at that drop.
    """
    posture = sitting_without_ankles(annotator)
    hip_y = posture.keypoints[PoseKeypoint.LEFT_HIP].y
    torso_height = hip_y - posture.keypoints[PoseKeypoint.LEFT_SHOULDER].y
    return posture.moved_to_height(
        hip_y + knee_drop_ratio * torso_height,
        PoseKeypoint.LEFT_KNEE,
        PoseKeypoint.RIGHT_KNEE,
    )


def test_knee_at_hip_height_sits_without_ankles(
    scene: GpsrScene, knee_rule_annotator: HumanSittingAnnotator
) -> None:
    """
    With the knee rule, a person with hidden ankles and the knees at hip height sits.
    """
    posture = sitting_without_ankles(knee_rule_annotator)

    assert scene.decide_activities(
        knee_rule_annotator, posture.keypoint_annotation()
    ) == ["sitting"]


def test_knee_rule_is_off_by_default(scene: GpsrScene) -> None:
    """
    Without the knee rule, a person with hidden ankles is not decided as sitting.
    """
    annotator = HumanSittingAnnotator(descriptor=HumanSittingAnnotator.Descriptor())
    posture = sitting_without_ankles(annotator)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == []


def test_knees_above_the_hips_sit_by_the_knee_rule(
    scene: GpsrScene, knee_rule_annotator: HumanSittingAnnotator
) -> None:
    """
    With the knee rule, knees slightly above the hips sit.
    """
    posture = sitting_without_ankles_with_knee_drop(knee_rule_annotator, -0.2)

    assert scene.decide_activities(
        knee_rule_annotator, posture.keypoint_annotation()
    ) == ["sitting"]


def test_knees_at_the_largest_knee_drop_sit_by_the_knee_rule(
    scene: GpsrScene, knee_rule_annotator: HumanSittingAnnotator
) -> None:
    """
    With the knee rule, knees exactly the largest knee drop below the hips sit.
    """
    largest_drop = knee_rule_annotator.descriptor.parameters.maximum_knee_drop_ratio
    posture = sitting_without_ankles_with_knee_drop(knee_rule_annotator, largest_drop)

    assert scene.decide_activities(
        knee_rule_annotator, posture.keypoint_annotation()
    ) == ["sitting"]


def test_knees_beyond_the_largest_knee_drop_do_not_sit_by_the_knee_rule(
    scene: GpsrScene, knee_rule_annotator: HumanSittingAnnotator
) -> None:
    """
    With the knee rule, knees farther below the hips than the largest knee drop do not
    sit.
    """
    largest_drop = knee_rule_annotator.descriptor.parameters.maximum_knee_drop_ratio
    posture = sitting_without_ankles_with_knee_drop(
        knee_rule_annotator, largest_drop + 0.01
    )

    assert (
        scene.decide_activities(knee_rule_annotator, posture.keypoint_annotation())
        == []
    )


def test_one_shoulder_is_enough_for_the_knee_rule(
    scene: GpsrScene, knee_rule_annotator: HumanSittingAnnotator
) -> None:
    """
    The knee rule judges the torso on the one reliable shoulder.
    """
    posture = sitting_without_ankles(knee_rule_annotator).with_confidence(
        unreliable_confidence(knee_rule_annotator), PoseKeypoint.LEFT_SHOULDER
    )

    assert scene.decide_activities(
        knee_rule_annotator, posture.keypoint_annotation()
    ) == ["sitting"]


def test_knee_rule_needs_a_shoulder(
    scene: GpsrScene, knee_rule_annotator: HumanSittingAnnotator
) -> None:
    """
    Without a reliable shoulder, the knee rule does not decide a person as sitting.
    """
    posture = sitting_without_ankles(knee_rule_annotator).with_confidence(
        unreliable_confidence(knee_rule_annotator),
        PoseKeypoint.LEFT_SHOULDER,
        PoseKeypoint.RIGHT_SHOULDER,
    )

    assert (
        scene.decide_activities(knee_rule_annotator, posture.keypoint_annotation())
        == []
    )


def test_knee_rule_skips_a_too_short_torso(
    scene: GpsrScene, knee_rule_annotator: HumanSittingAnnotator
) -> None:
    """
    The knee rule does not judge a leg whose hip is less than the smallest torso height
    below the shoulders.
    """
    posture = sitting_without_ankles(knee_rule_annotator)
    hip_y = posture.keypoints[PoseKeypoint.LEFT_HIP].y
    smallest_torso = knee_rule_annotator.descriptor.parameters.minimum_torso_height
    posture = posture.moved_to_height(
        hip_y - smallest_torso / 2,
        PoseKeypoint.LEFT_SHOULDER,
        PoseKeypoint.RIGHT_SHOULDER,
    )

    assert (
        scene.decide_activities(knee_rule_annotator, posture.keypoint_annotation())
        == []
    )


@pytest.mark.parametrize(
    "posture", [BodyPosture.standing, BodyPosture.lying], ids=["standing", "lying"]
)
def test_person_in_another_posture_does_not_sit_by_the_knee_rule(
    scene: GpsrScene,
    knee_rule_annotator: HumanSittingAnnotator,
    posture: Callable[[], BodyPosture],
) -> None:
    """
    With the knee rule, a standing or lying person is still not sitting.
    """
    assert (
        scene.decide_activities(knee_rule_annotator, posture().keypoint_annotation())
        == []
    )


# %% lying


def test_lying_person_is_lying(scene: GpsrScene) -> None:
    """
    A person in a lying body pose is decided as lying.
    """
    assert scene.decide_activities(
        HumanLyingAnnotator(), BodyPosture.lying().keypoint_annotation()
    ) == ["lying"]


@pytest.mark.parametrize(
    "posture", [BodyPosture.standing, BodyPosture.sitting], ids=["standing", "sitting"]
)
def test_person_in_another_posture_is_not_lying(
    scene: GpsrScene, posture: Callable[[], BodyPosture]
) -> None:
    """
    Person in another posture is not lying.
    """
    assert (
        scene.decide_activities(HumanLyingAnnotator(), posture().keypoint_annotation())
        == []
    )


def test_one_knee_at_hip_height_is_enough_to_lie(scene: GpsrScene) -> None:
    """
    One knee at hip height is enough to lie.
    """
    lying = BodyPosture.lying()
    far_below_the_hips = lying.keypoints[PoseKeypoint.LEFT_HIP].y + 20
    posture = lying.moved_to_height(far_below_the_hips, PoseKeypoint.LEFT_KNEE)

    assert scene.decide_activities(
        HumanLyingAnnotator(), posture.keypoint_annotation()
    ) == ["lying"]


def test_knees_far_below_the_hips_are_not_lying(scene: GpsrScene) -> None:
    """
    Knees far below the hips are not lying.
    """
    lying = BodyPosture.lying()
    far_below_the_hips = lying.keypoints[PoseKeypoint.LEFT_HIP].y + 20
    posture = lying.moved_to_height(
        far_below_the_hips, PoseKeypoint.LEFT_KNEE, PoseKeypoint.RIGHT_KNEE
    )

    assert (
        scene.decide_activities(HumanLyingAnnotator(), posture.keypoint_annotation())
        == []
    )


def test_head_far_from_the_shoulders_is_not_lying(scene: GpsrScene) -> None:
    """
    Head far from the shoulders is not lying.
    """
    lying = BodyPosture.lying()
    far_above_the_shoulders = lying.keypoints[PoseKeypoint.LEFT_SHOULDER].y - 15
    posture = lying.moved_to_height(far_above_the_shoulders, PoseKeypoint.NOSE)

    assert (
        scene.decide_activities(HumanLyingAnnotator(), posture.keypoint_annotation())
        == []
    )


def test_person_too_small_in_the_image_is_not_lying(scene: GpsrScene) -> None:
    """
    Person too small in the image is not lying.
    """
    posture = BodyPosture.lying().scaled(0.5)

    assert (
        scene.decide_activities(HumanLyingAnnotator(), posture.keypoint_annotation())
        == []
    )


def test_lying_is_not_decided_without_visible_legs(scene: GpsrScene) -> None:
    """
    Lying is not decided without visible legs.
    """
    annotator = HumanLyingAnnotator(descriptor=HumanLyingAnnotator.Descriptor())
    posture = BodyPosture.lying().with_confidence(
        unreliable_confidence(annotator),
        PoseKeypoint.LEFT_HIP,
        PoseKeypoint.LEFT_KNEE,
        PoseKeypoint.LEFT_ANKLE,
        PoseKeypoint.RIGHT_HIP,
        PoseKeypoint.RIGHT_KNEE,
        PoseKeypoint.RIGHT_ANKLE,
    )

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == []


# %% every posture rule


@dataclass
class PostureRule:
    """
    A posture rule together with a person holding the posture it decides.
    """

    annotator_class: type[BaseAnnotator]
    """
    The annotator that applies the rule.
    """

    posture: Callable[[], BodyPosture]
    """
    Builds a person holding the posture the rule decides.
    """

    posture_name: str
    """
    The name the rule annotates the posture with.
    """

    height_keypoints: tuple[PoseKeypoint, ...]
    """
    The keypoints whose vertical extent the rule takes as the height of the person.
    """

    smallest_decided_height: Callable[[BaseAnnotator], float]
    """
    Reads from an annotator of the rule the smallest height of a person in the image, in
    pixels, that the rule still decides.
    """


@pytest.fixture(
    params=[
        PostureRule(
            HumanStandingAnnotator,
            BodyPosture.standing,
            "standing",
            (PoseKeypoint.NOSE, PoseKeypoint.LEFT_KNEE),
            lambda annotator: annotator.descriptor.parameters.minimum_body_height,
        ),
        PostureRule(
            HumanSittingAnnotator,
            BodyPosture.sitting,
            "sitting",
            (PoseKeypoint.LEFT_HIP, PoseKeypoint.LEFT_ANKLE),
            lambda annotator: annotator.descriptor.parameters.minimum_leg_height,
        ),
        PostureRule(
            HumanLyingAnnotator,
            BodyPosture.lying,
            "lying",
            (PoseKeypoint.NOSE, PoseKeypoint.RIGHT_ANKLE),
            lambda annotator: annotator.descriptor.parameters.minimum_body_height,
        ),
    ],
    ids=["standing", "sitting", "lying"],
)
def posture_rule(request: pytest.FixtureRequest) -> PostureRule:
    """
    :param request: Holds the posture rule of this run.
    :return: Each posture rule in turn.
    """
    return request.param


def test_person_without_keypoints_is_skipped(
    scene: GpsrScene,
    posture_rule: PostureRule,
) -> None:
    """
    Person without keypoints is skipped.
    """
    annotator = posture_rule.annotator_class()
    scene.place_in_pipeline(annotator)
    human_hypothesis = scene.add_human()

    status = annotator.update()

    assert status == Status.SUCCESS
    assert scene.activities_of(human_hypothesis) == []


def test_person_with_only_keypoint_clouds_is_skipped(
    scene: GpsrScene,
    posture_rule: PostureRule,
) -> None:
    """
    Person with only keypoint clouds is skipped.
    """
    annotator = posture_rule.annotator_class()
    scene.place_in_pipeline(annotator)
    human_hypothesis = scene.add_human(BodyInSpace().keypoint_annotation())

    annotator.update()

    assert scene.activities_of(human_hypothesis) == []


def test_every_person_is_decided_on_their_own(
    scene: GpsrScene,
    posture_rule: PostureRule,
) -> None:
    """
    Every person is decided on their own.
    """
    annotator = posture_rule.annotator_class()
    scene.place_in_pipeline(annotator)
    holding_the_posture = scene.add_human(posture_rule.posture().keypoint_annotation())
    too_small = scene.add_human(
        posture_rule.posture().scaled(0.01).keypoint_annotation()
    )

    annotator.update()

    assert scene.activities_of(holding_the_posture) == [posture_rule.posture_name]
    assert scene.activities_of(too_small) == []


def test_posture_names_the_rule_as_its_source(
    scene: GpsrScene,
    posture_rule: PostureRule,
) -> None:
    """
    Posture names the rule as its source.
    """
    annotator = posture_rule.annotator_class()
    scene.place_in_pipeline(annotator)
    human_hypothesis = scene.add_human(posture_rule.posture().keypoint_annotation())

    annotator.update()

    assert [
        activity.source
        for activity in CAS.filter_by_type(
            ActivityAnnotation, human_hypothesis.annotations
        )
    ] == [posture_rule.annotator_class.__name__]


def test_person_exactly_the_smallest_decided_height_is_decided(
    scene: GpsrScene,
    posture_rule: PostureRule,
) -> None:
    """
    Person exactly the smallest decided height is decided.
    """
    annotator = posture_rule.annotator_class()
    posture = posture_rule.posture()
    heights = [
        posture.keypoints[keypoint].y for keypoint in posture_rule.height_keypoints
    ]
    posture = posture.scaled(
        posture_rule.smallest_decided_height(annotator) / (max(heights) - min(heights))
    )

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        posture_rule.posture_name
    ]
