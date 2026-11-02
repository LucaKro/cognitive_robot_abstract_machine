"""
Tests for deciding from the image keypoints of a person whether they wave or raise an
arm.
"""

from __future__ import annotations

# %% Imports
import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.waving import HumanWavingAnnotator
from robokudo.annotators.human_and_pose import PoseKeypoint

from ._body_postures import BodyPosture, BodySide
from ._gpsr_scene import GpsrScene

# %% gesture heights


def waving_annotator() -> HumanWavingAnnotator:
    """
    :return: A waving annotator with its own default parameters.
    """
    return HumanWavingAnnotator(descriptor=HumanWavingAnnotator.Descriptor())


def offset_above(
    annotator: HumanWavingAnnotator, posture: BodyPosture, side: BodySide
) -> float:
    """
    :return: How far a wrist has to be above eye or shoulder level to count, in pixels.
    """
    shoulder = posture.keypoints[side.shoulder]
    eye = posture.keypoints[side.eye]
    return (shoulder.y - eye.y) * annotator.descriptor.parameters.waving_decision_factor


def wrist_above_the_eyes(
    annotator: HumanWavingAnnotator, side: BodySide
) -> BodyPosture:
    """
    :return: A standing person holding one wrist above eye level.
    """
    standing = BodyPosture.standing()
    eye_y = standing.keypoints[side.eye].y
    return standing.moved_to_height(
        eye_y - offset_above(annotator, standing, side) - 1, side.wrist
    )


def wrist_above_the_shoulder(
    annotator: HumanWavingAnnotator, side: BodySide
) -> BodyPosture:
    """
    :return: A standing person holding one wrist above the shoulder but below the eyes.
    """
    standing = BodyPosture.standing()
    shoulder_y = standing.keypoints[side.shoulder].y
    return standing.moved_to_height(
        shoulder_y - offset_above(annotator, standing, side) - 1, side.wrist
    )


# %% waving


def test_left_wrist_above_the_eyes_waves_and_raises_the_left_arm(
    scene: GpsrScene,
) -> None:
    annotator = waving_annotator()
    posture = wrist_above_the_eyes(annotator, BodySide.LEFT)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        "raising left arm",
        "waving",
    ]


def test_right_wrist_above_the_eyes_waves_and_raises_the_right_arm(
    scene: GpsrScene,
) -> None:
    annotator = waving_annotator()
    posture = wrist_above_the_eyes(annotator, BodySide.RIGHT)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        "raising right arm",
        "waving",
    ]


@pytest.mark.parametrize(
    "side, raised_arm",
    [(BodySide.LEFT, "raising left arm"), (BodySide.RIGHT, "raising right arm")],
    ids=["left", "right"],
)
def test_wrist_exactly_at_the_waving_height_does_not_wave(
    scene: GpsrScene, side: BodySide, raised_arm: str
) -> None:
    annotator = waving_annotator()
    standing = BodyPosture.standing()
    eye_y = standing.keypoints[side.eye].y
    posture = standing.moved_to_height(
        eye_y - offset_above(annotator, standing, side), side.wrist
    )

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        raised_arm
    ]


# %% raising an arm


def test_left_wrist_above_the_shoulder_raises_the_left_arm(scene: GpsrScene) -> None:
    annotator = waving_annotator()
    posture = wrist_above_the_shoulder(annotator, BodySide.LEFT)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        "raising left arm"
    ]


def test_right_wrist_above_the_shoulder_raises_the_right_arm(scene: GpsrScene) -> None:
    annotator = waving_annotator()
    posture = wrist_above_the_shoulder(annotator, BodySide.RIGHT)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        "raising right arm"
    ]


def test_both_arms_raised_reports_only_the_left_arm(scene: GpsrScene) -> None:
    annotator = waving_annotator()
    left_raised = wrist_above_the_shoulder(annotator, BodySide.LEFT)
    right_wrist_y = (
        wrist_above_the_shoulder(annotator, BodySide.RIGHT)
        .keypoints[PoseKeypoint.RIGHT_WRIST]
        .y
    )
    posture = left_raised.moved_to_height(right_wrist_y, BodySide.RIGHT.wrist)

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == [
        "raising left arm"
    ]


@pytest.mark.parametrize("side", [BodySide.LEFT, BodySide.RIGHT], ids=["left", "right"])
def test_wrist_exactly_at_the_raising_height_does_not_raise(
    scene: GpsrScene, side: BodySide
) -> None:
    annotator = waving_annotator()
    standing = BodyPosture.standing()
    shoulder_y = standing.keypoints[side.shoulder].y
    posture = standing.moved_to_height(
        shoulder_y - offset_above(annotator, standing, side), side.wrist
    )

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == []


def test_arms_down_make_no_gesture(scene: GpsrScene) -> None:
    assert (
        scene.decide_activities(
            waving_annotator(), BodyPosture.standing().keypoint_annotation()
        )
        == []
    )


def test_wrist_at_the_confidence_threshold_makes_no_gesture(scene: GpsrScene) -> None:
    annotator = waving_annotator()
    threshold = annotator.descriptor.parameters.keypoint_confidence_threshold
    posture = wrist_above_the_eyes(annotator, BodySide.LEFT).with_confidence(
        threshold, PoseKeypoint.LEFT_WRIST
    )

    assert scene.decide_activities(annotator, posture.keypoint_annotation()) == []


# %% persons to decide


def test_person_without_keypoints_is_skipped(scene: GpsrScene) -> None:
    annotator = waving_annotator()
    scene.place_in_pipeline(annotator)
    person = scene.add_human()

    status = annotator.update()

    assert status == Status.SUCCESS
    assert scene.activities_of(person, HumanWavingAnnotator) == []


def test_activity_check_decides_only_persons_without_gesture(scene: GpsrScene) -> None:
    annotator = waving_annotator()
    annotator.descriptor.parameters.check_activity = True
    scene.place_in_pipeline(annotator)
    raising = wrist_above_the_shoulder(annotator, BodySide.LEFT).keypoint_annotation()
    without_gesture = scene.add_human(
        raising,
        GpsrScene.clip_activity("No gesture"),
    )
    already_pointing = scene.add_human(
        wrist_above_the_shoulder(annotator, BodySide.LEFT).keypoint_annotation(),
        GpsrScene.clip_activity("Pointing"),
    )

    annotator.update()

    assert scene.activities_of(without_gesture, HumanWavingAnnotator) == [
        "raising left arm"
    ]
    assert scene.activities_of(already_pointing, HumanWavingAnnotator) == []


def test_activity_check_without_persons_without_gesture_decides_nobody(
    scene: GpsrScene,
) -> None:
    annotator = waving_annotator()
    annotator.descriptor.parameters.check_activity = True
    scene.place_in_pipeline(annotator)
    person = scene.add_human(
        wrist_above_the_shoulder(annotator, BodySide.LEFT).keypoint_annotation()
    )

    status = annotator.update()

    assert status == Status.SUCCESS
    assert scene.activities_of(person, HumanWavingAnnotator) == []
