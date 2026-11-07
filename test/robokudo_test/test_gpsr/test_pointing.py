"""
Tests for deciding from the keypoint clouds of a person whether and to which side they
point.

Every keypoint cloud holds a single point, so its centroid is the keypoint position the
test chose.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr import pointing as pointing_module
from robokudo.annotators.gpsr.human_activity_annotator import HumanGesture
from robokudo.annotators.gpsr.pointing import HumanPointingAnnotator
from robokudo.annotators.human_and_pose import PoseKeypoint
from robokudo.exceptions import UnknownMode
from typing_extensions import TYPE_CHECKING, Iterator

from ._body_postures import BodyInSpace, BodySide, Forearm
from ._gpsr_scene import GpsrScene

if TYPE_CHECKING:
    from rclpy.node import Node

# %% annotator and arm poses


def test_unknown_pointing_target_is_rejected(
    node: Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Reject unsupported pointing target shapes.
    """
    monkeypatch.setattr(pointing_module, "Node", lambda node_name: node)
    descriptor = HumanPointingAnnotator.Descriptor()
    descriptor.parameters.pointing_target_structures_type = "unknown"
    with pytest.raises(UnknownMode):
        HumanPointingAnnotator(descriptor=descriptor)


@pytest.fixture
def pointing() -> Iterator[HumanPointingAnnotator]:
    """
    :return: An iterator over a pointing annotator with its own default parameters, whose
        ROS node is destroyed after the test.
    """
    annotator = HumanPointingAnnotator(descriptor=HumanPointingAnnotator.Descriptor())
    yield annotator
    annotator.node.destroy_node()


def body_with_left_wrist_beside(
    body: BodyInSpace, center: np.ndarray, distance: float
) -> BodyInSpace:
    """
    :param body: The body to change.
    :param center: The point the wrist is beside.
    :param distance: The distance of the wrist from it.
    :return: A copy of the body holding the left forearm horizontally, the wrist the
        given distance beside the center, and the right wrist hanging close to the
        center.
    """
    wrist = center + np.array([distance, 0.0, 0.0])
    return body.with_arm(
        BodySide.LEFT,
        Forearm(elbow=list(wrist - np.array([0.25, 0.0, 0.0])), wrist=list(wrist)),
    ).with_arm(
        BodySide.RIGHT,
        Forearm(
            elbow=list(center + np.array([-0.1, -0.15, 0.0])),
            wrist=list(center + np.array([-0.1, 0.05, 0.0])),
        ),
    )


def body_with_left_wrist_beside_the_torso(distance: float) -> BodyInSpace:
    """
    :param distance: The distance of the wrist from the torso center.
    :return: A person holding the left forearm horizontally, the wrist the given
        distance beside the torso center, and the right wrist hanging close to the
        torso.
    """
    body = BodyInSpace()
    return body_with_left_wrist_beside(body, body.torso_center(), distance)


def body_with_raised_left_arm() -> BodyInSpace:
    """
    :return: A person holding the left wrist above the shoulders.
    """
    return BodyInSpace().with_arm(
        BodySide.LEFT, Forearm(elbow=[0.45, -0.1, 2.0], wrist=[0.7, -0.2, 2.0])
    )


# %% pointing directions


def test_left_arm_out_to_the_image_right_points_left(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Left arm out to the image right points left.
    """
    assert scene.decide_activities(
        pointing, BodyInSpace().with_arm_held_out(BodySide.LEFT).keypoint_annotation()
    ) == ["pointing left"]


def test_right_arm_out_to_the_image_left_points_right(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Right arm out to the image left points right.
    """
    assert scene.decide_activities(
        pointing, BodyInSpace().with_arm_held_out(BodySide.RIGHT).keypoint_annotation()
    ) == ["pointing right"]


def test_arms_down_do_not_point(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Arms down do not point.
    """
    assert scene.decide_activities(pointing, BodyInSpace().keypoint_annotation()) == []


def test_forearm_just_inside_the_pointing_angle_points(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Forearm just inside the pointing angle points.
    """
    largest_angle = pointing.descriptor.parameters.maximum_pointing_angle
    body = BodyInSpace().with_arm_held_out(BodySide.LEFT, largest_angle - 1.0)

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == [
        "pointing left"
    ]


def test_forearm_just_beyond_the_pointing_angle_does_not_point(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Forearm just beyond the pointing angle does not point.
    """
    largest_angle = pointing.descriptor.parameters.maximum_pointing_angle
    body = BodyInSpace().with_arm_held_out(BodySide.LEFT, largest_angle + 1.0)

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == []


def test_wrist_above_the_shoulder_does_not_point(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Wrist above the shoulder does not point.
    """
    assert (
        scene.decide_activities(
            pointing, body_with_raised_left_arm().keypoint_annotation()
        )
        == []
    )


def test_without_shoulders_a_raised_arm_points(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Without shoulders a raised arm points.
    """
    body = body_with_raised_left_arm().without(
        PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.RIGHT_SHOULDER
    )

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == [
        "pointing left"
    ]


@pytest.mark.parametrize(
    "left_lowered_by, right_lowered_by, direction",
    [(0.0, 30.0, "pointing left"), (30.0, 0.0, "pointing right")],
    ids=["left arm higher", "right arm higher"],
)
def test_higher_arm_decides_the_direction(
    scene: GpsrScene,
    pointing: HumanPointingAnnotator,
    left_lowered_by: float,
    right_lowered_by: float,
    direction: str,
) -> None:
    """
    Higher arm decides the direction.
    """
    body = (
        BodyInSpace()
        .with_arm_held_out(BodySide.LEFT, left_lowered_by)
        .with_arm_held_out(BodySide.RIGHT, right_lowered_by)
    )

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == [direction]


def test_arms_held_out_equally_high_point_with_the_left_arm(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Arms held out equally high point with the left arm.
    """
    body = (
        BodyInSpace().with_arm_held_out(BodySide.LEFT).with_arm_held_out(BodySide.RIGHT)
    )

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == [
        "pointing left"
    ]


# %% arm extension


def test_wrists_just_inside_the_arm_extension_do_not_point(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Wrists just inside the arm extension do not point.
    """
    pointing.descriptor.parameters.minimum_arm_extension = 0.25

    body = body_with_left_wrist_beside_the_torso(
        pointing.descriptor.parameters.minimum_arm_extension - 0.01
    )

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == []


def test_wrist_just_beyond_the_arm_extension_points(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Wrist just beyond the arm extension points.
    """
    pointing.descriptor.parameters.minimum_arm_extension = 0.25

    body = body_with_left_wrist_beside_the_torso(
        pointing.descriptor.parameters.minimum_arm_extension + 0.01
    )

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == [
        "pointing left"
    ]


def test_no_arm_extension_lets_wrists_near_the_torso_point(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    With the minimum arm extension at zero, a wrist near the torso can point.
    """
    pointing.descriptor.parameters.minimum_arm_extension = 0.0

    body = body_with_left_wrist_beside_the_torso(0.24)

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == [
        "pointing left"
    ]


def test_without_hips_the_arm_extension_is_measured_from_the_shoulders(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Without hips the arm extension is measured from the shoulders.
    """
    pointing.descriptor.parameters.minimum_arm_extension = 0.25
    body = (
        BodyInSpace()
        .without(PoseKeypoint.LEFT_HIP, PoseKeypoint.RIGHT_HIP)
        .with_arm(
            BodySide.LEFT, Forearm(elbow=[-0.05, 0.05, 2.0], wrist=[0.2, 0.05, 2.0])
        )
        .with_arm(
            BodySide.RIGHT, Forearm(elbow=[-0.1, -0.15, 2.0], wrist=[-0.1, 0.05, 2.0])
        )
    )

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == []


@pytest.mark.parametrize(
    "missing_hip",
    [PoseKeypoint.RIGHT_HIP, PoseKeypoint.LEFT_HIP],
    ids=["right hip missing", "left hip missing"],
)
def test_with_one_hip_the_arm_extension_is_measured_from_the_shoulders(
    scene: GpsrScene,
    pointing: HumanPointingAnnotator,
    missing_hip: PoseKeypoint,
) -> None:
    """
    With one hip the arm extension is measured from the shoulders.
    """
    pointing.descriptor.parameters.minimum_arm_extension = 0.25
    body = BodyInSpace().without(missing_hip)
    between_the_shoulders = np.mean(
        [
            body.positions[PoseKeypoint.LEFT_SHOULDER],
            body.positions[PoseKeypoint.RIGHT_SHOULDER],
        ],
        axis=0,
    )

    body = body_with_left_wrist_beside(
        body,
        between_the_shoulders,
        pointing.descriptor.parameters.minimum_arm_extension - 0.01,
    )

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == []


# %% persons to decide


def test_person_without_keypoint_clouds_is_skipped(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Person without keypoint clouds is skipped.
    """
    scene.place_in_pipeline(pointing)
    human_hypothesis = scene.add_human()

    status = pointing.update()

    assert status == Status.SUCCESS
    assert scene.activities_of(human_hypothesis, HumanPointingAnnotator) == []


@pytest.mark.parametrize(
    "missing",
    [PoseKeypoint.LEFT_ELBOW, PoseKeypoint.RIGHT_WRIST],
    ids=["left elbow", "right wrist"],
)
def test_person_with_an_unmeasured_arm_keypoint_is_skipped(
    scene: GpsrScene, pointing: HumanPointingAnnotator, missing: PoseKeypoint
) -> None:
    """
    Person with an unmeasured arm keypoint is skipped.
    """
    body = BodyInSpace().with_arm_held_out(BodySide.LEFT).without(missing)

    assert scene.decide_activities(pointing, body.keypoint_annotation()) == []


@pytest.mark.parametrize(
    "known", [False, True], ids=["every person", "persons with known gestures"]
)
def test_no_person_is_no_failure(
    scene: GpsrScene, pointing: HumanPointingAnnotator, known: bool
) -> None:
    """
    A scene without persons does not fail the annotator.
    """
    pointing.descriptor.parameters.known = known
    scene.place_in_pipeline(pointing)

    status = pointing.update()

    assert status == Status.SUCCESS
    assert not scene.has_failed()


def test_known_gestures_decide_only_persons_pointing_or_without_gesture(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    With known gestures, only persons tagged pointing or no gesture are decided.
    """
    pointing.descriptor.parameters.known = True
    scene.place_in_pipeline(pointing)
    tagged_pointing = scene.add_human(
        BodyInSpace().with_arm_held_out(BodySide.LEFT).keypoint_annotation(),
        GpsrScene.clip_activity(HumanGesture.POINTING),
    )
    tagged_without_gesture = scene.add_human(
        BodyInSpace().with_arm_held_out(BodySide.LEFT).keypoint_annotation(),
        GpsrScene.clip_activity(HumanGesture.NONE),
    )
    tagged_waving = scene.add_human(
        BodyInSpace().with_arm_held_out(BodySide.LEFT).keypoint_annotation(),
        GpsrScene.clip_activity(HumanGesture.WAVING),
    )

    pointing.update()

    assert scene.activities_of(tagged_pointing, HumanPointingAnnotator) == [
        "pointing left"
    ]
    assert scene.activities_of(tagged_without_gesture, HumanPointingAnnotator) == [
        "pointing left"
    ]
    assert scene.activities_of(tagged_waving, HumanPointingAnnotator) == []


def test_without_known_gestures_only_the_first_person_is_decided(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    Without known gestures only the first person is decided.
    """
    scene.place_in_pipeline(pointing)
    first = scene.add_human(
        BodyInSpace().with_arm_held_out(BodySide.LEFT).keypoint_annotation()
    )
    second = scene.add_human(
        BodyInSpace().with_arm_held_out(BodySide.RIGHT).keypoint_annotation()
    )

    pointing.update()

    assert scene.activities_of(first, HumanPointingAnnotator) == ["pointing left"]
    assert scene.activities_of(second, HumanPointingAnnotator) == []


def test_with_known_gestures_every_person_is_decided(
    scene: GpsrScene, pointing: HumanPointingAnnotator
) -> None:
    """
    With known gestures every person is decided.
    """
    pointing.descriptor.parameters.known = True
    scene.place_in_pipeline(pointing)
    first = scene.add_human(
        BodyInSpace().with_arm_held_out(BodySide.LEFT).keypoint_annotation(),
        GpsrScene.clip_activity(HumanGesture.POINTING),
    )
    second = scene.add_human(
        BodyInSpace().with_arm_held_out(BodySide.RIGHT).keypoint_annotation(),
        GpsrScene.clip_activity(HumanGesture.POINTING),
    )

    pointing.update()

    assert scene.activities_of(first, HumanPointingAnnotator) == ["pointing left"]
    assert scene.activities_of(second, HumanPointingAnnotator) == ["pointing right"]
