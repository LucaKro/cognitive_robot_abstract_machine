"""
Tests for answering the human queries of GPSR with the persons that match them.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.belief_state import BeliefEntry, EntityType
from robokudo.annotators.gpsr.gpsr_results import (
    GenerateHumanActivityResult,
    GenerateHumanPostureResult,
    GeneratePointingResult,
    GenerateVisionLanguageResult,
    GenerateWavingRaisingResult,
    create_camera_header,
)
from robokudo.annotators.gpsr.human_activity_annotator import (
    HumanActivityAnnotator,
    HumanGesture,
)
from robokudo.annotators.gpsr.lying import HumanLyingAnnotator
from robokudo.annotators.gpsr.pointing import HumanPointingAnnotator
from robokudo.annotators.gpsr.sitting import HumanSittingAnnotator
from robokudo.annotators.gpsr.standing import HumanStandingAnnotator
from robokudo.annotators.gpsr.vision_language_annotator import VisionLanguageAnnotator
from robokudo.annotators.gpsr.waving import HumanWavingAnnotator
from robokudo.types.annotation import LocationAnnotation, PositionAnnotation
from robokudo.types.human import ActivityAnnotation
from robokudo.utils.type_conversion import (
    get_geometry_msgs_pose_stamped_from_position_annotation,
)
from robokudo_msgs.msg import ObjectDesignator
from typing_extensions import TYPE_CHECKING, Callable

from ._gpsr_scene import GpsrScene

if TYPE_CHECKING:
    from robokudo.annotators.core import BaseAnnotator

# %% activities as the annotators report them


def sitting_rule() -> ActivityAnnotation:
    """
    :return: Sitting as the keypoint rule reports it.
    """
    return ActivityAnnotation(type="sitting", source=HumanSittingAnnotator.__name__)


def standing_rule() -> ActivityAnnotation:
    """
    :return: Standing as the keypoint rule reports it.
    """
    return ActivityAnnotation(type="standing", source=HumanStandingAnnotator.__name__)


def gesture_rule(gesture: str) -> ActivityAnnotation:
    """
    :param gesture: The gesture.
    :return: A gesture as the waving and arm raising rules report it.
    """
    return ActivityAnnotation(type=gesture, source=HumanWavingAnnotator.__name__)


def pointing_rule(direction: str) -> ActivityAnnotation:
    """
    :param direction: The pointing direction.
    :return: A pointing direction as the pointing rules report it.
    """
    return ActivityAnnotation(type=direction, source=HumanPointingAnnotator.__name__)


def vision_language_answer(answer: str) -> ActivityAnnotation:
    """
    :param answer: The answer of the model.
    :return: An answer as the vision language annotator reports it.
    """
    return ActivityAnnotation(type=answer, source=VisionLanguageAnnotator.__name__)


# %% belief id and position of every answered person


@dataclass
class AnsweredQuery:
    """
    A result annotator and a person it answers the query with.
    """

    result_annotator: type[BaseAnnotator]
    """
    The annotator that builds the answer.
    """

    queried_attribute: list[str]
    """
    The attribute the query asks for.
    """

    matching_activity: Callable[[], ActivityAnnotation]
    """
    Builds the activity that makes a person match the query.
    """


@pytest.fixture(
    params=[
        AnsweredQuery(GenerateHumanPostureResult, ["sitting"], sitting_rule),
        AnsweredQuery(
            GenerateWavingRaisingResult, ["waving"], lambda: gesture_rule("waving")
        ),
        AnsweredQuery(
            GeneratePointingResult,
            ["pointing left"],
            lambda: pointing_rule("pointing left"),
        ),
        AnsweredQuery(
            GenerateHumanActivityResult,
            [],
            lambda: GpsrScene.clip_activity(HumanGesture.WAVING),
        ),
        AnsweredQuery(
            GenerateVisionLanguageResult,
            [],
            lambda: vision_language_answer("blue jeans"),
        ),
    ],
    ids=["posture", "waving", "pointing", "activity", "vision language"],
)
def answered_query(request: pytest.FixtureRequest) -> AnsweredQuery:
    """
    :param request: Holds the result annotator of this run.
    :return: Each result annotator in turn, with a person it answers the query with.
    """
    return request.param


def test_answered_person_carries_its_belief_id(
    scene: GpsrScene, answered_query: AnsweredQuery
) -> None:
    """
    The person's belief identifier appears in the answer.
    """
    scene.query.obj.attribute = answered_query.queried_attribute
    annotator = answered_query.result_annotator()
    scene.place_in_pipeline(annotator)
    belief = BeliefEntry(7, EntityType.PERSON, "human")
    scene.add_human(answered_query.matching_activity(), belief)

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert object_designator.uid == str(belief.id)


def test_answered_person_carries_its_position_in_the_camera_frame(
    scene: GpsrScene, answered_query: AnsweredQuery
) -> None:
    """
    The person's position is answered in the camera frame.
    """
    scene.query.obj.attribute = answered_query.queried_attribute
    annotator = answered_query.result_annotator()
    cas = scene.place_in_pipeline(annotator)
    scene.add_human(answered_query.matching_activity(), position=[1.0, 2.0, 3.0])

    annotator.update()

    [object_designator] = scene.sent_answer().res
    [pose] = object_designator.pose
    position = pose.pose.position
    assert (position.x, position.y, position.z) == (1.0, 2.0, 3.0)
    assert pose.header == create_camera_header(cas)


def test_answered_person_preserves_position_order_and_sources(
    scene: GpsrScene, answered_query: AnsweredQuery
) -> None:
    """
    The answer carries each position and its source in annotation order.
    """
    scene.query.obj.attribute = answered_query.queried_attribute
    annotator = answered_query.result_annotator()
    cas = scene.place_in_pipeline(annotator)
    positions = [
        PositionAnnotation(translation=[1.0, 2.0, 3.0], source="first"),
        PositionAnnotation(translation=[4.0, 5.0, 6.0], source="second"),
    ]
    scene.add_human(answered_query.matching_activity(), *positions)

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert object_designator.pose == [
        get_geometry_msgs_pose_stamped_from_position_annotation(
            position, create_camera_header(cas)
        )
        for position in positions
    ]
    assert object_designator.pose_source == [position.source for position in positions]


def test_answered_person_without_belief_has_empty_id(
    scene: GpsrScene, answered_query: AnsweredQuery
) -> None:
    """
    An answer without a belief entry keeps the default identifier.
    """
    scene.query.obj.attribute = answered_query.queried_attribute
    annotator = answered_query.result_annotator()
    scene.place_in_pipeline(annotator)
    scene.add_human(answered_query.matching_activity())

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert object_designator.uid == ObjectDesignator().uid


# %% camera header


def test_header_names_the_camera_frame_and_the_image_time(scene: GpsrScene) -> None:
    """
    The header identifies the camera frame and image acquisition time.
    """
    scene.timestamp_nanoseconds = 12_000_000_345
    cas = scene.place_in_pipeline(GenerateHumanPostureResult())

    header = create_camera_header(cas)

    assert header.frame_id == str(scene.camera_body.name)
    assert (header.stamp.sec, header.stamp.nanosec) == (12, 345)


# %% posture results


def test_person_in_the_queried_posture_is_answered(scene: GpsrScene) -> None:
    """
    A person whose posture matches the query appears in the answer.
    """
    scene.query.obj.type = "human"
    scene.query.obj.attribute = ["sitting"]
    annotator = GenerateHumanPostureResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(sitting_rule())

    status = annotator.update()

    assert status == Status.SUCCESS
    [object_designator] = scene.sent_answer().res
    assert object_designator.type == "human"
    assert list(object_designator.attribute) == ["sitting"]


def test_person_in_another_posture_is_not_answered(scene: GpsrScene) -> None:
    """
    A person with a different posture does not match the query.
    """
    scene.query.obj.attribute = ["sitting"]
    annotator = GenerateHumanPostureResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(standing_rule())

    annotator.update()

    assert list(scene.sent_answer().res) == []


def test_person_with_two_postures_is_undecided(scene: GpsrScene) -> None:
    """
    Conflicting posture annotations do not produce a match.
    """
    scene.query.obj.attribute = ["sitting"]
    annotator = GenerateHumanPostureResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(sitting_rule(), standing_rule())

    annotator.update()

    assert list(scene.sent_answer().res) == []


def test_person_without_posture_is_undecided(scene: GpsrScene) -> None:
    """
    A person without a posture annotation does not match the query.
    """
    scene.query.obj.attribute = ["sitting"]
    annotator = GenerateHumanPostureResult()
    scene.place_in_pipeline(annotator)
    scene.add_human()

    annotator.update()

    assert list(scene.sent_answer().res) == []


def test_lying_person_answers_a_laying_query(scene: GpsrScene) -> None:
    """
    Translate the lying annotation to the query's posture word.
    """
    scene.query.obj.attribute = ["laying"]
    annotator = GenerateHumanPostureResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(
        ActivityAnnotation(type="lying", source=HumanLyingAnnotator.__name__)
    )

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert list(object_designator.attribute) == ["laying"]


def test_clip_posture_does_not_decide_the_posture(scene: GpsrScene) -> None:
    """
    A CLIP posture alone does not decide the keypoint posture query.
    """
    scene.query.obj.attribute = ["sitting"]
    annotator = GenerateHumanPostureResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(GpsrScene.clip_activity("sitting"))

    annotator.update()

    assert list(scene.sent_answer().res) == []


def test_nobody_in_the_scene_answers_with_nobody(scene: GpsrScene) -> None:
    """
    An empty scene produces an empty posture answer.
    """
    scene.query.obj.attribute = ["sitting"]
    annotator = GenerateHumanPostureResult()
    scene.place_in_pipeline(annotator)

    status = annotator.update()

    assert status == Status.SUCCESS
    assert list(scene.sent_answer().res) == []


# %% waving and arm raising results


def test_person_making_the_queried_gesture_is_answered(scene: GpsrScene) -> None:
    """
    A matching gesture is reported for the person who made it.
    """
    scene.query.obj.type = "human"
    scene.query.obj.attribute = ["waving"]
    annotator = GenerateWavingRaisingResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(gesture_rule("waving"))

    status = annotator.update()

    assert status == Status.SUCCESS
    [object_designator] = scene.sent_answer().res
    assert list(object_designator.description) == ["waving"]


def test_person_making_another_gesture_is_not_answered(scene: GpsrScene) -> None:
    """
    A different gesture does not match the requested gesture.
    """
    scene.query.obj.attribute = ["raising left arm"]
    annotator = GenerateWavingRaisingResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(gesture_rule("waving"))

    annotator.update()

    assert list(scene.sent_answer().res) == []


def test_clip_gesture_does_not_answer_a_waving_query(scene: GpsrScene) -> None:
    """
    A waving query is not answered by a gesture from CLIP alone.
    """
    scene.query.obj.attribute = ["waving"]
    annotator = GenerateWavingRaisingResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(GpsrScene.clip_activity("waving"))

    annotator.update()

    assert list(scene.sent_answer().res) == []


def test_nobody_waving_answers_with_nobody(scene: GpsrScene) -> None:
    """
    Nobody waving answers with nobody.
    """
    scene.query.obj.attribute = ["waving"]
    annotator = GenerateWavingRaisingResult()
    scene.place_in_pipeline(annotator)

    annotator.update()

    assert list(scene.sent_answer().res) == []


# %% pointing results


def test_person_pointing_in_the_queried_direction_is_answered(scene: GpsrScene) -> None:
    """
    Person pointing in the queried direction is answered.
    """
    scene.query.obj.type = "human"
    scene.query.obj.attribute = ["pointing left"]
    annotator = GeneratePointingResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(pointing_rule("pointing left"))

    status = annotator.update()

    assert status == Status.SUCCESS
    [object_designator] = scene.sent_answer().res
    assert list(object_designator.description) == ["Behavior suggests pointing left."]


def test_person_pointing_in_another_direction_is_not_answered(scene: GpsrScene) -> None:
    """
    Person pointing in another direction is not answered.
    """
    scene.query.obj.attribute = ["pointing right"]
    annotator = GeneratePointingResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(pointing_rule("pointing left"))

    annotator.update()

    assert list(scene.sent_answer().res) == []


def test_person_pointing_and_doing_more_is_answered(scene: GpsrScene) -> None:
    """
    Person pointing and doing more is answered.
    """
    scene.query.obj.attribute = ["pointing left"]
    annotator = GeneratePointingResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(pointing_rule("pointing left"), standing_rule())

    annotator.update()

    assert [
        list(object_designator.description)
        for object_designator in scene.sent_answer().res
    ] == [["Behavior suggests pointing left."]]


def test_person_pointing_elsewhere_and_doing_more_is_not_answered(
    scene: GpsrScene,
) -> None:
    """
    Person pointing elsewhere and doing more is not answered.
    """
    scene.query.obj.attribute = ["pointing left"]
    annotator = GeneratePointingResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(pointing_rule("pointing right"), standing_rule())

    annotator.update()

    assert list(scene.sent_answer().res) == []


def test_nobody_pointing_answers_with_nobody(scene: GpsrScene) -> None:
    """
    Nobody pointing answers with nobody.
    """
    scene.query.obj.attribute = ["pointing left"]
    annotator = GeneratePointingResult()
    scene.place_in_pipeline(annotator)
    scene.add_human()

    annotator.update()

    assert list(scene.sent_answer().res) == []


# %% activity results


def test_clip_posture_is_answered_as_attribute(scene: GpsrScene) -> None:
    """
    CLIP posture is answered as attribute.
    """
    annotator = GenerateHumanActivityResult()
    scene.place_in_pipeline(annotator)
    posture = HumanActivityAnnotator.Descriptor.Parameters().posture_labels[0]
    scene.add_human(GpsrScene.clip_activity(posture))

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert list(object_designator.attribute) == [posture]
    assert list(object_designator.description) == []


def test_clip_gesture_is_answered_as_description(scene: GpsrScene) -> None:
    """
    CLIP gesture is answered as description.
    """
    annotator = GenerateHumanActivityResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(GpsrScene.clip_activity(HumanGesture.WAVING))

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert list(object_designator.description) == [HumanGesture.WAVING]


def test_pointing_direction_replaces_the_clip_pointing(scene: GpsrScene) -> None:
    """
    The direction from the pointing rule replaces the plain CLIP pointing.
    """
    annotator = GenerateHumanActivityResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(
        GpsrScene.clip_activity(HumanGesture.POINTING), pointing_rule("pointing left")
    )

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert list(object_designator.description) == ["pointing left"]


def test_skeleton_gesture_replaces_the_clip_no_gesture(scene: GpsrScene) -> None:
    """
    A gesture from the keypoint rules replaces the CLIP no gesture.
    """
    annotator = GenerateHumanActivityResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(
        GpsrScene.clip_activity(HumanGesture.NONE), gesture_rule("raising left arm")
    )

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert list(object_designator.description) == ["raising left arm"]


def test_clip_no_gesture_is_answered_without_skeleton_gesture(scene: GpsrScene) -> None:
    """
    Without a gesture from the keypoint rules, the CLIP no gesture is answered.
    """
    annotator = GenerateHumanActivityResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(GpsrScene.clip_activity(HumanGesture.NONE))

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert list(object_designator.description) == [HumanGesture.NONE]


def test_activity_result_answers_only_persons_in_the_queried_room(
    scene: GpsrScene,
) -> None:
    """
    Activity result answers only persons in the queried room.
    """
    scene.query.obj.location = "iai_kitchen"
    annotator = GenerateHumanActivityResult()
    scene.place_in_pipeline(annotator)
    in_the_room = BeliefEntry(1, EntityType.PERSON, "human")
    scene.add_human(
        GpsrScene.clip_activity(HumanGesture.WAVING),
        LocationAnnotation(name=scene.query.obj.location),
        in_the_room,
    )
    scene.add_human(
        GpsrScene.clip_activity(HumanGesture.WAVING), LocationAnnotation(name="office")
    )
    scene.add_human(GpsrScene.clip_activity(HumanGesture.WAVING))

    annotator.update()

    assert [object_designator.uid for object_designator in scene.sent_answer().res] == [
        str(in_the_room.id)
    ]


def test_person_without_activity_is_not_answered(scene: GpsrScene) -> None:
    """
    Person without activity is not answered.
    """
    annotator = GenerateHumanActivityResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(standing_rule())

    annotator.update()

    assert list(scene.sent_answer().res) == []


def test_nobody_active_answers_with_nobody(scene: GpsrScene) -> None:
    """
    Nobody active answers with nobody.
    """
    annotator = GenerateHumanActivityResult()
    scene.place_in_pipeline(annotator)

    status = annotator.update()

    assert status == Status.SUCCESS
    assert list(scene.sent_answer().res) == []


# %% vision language results


def test_vision_language_answer_is_split_into_attributes(scene: GpsrScene) -> None:
    """
    Vision language answer is split into attributes.
    """
    annotator = GenerateVisionLanguageResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(vision_language_answer("blue jeans, white t-shirt,"))

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert list(object_designator.attribute) == ["blue jeans", "white t-shirt"]


def test_person_without_vision_language_answer_is_not_answered(
    scene: GpsrScene,
) -> None:
    """
    Person without vision language answer is not answered.
    """
    annotator = GenerateVisionLanguageResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(GpsrScene.clip_activity(HumanGesture.WAVING))

    annotator.update()

    assert list(scene.sent_answer().res) == []


def test_nobody_described_answers_with_nobody(scene: GpsrScene) -> None:
    """
    Nobody described answers with nobody.
    """
    annotator = GenerateVisionLanguageResult()
    scene.place_in_pipeline(annotator)

    status = annotator.update()

    assert status == Status.SUCCESS
    assert list(scene.sent_answer().res) == []


def test_brackets_around_the_vision_language_answer_are_dropped(
    scene: GpsrScene,
) -> None:
    """
    Brackets around the vision language answer are dropped.
    """
    annotator = GenerateVisionLanguageResult()
    scene.place_in_pipeline(annotator)
    scene.add_human(vision_language_answer("[gray shirt, black shoes]"))

    annotator.update()

    [object_designator] = scene.sent_answer().res
    assert list(object_designator.attribute) == ["gray shirt", "black shoes"]
