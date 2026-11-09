"""
Tests for answering a query with the detections of the open-vocabulary detector.
"""

from __future__ import annotations

# %% Imports
from robokudo.annotators.gpsr.belief_state import BeliefEntry, EntityType
from robokudo.annotators.gpsr.exceptions import MissingObjectClassification
from robokudo.annotators.gpsr.gpsr_results import (
    GenerateOpenVocabularyResult,
    create_camera_header,
)
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.types.annotation import (
    Classification,
    LocationAnnotation,
    PositionAnnotation,
)
from robokudo.types.cv import Point2D, Rect
from robokudo.utils.type_conversion import (
    get_geometry_msgs_pose_stamped_from_position_annotation,
)
from robokudo_msgs.msg import ObjectDesignator
from py_trees.common import Status
from robokudo.utils.error_handling import get_blackboard_exception
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from ._gpsr_scene import GpsrScene


# %% open-vocabulary answer


def test_object_without_classification_reports_failure(scene: GpsrScene) -> None:
    """
    An unclassified object reports an error through the blackboard.
    """
    scene.query.obj.type = "object"
    annotator = GenerateOpenVocabularyResult()
    scene.place_in_pipeline(annotator)
    scene.add_object()

    status = annotator.update()

    assert status == Status.FAILURE
    assert isinstance(get_blackboard_exception(), MissingObjectClassification)


def test_clothing_answer_preserves_person_type_and_detected_description(
    scene: GpsrScene,
) -> None:
    """
    Report the garment on its associated person in the GPSR answer.
    """
    scene.query.obj.type = "human"
    annotator = GenerateOpenVocabularyResult()
    scene.place_in_pipeline(annotator)
    human_hypothesis = scene.add_human()
    human_hypothesis.roi.roi = Rect(pos=Point2D(x=10, y=10), width=80, height=120)
    garment = scene.add_object(
        Classification(
            classname="red shirt",
            source=OpenVocabularyObjectDetectionAnnotator.__name__,
        )
    )
    garment.roi.roi = Rect(pos=Point2D(x=20, y=30), width=40, height=40)

    annotator.update()

    assert scene.sent_answer().res == [
        ObjectDesignator(type="human", description=["red shirt"])
    ]


def test_clothing_answer_includes_person_identity_and_position(
    scene: GpsrScene,
) -> None:
    """
    A clothing match carries the person's identity and position.
    """
    scene.query.obj.type = "human"
    annotator = GenerateOpenVocabularyResult()
    cas = scene.place_in_pipeline(annotator)
    belief = BeliefEntry(7, EntityType.PERSON, "human")
    position = PositionAnnotation(translation=[1.0, 2.0, 3.0], source="camera")
    human_hypothesis = scene.add_human(belief, position)
    human_hypothesis.roi.roi = Rect(pos=Point2D(x=10, y=10), width=80, height=120)
    garment = scene.add_object(
        Classification(
            classname="red shirt",
            source=OpenVocabularyObjectDetectionAnnotator.__name__,
        )
    )
    garment.roi.roi = Rect(pos=Point2D(x=20, y=30), width=40, height=40)

    annotator.update()

    assert scene.sent_answer().res == [
        ObjectDesignator(
            type="human",
            description=["red shirt"],
            uid=str(belief.id),
            pose=[
                get_geometry_msgs_pose_stamped_from_position_annotation(
                    position, create_camera_header(cas)
                )
            ],
            pose_source=[position.source],
        )
    ]


def test_object_answer_includes_position_without_belief_id(scene: GpsrScene) -> None:
    """
    An object answer carries its position without its belief identifier.
    """
    scene.query.obj.type = "object"
    annotator = GenerateOpenVocabularyResult()
    cas = scene.place_in_pipeline(annotator)
    position = PositionAnnotation(translation=[1.0, 2.0, 3.0], source="camera")
    scene.add_object(
        Classification(
            classname="cup", source=OpenVocabularyObjectDetectionAnnotator.__name__
        ),
        BeliefEntry(7, EntityType.OBJECT, "cup"),
        position,
    )

    annotator.update()

    assert scene.sent_answer().res == [
        ObjectDesignator(
            type="object",
            description=["cup"],
            pose=[
                get_geometry_msgs_pose_stamped_from_position_annotation(
                    position, create_camera_header(cas)
                )
            ],
            pose_source=[position.source],
        )
    ]


def test_clothing_answer_filters_persons_by_queried_room(scene: GpsrScene) -> None:
    """
    A room query answers only matching persons in the requested room.
    """
    scene.query.obj.type = "human"
    scene.query.obj.location = "kitchen"
    annotator = GenerateOpenVocabularyResult(with_room_filter=True)
    scene.place_in_pipeline(annotator)
    human_hypothesis = scene.add_human(
        LocationAnnotation(name=scene.query.obj.location)
    )
    human_hypothesis.roi.roi = Rect(pos=Point2D(x=10, y=10), width=80, height=120)
    other = scene.add_human(LocationAnnotation(name="office"))
    other.roi.roi = human_hypothesis.roi.roi
    garment = scene.add_object(
        Classification(
            classname="red shirt",
            source=OpenVocabularyObjectDetectionAnnotator.__name__,
        )
    )
    garment.roi.roi = Rect(pos=Point2D(x=20, y=30), width=40, height=40)

    annotator.update()

    assert scene.sent_answer().res == [
        ObjectDesignator(type="human", description=["red shirt"])
    ]
