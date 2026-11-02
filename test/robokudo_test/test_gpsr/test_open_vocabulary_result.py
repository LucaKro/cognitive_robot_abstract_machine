"""
Tests for answering a query with the detections of the open-vocabulary detector.
"""

from __future__ import annotations

# %% Imports
from robokudo.annotators.gpsr.gpsr_results import GenerateOpenVocabularyResult
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.types.annotation import Classification
from robokudo.types.cv import Point2D, Rect
from robokudo_msgs.msg import ObjectDesignator
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from ._gpsr_scene import GpsrScene


# %% open-vocabulary answer


def test_clothing_answer_preserves_person_type_and_detected_description(
    scene: GpsrScene,
) -> None:
    """
    Report the garment on its associated person in the GPSR answer.
    """
    scene.query.obj.type = "human"
    annotator = GenerateOpenVocabularyResult()
    scene.place_in_pipeline(annotator)
    person = scene.add_human()
    person.roi.roi = Rect(pos=Point2D(x=10, y=10), width=80, height=120)
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
