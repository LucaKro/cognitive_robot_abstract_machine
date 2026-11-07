"""
The GPSR subtree that finds doors and handles with open-vocabulary detection and
estimates a grasp pose for each handle.
"""

from __future__ import annotations

# %% Imports
from py_trees.composites import Sequence

from robokudo.analysis_engine import SubtreeInterface
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.door_handle_pose_annotator import (
    DoorObjectClass,
    DoorHandlePoseAnnotator,
)
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.annotators.query import GenerateQueryResult

# %% subtree


class Subtree(SubtreeInterface):
    """
    Finds doors and handles by text and estimates a grasp pose for each handle.
    """

    def __init__(
        self, detection_threshold: float = 0.25, *args: object, **kwargs: object
    ) -> None:
        """
        :param detection_threshold: Smallest detector score of a door or handle.
        """
        super().__init__(*args, **kwargs)
        self.detection_threshold = detection_threshold
        """
        Smallest detector score of a door or handle.
        """

    def implementation(self, *args: object, **kwargs: object) -> Sequence:
        """
        :return: The sequence that finds doors and handles by text and answers with their poses.
        """
        detector_descriptor = OpenVocabularyObjectDetectionAnnotator.Descriptor()
        detector_descriptor.parameters.classes = [
            DoorObjectClass.DOOR.value,
            DoorObjectClass.HANDLE.value,
        ]
        detector_descriptor.parameters.detection_threshold = self.detection_threshold

        sequence = Sequence("subtree_door_open_vocabulary_pipeline", memory=True)
        sequence.add_children(
            [
                OpenVocabularyObjectDetectionAnnotator(descriptor=detector_descriptor),
                DoorHandlePoseAnnotator(),
                BeliefStateAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return sequence
