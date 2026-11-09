"""
The GPSR subtree that detects, classifies and stores faces.
"""

from __future__ import annotations

# %% Imports
import os
from pathlib import Path
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.face_classification import FaceClassification
from robokudo.annotators.gpsr.face_detector import FaceDetector
from robokudo.annotators.gpsr.store_unknown_faces import StoreFaces
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.filter_objects_by_area import FilterObjectsByArea
from robokudo.annotators.gpsr.generate_all_annotations import GenerateAllQueryResult
from robokudo.annotators.gpsr.parametrize_human_classifier import (
    ParametrizeHumanClassifyAnnotator,
)


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Detects, classifies and stores faces for queries.
    """

    def __init__(self, face_directory: Path | None = None) -> None:
        """
        :param face_directory: Directory containing face images.
        """
        self.face_directory = face_directory
        """
        Directory containing face images, ``~/.local/share/robokudo/faces`` if *None*.
        """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        Build the sequence with one face directory shared by all of its annotators.

        :param name: Name of this subtree.
        :return: The sequence of the face route.
        """
        face_classification_descriptor = FaceClassification.Descriptor()
        face_directory = self.face_directory
        if face_directory is None:
            face_directory = Path.home() / ".local" / "share" / "robokudo" / "faces"
        face_directory.mkdir(parents=True, exist_ok=True)
        face_classification_descriptor.parameters.face_directory = face_directory

        file_names = []
        labels = []
        for file_name in os.listdir(face_directory):
            file_names.append(file_name)
            class_label = file_name.split(".")[0]
            labels.append(class_label)

        face_classification_descriptor.parameters.file_names = file_names
        face_classification_descriptor.parameters.labels = labels

        store_faces_descriptor = StoreFaces.Descriptor()
        store_faces_descriptor.parameters.face_directory = face_directory
        store_faces_descriptor.parameters.padding = 5

        face_classifier = FaceClassification(descriptor=face_classification_descriptor)

        sequence = py_trees.composites.Sequence("subtree_faces", memory=True)
        sequence.add_children(
            [
                FaceDetector(),
                ClusterPositionAnnotator(),
                FilterObjectsByArea(),
                ParametrizeHumanClassifyAnnotator(
                    face_classifier, face_directory=face_directory
                ),
                face_classifier,
                StoreFaces(descriptor=store_faces_descriptor),
                BeliefStateAnnotator(),
                GenerateAllQueryResult(),
            ]
        )
        return sequence
