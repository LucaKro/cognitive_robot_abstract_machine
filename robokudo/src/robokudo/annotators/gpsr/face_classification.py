"""
Classification of detected faces against a directory of known face images.
"""

from __future__ import annotations

# %% Imports
import copy
from timeit import default_timer

import face_recognition
import py_trees
from typing_extensions import TYPE_CHECKING

import robokudo.annotators.core
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.error_handling
from robokudo.cas import CASViews
from robokudo.annotators.gpsr.face_detector import FaceDetector
from robokudo.annotators.gpsr.exceptions import (
    FaceImageDirectoryMissing,
    FaceLabelsMismatch,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


# %% face classification
class FaceClassification(robokudo.annotators.core.BaseAnnotator):
    """
    Classifies detected faces with the face_recognition library.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """
        Configuration of face classification.
        """

        class Parameters:
            """
            Known face images and classification threshold.
            """

            def __init__(self) -> None:
                """
                Start without a face directory or known faces.
                """
                self.face_directory: Path | None = None
                """Directory containing face images."""
                self.file_names: list[str] = []
                """Names of face image files to load."""
                self.labels: list[str] = []
                """Person label for each face image."""
                self.maximum_face_distance: float = 0.45
                """Maximum distance for matching face encodings."""

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self,
        name: str = "FaceClassification",
        descriptor: FaceClassification.Descriptor | None = None,
    ) -> None:
        """
        Load the known faces of the descriptor.

        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        :raises FaceImageDirectoryMissing: If no face directory is configured.
        :raises FaceLabelsMismatch: If images and labels differ in number.
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.get_class_name())
        self.known_face_encodings = []
        """Encodings of the known faces."""
        self.known_face_image = []
        """Face image loaded last."""
        self.load_faces()

    def load_faces(self) -> py_trees.common.Status | None:
        """
        Load face encodings, reporting failure if an image contains no face.

        :return: :attr:`Status.FAILURE` if an image contains no face, otherwise *None*.
        :raises FaceImageDirectoryMissing: If no image directory is configured.
        :raises FaceLabelsMismatch: If images and labels differ in number.
        """
        face_directory = self.descriptor.parameters.face_directory
        if face_directory is None:
            raise FaceImageDirectoryMissing()

        if len(self.descriptor.parameters.file_names) != len(
            self.descriptor.parameters.labels
        ):
            raise FaceLabelsMismatch()

        for file_name in self.descriptor.parameters.file_names:
            file_path = face_directory / file_name
            if not file_path.exists():
                self.rk_logger.error(f"No face image file found at '{file_path}'")

            self.known_face_image = face_recognition.load_image_file(str(file_path))
            known_face_encodings = face_recognition.face_encodings(
                self.known_face_image
            )
            if known_face_encodings:
                known_face_encoding = known_face_encodings[0]
                self.known_face_encodings.append(known_face_encoding)
            else:
                self.rk_logger.warning(f"No face detected in image '{file_path}'")
                return py_trees.common.Status.FAILURE
        return None

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
        """
        Overrides :meth:`BaseAnnotator.update` to add a classification to each person
        whose face matches a known face.

        :return: :attr:`Status.SUCCESS`.
        """
        start_timer = default_timer()
        self.load_faces()

        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        human_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.HumanHypothesis
        )

        if len(human_hypotheses) == 0:
            self.rk_logger.warning("Could not find any HumanHypothesis")
            return py_trees.common.Status.SUCCESS

        for human_hypothesis in human_hypotheses:
            encoding_annotations = robokudo.cas.CAS.filter_by_type(
                robokudo.types.annotation.Encoding, human_hypothesis.annotations
            )
            for encoding_annotation in encoding_annotations:
                if encoding_annotation.source != FaceDetector.__name__:
                    continue

                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, encoding_annotation.encoding[0]
                )

                detected_classes = [
                    (match, name)
                    for (match, name) in zip(
                        face_distances, self.descriptor.parameters.labels
                    )
                    if match < self.descriptor.parameters.maximum_face_distance
                ]
                if len(detected_classes) > 0:
                    detected_class = min(
                        detected_classes, key=lambda detection: detection[0]
                    )[1]
                    classification = robokudo.types.annotation.Classification()
                    classification.source = self.name
                    classification.classification_type = "face"
                    classification.classname = detected_class
                    human_hypothesis.annotations.append(classification)
                break

        visualization_image = copy.deepcopy(color)
        self.draw_visualization(visualization_image)
        self.get_annotator_output_struct().set_image(visualization_image)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def draw_visualization(self, visualization_image: np.ndarray) -> np.ndarray:
        """
        Draw the faces classified by this annotator on an image.

        :param visualization_image: Image the bounding boxes are drawn on.
        :return: The image with the bounding boxes and class information.
        """
        human_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.HumanHypothesis
        )
        human_hypotheses_with_face_encodings = []

        for human_hypothesis in human_hypotheses:
            encoding_annotations = robokudo.cas.CAS.filter_by_type(
                robokudo.types.annotation.Encoding, human_hypothesis.annotations
            )
            for encoding_annotation in encoding_annotations:
                if encoding_annotation.source != FaceDetector.__name__:
                    continue

                human_hypotheses_with_face_encodings.append(human_hypothesis)
                break

        # Multiple annotators of this type in one behavior tree need different names,
        # and no annotation of this annotator may exist in the CAS before it runs.
        def get_classification_label(
            human_hypothesis: robokudo.types.scene.HumanHypothesis,
        ) -> str:
            """
            :param human_hypothesis: The person to read the classification of.
            :return: The classifier label of the person, or ``NO-CLASS`` if it is ambiguous.
            """
            classifications = [
                annotation
                for annotation in human_hypothesis.annotations
                if isinstance(annotation, robokudo.types.annotation.Classification)
                and annotation.source == self.name
            ]

            if len(classifications) == 1:
                return classifications[0].classname
            return "NO-CLASS"

        robokudo.utils.annotator_helper.draw_bounding_boxes_from_object_hypotheses(
            visualization_image,
            human_hypotheses_with_face_encodings,
            get_classification_label,
        )

        return visualization_image
