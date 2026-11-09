"""
Storage of the face images of persons without a face classification.
"""

from __future__ import annotations

# %% Imports
from timeit import default_timer
from PIL import Image
import py_trees

import robokudo.annotators.core
import robokudo.utils.error_handling
import robokudo.types.human
import robokudo.types.scene
from robokudo.annotators.gpsr.exceptions import (
    FaceStoragePathMissing,
    InvalidFaceHypothesis,
    NoAvailableFaceIdentifier,
)
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification

import os

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# %% environment
os.environ["QT_API"] = "pyqt5"


# %% face storage
class StoreFaces(robokudo.annotators.core.ThreadedAnnotator):
    """
    Stores images of persons without a face classification.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """
        Configuration of face image storage.
        """

        class Parameters:
            """
            Destination and crop settings for face images.
            """

            def __init__(self) -> None:
                """
                Start without a face directory and without padding.
                """
                self.face_directory: Path | None = None
                """Directory where face images are stored."""
                self.padding: int = 0
                """Padding around each stored face image, in pixels."""

        parameters = Parameters()
        """
        Default parameters on the class. Each descriptor instance gets its own copy.
        """

    def __init__(
        self, name: str = "StoreFaces", descriptor: StoreFaces.Descriptor | None = None
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.get_class_name())

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self) -> py_trees.common.Status:
        """
        Overrides :meth:`ThreadedAnnotator.compute` to store the face of each person
        without a classification and classify the person.

        :return: :attr:`Status.SUCCESS`.
        :raises FaceStoragePathMissing: If no face directory is configured.
        """
        if self.descriptor.parameters.face_directory is None:
            raise FaceStoragePathMissing()

        start_timer = default_timer()

        human_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.HumanHypothesis
        )

        for human_hypothesis in human_hypotheses:
            if self._has_classification(human_hypothesis):
                continue

            for annotation in human_hypothesis.annotations:
                if isinstance(annotation, robokudo.types.human.FaceAnnotation):
                    # The region of the face, its position is the top left corner
                    region = annotation.roi.roi

                    color_image = self.get_cas().get(CASViews.COLOR_IMAGE)
                    padding = self.descriptor.parameters.padding
                    left = region.pos.x - padding
                    right = region.pos.x + region.width + padding
                    top = region.pos.y - padding
                    bottom = region.pos.y + region.height + padding

                    face_directory = self.descriptor.parameters.face_directory
                    file_name = self._get_id(face_directory) + ".png"
                    file_path = face_directory / file_name
                    cropped_image = color_image[top:bottom, left:right]
                    image = Image.fromarray(cropped_image)
                    image.save(file_path)

                    classification = Classification()
                    classification.source = self.name
                    classification.classification_type = "face"
                    classification.classname = file_name.split(".")[0]
                    human_hypothesis.annotations.append(classification)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def _has_classification(self, human_hypothesis: object) -> bool:
        """
        :param human_hypothesis: The person to check.
        :return: Whether the person already has a classification.
        :raises InvalidFaceHypothesis: If the hypothesis is not a person.
        """
        if not isinstance(human_hypothesis, robokudo.types.scene.HumanHypothesis):
            raise InvalidFaceHypothesis(type(human_hypothesis))

        for classification in human_hypothesis.annotations:
            if isinstance(classification, Classification):
                print(f"Detect {classification.classname}, it's a familiar face")
                return True
        print("Detect unfamiliar face, create a new file for his face!")
        return False

    def _get_id(self, face_directory: Path) -> str:
        """
        :param face_directory: Directory containing the face images.
        :return: An unused face image identifier in the directory.
        :raises NoAvailableFaceIdentifier: If every identifier is in use.
        """
        names = []
        for file_name in os.listdir(face_directory):
            names.append(file_name.split(".")[0])
        print(len(names))
        for identifier in range(1000):
            if not str(identifier) in names:
                return str(identifier)
        raise NoAvailableFaceIdentifier()
