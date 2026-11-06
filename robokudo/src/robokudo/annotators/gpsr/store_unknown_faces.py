"""
Storage of the face images of people without a face classification.
"""

from __future__ import annotations

# %% imports
from timeit import default_timer
from PIL import Image
import py_trees

import robokudo.annotators.core
import robokudo.utils.cv_helper
import robokudo.utils.error_handling
import robokudo.types.annotation
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
    Store images of people without a face classification.
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
                self.noise: int = 0
                """Padding around each stored face image."""

        parameters = Parameters()

    def __init__(
        self, name: str = "StoreFaces", descriptor: StoreFaces.Descriptor | None = None
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Configuration of this annotator.
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self) -> py_trees.common.Status:
        """
        Store the face of each person without a classification and classify the person.

        :raises FaceStoragePathMissing: If no face directory is configured.
        """
        if self.descriptor.parameters.face_directory is None:
            raise FaceStoragePathMissing()

        start_timer = default_timer()

        # Get all human hypothesis
        human_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.HumanHypothesis
        )

        for annotation in human_hypotheses:
            # Check if the human_hypothesis already has a classification
            if self._check_if_class_exists(annotation):
                continue

            # Searching for Face annotations
            for face_annotation in annotation.annotations:
                if isinstance(face_annotation, robokudo.types.human.FaceAnnotation):
                    # The region of the face, its position is the top left corner
                    region = face_annotation.roi.roi

                    # Define area of points with face
                    color_image = self.get_cas().get(CASViews.COLOR_IMAGE)
                    noise = self.descriptor.parameters.noise
                    left = region.pos.x - noise
                    right = region.pos.x + region.width + noise
                    top = region.pos.y - noise
                    bottom = region.pos.y + region.height + noise

                    # Store unknown image under a unique id
                    face_directory = self.descriptor.parameters.face_directory
                    file_name = self._get_id(face_directory) + ".png"
                    file = face_directory / file_name
                    cropped_image = color_image[top:bottom, left:right]
                    image = Image.fromarray(cropped_image)
                    image.save(file)

                    # Write Annotation with id of people
                    classification = Classification()
                    classification.source = self.name
                    classification.classification_type = "face"
                    classification.classname = file_name.split(".")[0]
                    annotation.annotations.append(classification)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def _check_if_class_exists(self, human_hypothesis: object) -> bool:
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

    def _get_id(self, path: Path) -> str:
        """
        :param path: Directory containing the face images.
        :return: An unused face image identifier in the directory.
        :raises NoAvailableFaceIdentifier: If every identifier is in use.
        """
        names = []
        for filename in os.listdir(path):
            names.append(filename.split(".")[0])
        print(len(names))
        for identifier in range(1000):
            if not str(identifier) in names:
                return str(identifier)
        raise NoAvailableFaceIdentifier()
