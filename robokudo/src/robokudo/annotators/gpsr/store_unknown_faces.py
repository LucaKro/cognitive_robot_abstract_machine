from __future__ import annotations

# %% imports
from timeit import default_timer
from PIL import Image
import py_trees

import robokudo.annotators.core
import robokudo.utils.cv_helper
import robokudo.utils.error_handling
import robokudo.utils.file_loader
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
                self.ros_pkg_path: str | None = None
                """ROS package containing the face image directory."""
                self.data_path: str | None = None
                """Face image directory relative to the ROS package."""
                self.face_directory: Path | None = None
                """
                Directory where face images are stored, if configured directly.
                """
                self.noise: int = 0
                """Padding around each stored face image."""

        parameters = Parameters()

    def __init__(
        self, name: str = "StoreFaces", descriptor: StoreFaces.Descriptor | None = None
    ) -> None:
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self) -> py_trees.common.Status:
        if (
            self.descriptor.parameters.face_directory is None
            and self.descriptor.parameters.ros_pkg_path is None
        ):
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
                    # Get the roi of face, extracted point is top left
                    width = face_annotation.roi.roi.width
                    length = face_annotation.roi.roi.height
                    x = face_annotation.roi.roi.pos.x
                    y = face_annotation.roi.roi.pos.y

                    # Define area of points with face
                    color_image = self.get_cas().get(CASViews.COLOR_IMAGE)
                    noise = self.descriptor.parameters.noise
                    x_min = x - noise
                    x_max = x + width + noise
                    y_min = y - noise
                    y_max = y + length + noise

                    # Load Path to data
                    full_path = self.descriptor.parameters.face_directory
                    if full_path is None:
                        file_loader = robokudo.utils.file_loader.FileLoader()
                        full_path = file_loader.get_path_to_file_in_ros_package(
                            ros_pkg_name=self.descriptor.parameters.ros_pkg_path,
                            relative_path=self.descriptor.parameters.data_path,
                        )
                    file_path = full_path

                    # Store unknown image under a unique id
                    file_name = self._get_id(full_path) + ".png"
                    file = os.path.join(file_path, file_name)
                    cropped_image = color_image[y_min:y_max, x_min:x_max]
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
        :return: Whether the person already has a classification.
        """
        if not isinstance(human_hypothesis, robokudo.types.scene.HumanHypothesis):
            raise InvalidFaceHypothesis(type(human_hypothesis))

        for classification in human_hypothesis.annotations:
            if isinstance(classification, Classification):
                print(f"Detect {classification.classname}, it's a familiar face")
                return True
        print("Detect unfamiliar face, create a new file for his face!")
        return False

    def _get_id(self, path: str | os.PathLike[str]) -> str:
        """
        :return: An unused face image identifier in the directory.
        """
        names = []
        for filename in os.listdir(path):
            names.append(filename.split(".")[0])
        print(len(names))
        for i in range(1000):
            if not str(i) in names:
                return str(i)
        raise NoAvailableFaceIdentifier()
