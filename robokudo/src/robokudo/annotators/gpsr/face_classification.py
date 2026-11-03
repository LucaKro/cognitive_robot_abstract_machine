from __future__ import annotations

# %% imports
import copy
from pathlib import Path
from timeit import default_timer

import face_recognition
import py_trees
from typing_extensions import TYPE_CHECKING

import robokudo
import robokudo.annotators.core
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.cv_helper
import robokudo.utils.error_handling
from robokudo.cas import CASViews
from robokudo.utils.file_loader import FileLoader

if TYPE_CHECKING:
    import numpy as np


# %% face classification
class FaceClassification(robokudo.annotators.core.BaseAnnotator):
    """
    Classifies detected faces based on face_recognition library.
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
                self.ros_pkg_path = None  # If set, use data_path as a relative path to self.ros_pkg_path
                self.face_directory: Path | None = None
                """
                Directory containing face images, if configured directly.
                """
                self.data_path = (
                    None  # Relative Path to the folder containing the models
                )
                self.file_names = []  # files in self.data_path to load
                self.labels = []  # 'class labels' for each of the file
                self.distance = 0.45  # WIP

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion

    def __init__(
        self,
        name: str = "FaceClassification",
        descriptor: FaceClassification.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.known_face_encodings = []
        self.known_person_image = []
        self.load_faces()

    def load_faces(self) -> py_trees.common.Status | None:
        """
        Load face encodings, reporting failure if an image contains no face.
        """
        data_folder_path = self.descriptor.parameters.face_directory
        if data_folder_path is None:
            if (
                self.descriptor.parameters.ros_pkg_path is None
                or self.descriptor.parameters.data_path is None
            ):
                raise RuntimeError(
                    "Can't look up necessary input data. Please check Parameters of this Annotator."
                )
            data_folder_path = FileLoader.get_path_to_file_in_ros_package(
                ros_pkg_name=self.descriptor.parameters.ros_pkg_path,
                relative_path=self.descriptor.parameters.data_path,
            )

        if len(self.descriptor.parameters.file_names) != len(
            self.descriptor.parameters.labels
        ):
            raise RuntimeError(
                "File names and corresponding label list must be of equal size"
            )

        for file_name in self.descriptor.parameters.file_names:
            file_path: Path = data_folder_path.joinpath(f"{file_name}")
            if not file_path.exists():
                self.rk_logger.error(f"No face image file found at '{file_path}'")

            self.known_person_image = face_recognition.load_image_file(str(file_path))
            known_face_encodings = face_recognition.face_encodings(
                self.known_person_image
            )
            if known_face_encodings:
                known_person_encoding = known_face_encodings[0]
                self.known_face_encodings.append(known_person_encoding)
            else:
                self.rk_logger.warning(f"No face detected in image '{file_path}'")
                return py_trees.common.Status.FAILURE
        return None

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
        start_timer = default_timer()
        self.load_faces()  # Reload faces before comparison

        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        human_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.HumanHypothesis
        )

        if len(human_hypotheses) == 0:
            self.rk_logger.warning(f"Could not find any HumanHypothesis")
            return py_trees.common.Status.SUCCESS

        for human_hypothesis in human_hypotheses:
            encoding_annotations = robokudo.cas.CAS.filter_by_type(
                robokudo.types.annotation.Encoding, human_hypothesis.annotations
            )
            for encoding_annotation in encoding_annotations:
                if encoding_annotation.source != "FaceDetector":
                    continue

                # Classify the given face by comparing against the loaded face encodings
                # results = face_recognition.compare_faces(known_face_encodings=self.known_face_encodings,
                #                                         face_encoding_to_check=encoding_annotation.encoding[0],
                #                                         tolerance=self.descriptor.parameters.tolerance)
                # print(f'Results are: {results}')

                distance = face_recognition.face_distance(
                    self.known_face_encodings, encoding_annotation.encoding[0]
                )

                detected_classes = [
                    (match, name)
                    for (match, name) in zip(
                        distance, self.descriptor.parameters.labels
                    )
                    if match < self.descriptor.parameters.distance
                ]
                if len(detected_classes) > 0:
                    detected_class = min(detected_classes, key=lambda x: x[0])[1]
                    classification = robokudo.types.annotation.Classification()
                    classification.source = self.name
                    classification.classification_type = "face"
                    classification.classname = detected_class
                    human_hypothesis.annotations.append(classification)
                break

        visualization_img = copy.deepcopy(color)
        self.draw_visualization(visualization_img)
        self.get_annotator_output_struct().set_image(visualization_img)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def draw_visualization(self, visualization_img: np.ndarray) -> np.ndarray:
        """
        Fetch object hypothesis with Annotations done by this Annotator and visualize them in 2D
        :param visualization_img:
        :return: The updated visualization_img with the bounding boxes and class information
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
                if encoding_annotation.source != "FaceDetector":
                    continue

                human_hypotheses_with_face_encodings.append(human_hypothesis)
                # only add one hypothesis with an encoding
                break

        # Fetch the class annotations done by this annotator
        # Warning:If you have multiple Annotators of this type in your BT, you need to switch from self.get_class_name()
        # in the annotation to self.name and set different names in the CTOR. We also assume,
        # that no annotations from this Annotator have existed in the CAS prior to the call of this Annotator
        def get_class_label_from_h(h: robokudo.types.scene.HumanHypothesis) -> str:
            """
            Return the classifier label for a person, if unambiguous.
            """
            class_annotations = [
                a
                for a in h.annotations
                if isinstance(a, robokudo.types.annotation.Classification)
                and a.source == self.name
            ]

            if len(class_annotations) == 1:
                return class_annotations[0].classname
            return "NO-CLASS"

        robokudo.utils.annotator_helper.draw_bounding_boxes_from_object_hypotheses(
            visualization_img,
            human_hypotheses_with_face_encodings,
            lambda h: f"{get_class_label_from_h(h)}",
        )

        return visualization_img
