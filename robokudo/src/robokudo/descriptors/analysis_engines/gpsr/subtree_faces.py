from __future__ import annotations

# %% imports
import os
from pathlib import Path
import py_trees
import robokudo.analysis_engine
import robokudo
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.face_classification import FaceClassification
import robokudo.utils.file_loader
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
    Detect, classify and store faces for queries.
    """

    def __init__(
        self,
        ros_pkg_path: str | None = None,
        face_directory: Path | None = None,
    ) -> None:
        """
        :param ros_pkg_path: ROS package containing the face image directory.
        :param face_directory: Directory containing face images.
        """
        self.ros_pkg_path = ros_pkg_path
        """
        ROS package containing the face image directory.
        """
        self.face_directory = face_directory
        """
        Directory containing face images, if configured directly.
        """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        face_classification_descriptor = FaceClassification.Descriptor()
        full_path = self.face_directory
        if full_path is None and self.ros_pkg_path is not None:
            file_loader = robokudo.utils.file_loader.FileLoader()
            package_path = Path(
                file_loader.get_path_to_file_in_ros_package(
                    ros_pkg_name=self.ros_pkg_path,
                    relative_path="package.xml",
                )
            )
            full_path = package_path.parent / "faces"
        if full_path is None:
            full_path = Path.home() / ".local" / "share" / "robokudo" / "faces"
        full_path.mkdir(parents=True, exist_ok=True)
        face_classification_descriptor.parameters.face_directory = full_path

        names = []
        classes = []
        for filename in os.listdir(full_path):
            names.append(filename)
            class_label = filename.split(".")[0]
            classes.append(class_label)

        face_classification_descriptor.parameters.file_names = names
        face_classification_descriptor.parameters.labels = classes

        store_faces_descriptor = StoreFaces.Descriptor()
        store_faces_descriptor.parameters.face_directory = full_path
        store_faces_descriptor.parameters.noise = 5

        human_classifier = FaceClassification(descriptor=face_classification_descriptor)

        seq = py_trees.composites.Sequence("subtree_faces", memory=True)
        seq.add_children(
            [
                FaceDetector(),
                ClusterPositionAnnotator(),
                FilterObjectsByArea(),
                ParametrizeHumanClassifyAnnotator(
                    human_classifier, face_directory=full_path
                ),
                human_classifier,
                StoreFaces(descriptor=store_faces_descriptor),
                BeliefStateAnnotator(),
                GenerateAllQueryResult(),
            ]
        )
        return seq
