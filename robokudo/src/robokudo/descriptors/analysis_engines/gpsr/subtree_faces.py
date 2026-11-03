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

    def __init__(self, ros_pkg_path: str = "robokudo_robocup_receptionist") -> None:
        """:param ros_pkg_path: ROS package containing the face image directory."""
        self.ros_pkg_path = ros_pkg_path
        """
        ROS package containing the face image directory.
        """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        face_classifcation_descriptor = FaceClassification.Descriptor()
        face_classifcation_descriptor.parameters.ros_pkg_path = self.ros_pkg_path
        face_classifcation_descriptor.parameters.data_path = "faces"
        file_loader = robokudo.utils.file_loader.FileLoader()
        try:
            full_path = Path(
                file_loader.get_path_to_file_in_ros_package(
                    ros_pkg_name=self.ros_pkg_path,
                    relative_path="faces",
                )
            )
        except Exception:
            pkg_xml = Path(
                file_loader.get_path_to_file_in_ros_package(
                    ros_pkg_name=self.ros_pkg_path,
                    relative_path="package.xml",
                )
            )
            full_path = pkg_xml.parent / "faces"
        full_path.mkdir(parents=True, exist_ok=True)

        names = []
        classes = []
        for filename in os.listdir(full_path):
            names.append(filename)
            classe = filename.split(".")[0]
            classes.append(classe)

        face_classifcation_descriptor.parameters.file_names = names
        face_classifcation_descriptor.parameters.labels = classes

        store_faces_descriptor = StoreFaces.Descriptor()
        store_faces_descriptor.parameters.ros_pkg_path = self.ros_pkg_path
        store_faces_descriptor.parameters.data_path = "faces"
        store_faces_descriptor.parameters.noise = 5

        human_classifier = FaceClassification(descriptor=face_classifcation_descriptor)

        seq = py_trees.composites.Sequence("subtree_faces", memory=True)
        seq.add_children(
            [
                FaceDetector(),
                ClusterPositionAnnotator(),
                FilterObjectsByArea(),
                ParametrizeHumanClassifyAnnotator(
                    human_classifier, ros_pkg_path=self.ros_pkg_path
                ),
                human_classifier,
                StoreFaces(descriptor=store_faces_descriptor),
                BeliefStateAnnotator(),
                GenerateAllQueryResult(),
            ]
        )
        return seq
