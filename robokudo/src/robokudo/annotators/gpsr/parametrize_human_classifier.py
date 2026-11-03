from __future__ import annotations

# %% imports
import os

import py_trees
import robokudo.annotators.gpsr.face_classification

import robokudo.annotators.core
import robokudo.utils.annotator_helper
import robokudo.utils.error_handling
import robokudo.utils.transform
import robokudo.utils.file_loader
import robokudo.annotators.core


# %% classifier configuration
class ParametrizeHumanClassifyAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Update face classifier parameters from the stored face images.
    """

    def __init__(
        self,
        classify_annotator: robokudo.annotators.gpsr.face_classification.FaceClassification,
        name: str = "ParametrizeHumanClassifyAnnotator",
        ros_pkg_path: str = "robokudo_robocup_receptionist",
    ) -> None:
        """
        :param classify_annotator: Classifier whose face images are configured.
        :param ros_pkg_path: ROS package containing the face image directory.
        """
        super().__init__(str(name))

        self.classify_annotator = classify_annotator
        """
        Classifier whose face images are configured.
        """
        self.ros_pkg_path = ros_pkg_path
        """
        ROS package containing the face image directory.
        """

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:

        file_loader = robokudo.utils.file_loader.FileLoader()
        full_path = file_loader.get_path_to_file_in_ros_package(
            ros_pkg_name=self.ros_pkg_path,
            relative_path="faces",
        )
        self.classify_annotator.descriptor.parameters.ros_pkg_path = (
            self.ros_pkg_path
        )  # If set, use data_path as a relative path to self.ros_pkg_path
        self.classify_annotator.descriptor.parameters.data_path = "faces"
        filenames = []
        classes = []
        for filename in os.listdir(full_path):
            filenames.append(filename)
            classe = filename.split(".")[0]
            classes.append(classe)
        self.classify_annotator.descriptor.parameters.file_names = (
            filenames  # files in self.data_path to load
        )
        self.classify_annotator.descriptor.parameters.labels = (
            classes  # 'class labels' for each of the fil
        )

        return py_trees.common.Status.SUCCESS
