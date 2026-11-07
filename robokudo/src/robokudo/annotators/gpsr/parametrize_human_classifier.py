"""
Configuration of the face classifier from the stored face images.
"""

from __future__ import annotations

# %% Imports
import os

import py_trees
from typing_extensions import TYPE_CHECKING
import robokudo.annotators.gpsr.face_classification

import robokudo.annotators.core
import robokudo.utils.error_handling

if TYPE_CHECKING:
    from pathlib import Path


# %% classifier configuration
class ParametrizeHumanClassifyAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Updates face classifier parameters from the stored face images.
    """

    def __init__(
        self,
        classify_annotator: robokudo.annotators.gpsr.face_classification.FaceClassification,
        face_directory: Path,
        name: str = "ParametrizeHumanClassifyAnnotator",
    ) -> None:
        """
        :param classify_annotator: Classifier whose face images are configured.
        :param face_directory: Directory containing face images.
        :param name: Name of this annotator in the tree.
        """
        super().__init__(str(name))

        self.classify_annotator = classify_annotator
        """
        Classifier whose face images are configured.
        """
        self.face_directory = face_directory
        """
        Directory containing face images.
        """

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
        """
        Overrides :meth:`BaseAnnotator.update` to point the classifier to the images in
        the directory, labeled by file name.

        :return: :attr:`Status.SUCCESS`.
        """
        self.classify_annotator.descriptor.parameters.face_directory = (
            self.face_directory
        )
        file_names = []
        labels = []
        for file_name in os.listdir(self.face_directory):
            file_names.append(file_name)
            class_label = file_name.split(".")[0]
            labels.append(class_label)
        self.classify_annotator.descriptor.parameters.file_names = file_names
        self.classify_annotator.descriptor.parameters.labels = labels

        return py_trees.common.Status.SUCCESS
