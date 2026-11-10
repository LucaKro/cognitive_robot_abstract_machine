"""
The object detection shared by the GPSR object subtrees.
"""

from __future__ import annotations

# %% Imports
import py_trees.composites
import py_trees.decorators

import robokudo.analysis_engine
from robokudo.annotators.cas_check import CASCheckOHExists
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.outlier_removal_objecthypothesis import (
    OutlierRemovalOnObjectHypothesisAnnotator,
)
from robokudo.annotators.gpsr.filter_object_by_semantic_annotation import (
    FilterObjectBySemanticAnnotation,
)
from robokudo.annotators.gpsr.type_mapping_annotator import TypeMappingAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Detects objects and prepares their spatial and semantic annotations.
    """

    def __init__(
        self,
        model: str = "ultimate_model_v11_1.26.4",
        skip_outlier_removal_classes: list[str] | None = None,
        with_semantic_filter: bool = False,
        outlier_removal_name: str = "OutlierRemovalOnObjectHypothesis",
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        :param model: Detector model name.
        :param skip_outlier_removal_classes: Classes excluded from outlier removal.
        :param with_semantic_filter: Whether to filter detections by the query.
        :param outlier_removal_name: Name of the outlier removal annotator.
        """
        super().__init__(*args, **kwargs)
        # the outlier removal declares ROS parameters under its name, so each instance needs its own
        self.outlier_removal_name = outlier_removal_name
        """
        Name of the outlier removal annotator.
        """
        self.model = model
        """
        Detector model name.
        """
        self.skip_outlier_removal_classes = skip_outlier_removal_classes
        """
        Classes excluded from outlier removal.
        """
        self.with_semantic_filter = with_semantic_filter
        """
        Whether to filter detections by the query.
        """
        if skip_outlier_removal_classes is None:
            # Thin cutlery can have too few depth points for outlier removal.
            self.skip_outlier_removal_classes = [
                "spoon_dinner_blackgrip",
                "spoon_dinner_redgrip",
                "fork_dinner_redgrip",
                "fork_dinner_blackgrip",
                "knife_butter_blackgrip",
                "knife_butter_redgrip",
                "knife_dinner_blackgrip",
                "knife_dinner_redgrip",
            ]

    def implementation(self) -> py_trees.composites.Sequence:
        """
        :return: The sequence that detects objects and adds their poses and semantic types.
        """
        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.precision_mode = True
        yolo_descriptor.parameters.ros_package_name = "robokudo_suturo"
        yolo_descriptor.parameters.weights_path = f"weights/{self.model}.pt"
        yolo_descriptor.parameters.class_names_path = (
            f"data/json/id2name_{self.model}.json"
        )
        yolo_descriptor.parameters.threshold = 0.6

        outlier_removal_descriptor = (
            OutlierRemovalOnObjectHypothesisAnnotator.Descriptor()
        )
        outlier_removal_descriptor.parameters.dbscan_neighbors = 90
        outlier_removal_descriptor.parameters.dbscan_epsilon = 0.02
        outlier_removal_descriptor.parameters.stat_neighbors = 300
        outlier_removal_descriptor.parameters.stat_std = 0.45
        outlier_removal_descriptor.parameters.skip_removal_on_classes = (
            self.skip_outlier_removal_classes
        )

        object_processing = py_trees.composites.Sequence(
            "object_processing", memory=True
        )
        object_processing.add_children(
            [
                CASCheckOHExists(name="ObjectHypothesesFound"),
                OutlierRemovalOnObjectHypothesisAnnotator(
                    name=self.outlier_removal_name,
                    descriptor=outlier_removal_descriptor,
                ),
                ClusterPoseBBAnnotator(),
                TypeMappingAnnotator(),
            ]
        )
        # the outlier removal raises without object hypotheses: an empty scene gets an empty
        # answer, any other failure of the processing still fails the query
        empty_scene = py_trees.decorators.Inverter(
            name="NoObjectHypotheses",
            child=CASCheckOHExists(name="AnyObjectHypothesis"),
        )
        processing_unless_empty = py_trees.composites.Selector(
            "object_processing_unless_empty", memory=True
        )
        processing_unless_empty.add_children([object_processing, empty_scene])

        sequence = py_trees.composites.Sequence("object_detection", memory=True)
        sequence.add_children(
            [
                YoloAnnotator(descriptor=yolo_descriptor),
                processing_unless_empty,
            ]
        )
        if self.with_semantic_filter:
            sequence.add_children([FilterObjectBySemanticAnnotation()])
        return sequence
