"""
The GPSR subtree that detects doors with a trained detector and estimates grasp poses
for their handles.
"""

from __future__ import annotations

# %% Imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.query import GenerateQueryResult
from robokudo.annotators.yolo_annotator import YoloAnnotator

from robokudo.annotators.gpsr.door_handle_pose_annotator import DoorHandlePoseAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Detects doors and estimates door handle poses.
    """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        Detect doors and handles and estimate a grasp pose for each handle.

        :param name: Name of this subtree.
        :return: The sequence of the door route.
        """
        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.ros_package_name = "robokudo_suturo"
        yolo_descriptor.parameters.weights_path = "weights/door.pt"
        yolo_descriptor.parameters.class_names_path = "data/json/id2name_door.json"
        yolo_descriptor.parameters.threshold = 0.35
        yolo_descriptor.parameters.precision_mode = False

        sequence = py_trees.composites.Sequence("subtree_door_pipeline", memory=True)
        sequence.add_children(
            [
                YoloAnnotator(descriptor=yolo_descriptor),
                DoorHandlePoseAnnotator(),
                BeliefStateAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return sequence
