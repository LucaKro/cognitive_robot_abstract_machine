from __future__ import annotations

# %% imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.query import GenerateQueryResult
from robokudo.annotators.yolo_annotator import YoloAnnotator

from robokudo.annotators.gpsr.door_handle_pose_annotator import DoorHandlePoseAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Detect doors and estimate door handle poses.
    """

    def __init__(self, ros_pkg_path: str = "robokudo_suturo") -> None:
        """:param ros_pkg_path: ROS package containing the detector files."""
        self.ros_pkg_path = ros_pkg_path
        """
        ROS package containing the detector files.
        """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        Detect doors and handles and estimate a grasp pose for each handle.
        """
        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.ros_package_name = self.ros_pkg_path
        # the SAM model lies next to our weights, outside CRAM
        yolo_descriptor.parameters.models_directory = (
            yolo_descriptor.parameters.file_in_ros_package("weights")
        )
        yolo_descriptor.parameters.weights_path = "weights/door.pt"
        yolo_descriptor.parameters.class_names_path = "data/json/id2name_door.json"
        yolo_descriptor.parameters.threshold = 0.35
        yolo_descriptor.parameters.precision_mode = False

        seq = py_trees.composites.Sequence("subtree_door_pipeline", memory=True)
        seq.add_children(
            [
                YoloAnnotator(descriptor=yolo_descriptor),
                DoorHandlePoseAnnotator(),
                BeliefStateAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return seq
