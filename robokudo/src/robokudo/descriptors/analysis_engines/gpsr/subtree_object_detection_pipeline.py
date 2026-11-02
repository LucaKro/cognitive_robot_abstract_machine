import py_trees
import robokudo.analysis_engine
import robokudo.annotators.query
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.descriptors.analysis_engines.gpsr.subtree_object_detection import (
    Subtree as ObjectDetectionSubtree,
)
from robokudo.annotators.query import GenerateQueryResult


class Subtree(robokudo.analysis_engine.SubtreeInterface):
    def implementation(self, name=None, *args, **kwargs):

        seq = py_trees.composites.Sequence(
            "subtree_object_detection_pipeline", memory=True
        )
        seq.add_children(
            [
                ObjectDetectionSubtree(
                    ros_pkg="robokudo_suturo",
                    model="ultimate_model_v11_1.26.4",
                    with_semantic_filter=True,
                ).implementation(),
                BeliefStateAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return seq
