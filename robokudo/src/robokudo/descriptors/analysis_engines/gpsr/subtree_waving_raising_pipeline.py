from __future__ import annotations

# %% imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.waving import HumanWavingAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.utils.file_loader import FileLoader
from robokudo.annotators.gpsr.gpsr_results import GenerateWavingRaisingResult
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Find people making the queried waving or arm-raising gesture.
    """

    def __init__(
        self, attribute: object = None, *args: object, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attribute = attribute

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        # CRAM's HumanAndPoseAnnotator works without depth by default, the person's position needs it
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True
        # the pose model lies next to our weights, outside CRAM
        human_descriptor.parameters.models_directory = (
            FileLoader.get_path_to_file_in_ros_package(
                ros_pkg_name="robokudo_suturo", relative_path="weights"
            )
        )

        seq = py_trees.composites.Sequence(
            "subtree_waving_raising_pipeline", memory=True
        )
        seq.add_children(
            [
                HumanAndPoseAnnotator(descriptor=human_descriptor),
                ClusterPositionAnnotator(),
                HumanWavingAnnotator(),
                BeliefStateAnnotator(),
                GenerateWavingRaisingResult(),
            ]
        )
        return seq
