from __future__ import annotations

# %% imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.utils.file_loader import FileLoader
from robokudo.annotators.gpsr.gpsr_results import GenerateHumanPostureResults
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.lying import HumanLyingAnnotator
from robokudo.annotators.gpsr.sitting import HumanSittingAnnotator
from robokudo.annotators.gpsr.standing import HumanStandingAnnotator


# %% posture pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Find people whose posture matches the query.
    """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        Returns the list of annotators to be used in this subtree.
        """
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
            "subtree_human_position_pipeline", memory=True
        )
        seq.add_children(
            [
                HumanAndPoseAnnotator(descriptor=human_descriptor),
                # posture from the body keypoints only; CLIP and keypoints together are the hybrid activity route
                HumanSittingAnnotator(),
                HumanStandingAnnotator(),
                HumanLyingAnnotator(),
                ClusterPositionAnnotator(),
                BeliefStateAnnotator(),
                GenerateHumanPostureResults(),
            ]
        )
        return seq
