from __future__ import annotations

# %% imports
import py_trees
import robokudo.analysis_engine
import robokudo.annotators.query
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.pointing import HumanPointingAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.gpsr.gpsr_results import GeneratePointingResult
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Find people pointing in the queried direction.
    """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True
        human_descriptor.parameters.generate_clouds_for_keypoints = True

        human_pointing_descriptor = HumanPointingAnnotator.Descriptor()
        human_pointing_descriptor.parameters.pointing_length_meters = 5
        # To which degree the arm has to be raised higher to be considered pointing
        human_pointing_descriptor.parameters.max_pointing_angle = 35

        seq = py_trees.composites.Sequence("subtree_pointing_pipeline", memory=True)
        seq.add_children(
            [
                HumanAndPoseAnnotator(descriptor=human_descriptor),
                ClusterPositionAnnotator(),
                HumanPointingAnnotator(descriptor=human_pointing_descriptor),
                BeliefStateAnnotator(),
                GeneratePointingResult(),
            ]
        )
        return seq
