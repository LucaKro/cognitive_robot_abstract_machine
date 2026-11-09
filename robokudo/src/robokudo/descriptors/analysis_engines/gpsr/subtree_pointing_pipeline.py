"""
The GPSR subtree that finds the persons pointing in the queried direction.
"""

from __future__ import annotations

# %% Imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.pointing import HumanPointingAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.gpsr.gpsr_results import GeneratePointingResult
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Finds persons pointing in the queried direction.
    """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        :param name: Name of this subtree.
        :return: The sequence that answers with the persons pointing in the queried direction.
        """
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True
        human_descriptor.parameters.generate_clouds_for_keypoints = True

        pointing_descriptor = HumanPointingAnnotator.Descriptor()
        pointing_descriptor.parameters.pointing_length_meters = 5
        pointing_descriptor.parameters.maximum_pointing_angle = 35

        sequence = py_trees.composites.Sequence(
            "subtree_pointing_pipeline", memory=True
        )
        sequence.add_children(
            [
                HumanAndPoseAnnotator(descriptor=human_descriptor),
                ClusterPositionAnnotator(),
                HumanPointingAnnotator(descriptor=pointing_descriptor),
                BeliefStateAnnotator(),
                GeneratePointingResult(),
            ]
        )
        return sequence
