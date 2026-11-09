"""
The GPSR subtree that finds the persons with the queried posture.
"""

from __future__ import annotations

# %% Imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.gpsr.gpsr_results import GenerateHumanPostureResult
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.lying import HumanLyingAnnotator
from robokudo.annotators.gpsr.sitting import HumanSittingAnnotator
from robokudo.annotators.gpsr.standing import HumanStandingAnnotator


# %% posture pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Finds persons whose posture matches the query.
    """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        :param name: Name of this subtree.
        :return: The sequence that answers with the persons in the queried posture.
        """
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True

        sequence = py_trees.composites.Sequence(
            "subtree_human_posture_pipeline", memory=True
        )
        sequence.add_children(
            [
                HumanAndPoseAnnotator(descriptor=human_descriptor),
                # posture from the body keypoints only
                HumanSittingAnnotator(),
                HumanStandingAnnotator(),
                HumanLyingAnnotator(),
                ClusterPositionAnnotator(),
                BeliefStateAnnotator(),
                GenerateHumanPostureResult(),
            ]
        )
        return sequence
