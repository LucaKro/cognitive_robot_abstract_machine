"""
The GPSR subtree that finds the persons waving or raising the queried arm.
"""

from __future__ import annotations

# %% Imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.waving import HumanWavingAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.gpsr.gpsr_results import GenerateWavingRaisingResult
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Finds persons making the queried waving or arm raising gesture.
    """

    def __init__(
        self, attribute: str | None = None, *args: object, **kwargs: object
    ) -> None:
        """
        :param attribute: Attribute of the queried gesture.
        """
        super().__init__(*args, **kwargs)
        self.attribute = attribute
        """
        Attribute of the queried gesture.
        """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        :param name: Name of this subtree.
        :return: The sequence that answers with the persons showing the queried gesture.
        """
        # HumanAndPoseAnnotator defaults to False, the position of a person needs depth
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True

        sequence = py_trees.composites.Sequence(
            "subtree_waving_raising_pipeline", memory=True
        )
        sequence.add_children(
            [
                HumanAndPoseAnnotator(descriptor=human_descriptor),
                ClusterPositionAnnotator(),
                HumanWavingAnnotator(),
                BeliefStateAnnotator(),
                GenerateWavingRaisingResult(),
            ]
        )
        return sequence
