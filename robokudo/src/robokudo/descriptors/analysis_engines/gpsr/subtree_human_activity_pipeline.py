"""
The GPSR subtree that detects the postures and gestures of persons.
"""

from __future__ import annotations

# %% Imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.pointing import HumanPointingAnnotator
from robokudo.annotators.gpsr.waving import HumanWavingAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.gpsr.gpsr_results import GenerateHumanActivityResult
from robokudo.annotators.gpsr.human_activity_annotator import HumanActivityAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.check_for_human_hypothesis import CheckForHumanHypothesis
from robokudo.annotators.gpsr.filter_human_by_activity_annotator import (
    FilterHumanByActivityAnnotator,
)


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Detects human activities and answers queries about them.
    """

    def __init__(
        self, with_filter: bool = False, *args: object, **kwargs: object
    ) -> None:
        """:param with_filter: Whether to filter by the queried activities."""
        super().__init__(*args, **kwargs)
        self.with_filter = with_filter
        """
        Whether to filter by the queried activities.
        """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        :param name: Name of this subtree.
        :return: The sequence that detects the activities of the persons and answers with them.
        """
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True
        human_descriptor.parameters.generate_clouds_for_keypoints = True

        waving_descriptor = HumanWavingAnnotator.Descriptor()
        waving_descriptor.parameters.check_activity = True

        pointing_descriptor = HumanPointingAnnotator.Descriptor()
        pointing_descriptor.parameters.pointing_length_meters = 5
        pointing_descriptor.parameters.known = True
        pointing_descriptor.parameters.minimum_arm_extension = 0.25

        sequence = py_trees.composites.Sequence(
            "subtree_human_activity_pipeline", memory=True
        )
        sequence.add_children(
            [
                HumanAndPoseAnnotator(descriptor=human_descriptor),
                CheckForHumanHypothesis(),
                HumanActivityAnnotator(),
                ClusterPositionAnnotator(),
                HumanWavingAnnotator(descriptor=waving_descriptor),
                HumanPointingAnnotator(descriptor=pointing_descriptor),
                BeliefStateAnnotator(),
            ]
        )
        if self.with_filter:
            sequence.add_child(FilterHumanByActivityAnnotator())
        sequence.add_child(GenerateHumanActivityResult())
        return sequence
