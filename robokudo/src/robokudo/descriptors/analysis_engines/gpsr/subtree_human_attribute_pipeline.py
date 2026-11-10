"""
The GPSR subtree that describes every detected person with a vision language model.
"""

from __future__ import annotations

# %% Imports
from py_trees.composites import Sequence

from robokudo.analysis_engine import SubtreeInterface
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.gpsr.check_for_human_hypothesis import CheckForHumanHypothesis
from robokudo.annotators.gpsr.gpsr_results import GenerateVisionLanguageResult
from robokudo.annotators.gpsr.vision_language_annotator import VisionLanguageAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator


# %% subtree


class Subtree(SubtreeInterface):
    """
    Asks a vision language model about every detected person and answers with the
    persons and their descriptions.
    """

    def __init__(
        self,
        question: str | None = None,
        system_prompt: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        :param question: The question asked about every person, or *None* for the
            default question of the :class:`VisionLanguageAnnotator`.
        :param system_prompt: The instructions given to the model, or *None* for the
            default instructions of the :class:`VisionLanguageAnnotator`.
        """
        super().__init__(*args, **kwargs)
        self.question = question
        """
        The question asked about every person, or *None* for the default question.
        """
        self.system_prompt = system_prompt
        """
        The instructions given to the model, or *None* for the default instructions.
        """

    def implementation(self, *args: object, **kwargs: object) -> Sequence:
        """
        :return: The sequence that describes every person and answers with the descriptions.
        """
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True

        vision_language_descriptor = VisionLanguageAnnotator.Descriptor()
        if self.question is not None:
            vision_language_descriptor.parameters.question = self.question
        if self.system_prompt is not None:
            vision_language_descriptor.parameters.system_prompt = self.system_prompt

        sequence = Sequence("subtree_human_attribute_pipeline", memory=True)
        sequence.add_children(
            [
                HumanAndPoseAnnotator(descriptor=human_descriptor),
                CheckForHumanHypothesis(),
                ClusterPositionAnnotator(),
                VisionLanguageAnnotator(descriptor=vision_language_descriptor),
                BeliefStateAnnotator(),
                GenerateVisionLanguageResult(),
            ]
        )
        return sequence
