"""
The GPSR subtree that detects objects and answers with them.
"""

from __future__ import annotations

# %% Imports
import py_trees
import robokudo.analysis_engine
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.descriptors.analysis_engines.gpsr.subtree_object_detection import (
    Subtree as ObjectDetectionSubtree,
)
from robokudo.annotators.query import GenerateQueryResult


# %% pipeline
class Subtree(robokudo.analysis_engine.SubtreeInterface):
    """
    Detects objects and answers the query with the observed hypotheses.
    """

    def implementation(
        self, name: str | None = None, *args: object, **kwargs: object
    ) -> py_trees.composites.Sequence:
        """
        :param name: Name of this subtree.
        :return: The sequence that detects objects and answers with them.
        """
        sequence = py_trees.composites.Sequence(
            "subtree_object_detection_pipeline", memory=True
        )
        sequence.add_children(
            [
                ObjectDetectionSubtree(
                    model="ultimate_model_v11_1.26.4",
                    with_semantic_filter=True,
                ).implementation(),
                BeliefStateAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return sequence
